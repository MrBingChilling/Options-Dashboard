from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import requests

from src.chain_archive import CALCULATION_VERSION as STORED_CALCULATION_VERSION
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS, skew_snapshots_from_chain
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import save_volatility_snapshots, volatility_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute saved 1W/1M IV and skew as true constant-maturity "
            "points using already archived option chains. No MarketData calls."
        )
    )
    parser.add_argument(
        "--days",
        type=int,
        default=45,
        help="Calendar-day lookback to scan for archived rows.",
    )
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--symbols", default="AUTO")
    return parser.parse_args()


def _symbols(raw: str) -> list[str]:
    if raw.strip().upper() in {"", "AUTO"}:
        return list(AUTO_SYMBOLS)
    values = [
        value.strip().upper()
        for value in raw.replace("\n", ",").replace(" ", ",").split(",")
        if value.strip()
    ]
    return list(dict.fromkeys(values))


def _store() -> SnapshotStore:
    return SnapshotStore(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SECRET_KEY", "")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def _download_archived_chain(store: SnapshotStore, archive_path: str) -> pd.DataFrame:
    cleaned = str(archive_path).strip().lstrip("/")
    if "/" not in cleaned:
        raise SnapshotStoreError(f"Invalid archive path: {archive_path}")
    bucket, object_path = cleaned.split("/", 1)
    headers = {"apikey": store.key, "Accept": "application/vnd.apache.parquet"}
    if store.key and not store.key.startswith(("sb_secret_", "sb_publishable_")):
        headers["Authorization"] = f"Bearer {store.key}"

    last_response = None
    for endpoint in (
        f"{store.url}/storage/v1/object/authenticated/{bucket}/{object_path}",
        f"{store.url}/storage/v1/object/{bucket}/{object_path}",
    ):
        response = requests.get(endpoint, headers=headers, timeout=max(store.timeout, 60))
        last_response = response
        if response.status_code == 200:
            try:
                frame = pd.read_parquet(BytesIO(response.content))
            except Exception as exc:
                raise SnapshotStoreError(
                    f"Archived chain could not be decoded: {exc}"
                ) from exc
            if "expiration" in frame.columns:
                frame["expiration"] = pd.to_datetime(
                    frame["expiration"], errors="coerce"
                )
            return frame
        if response.status_code not in {400, 401, 403, 404}:
            break

    status = last_response.status_code if last_response is not None else "unknown"
    text = last_response.text[:300] if last_response is not None else ""
    raise SnapshotStoreError(
        f"Supabase archived-chain download failed ({status}): {text}"
    )


def _archived_sessions(
    store: SnapshotStore,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for tenor in DAILY_TENORS:
        frame = volatility_history(
            store,
            symbols,
            tenor,
            start_date=start_date,
            end_date=end_date,
            limit=max(50000, len(symbols) * 200),
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    history = pd.concat(frames, ignore_index=True)
    history = history[history["archive_path"].notna()].copy()
    if history.empty:
        return history

    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"]).dt.date
    history["target_dte"] = pd.to_numeric(history["target_dte"], errors="coerce")
    history["actual_dte"] = pd.to_numeric(history["actual_dte"], errors="coerce")
    synthetic_expiration = pd.to_datetime(history["snapshot_date"]) + pd.to_timedelta(
        history["target_dte"], unit="D"
    )
    stored_expiration = pd.to_datetime(history["expiration"], errors="coerce")
    history = history[
        history["target_dte"].notna()
        & (
            history["actual_dte"].ne(history["target_dte"])
            | stored_expiration.ne(synthetic_expiration)
        )
    ].copy()
    if history.empty:
        return history

    return (
        history.sort_values(["snapshot_date", "symbol"])
        .drop_duplicates(["symbol", "snapshot_date", "archive_path"])
        [["symbol", "snapshot_date", "archive_path", "chain_contract_count"]]
        .reset_index(drop=True)
    )


def main() -> int:
    args = parse_args()
    if args.days < 1:
        print("--days must be at least 1.", file=sys.stderr)
        return 2

    store = _store()
    if not store.enabled:
        print("Supabase server credentials are not configured.", file=sys.stderr)
        return 2

    symbols = _symbols(args.symbols)
    end_date = args.end
    start_date = args.start or (end_date - timedelta(days=args.days))
    if start_date > end_date:
        print("--start must be on or before --end.", file=sys.stderr)
        return 2

    try:
        sessions = _archived_sessions(store, symbols, start_date, end_date)
    except SnapshotStoreError as exc:
        print(f"Could not read archived resume state: {exc}", file=sys.stderr)
        return 2

    if sessions.empty:
        print(
            "No archived rows need constant-maturity recomputation "
            f"for {start_date}..{end_date}.",
            flush=True,
        )
        return 0

    print(
        f"Recomputing {len(sessions)} archived ticker-days for "
        f"{start_date}..{end_date} with total-variance interpolation; "
        "MarketData requests: 0.",
        flush=True,
    )
    failures: list[str] = []
    saved = 0

    for index, row in enumerate(sessions.itertuples(index=False), start=1):
        symbol = str(row.symbol).upper()
        session_date = row.snapshot_date
        archive_path = str(row.archive_path)
        try:
            chain = _download_archived_chain(store, archive_path)
            snapshots = skew_snapshots_from_chain(symbol, session_date, chain)
            snapshots = [
                replace(
                    snapshot,
                    archive_path=archive_path,
                    chain_contract_count=(
                        int(row.chain_contract_count)
                        if pd.notna(row.chain_contract_count)
                        else len(chain)
                    ),
                    calculation_version=STORED_CALCULATION_VERSION,
                )
                for snapshot in snapshots
            ]
            save_volatility_snapshots(store, snapshots)
            saved += 1
            if index == 1 or index % 100 == 0 or index == len(sessions):
                print(
                    f"[{index}/{len(sessions)}] saved={saved} failures={len(failures)}",
                    flush=True,
                )
        except (SnapshotStoreError, ValueError, TypeError) as exc:
            failures.append(f"{session_date} {symbol}: {exc}")
            print(f"[failed] {session_date} {symbol}: {exc}", flush=True)

    print(
        f"Constant-maturity recompute complete: saved={saved}; "
        f"failures={len(failures)}; MarketData requests=0.",
        flush=True,
    )
    if failures:
        print("\n".join(failures[:50]), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
