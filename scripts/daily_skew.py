from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.chain_archive import CALCULATION_VERSION, archive_chain
from src.collection_storage import save_collection_run_best_effort, usage_since
from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS, previous_weekday, skew_snapshots_from_chain
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import latest_volatility, save_volatility_snapshots


EASTERN = ZoneInfo("America/New_York")


def _store() -> SnapshotStore:
    return SnapshotStore(
        os.environ.get("SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SECRET_KEY", "")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def _symbols() -> list[str]:
    raw = os.environ.get("SKEW_SYMBOLS", "").strip()
    if not raw:
        return list(AUTO_SYMBOLS)
    requested = [
        value.strip().upper()
        for value in raw.replace("\n", ",").replace(" ", ",").split(",")
        if value.strip()
    ]
    return list(dict.fromkeys(requested))


def _completed_symbols(
    store: SnapshotStore,
    symbols: list[str],
    session_date,
) -> set[str]:
    complete: dict[str, set[str]] = {symbol: set() for symbol in symbols}

    for tenor in DAILY_TENORS:
        frame = latest_volatility(store, symbols, tenor)
        if frame.empty:
            continue

        work = frame.copy()
        work["snapshot_date"] = pd.to_datetime(
            work["snapshot_date"], errors="coerce"
        ).dt.date

        for symbol in work.loc[
            work["snapshot_date"] == session_date, "symbol"
        ].astype(str):
            complete.setdefault(symbol.upper(), set()).add(tenor)

    needed = set(DAILY_TENORS)
    return {symbol for symbol, tenors in complete.items() if needed.issubset(tenors)}


def _fetch_and_save(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    requested_date,
):
    print(
        f"[request] {symbol} date={requested_date} "
        "DTE=0..45 range=otm strikeLimit=30; local IV/delta; one request for 1W + 1M",
        flush=True,
    )

    usage_start = len(client.usage_events)
    result = None
    archive = None
    saved_rows = 0
    try:
        result = client.fetch_skew_chain(symbol, requested_date)

        vendor_iv = pd.to_numeric(result.data.get("iv"), errors="coerce")
        usable_vendor_iv = int((vendor_iv.notna() & (vendor_iv > 0)).sum())
        print(
            f"[chain] {symbol} rows={len(result.data)} "
            f"vendor_iv_usable={usable_vendor_iv}/{len(result.data)}",
            flush=True,
        )

        # Preserve the entire already-paid-for bounded chain before reducing it
        # to dashboard summaries. Reruns upsert the same immutable date/version path.
        archive = archive_chain(store, symbol, result.snapshot_date, result.data)
        snapshots = skew_snapshots_from_chain(
            symbol,
            result.snapshot_date,
            result.data,
        )
        snapshots = [
            replace(
                snapshot,
                archive_path=archive.path,
                chain_contract_count=archive.contract_count,
                calculation_version=CALCULATION_VERSION,
            )
            for snapshot in snapshots
        ]
        save_volatility_snapshots(store, snapshots)
        saved_rows = len(snapshots)

        consumed, remaining = usage_since(client, usage_start)
        save_collection_run_best_effort(
            store,
            {
                "collector": "daily_skew",
                "symbol": symbol.upper(),
                "requested_date": requested_date.isoformat(),
                "snapshot_date": result.snapshot_date.isoformat(),
                "status": "saved",
                "contract_count": len(result.data),
                "archive_path": archive.path,
                "archive_bytes": archive.byte_count,
                "summary_rows_saved": saved_rows,
                "api_credits_consumed": consumed,
                "api_credits_remaining": remaining,
                "min_dte": 0,
                "max_dte": 45,
                "range_filter": "otm",
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
            },
        )

        print(
            f"[saved] {symbol} actual_session={result.snapshot_date} "
            f"rows={len(snapshots)} archive={archive.path} bytes={archive.byte_count}",
            flush=True,
        )
        return result.snapshot_date
    except (MarketDataError, SnapshotStoreError, ValueError) as exc:
        consumed, remaining = usage_since(client, usage_start)
        save_collection_run_best_effort(
            store,
            {
                "collector": "daily_skew",
                "symbol": symbol.upper(),
                "requested_date": requested_date.isoformat(),
                "snapshot_date": result.snapshot_date.isoformat() if result is not None else None,
                "status": "failed" if archive is None else "archive_saved_summary_failed",
                "contract_count": len(result.data) if result is not None else None,
                "archive_path": archive.path if archive is not None else None,
                "archive_bytes": archive.byte_count if archive is not None else None,
                "summary_rows_saved": saved_rows,
                "api_credits_consumed": consumed,
                "api_credits_remaining": remaining,
                "min_dte": 0,
                "max_dte": 45,
                "range_filter": "otm",
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
                "error": str(exc)[:2000],
            },
        )
        raise


def main() -> int:
    token = os.environ.get("MARKETDATA_TOKEN", "")
    if not token:
        raise SystemExit("MARKETDATA_TOKEN is not configured.")

    store = _store()
    if not store.enabled:
        raise SystemExit("Supabase is not configured.")

    client = MarketDataClient(token)
    symbols = _symbols()
    if not symbols:
        raise SystemExit("No skew symbols were configured.")
    requested_date = previous_weekday(datetime.now(EASTERN).date())

    print(
        f"Daily skew collector: {len(symbols)} symbols; "
        f"requested session={requested_date}",
        flush=True,
    )

    try:
        completed = _completed_symbols(store, symbols, requested_date)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not read Supabase before requesting data: {exc}")

    if len(completed) == len(symbols):
        print(
            "All configured symbols already have 1W + 1M rows for "
            f"{requested_date}. MarketData requests: 0.",
            flush=True,
        )
        return 0

    probe = "SPY" if "SPY" in symbols else symbols[0]
    actual_session = requested_date
    attempted: set[str] = set()
    successes = 0
    failures: list[str] = []

    if probe not in completed:
        attempted.add(probe)
        try:
            actual_session = _fetch_and_save(client, store, probe, requested_date)
            successes += 1
        except (MarketDataError, SnapshotStoreError, ValueError) as exc:
            failures.append(f"{probe}: {exc}")
            print(f"[failed] {probe}: {exc}", flush=True)

    if actual_session != requested_date:
        print(
            f"Requested {requested_date}, but MarketData returned "
            f"{actual_session}. Re-checking Supabase before any more requests.",
            flush=True,
        )

    try:
        completed = _completed_symbols(store, symbols, actual_session)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not re-check Supabase: {exc}")

    missing = [
        symbol
        for symbol in symbols
        if symbol not in completed and symbol not in attempted
    ]

    if not missing:
        if successes == 0 and failures:
            print("No skew rows were saved.", flush=True)
            return 1
        print(
            f"{actual_session} is complete after the probe. "
            "No further MarketData requests are needed.",
            flush=True,
        )
        return 0

    for index, symbol in enumerate(missing, start=1):
        attempted.add(symbol)
        print(f"[{index}/{len(missing)}] collecting {symbol}", flush=True)
        try:
            _fetch_and_save(client, store, symbol, actual_session)
            successes += 1
        except (MarketDataError, SnapshotStoreError, ValueError) as exc:
            failures.append(f"{symbol}: {exc}")
            print(f"[failed] {symbol}: {exc}", flush=True)

    print(
        f"Finished. Successful symbols: {successes}. "
        f"Failed symbols: {len(failures)}.",
        flush=True,
    )
    if failures:
        print("\n".join(failures), flush=True)

    return 1 if successes == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
