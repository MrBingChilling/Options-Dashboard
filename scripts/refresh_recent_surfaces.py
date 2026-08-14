from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from datetime import date

import pandas as pd

from src.chain_archive import CALCULATION_VERSION, archive_chain
from src.collection_storage import save_collection_run_best_effort, usage_since
from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import AUTO_SYMBOLS, skew_snapshots_from_chain
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import save_volatility_snapshots


COLLECTOR = "surface_v3_refresh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force-refresh exact historical sessions with surface_v3_gex data."
    )
    parser.add_argument("--dates", nargs="+", type=date.fromisoformat, required=True)
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


def refresh_one(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    session_date: date,
) -> None:
    usage_start = len(client.usage_events)
    result = None
    archive = None
    saved_rows = 0
    try:
        result = client.fetch_surface_chain(symbol, session_date)
        if result.snapshot_date != session_date:
            raise ValueError(
                f"Requested exact session {session_date}, provider resolved to {result.snapshot_date}."
            )

        archive = archive_chain(store, symbol, session_date, result.data)
        snapshots = skew_snapshots_from_chain(symbol, session_date, result.data)
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
        charged = consumed if consumed is not None else 1
        save_collection_run_best_effort(
            store,
            {
                "collector": COLLECTOR,
                "symbol": symbol,
                "requested_date": session_date.isoformat(),
                "snapshot_date": session_date.isoformat(),
                "status": "saved",
                "contract_count": len(result.data),
                "archive_path": archive.path,
                "archive_bytes": archive.byte_count,
                "summary_rows_saved": saved_rows,
                "api_credits_consumed": charged,
                "api_credits_remaining": remaining,
                "min_dte": 0,
                "max_dte": 45,
                "range_filter": "all",
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
                "collection_tier": "full_surface",
            },
        )
        print(
            f"[saved] {session_date} {symbol}: contracts={len(result.data)} "
            f"credits={charged} archive={archive.path}",
            flush=True,
        )
    except (MarketDataError, SnapshotStoreError, ValueError) as exc:
        consumed, remaining = usage_since(client, usage_start)
        save_collection_run_best_effort(
            store,
            {
                "collector": COLLECTOR,
                "symbol": symbol,
                "requested_date": session_date.isoformat(),
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
                "range_filter": "all",
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
                "collection_tier": "full_surface",
                "error": str(exc)[:2000],
            },
        )
        raise


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = _store()
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr)
        return 2

    symbols = _symbols(args.symbols)
    dates = list(dict.fromkeys(args.dates))
    print(
        f"Forced surface_v3_gex refresh: {len(symbols)} symbols x {len(dates)} sessions "
        f"({', '.join(map(str, dates))}); DTE=0..45, range=all, strikeLimit=30.",
        flush=True,
    )

    client = MarketDataClient(token)
    failures: list[str] = []
    for session_date in dates:
        for index, symbol in enumerate(symbols, start=1):
            print(f"[{session_date} {index}/{len(symbols)}] refreshing {symbol}", flush=True)
            try:
                refresh_one(client, store, symbol, session_date)
            except (MarketDataError, SnapshotStoreError, ValueError) as exc:
                failures.append(f"{session_date} {symbol}: {exc}")
                print(f"[failed] {session_date} {symbol}: {exc}", flush=True)

    usage = client.usage_summary()
    print(
        f"Refresh complete: failures={len(failures)}; "
        f"reported_credits={usage.get('consumed', 'unknown')}; "
        f"provider_remaining={usage.get('remaining', 'unknown')}.",
        flush=True,
    )
    if failures:
        print("\n".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
