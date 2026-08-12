from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import pandas as pd

from src.marketdata_client import MarketDataClient
from src.storage import SnapshotStore
from src.volatility import TENORS, snapshot_from_chain
from src.volatility_storage import save_volatility_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill constant-tenor ATM IV, 25-delta IV, and 25-delta skew history."
    )
    parser.add_argument("--symbols", default=os.getenv("WATCHLIST", "SPY,QQQ"))
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--tenors",
        default="1W,1M,3M,6M",
        help="Comma-separated subset of 1W,1M,3M,6M.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = SnapshotStore(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr)
        return 2
    if args.start > args.end:
        print("--start must be on or before --end.", file=sys.stderr)
        return 2

    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    tenors = [value.strip().upper() for value in args.tenors.split(",") if value.strip()]
    invalid = [value for value in tenors if value not in TENORS]
    if invalid:
        print(f"Unsupported tenors: {', '.join(invalid)}", file=sys.stderr)
        return 2

    client = MarketDataClient(token)
    failures = 0
    sessions = [timestamp.date() for timestamp in pd.bdate_range(args.start, args.end)]
    max_target = max(TENORS[tenor] for tenor in tenors)

    for symbol in symbols:
        seen_snapshot_dates: set[date] = set()
        for requested_date in sessions:
            try:
                result = client.fetch_chain(
                    symbol,
                    requested_date,
                    min_dte=0,
                    max_dte=max_target + 30,
                    min_open_interest=0,
                )
                if result.snapshot_date in seen_snapshot_dates:
                    continue
                seen_snapshot_dates.add(result.snapshot_date)
                saved = 0
                for tenor in tenors:
                    try:
                        snap = snapshot_from_chain(
                            symbol,
                            result.data,
                            result.snapshot_date,
                            tenor,
                        )
                        save_volatility_snapshot(store, snap)
                        saved += 1
                    except Exception as exc:
                        print(
                            f"Warning {symbol} {result.snapshot_date} {tenor}: {exc}",
                            file=sys.stderr,
                        )
                print(f"Saved {symbol} {result.snapshot_date}: {saved} tenor rows.")
            except Exception as exc:
                failures += 1
                print(f"Failed {symbol} {requested_date}: {exc}", file=sys.stderr)

    usage = client.usage_summary()
    if usage.get("consumed") is not None:
        print(
            f"MarketData credits reported: {usage['consumed']} consumed; "
            f"{usage.get('remaining', 'unknown')} remaining."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
