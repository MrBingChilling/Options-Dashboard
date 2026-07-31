from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from src.analytics import ASSUMPTIONS, STANDARD, enrich_chain, gamma_curve, snapshot_record, strike_profile, summarize
from src.marketdata_client import MarketDataClient
from src.storage import SnapshotStore


EASTERN = ZoneInfo("America/New_York")


def env_text(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    return int(value) if value else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save end-of-day option positioning snapshots.")
    parser.add_argument("--symbols", default=env_text("WATCHLIST", "SPY,QQQ"))
    parser.add_argument("--min-dte", type=int, default=env_int("MIN_DTE", 7))
    parser.add_argument("--max-dte", type=int, default=env_int("MAX_DTE", 365))
    parser.add_argument("--min-open-interest", type=int, default=env_int("MIN_OPEN_INTEREST", 1))
    parser.add_argument("--assumption", default=env_text("DEALER_ASSUMPTION", STANDARD), choices=ASSUMPTIONS)
    parser.add_argument("--risk-free-rate", type=float, default=float(env_text("RISK_FREE_RATE", "0.04")))
    parser.add_argument("--dividend-yield", type=float, default=float(env_text("DIVIDEND_YIELD", "0.0")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = SnapshotStore(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr)
        return 2

    client = MarketDataClient(token)
    requested_date = datetime.now(EASTERN).date()
    failures = 0
    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    for symbol in symbols:
        try:
            result = client.fetch_chain(
                symbol,
                requested_date,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                min_open_interest=args.min_open_interest,
            )
            enriched = enrich_chain(
                result.data,
                args.assumption,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
            )
            curve = gamma_curve(
                enriched,
                args.assumption,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
            )
            profile = strike_profile(enriched)
            summary = summarize(
                symbol,
                result.snapshot_date,
                enriched,
                curve,
                args.assumption,
                args.min_dte,
                args.max_dte,
            )
            store.save(snapshot_record(summary, profile))
            print(f"Saved {symbol} snapshot for {result.snapshot_date.isoformat()} ({len(enriched):,} contracts).")
        except Exception as exc:  # keep processing the remainder of the watchlist
            failures += 1
            print(f"Failed {symbol}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
