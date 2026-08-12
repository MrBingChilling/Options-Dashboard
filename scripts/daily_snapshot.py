from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from src.analytics import ASSUMPTIONS, STANDARD, enrich_chain, gamma_curve, snapshot_record, strike_profile, summarize
from src.expiration_filters import EXPIRATION_FILTERS, custom_expiration_selection, resolve_expiration_filter
from src.marketdata_client import MarketDataClient
from src.storage import SnapshotStore
from src.volatility import TENORS, snapshot_from_chain
from src.volatility_storage import save_volatility_snapshot


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
    parser.add_argument(
        "--expiration-filter",
        default=os.getenv("EXPIRATION_FILTER", "").strip() or None,
        choices=EXPIRATION_FILTERS,
    )
    parser.add_argument("--dealer-call-weight", type=float, default=float(env_text("DEALER_CALL_WEIGHT", "-0.40")))
    parser.add_argument("--dealer-put-weight", type=float, default=float(env_text("DEALER_PUT_WEIGHT", "-0.70")))
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
        usage_start = len(client.usage_events)
        try:
            result = client.fetch_chain(
                symbol,
                requested_date,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                min_open_interest=args.min_open_interest,
                expiration_filter=args.expiration_filter,
            )
            selection = (
                resolve_expiration_filter(args.expiration_filter)
                if args.expiration_filter
                else custom_expiration_selection(args.min_dte, args.max_dte)
            )
            enriched = enrich_chain(
                result.data,
                args.assumption,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
                call_weight=args.dealer_call_weight,
                put_weight=args.dealer_put_weight,
            )
            curve = gamma_curve(
                enriched,
                args.assumption,
                risk_free_rate=args.risk_free_rate,
                dividend_yield=args.dividend_yield,
                call_weight=args.dealer_call_weight,
                put_weight=args.dealer_put_weight,
            )
            profile = strike_profile(enriched)
            summary = summarize(
                symbol,
                result.snapshot_date,
                enriched,
                curve,
                args.assumption,
                selection.min_dte,
                selection.max_dte,
                expiration_filter=selection.label,
                call_weight=args.dealer_call_weight,
                put_weight=args.dealer_put_weight,
            )
            store.save(snapshot_record(summary, profile))

            volatility_saved = 0
            for tenor, target_dte in TENORS.items():
                if target_dte < selection.min_dte or target_dte > selection.max_dte:
                    continue
                try:
                    volatility = snapshot_from_chain(
                        symbol,
                        result.data,
                        result.snapshot_date,
                        tenor,
                        target_dte=target_dte,
                    )
                    save_volatility_snapshot(store, volatility)
                    volatility_saved += 1
                except Exception as volatility_exc:
                    print(
                        f"IV/skew warning for {symbol} {tenor}: {volatility_exc}",
                        file=sys.stderr,
                    )

            try:
                candles = client.fetch_candles(symbol, result.snapshot_date, result.snapshot_date)
                store.save_candles(symbol, candles)
            except Exception as candle_exc:
                print(f"Price candle warning for {symbol}: {candle_exc}", file=sys.stderr)
            print(
                f"Saved {symbol} snapshot for {result.snapshot_date.isoformat()} "
                f"({len(enriched):,} contracts; {volatility_saved} IV/skew tenors)."
            )
        except Exception as exc:  # keep processing the remainder of the watchlist
            failures += 1
            print(f"Failed {symbol}: {exc}", file=sys.stderr)
        finally:
            symbol_usage = client.usage_events[usage_start:]
            if symbol_usage:
                detail = ", ".join(
                    f"{event.endpoint}={event.consumed if event.consumed is not None else '?'}"
                    for event in symbol_usage
                )
                consumed = [event.consumed for event in symbol_usage if event.consumed is not None]
                remaining = next(
                    (event.remaining for event in reversed(symbol_usage) if event.remaining is not None),
                    None,
                )
                total_text = str(sum(consumed)) if consumed else "unknown"
                remaining_text = f"; {remaining} remaining" if remaining is not None else ""
                print(f"MarketData credits for {symbol}: {total_text} ({detail}){remaining_text}.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
