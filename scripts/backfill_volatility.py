from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

from src.marketdata_client import MarketDataClient
from src.storage import SnapshotStore
from src.volatility import TENORS, snapshot_from_chain
from src.volatility_storage import (
    save_volatility_snapshot,
    volatility_history,
)


DELTA_FILTER = "0.50,0.25"
DTE_BUFFER_BEFORE = 14
DTE_BUFFER_AFTER = 30


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
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1000,
        help="Maximum missing ticker-day MarketData requests in this run. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--credit-reserve",
        type=int,
        default=500,
        help="Stop before the next ticker-day when MarketData reports this many or fewer credits remaining. Use 0 to disable.",
    )
    return parser.parse_args()


def existing_tenors_by_date(
    store: SnapshotStore,
    symbol: str,
    tenors: list[str],
    start_date: date,
    end_date: date,
) -> dict[date, set[str]]:
    """Return already-saved tenors keyed by session date, without using MarketData."""
    completed: dict[date, set[str]] = {}
    for tenor in tenors:
        history = volatility_history(
            store,
            [symbol],
            tenor,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )
        if history.empty:
            continue
        for value in history["snapshot_date"].dropna():
            session_date = pd.Timestamp(value).date()
            completed.setdefault(session_date, set()).add(tenor)
    return completed


def _last_remaining_credits(client: MarketDataClient) -> int | None:
    for event in reversed(client.usage_events):
        if event.remaining is not None:
            return event.remaining
    return None


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = SnapshotStore(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr, flush=True)
        return 2
    if args.start > args.end:
        print("--start must be on or before --end.", file=sys.stderr, flush=True)
        return 2
    if args.max_requests < 0 or args.credit_reserve < 0:
        print("--max-requests and --credit-reserve must be zero or positive.", file=sys.stderr, flush=True)
        return 2

    symbols = [value.strip().upper() for value in args.symbols.split(",") if value.strip()]
    tenors = [value.strip().upper() for value in args.tenors.split(",") if value.strip()]
    invalid = [value for value in tenors if value not in TENORS]
    if invalid:
        print(f"Unsupported tenors: {', '.join(invalid)}", file=sys.stderr, flush=True)
        return 2
    if not symbols or not tenors:
        print("At least one symbol and one tenor are required.", file=sys.stderr, flush=True)
        return 2

    client = MarketDataClient(token)
    failures = 0
    requests_started = 0
    rows_saved = 0
    rows_skipped = 0
    sessions = [timestamp.date() for timestamp in pd.bdate_range(args.start, args.end)]
    selected_tenors = set(tenors)
    min_target = min(TENORS[tenor] for tenor in tenors)
    max_target = max(TENORS[tenor] for tenor in tenors)
    min_dte = max(0, min_target - DTE_BUFFER_BEFORE)
    max_dte = max_target + DTE_BUFFER_AFTER
    stop_reason: str | None = None

    print(
        "Backfill starting: "
        f"{len(symbols)} symbols × {len(sessions)} business dates; "
        f"tenors={','.join(tenors)}; DTE={min_dte}-{max_dte}; "
        f"delta filter={DELTA_FILTER}; max requests={args.max_requests or 'unlimited'}.",
        flush=True,
    )

    for symbol in symbols:
        print(f"[{symbol}] Reading already-saved Supabase rows...", flush=True)
        try:
            existing = existing_tenors_by_date(store, symbol, tenors, args.start, args.end)
        except Exception as exc:
            print(f"[{symbol}] Failed to read resume state: {exc}", file=sys.stderr, flush=True)
            return 2

        complete_dates = sum(1 for saved in existing.values() if selected_tenors.issubset(saved))
        print(
            f"[{symbol}] Resume state: {complete_dates} dates already complete; "
            "completed dates will not call MarketData.",
            flush=True,
        )

        seen_snapshot_dates: set[date] = set()
        for requested_date in sessions:
            already_saved = existing.get(requested_date, set())
            if selected_tenors.issubset(already_saved):
                rows_skipped += len(tenors)
                continue

            if args.max_requests and requests_started >= args.max_requests:
                stop_reason = f"Reached max-requests cap ({args.max_requests})."
                break

            remaining = _last_remaining_credits(client)
            if args.credit_reserve and remaining is not None and remaining <= args.credit_reserve:
                stop_reason = (
                    f"MarketData reported {remaining} credits remaining, at or below "
                    f"the {args.credit_reserve}-credit reserve."
                )
                break

            missing_hint = sorted(selected_tenors.difference(already_saved))
            print(
                f"[{symbol} {requested_date}] Loading narrow chain for missing "
                f"{','.join(missing_hint)}...",
                flush=True,
            )
            requests_started += 1
            try:
                result = client.fetch_chain(
                    symbol,
                    requested_date,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    min_open_interest=0,
                    delta_filter=DELTA_FILTER,
                )
                actual_date = result.snapshot_date
                if actual_date in seen_snapshot_dates:
                    print(
                        f"[{symbol} {requested_date}] Provider resolved to already-seen "
                        f"session {actual_date}; skipping.",
                        flush=True,
                    )
                    continue
                seen_snapshot_dates.add(actual_date)

                actual_saved = existing.get(actual_date, set())
                missing_tenors = [tenor for tenor in tenors if tenor not in actual_saved]
                if not missing_tenors:
                    rows_skipped += len(tenors)
                    print(
                        f"[{symbol} {actual_date}] Already complete after provider date resolution; "
                        "nothing to save.",
                        flush=True,
                    )
                    continue

                saved = 0
                for tenor in missing_tenors:
                    try:
                        snap = snapshot_from_chain(
                            symbol,
                            result.data,
                            actual_date,
                            tenor,
                        )
                        save_volatility_snapshot(store, snap)
                        existing.setdefault(actual_date, set()).add(tenor)
                        saved += 1
                        rows_saved += 1
                    except Exception as exc:
                        print(
                            f"Warning {symbol} {actual_date} {tenor}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                remaining = _last_remaining_credits(client)
                credit_text = f"; {remaining} credits remaining" if remaining is not None else ""
                print(
                    f"[{symbol} {actual_date}] Saved {saved}/{len(missing_tenors)} missing tenor rows"
                    f"{credit_text}.",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(f"Failed {symbol} {requested_date}: {exc}", file=sys.stderr, flush=True)

        if stop_reason:
            break

    if stop_reason:
        print(f"Backfill stopped safely: {stop_reason}", flush=True)
        print("Run the workflow again later; completed Supabase dates will be skipped.", flush=True)

    usage = client.usage_summary()
    if usage.get("consumed") is not None:
        print(
            f"MarketData credits reported: {usage['consumed']} consumed; "
            f"{usage.get('remaining', 'unknown')} remaining.",
            flush=True,
        )
    print(
        f"Backfill summary: {requests_started} missing ticker-day requests started; "
        f"{rows_saved} rows saved; {rows_skipped} already-saved rows skipped; "
        f"{failures} failures.",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
