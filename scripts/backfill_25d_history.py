from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.easter import easter

from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS, skew_snapshots_from_chain
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import save_volatility_snapshots, volatility_history


EASTERN = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill 1W/1M 25-delta call IV, put IV, and skew using the same "
            "bounded local-IV/local-delta path as the daily collector."
        )
    )
    parser.add_argument(
        "--symbols",
        default="AUTO",
        help="Comma-separated tickers, or AUTO for the daily automatic basket.",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=9000,
        help="Maximum ticker-day MarketData requests in this run. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--credit-reserve",
        type=int,
        default=750,
        help=(
            "Stop when MarketData reports this many or fewer credits remaining. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--probe-symbol",
        default="SPY",
        help="Validate one historical ticker-day before starting the bulk run.",
    )
    return parser.parse_args()


def _store() -> SnapshotStore:
    return SnapshotStore(
        os.getenv("SUPABASE_URL", ""),
        os.getenv("SUPABASE_SECRET_KEY", "")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def _symbols(raw: str) -> list[str]:
    if raw.strip().upper() in {"", "AUTO"}:
        return list(AUTO_SYMBOLS)
    values = [
        value.strip().upper()
        for value in raw.replace("\n", ",").replace(" ", ",").split(",")
        if value.strip()
    ]
    return list(dict.fromkeys(values))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    value = date(year, month, 1)
    value += timedelta(days=(weekday - value.weekday()) % 7)
    return value + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        value = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        value = date(year, month + 1, 1) - timedelta(days=1)
    value -= timedelta(days=(value.weekday() - weekday) % 7)
    return value


def _observed_fixed(value: date) -> date:
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nyse_holidays(year: int) -> set[date]:
    return {
        _observed_fixed(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
        _nth_weekday(year, 2, 0, 3),  # Presidents Day
        easter(year) - timedelta(days=2),  # Good Friday
        _last_weekday(year, 5, 0),  # Memorial Day
        _observed_fixed(date(year, 6, 19)),  # Juneteenth
        _observed_fixed(date(year, 7, 4)),  # Independence Day
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed_fixed(date(year, 12, 25)),  # Christmas
    }


def _sessions(start_date: date, end_date: date) -> list[date]:
    holidays: set[date] = set()
    for year in range(start_date.year - 1, end_date.year + 2):
        holidays.update(_nyse_holidays(year))
    return [
        timestamp.date()
        for timestamp in pd.bdate_range(start_date, end_date)
        if timestamp.date() not in holidays
    ]


def _last_remaining_credits(client: MarketDataClient) -> int | None:
    for event in reversed(client.usage_events):
        if event.remaining is not None:
            return event.remaining
    return None


def _completed_dates(
    store: SnapshotStore,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, set[date]]:
    by_symbol: dict[str, dict[date, set[str]]] = {
        symbol: {} for symbol in symbols
    }
    for tenor in DAILY_TENORS:
        history = volatility_history(
            store,
            symbols,
            tenor,
            start_date=start_date,
            end_date=end_date,
            limit=50000,
        )
        if history.empty:
            continue
        for row in history[["symbol", "snapshot_date"]].itertuples(index=False):
            symbol = str(row.symbol).upper()
            session_date = pd.Timestamp(row.snapshot_date).date()
            by_symbol.setdefault(symbol, {}).setdefault(session_date, set()).add(tenor)

    needed = set(DAILY_TENORS)
    return {
        symbol: {
            session_date
            for session_date, tenors in dates.items()
            if needed.issubset(tenors)
        }
        for symbol, dates in by_symbol.items()
    }


def _collect_one(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    requested_date: date,
):
    result = client.fetch_skew_chain(symbol, requested_date)
    snapshots = skew_snapshots_from_chain(
        symbol,
        result.snapshot_date,
        result.data,
    )
    save_volatility_snapshots(store, snapshots)
    return result.snapshot_date, len(snapshots), len(result.data)


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = _store()
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr)
        return 2
    if args.start > args.end:
        print("--start must be on or before --end.", file=sys.stderr)
        return 2
    if args.max_requests < 0 or args.credit_reserve < 0:
        print("--max-requests and --credit-reserve must be zero or positive.", file=sys.stderr)
        return 2

    symbols = _symbols(args.symbols)
    if not symbols:
        print("No symbols were configured.", file=sys.stderr)
        return 2

    sessions = _sessions(args.start, args.end)
    if not sessions:
        print("No NYSE sessions were found in the requested range.", file=sys.stderr)
        return 2

    client = MarketDataClient(token)
    print(
        f"25D history backfill: {len(symbols)} symbols × {len(sessions)} NYSE sessions; "
        f"range={sessions[0]}..{sessions[-1]}; tenors=1W,1M; "
        f"max_requests={args.max_requests or 'unlimited'}; "
        f"credit_reserve={args.credit_reserve}.",
        flush=True,
    )
    print(
        "Request path: DTE=0..45, range=otm, strikeLimit=30; "
        "missing IV/delta derived locally from historical option prices.",
        flush=True,
    )

    try:
        completed = _completed_dates(store, symbols, args.start, args.end)
    except SnapshotStoreError as exc:
        print(f"Could not read Supabase resume state: {exc}", file=sys.stderr)
        return 2

    total_pairs = len(symbols) * len(sessions)
    complete_pairs = sum(
        1
        for symbol in symbols
        for session_date in sessions
        if session_date in completed.get(symbol, set())
    )
    print(
        f"Resume state: {complete_pairs}/{total_pairs} ticker-days already complete; "
        "those dates will use zero MarketData credits.",
        flush=True,
    )
    if complete_pairs == total_pairs:
        print("Backfill is already complete. MarketData requests: 0.", flush=True)
        return 0

    requests_started = 0
    rows_saved = 0
    failures = 0
    stop_reason: str | None = None

    probe_symbol = args.probe_symbol.strip().upper()
    if probe_symbol not in symbols:
        probe_symbol = "SPY" if "SPY" in symbols else symbols[0]
    probe_date = next(
        (
            session_date
            for session_date in sessions
            if session_date not in completed.get(probe_symbol, set())
        ),
        None,
    )

    if probe_date is not None:
        print(
            f"Safety probe: validating {probe_symbol} on oldest missing session {probe_date}...",
            flush=True,
        )
        requests_started += 1
        try:
            actual_date, saved, chain_rows = _collect_one(
                client, store, probe_symbol, probe_date
            )
            completed.setdefault(probe_symbol, set()).add(actual_date)
            rows_saved += saved
            print(
                f"Safety probe passed: requested={probe_date}, actual={actual_date}, "
                f"chain_rows={chain_rows}, saved={saved}.",
                flush=True,
            )
        except (MarketDataError, SnapshotStoreError, ValueError) as exc:
            print(
                f"Safety probe failed; bulk backfill aborted before spending more credits: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return 1

    # Oldest session first across every ticker. This protects the rolling one-year
    # trial window: the oldest available dates are captured before newer dates.
    for session_date in sessions:
        if stop_reason:
            break
        for symbol in symbols:
            if session_date in completed.get(symbol, set()):
                continue

            if args.max_requests and requests_started >= args.max_requests:
                stop_reason = f"Reached max-requests cap ({args.max_requests})."
                break

            remaining = _last_remaining_credits(client)
            if (
                args.credit_reserve
                and remaining is not None
                and remaining <= args.credit_reserve
            ):
                stop_reason = (
                    f"MarketData reports {remaining} credits remaining, at or below "
                    f"the {args.credit_reserve}-credit reserve."
                )
                break

            requests_started += 1
            if requests_started == 2 or requests_started % 50 == 0:
                print(
                    f"Progress: request {requests_started}; now {session_date} {symbol}; "
                    f"remaining_credits={remaining if remaining is not None else 'unknown'}.",
                    flush=True,
                )

            try:
                actual_date, saved, _ = _collect_one(
                    client, store, symbol, session_date
                )
                completed.setdefault(symbol, set()).add(actual_date)
                rows_saved += saved
            except (MarketDataError, SnapshotStoreError, ValueError) as exc:
                message = str(exc)
                if "HTTP 429" in message or "credit" in message.lower() and "limit" in message.lower():
                    stop_reason = f"Provider credit limit reached while requesting {symbol} {session_date}: {message}"
                    break
                failures += 1
                print(
                    f"Warning {symbol} {session_date}: {message}",
                    file=sys.stderr,
                    flush=True,
                )

    remaining_pairs = sum(
        1
        for symbol in symbols
        for session_date in sessions
        if session_date not in completed.get(symbol, set())
    )
    usage = client.usage_summary()
    if usage.get("consumed") is not None:
        print(
            f"MarketData reported this run: {usage['consumed']} credits consumed; "
            f"{usage.get('remaining', 'unknown')} remaining.",
            flush=True,
        )
    if stop_reason:
        print(f"Backfill paused safely: {stop_reason}", flush=True)
    print(
        f"Backfill summary: {requests_started} ticker-day requests started; "
        f"{rows_saved} volatility rows saved; {failures} warnings; "
        f"{remaining_pairs}/{total_pairs} ticker-days still missing.",
        flush=True,
    )
    if remaining_pairs:
        print(
            "Run again after the MarketData daily reset; completed Supabase dates are skipped.",
            flush=True,
        )
    else:
        print("One-year 25D call/put IV history is complete.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
