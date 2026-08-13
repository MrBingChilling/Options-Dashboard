from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.easter import easter

from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility import VolatilitySnapshot, snapshot_from_chain
from src.volatility_storage import save_volatility_snapshot, volatility_history

EASTERN = ZoneInfo("America/New_York")
UNAVAILABLE_PREFIX = "NA_"
OPTIONS_NOT_BEFORE = {
    "CBRS": date(2026, 5, 18),
    "SKHY": date(2026, 7, 14),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="AUTO")
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--credit-reserve", type=int, default=100)
    parser.add_argument("--probe-symbol", default="SPY")
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
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_fixed(date(year, 6, 19)),
        _observed_fixed(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_fixed(date(year, 12, 25)),
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


def _eligible(symbol: str, session_date: date) -> bool:
    first_option_date = OPTIONS_NOT_BEFORE.get(symbol.upper())
    return first_option_date is None or session_date >= first_option_date


def _last_remaining_credits(client: MarketDataClient) -> int | None:
    for event in reversed(client.usage_events):
        if event.remaining is not None:
            return event.remaining
    return None


def _marker_tenor(tenor: str) -> str:
    return f"{UNAVAILABLE_PREFIX}{tenor}"


def _attempted_tenors_by_date(
    store: SnapshotStore,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, dict[date, set[str]]]:
    attempted: dict[str, dict[date, set[str]]] = {symbol: {} for symbol in symbols}
    stored_tenors = list(DAILY_TENORS) + [_marker_tenor(t) for t in DAILY_TENORS]
    for stored_tenor in stored_tenors:
        history = volatility_history(
            store,
            symbols,
            stored_tenor,
            start_date=start_date,
            end_date=end_date,
            limit=50000,
        )
        if history.empty:
            continue
        logical_tenor = (
            stored_tenor[len(UNAVAILABLE_PREFIX):]
            if stored_tenor.startswith(UNAVAILABLE_PREFIX)
            else stored_tenor
        )
        for row in history[["symbol", "snapshot_date"]].itertuples(index=False):
            symbol = str(row.symbol).upper()
            session_date = pd.Timestamp(row.snapshot_date).date()
            attempted.setdefault(symbol, {}).setdefault(session_date, set()).add(logical_tenor)
    return attempted


def _save_marker(store: SnapshotStore, symbol: str, snapshot_date: date, tenor: str) -> None:
    marker = VolatilitySnapshot(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date,
        tenor=_marker_tenor(tenor),
        target_dte=int(DAILY_TENORS[tenor]),
        actual_dte=0,
        expiration=snapshot_date,
        spot=0.0,
        atm_iv=None,
        call_25d_iv=None,
        put_25d_iv=None,
        skew_25d=None,
    )
    save_volatility_snapshot(store, marker)


def _collect_one(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    requested_date: date,
    already_attempted: set[str],
):
    result = client.fetch_skew_chain(symbol, requested_date)
    actual_date = result.snapshot_date
    saved: list[str] = []
    unavailable: list[str] = []

    for tenor, target_dte in DAILY_TENORS.items():
        if tenor in already_attempted:
            continue
        try:
            snapshot = snapshot_from_chain(
                symbol,
                result.data,
                actual_date,
                tenor,
                target_dte=int(target_dte),
            )
            if (
                snapshot.call_25d_iv is None
                or snapshot.put_25d_iv is None
                or snapshot.skew_25d is None
            ):
                raise ValueError("no usable 25D call/put IV pair")
            clean = VolatilitySnapshot(
                symbol=snapshot.symbol,
                snapshot_date=snapshot.snapshot_date,
                tenor=snapshot.tenor,
                target_dte=snapshot.target_dte,
                actual_dte=snapshot.actual_dte,
                expiration=snapshot.expiration,
                spot=snapshot.spot,
                atm_iv=None,
                call_25d_iv=snapshot.call_25d_iv,
                put_25d_iv=snapshot.put_25d_iv,
                skew_25d=snapshot.skew_25d,
            )
            save_volatility_snapshot(store, clean)
            saved.append(tenor)
        except ValueError:
            _save_marker(store, symbol, actual_date, tenor)
            unavailable.append(tenor)

    return actual_date, saved, unavailable, len(result.data)


def _mark_404(
    store: SnapshotStore,
    symbol: str,
    requested_date: date,
    already_attempted: set[str],
) -> list[str]:
    marked: list[str] = []
    for tenor in DAILY_TENORS:
        if tenor not in already_attempted:
            _save_marker(store, symbol, requested_date, tenor)
            marked.append(tenor)
    return marked


def main() -> int:
    args = parse_args()
    token = os.getenv("MARKETDATA_TOKEN", "")
    store = _store()
    if not token or not store.enabled:
        print("Missing MARKETDATA_TOKEN or Supabase server credentials.", file=sys.stderr)
        return 2
    if args.start > args.end or args.max_requests < 0 or args.credit_reserve < 0:
        print("Invalid backfill arguments.", file=sys.stderr)
        return 2

    symbols = _symbols(args.symbols)
    sessions = _sessions(args.start, args.end)
    if not symbols or not sessions:
        print("No symbols or NYSE sessions were found.", file=sys.stderr)
        return 2

    client = MarketDataClient(token)
    eligible_pairs = [
        (session_date, symbol)
        for session_date in sessions
        for symbol in symbols
        if _eligible(symbol, session_date)
    ]
    skipped_pre_options = len(symbols) * len(sessions) - len(eligible_pairs)

    print(
        f"25D history backfill: {len(symbols)} symbols x {len(sessions)} NYSE sessions; "
        f"range={sessions[0]}..{sessions[-1]}; tenors=1W,1M; "
        f"max_requests={args.max_requests or 'unlimited'}; credit_reserve={args.credit_reserve}.",
        flush=True,
    )
    print(
        "Request path: DTE=0..45, range=otm, strikeLimit=30; missing IV/delta "
        "derived locally from historical option prices.",
        flush=True,
    )
    print(
        f"Pre-options history guard: {skipped_pre_options} ticker-days skipped "
        "without MarketData requests.",
        flush=True,
    )

    try:
        attempted = _attempted_tenors_by_date(store, symbols, args.start, args.end)
    except SnapshotStoreError as exc:
        print(f"Could not read Supabase resume state: {exc}", file=sys.stderr)
        return 2

    needed = set(DAILY_TENORS)
    total_pairs = len(eligible_pairs)
    complete_pairs = sum(
        1
        for session_date, symbol in eligible_pairs
        if needed.issubset(attempted.get(symbol, {}).get(session_date, set()))
    )
    print(
        f"Resume state: {complete_pairs}/{total_pairs} eligible ticker-days already "
        "saved or attempted; those dates will use zero MarketData credits.",
        flush=True,
    )
    if complete_pairs == total_pairs:
        print("Backfill is already complete. MarketData requests: 0.", flush=True)
        return 0

    requests_started = 0
    rows_saved = 0
    markers_saved = 0
    failures = 0
    stop_reason: str | None = None

    probe_symbol = args.probe_symbol.strip().upper()
    if probe_symbol not in symbols:
        probe_symbol = "SPY" if "SPY" in symbols else symbols[0]
    probe_date = next(
        (
            d for d in sessions
            if _eligible(probe_symbol, d)
            and not needed.issubset(attempted.get(probe_symbol, {}).get(d, set()))
        ),
        None,
    )

    if probe_date is not None:
        print(f"Safety probe: validating {probe_symbol} on {probe_date}...", flush=True)
        requests_started += 1
        prior = attempted.get(probe_symbol, {}).get(probe_date, set())
        try:
            actual_date, saved, unavailable, chain_rows = _collect_one(
                client, store, probe_symbol, probe_date, prior
            )
            attempted.setdefault(probe_symbol, {}).setdefault(actual_date, set()).update(
                saved + unavailable
            )
            rows_saved += len(saved)
            markers_saved += len(unavailable)
            if unavailable or not needed.issubset(attempted[probe_symbol][actual_date]):
                raise ValueError("SPY probe did not resolve both required tenors")
            print(
                f"Safety probe passed: requested={probe_date}, actual={actual_date}, "
                f"chain_rows={chain_rows}, saved={len(saved)}.",
                flush=True,
            )
        except (MarketDataError, SnapshotStoreError, ValueError) as exc:
            print(f"Safety probe failed; bulk backfill aborted: {exc}", file=sys.stderr)
            return 1

    for session_date in sessions:
        if stop_reason:
            break
        for symbol in symbols:
            if not _eligible(symbol, session_date):
                continue
            prior = attempted.get(symbol, {}).get(session_date, set())
            if needed.issubset(prior):
                continue
            if args.max_requests and requests_started >= args.max_requests:
                stop_reason = f"Reached max-requests cap ({args.max_requests})."
                break

            remaining = _last_remaining_credits(client)
            if args.credit_reserve and remaining is not None and remaining <= args.credit_reserve:
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
                actual_date, saved, unavailable, _ = _collect_one(
                    client, store, symbol, session_date, prior
                )
                attempted.setdefault(symbol, {}).setdefault(actual_date, set()).update(
                    saved + unavailable
                )
                rows_saved += len(saved)
                markers_saved += len(unavailable)
                if unavailable:
                    print(
                        f"Unavailable {symbol} {actual_date}: {','.join(unavailable)} marked; "
                        "future runs will not retry them.",
                        flush=True,
                    )
            except MarketDataError as exc:
                message = str(exc)
                if "HTTP 429" in message or (
                    "credit" in message.lower() and "limit" in message.lower()
                ):
                    stop_reason = f"Provider credit limit reached at {symbol} {session_date}: {message}"
                    break
                if "HTTP 404" in message:
                    try:
                        marked = _mark_404(store, symbol, session_date, prior)
                    except SnapshotStoreError as marker_exc:
                        print(f"Could not persist 404 marker: {marker_exc}", file=sys.stderr)
                        return 2
                    attempted.setdefault(symbol, {}).setdefault(session_date, set()).update(marked)
                    markers_saved += len(marked)
                    print(
                        f"Unavailable {symbol} {session_date}: provider 404; marked "
                        f"{','.join(marked)} so future runs will not retry.",
                        flush=True,
                    )
                    continue
                failures += 1
                print(f"Warning {symbol} {session_date}: {message}", file=sys.stderr, flush=True)
            except (SnapshotStoreError, ValueError) as exc:
                failures += 1
                print(f"Warning {symbol} {session_date}: {exc}", file=sys.stderr, flush=True)

    remaining_pairs = sum(
        1
        for session_date, symbol in eligible_pairs
        if not needed.issubset(attempted.get(symbol, {}).get(session_date, set()))
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
        f"Backfill summary: {requests_started} ticker-day requests; {rows_saved} IV rows; "
        f"{markers_saved} unavailable-tenor markers; {failures} warnings; "
        f"{remaining_pairs}/{total_pairs} eligible ticker-days still unattempted.",
        flush=True,
    )
    if remaining_pairs:
        print(
            "Run again after the MarketData daily reset; saved rows and unavailable "
            "markers are both skipped.",
            flush=True,
        )
    else:
        print("One-year 25D call/put IV history is complete where data exists.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
