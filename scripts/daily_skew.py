from __future__ import annotations

import os
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.chain_archive import CALCULATION_VERSION, archive_chain
from src.collection_storage import (
    credits_consumed_for_requested_date,
    resolved_snapshot_date_for_requested_date,
    save_collection_run_best_effort,
    usage_since,
)
from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import (
    AUTO_SYMBOLS,
    DAILY_TENORS,
    INDEX_SYMBOLS,
    MAG7_SYMBOLS,
    previous_weekday,
    skew_snapshots_from_chain,
)
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import latest_volatility, save_volatility_snapshots


EASTERN = ZoneInfo("America/New_York")
COLLECTOR = "daily_skew"
DEFAULT_DAILY_CREDIT_LIMIT = 99
FULL_SURFACE_CREDIT_RESERVE = 6


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


def _daily_credit_limit() -> int:
    raw = os.environ.get("DAILY_MARKETDATA_CREDIT_LIMIT", "").strip()
    try:
        configured = int(raw) if raw else DEFAULT_DAILY_CREDIT_LIMIT
    except ValueError:
        configured = DEFAULT_DAILY_CREDIT_LIMIT
    # Never let repository configuration turn the automatic task into a
    # 100+ credit job. The user can still make manual requests separately.
    return max(1, min(configured, 99))


def _priority_credit_reserve(symbol: str) -> int:
    """Conservative reserve for the cheaper OTM 25D request.

    Index ETFs carry many more short-dated expirations; Mag-7 names also tend to
    have denser chains. These reserves are used only to decide when to stop
    spending the optional GEX budget and switch to 25D-first mode.
    """
    symbol = symbol.upper()
    if symbol in INDEX_SYMBOLS:
        return 3
    if symbol in MAG7_SYMBOLS:
        return 2
    return 1


def _provider_remaining(client: MarketDataClient) -> int | None:
    for event in reversed(client.usage_events):
        if event.remaining is not None:
            return max(int(event.remaining), 0)
    return None


def _completed_symbols(
    store: SnapshotStore,
    symbols: list[str],
    session_date: date,
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


def _choose_collection_tier(
    symbol: str,
    later_missing: list[str],
    effective_credits_left: int,
) -> str:
    """Spend on GEX only while every later ticker retains a 25D reserve."""
    reserve_for_later_25d = sum(_priority_credit_reserve(item) for item in later_missing)
    if effective_credits_left >= FULL_SURFACE_CREDIT_RESERVE + reserve_for_later_25d:
        return "full_surface"
    return "priority_25d"


def _fetch_and_save(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    request_date: date,
    budget_date: date,
    collection_tier: str,
) -> tuple[date, int, int | None]:
    if collection_tier == "full_surface":
        range_filter = "all"
        fetcher = client.fetch_surface_chain
        request_description = "range=all (IV + GEX/volume)"
    elif collection_tier == "priority_25d":
        range_filter = "otm"
        fetcher = client.fetch_skew_chain
        request_description = "range=otm (25D priority fallback)"
    else:
        raise ValueError(f"Unknown collection tier: {collection_tier}")

    print(
        f"[request:{collection_tier}] {symbol} date={request_date} "
        f"DTE=0..45 {request_description} strikeLimit=30; local IV/delta/gamma",
        flush=True,
    )

    usage_start = len(client.usage_events)
    result = None
    archive = None
    saved_rows = 0
    try:
        result = fetcher(symbol, request_date)

        vendor_iv = pd.to_numeric(result.data.get("iv"), errors="coerce")
        usable_vendor_iv = int((vendor_iv.notna() & (vendor_iv > 0)).sum())
        print(
            f"[chain] {symbol} rows={len(result.data)} "
            f"vendor_iv_usable={usable_vendor_iv}/{len(result.data)}",
            flush=True,
        )

        # Archive every useful field from the paid-for chain. surface_v3_gex
        # also reconstructs missing IV/delta/gamma locally for future analytics.
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
        charged = consumed if consumed is not None else 1
        save_collection_run_best_effort(
            store,
            {
                "collector": COLLECTOR,
                "symbol": symbol.upper(),
                # Keep the original target date here so all backup runs share
                # one credit ledger even if MarketData resolves a holiday back.
                "requested_date": budget_date.isoformat(),
                "snapshot_date": result.snapshot_date.isoformat(),
                "status": "saved",
                "contract_count": len(result.data),
                "archive_path": archive.path,
                "archive_bytes": archive.byte_count,
                "summary_rows_saved": saved_rows,
                "api_credits_consumed": charged,
                "api_credits_remaining": remaining,
                "min_dte": 0,
                "max_dte": 45,
                "range_filter": range_filter,
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
                "collection_tier": collection_tier,
            },
        )

        print(
            f"[saved:{collection_tier}] {symbol} actual_session={result.snapshot_date} "
            f"rows={len(snapshots)} credits={charged} archive={archive.path} "
            f"bytes={archive.byte_count}",
            flush=True,
        )
        return result.snapshot_date, charged, remaining
    except (MarketDataError, SnapshotStoreError, ValueError) as exc:
        consumed, remaining = usage_since(client, usage_start)
        charged = consumed if consumed is not None else 0
        save_collection_run_best_effort(
            store,
            {
                "collector": COLLECTOR,
                "symbol": symbol.upper(),
                "requested_date": budget_date.isoformat(),
                "snapshot_date": result.snapshot_date.isoformat() if result is not None else None,
                "status": "failed" if archive is None else "archive_saved_summary_failed",
                "contract_count": len(result.data) if result is not None else None,
                "archive_path": archive.path if archive is not None else None,
                "archive_bytes": archive.byte_count if archive is not None else None,
                "summary_rows_saved": saved_rows,
                "api_credits_consumed": charged,
                "api_credits_remaining": remaining,
                "min_dte": 0,
                "max_dte": 45,
                "range_filter": range_filter,
                "strike_limit": 30,
                "calculation_version": CALCULATION_VERSION,
                "collection_tier": collection_tier,
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

    budget_date = previous_weekday(datetime.now(EASTERN).date())
    credit_limit = _daily_credit_limit()

    try:
        logged_credits = credits_consumed_for_requested_date(store, COLLECTOR, budget_date)
        resolved_date = resolved_snapshot_date_for_requested_date(store, COLLECTOR, budget_date)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not read the shared MarketData credit ledger: {exc}")

    session_date = resolved_date or budget_date
    print(
        f"Daily IV/GEX collector: {len(symbols)} symbols; target={budget_date}; "
        f"resolved_session={session_date}; hard_task_limit={credit_limit}; "
        f"already_logged={logged_credits}",
        flush=True,
    )

    try:
        completed = _completed_symbols(store, symbols, session_date)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not read Supabase before requesting data: {exc}")

    if len(completed) == len(symbols):
        print(
            "All configured symbols already have 1W + 1M 25D rows for "
            f"{session_date}. MarketData requests: 0.",
            flush=True,
        )
        return 0

    # SPY is kept first when it is missing so a stale/holiday requested date is
    # resolved before spending the rest of the daily budget.
    missing = [symbol for symbol in symbols if symbol not in completed]
    if "SPY" in missing:
        missing = ["SPY"] + [symbol for symbol in missing if symbol != "SPY"]

    credits_used = logged_credits
    successes = 0
    full_surface_saved = 0
    priority_saved = 0
    failures: list[str] = []
    index = 0

    while index < len(missing):
        symbol = missing[index]
        later_missing = missing[index + 1 :]
        task_left = max(credit_limit - credits_used, 0)
        provider_left = _provider_remaining(client)
        effective_left = min(task_left, provider_left) if provider_left is not None else task_left

        reserve_all_pending = _priority_credit_reserve(symbol) + sum(
            _priority_credit_reserve(item) for item in later_missing
        )
        if effective_left <= 0:
            failures.append(
                f"Budget exhausted before {symbol}; remaining tickers were not requested."
            )
            break
        if effective_left < reserve_all_pending:
            print(
                f"[priority-warning] only {effective_left} effective credits remain versus "
                f"a {reserve_all_pending}-credit 25D reserve. GEX is disabled for the rest "
                "of this run and every remaining request is 25D-priority.",
                flush=True,
            )

        tier = _choose_collection_tier(symbol, later_missing, effective_left)
        if effective_left < _priority_credit_reserve(symbol):
            failures.append(
                f"Insufficient reserved credits to safely request {symbol} 25D data."
            )
            break

        print(
            f"[{index + 1}/{len(missing)}] {symbol}: effective_credits_left={effective_left}; "
            f"reserve_for_later_25d={sum(_priority_credit_reserve(item) for item in later_missing)}; "
            f"tier={tier}",
            flush=True,
        )

        try:
            actual_date, charged, _ = _fetch_and_save(
                client,
                store,
                symbol,
                session_date,
                budget_date,
                tier,
            )
            credits_used += max(charged, 0)
            successes += 1
            if tier == "full_surface":
                full_surface_saved += 1
            else:
                priority_saved += 1

            # If the first successful request resolved the target to an earlier
            # closed session, re-check the database before continuing so no
            # already-saved ticker is paid for twice.
            if actual_date != session_date:
                session_date = actual_date
                print(
                    f"Provider resolved {budget_date} to {session_date}; refreshing resume state.",
                    flush=True,
                )
                completed = _completed_symbols(store, symbols, session_date)
                missing = [
                    item for item in missing[index + 1 :] if item not in completed
                ]
                index = 0
                continue
        except (MarketDataError, SnapshotStoreError, ValueError) as exc:
            failures.append(f"{symbol}: {exc}")
            print(f"[failed] {symbol}: {exc}", flush=True)
            try:
                # Failed HTTP responses may still consume credits. Re-read the
                # shared ledger so the next decision cannot accidentally ignore them.
                credits_used = max(
                    credits_used,
                    credits_consumed_for_requested_date(store, COLLECTOR, budget_date),
                )
            except SnapshotStoreError:
                pass
        index += 1

    print(
        f"Finished {session_date}. Successful symbols: {successes}; "
        f"full IV+GEX surfaces: {full_surface_saved}; 25D-priority fallbacks: {priority_saved}; "
        f"task credits logged/estimated: {credits_used}/{credit_limit}; failures: {len(failures)}.",
        flush=True,
    )
    if failures:
        print("\n".join(failures), flush=True)

    # A partial success returns 0 so the scheduled backup can retry only missing
    # 25D rows; a total failure remains visible as a failed action.
    return 1 if successes == 0 and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
