from __future__ import annotations

"""Upgrade legacy 1W/1M rows with resumable surface_v3_gex archives.

The mature session, credit, marker, and local-IV logic remains in
backfill_25d_history.py. This wrapper changes the request path to the bounded
all-moneyness surface and changes resume state so legacy summary-only rows do
not prevent a paid full-format upgrade.
"""

from datetime import date

import pandas as pd
import requests

from src.chain_archive import CALCULATION_VERSION
from src.marketdata_client import MarketDataClient
from src.skew_collector import DAILY_TENORS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import volatility_history

import scripts.backfill_25d_history as legacy


_original_audit_save = legacy.save_collection_run_best_effort


def _surface_v3_audit(store, record):
    upgraded = {
        **record,
        "collector": "backfill_surface_v3",
        "range_filter": "all",
        "strike_limit": 30,
        "calculation_version": CALCULATION_VERSION,
        "collection_tier": "full_surface",
    }
    _original_audit_save(store, upgraded)


def _successful_full_surface_pairs(
    store: SnapshotStore,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> set[tuple[str, date]]:
    """Return paid ticker-days already archived by any surface_v3 full run."""
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")

    params = {
        "select": "symbol,requested_date,status,archive_path",
        "symbol": f"in.({','.join(symbols)})",
        "and": (
            f"(requested_date.gte.{start_date.isoformat()},"
            f"requested_date.lte.{end_date.isoformat()})"
        ),
        "calculation_version": f"eq.{CALCULATION_VERSION}",
        "collection_tier": "eq.full_surface",
        "status": "like.saved*",
        "archive_path": "not.is.null",
        "order": "requested_date.asc,symbol.asc",
    }

    pairs: set[tuple[str, date]] = set()
    offset = 0
    while True:
        response = requests.get(
            f"{store.url}/rest/v1/collection_runs",
            headers=store.headers,
            params={**params, "limit": 1000, "offset": offset},
            timeout=store.timeout,
        )
        if response.status_code != 200:
            raise SnapshotStoreError(
                "Supabase surface-v3 resume read failed "
                f"({response.status_code}): {response.text[:300]}"
            )
        try:
            batch = response.json()
        except requests.JSONDecodeError as exc:
            raise SnapshotStoreError(
                "Supabase surface-v3 resume read returned invalid JSON."
            ) from exc
        if not isinstance(batch, list):
            raise SnapshotStoreError(
                "Supabase surface-v3 resume read returned an invalid payload."
            )
        for row in batch:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol", "")).upper()
            raw_date = str(row.get("requested_date", ""))[:10]
            if symbol not in symbols or not raw_date:
                continue
            try:
                pairs.add((symbol, date.fromisoformat(raw_date)))
            except ValueError:
                continue
        if len(batch) < 1000:
            break
        offset += len(batch)
    return pairs


def _surface_v3_attempted_tenors_by_date(
    store: SnapshotStore,
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, dict[date, set[str]]]:
    """Ignore lean legacy rows; resume only from verified full v3 archives."""
    attempted: dict[str, dict[date, set[str]]] = {symbol: {} for symbol in symbols}

    # Normal case: both summary rows exist and point to the archived v3 chain.
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
        work = history[
            history["calculation_version"].eq(CALCULATION_VERSION)
            & history["archive_path"].notna()
        ]
        for row in work[["symbol", "snapshot_date"]].itertuples(index=False):
            symbol = str(row.symbol).upper()
            session_date = pd.Timestamp(row.snapshot_date).date()
            attempted.setdefault(symbol, {}).setdefault(session_date, set()).add(tenor)

    # Audit fallback covers a successfully archived chain whose 1W or 1M 25D
    # pair was unavailable and therefore stored as an NA marker.
    for symbol, session_date in _successful_full_surface_pairs(
        store, symbols, start_date, end_date
    ):
        attempted.setdefault(symbol, {}).setdefault(session_date, set()).update(
            DAILY_TENORS
        )

    return attempted


def main() -> int:
    legacy.MarketDataClient.fetch_skew_chain = MarketDataClient.fetch_surface_chain
    legacy.save_collection_run_best_effort = _surface_v3_audit
    legacy._attempted_tenors_by_date = _surface_v3_attempted_tenors_by_date
    print(
        "surface_v3_gex upgrade enabled: DTE=0..45, range=all, strikeLimit=30; "
        "legacy summary-only rows are upgraded, while completed v3 archives are skipped.",
        flush=True,
    )
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
