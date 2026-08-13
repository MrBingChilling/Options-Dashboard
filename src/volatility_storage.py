from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests

from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility import VolatilitySnapshot


VOLATILITY_TABLE = "volatility_snapshots"
SUPABASE_PAGE_SIZE = 1000
DEFAULT_HISTORY_LIMIT = 50000


def endpoint(store: SnapshotStore) -> str:
    return f"{store.url}/rest/v1/{VOLATILITY_TABLE}"


def save_volatility_snapshot(store: SnapshotStore, snapshot: VolatilitySnapshot) -> None:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    response = requests.post(
        endpoint(store),
        params={"on_conflict": "symbol,snapshot_date,tenor"},
        headers={**store.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=snapshot.record(),
        timeout=store.timeout,
    )
    if response.status_code not in {200, 201, 204}:
        raise SnapshotStoreError(
            f"Supabase volatility save failed ({response.status_code}): {response.text[:300]}"
        )


def save_volatility_snapshots(
    store: SnapshotStore, snapshots: Iterable[VolatilitySnapshot]
) -> None:
    for snapshot in snapshots:
        save_volatility_snapshot(store, snapshot)


def volatility_history(
    store: SnapshotStore,
    symbols: Iterable[str],
    tenor: str,
    start_date=None,
    end_date=None,
    limit: int = DEFAULT_HISTORY_LIMIT,
    newest_first: bool = False,
) -> pd.DataFrame:
    if not store.enabled:
        return pd.DataFrame()
    cleaned = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not cleaned:
        return pd.DataFrame()

    requested_limit = max(0, int(limit))
    if requested_limit == 0:
        return pd.DataFrame()

    symbol_filter = ",".join(cleaned)
    base_params: dict[str, str] = {
        "select": (
            "symbol,snapshot_date,tenor,target_dte,actual_dte,expiration,spot,"
            "atm_iv,call_25d_iv,put_25d_iv,skew_25d"
        ),
        "symbol": f"in.({symbol_filter})",
        "tenor": f"eq.{tenor}",
        "order": (
            "snapshot_date.desc,symbol.asc"
            if newest_first
            else "snapshot_date.asc,symbol.asc"
        ),
    }
    date_terms: list[str] = []
    if start_date is not None:
        date_terms.append(f"snapshot_date.gte.{pd.Timestamp(start_date).date().isoformat()}")
    if end_date is not None:
        date_terms.append(f"snapshot_date.lte.{pd.Timestamp(end_date).date().isoformat()}")
    if len(date_terms) == 1:
        field, op, value = date_terms[0].split(".", 2)
        base_params[field] = f"{op}.{value}"
    elif date_terms:
        base_params["and"] = "(" + ",".join(date_terms) + ")"

    rows: list[dict] = []
    offset = 0
    while len(rows) < requested_limit:
        page_limit = min(SUPABASE_PAGE_SIZE, requested_limit - len(rows))
        params = {
            **base_params,
            "limit": str(page_limit),
            "offset": str(offset),
        }
        response = requests.get(
            endpoint(store), params=params, headers=store.headers, timeout=store.timeout
        )
        if response.status_code != 200:
            raise SnapshotStoreError(
                f"Supabase volatility history load failed ({response.status_code}): {response.text[:300]}"
            )
        batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)

    frame = pd.DataFrame(rows[:requested_limit])
    if frame.empty:
        return frame
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
    frame["expiration"] = pd.to_datetime(frame["expiration"])
    for column in (
        "target_dte",
        "actual_dte",
        "spot",
        "atm_iv",
        "call_25d_iv",
        "put_25d_iv",
        "skew_25d",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # A long-running backfill can insert rows while offset-based pagination is
    # in progress. That can make the same logical row appear on adjacent pages
    # even though the database key itself is unique. Deduplicate the read result
    # so charts always receive one symbol/date/tenor observation.
    frame = frame.drop_duplicates(
        subset=["symbol", "snapshot_date", "tenor"], keep="last"
    )
    return frame.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)


def latest_volatility(store: SnapshotStore, symbols: Iterable[str], tenor: str) -> pd.DataFrame:
    cleaned = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not cleaned:
        return pd.DataFrame()

    # Read newest rows first. The old implementation read ascending history and
    # then took the last row per symbol; once the table exceeded Supabase's
    # per-response row cap, that silently turned "latest" into the oldest ~1000
    # rows (for the 49-name basket this landed around Sep 2025).
    probe_limit = max(1000, len(cleaned) * 40)
    history = volatility_history(
        store,
        cleaned,
        tenor,
        limit=probe_limit,
        newest_first=True,
    )
    if history.empty:
        return history

    latest = (
        history.sort_values("snapshot_date")
        .groupby("symbol", as_index=False)
        .tail(1)
    )

    # If a stale/rare ticker was not present in the newest probe window, fetch
    # only those missing names so they do not prevent current names from loading.
    present = set(latest["symbol"].astype(str).str.upper())
    missing = [symbol for symbol in cleaned if symbol not in present]
    if missing:
        fallback = volatility_history(
            store,
            missing,
            tenor,
            limit=max(1000, len(missing) * 250),
            newest_first=True,
        )
        if not fallback.empty:
            fallback_latest = (
                fallback.sort_values("snapshot_date")
                .groupby("symbol", as_index=False)
                .tail(1)
            )
            latest = pd.concat([latest, fallback_latest], ignore_index=True)

    if latest.empty:
        return latest

    # Cross-sections should compare one market session, never mix yesterday with
    # months-old fallback rows. Show the newest saved session and omit symbols
    # that do not have that session yet.
    newest_session = pd.to_datetime(latest["snapshot_date"]).max()
    return (
        latest[pd.to_datetime(latest["snapshot_date"]) == newest_session]
        .sort_values("symbol")
        .reset_index(drop=True)
    )