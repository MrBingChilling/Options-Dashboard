from __future__ import annotations

from typing import Iterable

import pandas as pd
import requests

from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility import VolatilitySnapshot


VOLATILITY_TABLE = "volatility_snapshots"


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
    limit: int = 10000,
) -> pd.DataFrame:
    if not store.enabled:
        return pd.DataFrame()
    cleaned = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not cleaned:
        return pd.DataFrame()
    symbol_filter = ",".join(cleaned)
    params: dict[str, str] = {
        "select": (
            "symbol,snapshot_date,tenor,target_dte,actual_dte,expiration,spot,"
            "atm_iv,call_25d_iv,put_25d_iv,skew_25d"
        ),
        "symbol": f"in.({symbol_filter})",
        "tenor": f"eq.{tenor}",
        "order": "snapshot_date.asc,symbol.asc",
        "limit": str(limit),
    }
    date_terms: list[str] = []
    if start_date is not None:
        date_terms.append(f"snapshot_date.gte.{pd.Timestamp(start_date).date().isoformat()}")
    if end_date is not None:
        date_terms.append(f"snapshot_date.lte.{pd.Timestamp(end_date).date().isoformat()}")
    if len(date_terms) == 1:
        field, op, value = date_terms[0].split(".", 2)
        params[field] = f"{op}.{value}"
    elif date_terms:
        params["and"] = "(" + ",".join(date_terms) + ")"

    response = requests.get(endpoint(store), params=params, headers=store.headers, timeout=store.timeout)
    if response.status_code != 200:
        raise SnapshotStoreError(
            f"Supabase volatility history load failed ({response.status_code}): {response.text[:300]}"
        )
    frame = pd.DataFrame(response.json())
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
    return frame.sort_values(["snapshot_date", "symbol"]).reset_index(drop=True)


def latest_volatility(store: SnapshotStore, symbols: Iterable[str], tenor: str) -> pd.DataFrame:
    history = volatility_history(store, symbols, tenor)
    if history.empty:
        return history
    return (
        history.sort_values("snapshot_date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
        .reset_index(drop=True)
    )
