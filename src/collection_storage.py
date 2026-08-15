from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any, Iterable

import requests

from src.marketdata_client import ApiUsage, MarketDataClient
from src.storage import SnapshotStore, SnapshotStoreError


COLLECTION_RUNS_TABLE = "collection_runs"


def usage_since(client: MarketDataClient, start_index: int) -> tuple[int | None, int | None]:
    events: Iterable[ApiUsage] = client.usage_events[start_index:]
    consumed_values = [event.consumed for event in events if event.consumed is not None]
    remaining = next(
        (event.remaining for event in reversed(client.usage_events[start_index:]) if event.remaining is not None),
        None,
    )
    return (sum(consumed_values) if consumed_values else None, remaining)


def _collection_rows(
    store: SnapshotStore,
    collector: str,
    requested_date: date,
    select: str,
    *,
    order: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    params: dict[str, str | int] = {
        "select": select,
        "collector": f"eq.{collector}",
        "requested_date": f"eq.{requested_date.isoformat()}",
    }
    if order:
        params["order"] = order
    if limit is not None:
        params["limit"] = int(limit)
    response = requests.get(
        f"{store.url}/rest/v1/{COLLECTION_RUNS_TABLE}",
        headers=store.headers,
        params=params,
        timeout=store.timeout,
    )
    if response.status_code != 200:
        raise SnapshotStoreError(
            f"Supabase collection audit read failed ({response.status_code}): "
            f"{response.text[:300]}"
        )
    try:
        rows = response.json()
    except requests.JSONDecodeError as exc:
        raise SnapshotStoreError("Supabase collection audit returned invalid JSON.") from exc
    if not isinstance(rows, list):
        raise SnapshotStoreError("Supabase collection audit returned an invalid payload.")
    return [row for row in rows if isinstance(row, dict)]


def credits_consumed_for_requested_date(
    store: SnapshotStore,
    collector: str,
    requested_date: date,
) -> int:
    """Return credits already logged for one collector/session across retries."""
    rows = _collection_rows(
        store,
        collector,
        requested_date,
        "api_credits_consumed",
    )
    total = 0
    for row in rows:
        value = row.get("api_credits_consumed")
        try:
            if value is not None:
                total += max(int(value), 0)
        except (TypeError, ValueError):
            continue
    return total


def resolved_snapshot_date_for_requested_date(
    store: SnapshotStore,
    collector: str,
    requested_date: date,
) -> date | None:
    """Reuse a provider-resolved session without locking in a weekend stale row.

    MarketData historical option chains are one trading day old. A weekend probe
    for Friday can therefore return Thursday even though Friday is a valid session.
    Ignore that mismatch so Monday can retry Friday after the next rollover. A
    mismatch first observed on a weekday is retained for genuine market holidays.
    """
    rows = _collection_rows(
        store,
        collector,
        requested_date,
        "snapshot_date,status,created_at",
        order="created_at.desc",
        limit=20,
    )
    for row in rows:
        if not str(row.get("status", "")).startswith("saved"):
            continue
        raw = row.get("snapshot_date")
        if not raw:
            continue
        try:
            resolved = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if resolved == requested_date:
            return resolved
        if resolved > requested_date:
            continue

        created_at = row.get("created_at")
        if not created_at:
            continue
        try:
            created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if created.weekday() < 5:
            return resolved
    return None


def save_collection_run(store: SnapshotStore, record: dict[str, Any]) -> None:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    payload = {
        "github_run_id": os.getenv("GITHUB_RUN_ID") or None,
        **record,
    }
    response = requests.post(
        f"{store.url}/rest/v1/{COLLECTION_RUNS_TABLE}",
        headers={**store.headers, "Prefer": "return=minimal"},
        json=payload,
        timeout=store.timeout,
    )
    if response.status_code not in {200, 201, 204}:
        raise SnapshotStoreError(
            f"Supabase collection audit save failed ({response.status_code}): "
            f"{response.text[:300]}"
        )


def save_collection_run_best_effort(store: SnapshotStore, record: dict[str, Any]) -> None:
    try:
        save_collection_run(store, record)
    except SnapshotStoreError as exc:
        print(f"[audit-warning] {exc}", flush=True)
