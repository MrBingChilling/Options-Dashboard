from __future__ import annotations

import os
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
