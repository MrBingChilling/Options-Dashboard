from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import requests

from src.storage import SnapshotStore, SnapshotStoreError


SUMMARY_TABLE = "daily_ai_summaries"
GENERATOR_VERSION = "daily_ai_summary_llm_v1"


class SummaryNotReady(ValueError):
    """Raised when fully comparable daily sessions are not available."""


@dataclass(frozen=True)
class SummaryBullet:
    title: str
    body: str

    def record(self) -> dict[str, str]:
        return {"title": self.title, "body": self.body}


@dataclass(frozen=True)
class DailySummary:
    snapshot_date: date
    comparison_date: date
    symbol_count: int
    expected_symbol_count: int
    bullets: tuple[SummaryBullet, ...]
    bottom_line: str
    input_signature: str | None = None
    week_comparison_date: date | None = None
    month_comparison_date: date | None = None
    generator_version: str = GENERATOR_VERSION

    def record(self) -> dict[str, Any]:
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "comparison_date": self.comparison_date.isoformat(),
            "symbol_count": self.symbol_count,
            "expected_symbol_count": self.expected_symbol_count,
            "generator_version": self.generator_version,
            "summary": {
                "bullets": [bullet.record() for bullet in self.bullets],
                "bottom_line": self.bottom_line,
                "input_signature": self.input_signature,
                "comparison_dates": {
                    "1D": self.comparison_date.isoformat(),
                    **(
                        {"1W": self.week_comparison_date.isoformat()}
                        if self.week_comparison_date
                        else {}
                    ),
                    **(
                        {"1M": self.month_comparison_date.isoformat()}
                        if self.month_comparison_date
                        else {}
                    ),
                },
                "data_note": (
                    "Fresh model-written analysis generated only from saved spot and "
                    "volatility-surface data. It does not use news, event calendars or "
                    "observed option order flow."
                ),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DailySummary":
        payload = record.get("summary") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        bullets = tuple(
            SummaryBullet(str(item.get("title", "")), str(item.get("body", "")))
            for item in payload.get("bullets", [])
            if isinstance(item, dict)
        )
        comparison_dates = payload.get("comparison_dates") or {}

        def optional_date(key: str) -> date | None:
            value = comparison_dates.get(key)
            return date.fromisoformat(str(value)[:10]) if value else None

        return cls(
            snapshot_date=date.fromisoformat(str(record["snapshot_date"])[:10]),
            comparison_date=date.fromisoformat(str(record["comparison_date"])[:10]),
            symbol_count=int(record.get("symbol_count") or 0),
            expected_symbol_count=int(record.get("expected_symbol_count") or 0),
            bullets=bullets,
            bottom_line=str(payload.get("bottom_line", "")),
            input_signature=(
                str(payload["input_signature"])
                if payload.get("input_signature")
                else None
            ),
            week_comparison_date=optional_date("1W"),
            month_comparison_date=optional_date("1M"),
            generator_version=str(record.get("generator_version") or GENERATOR_VERSION),
        )


def summary_endpoint(store: SnapshotStore) -> str:
    return f"{store.url}/rest/v1/{SUMMARY_TABLE}"


def save_daily_summary(store: SnapshotStore, summary: DailySummary) -> None:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    response = requests.post(
        summary_endpoint(store),
        params={"on_conflict": "snapshot_date"},
        headers={**store.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=summary.record(),
        timeout=store.timeout,
    )
    if response.status_code not in {200, 201, 204}:
        raise SnapshotStoreError(
            f"Supabase daily-summary save failed ({response.status_code}): "
            f"{response.text[:300]}"
        )


def load_daily_summaries(store: SnapshotStore, limit: int = 90) -> list[DailySummary]:
    if not store.enabled:
        return []
    response = requests.get(
        summary_endpoint(store),
        params={
            "select": (
                "snapshot_date,comparison_date,symbol_count,expected_symbol_count,"
                "generator_version,summary,generated_at"
            ),
            "order": "snapshot_date.desc",
            "limit": str(max(1, int(limit))),
        },
        headers=store.headers,
        timeout=store.timeout,
    )
    if response.status_code != 200:
        raise SnapshotStoreError(
            f"Supabase daily-summary load failed ({response.status_code}): "
            f"{response.text[:300]}"
        )
    try:
        rows = response.json()
    except requests.JSONDecodeError as exc:
        raise SnapshotStoreError("Supabase daily-summary load returned invalid JSON.") from exc
    if not isinstance(rows, list):
        raise SnapshotStoreError("Supabase daily-summary load returned an invalid payload.")
    return [DailySummary.from_record(row) for row in rows if isinstance(row, dict)]
