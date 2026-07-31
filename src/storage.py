from __future__ import annotations

from typing import Any

import pandas as pd
import requests


class SnapshotStoreError(RuntimeError):
    pass


class SnapshotStore:
    def __init__(self, url: str | None, secret_key: str | None, timeout: int = 30) -> None:
        self.url = (url or "").rstrip("/")
        self.key = secret_key or ""
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.key)

    @property
    def endpoint(self) -> str:
        return f"{self.url}/rest/v1/options_snapshots"

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "apikey": self.key,
            "Content-Type": "application/json",
        }
        # New sb_secret_* keys are not JWTs and must not be sent as Bearer
        # tokens. Keep legacy service_role JWTs working during migration.
        if self.key and not self.key.startswith(("sb_secret_", "sb_publishable_")):
            headers["Authorization"] = f"Bearer {self.key}"
        return headers

    def save(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            raise SnapshotStoreError("Supabase is not configured.")
        response = requests.post(
            self.endpoint,
            params={"on_conflict": "symbol,snapshot_date,min_dte,max_dte,assumption"},
            headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
            json=record,
            timeout=self.timeout,
        )
        if response.status_code not in {200, 201, 204}:
            raise SnapshotStoreError(
                f"Supabase save failed ({response.status_code}): {response.text[:300]}"
            )

    def history(
        self,
        symbol: str,
        assumption: str,
        min_dte: int,
        max_dte: int,
        limit: int = 1000,
    ) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        params = {
            "select": (
                "snapshot_date,spot,net_gex,gross_gex,gamma_flip,flip_distance_pct,"
                "call_wall,put_wall,call_open_interest,put_open_interest,"
                "put_call_oi_ratio,net_delta_exposure,contract_count"
            ),
            "symbol": f"eq.{symbol.upper()}",
            "assumption": f"eq.{assumption}",
            "min_dte": f"eq.{min_dte}",
            "max_dte": f"eq.{max_dte}",
            "order": "snapshot_date.asc",
            "limit": str(limit),
        }
        response = requests.get(
            self.endpoint,
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise SnapshotStoreError(
                f"Supabase history load failed ({response.status_code}): {response.text[:300]}"
            )
        frame = pd.DataFrame(response.json())
        if not frame.empty:
            frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"])
        return frame
