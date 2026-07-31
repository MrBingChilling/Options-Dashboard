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
    def candles_endpoint(self) -> str:
        return f"{self.url}/rest/v1/stock_candles"

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
            params={
                "on_conflict": (
                    "symbol,snapshot_date,expiration_filter,assumption,"
                    "dealer_call_weight,dealer_put_weight"
                )
            },
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
        expiration_filter: str | None = None,
        dealer_call_weight: float | None = None,
        dealer_put_weight: float | None = None,
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
            "order": "snapshot_date.asc",
            "limit": str(limit),
        }
        if expiration_filter:
            params["expiration_filter"] = f"eq.{expiration_filter}"
        else:
            params["min_dte"] = f"eq.{min_dte}"
            params["max_dte"] = f"eq.{max_dte}"
        if dealer_call_weight is not None:
            params["dealer_call_weight"] = f"eq.{dealer_call_weight:.4f}"
        if dealer_put_weight is not None:
            params["dealer_put_weight"] = f"eq.{dealer_put_weight:.4f}"
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

    def save_candles(self, symbol: str, candles: pd.DataFrame) -> None:
        if not self.enabled or candles.empty:
            return
        records = []
        for row in candles.itertuples(index=False):
            records.append(
                {
                    "symbol": symbol.upper(),
                    "session_date": pd.Timestamp(row.time).date().isoformat(),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                    "volume": int(row.volume) if pd.notna(row.volume) else None,
                }
            )
        for start in range(0, len(records), 500):
            response = requests.post(
                self.candles_endpoint,
                params={"on_conflict": "symbol,session_date"},
                headers={
                    **self.headers,
                    "Prefer": "resolution=merge-duplicates,return=minimal",
                },
                json=records[start : start + 500],
                timeout=self.timeout,
            )
            if response.status_code not in {200, 201, 204}:
                raise SnapshotStoreError(
                    f"Supabase candle save failed ({response.status_code}): "
                    f"{response.text[:300]}"
                )

    def price_history(
        self,
        symbol: str,
        start_date: Any | None = None,
        end_date: Any | None = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        if not self.enabled:
            return pd.DataFrame()
        params: dict[str, str] = {
            "select": "session_date,open,high,low,close,volume",
            "symbol": f"eq.{symbol.upper()}",
            "order": "session_date.asc",
            "limit": str(limit),
        }
        if start_date is not None:
            params["session_date"] = f"gte.{pd.Timestamp(start_date).date().isoformat()}"
        if end_date is not None:
            end_filter = f"lte.{pd.Timestamp(end_date).date().isoformat()}"
            if "session_date" in params:
                params["and"] = f"(session_date.{params.pop('session_date')},session_date.{end_filter})"
            else:
                params["session_date"] = end_filter
        response = requests.get(
            self.candles_endpoint,
            params=params,
            headers=self.headers,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise SnapshotStoreError(
                f"Supabase price-history load failed ({response.status_code}): "
                f"{response.text[:300]}"
            )
        frame = pd.DataFrame(response.json())
        if frame.empty:
            return frame
        frame = frame.rename(columns={"session_date": "time"})
        frame["time"] = pd.to_datetime(frame["time"])
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.sort_values("time").reset_index(drop=True)
