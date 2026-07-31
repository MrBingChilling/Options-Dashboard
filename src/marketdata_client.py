from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


EASTERN = ZoneInfo("America/New_York")
NUMERIC_COLUMNS = {
    "ask",
    "bid",
    "mid",
    "last",
    "strike",
    "dte",
    "volume",
    "openInterest",
    "underlyingPrice",
    "iv",
    "delta",
    "gamma",
    "theta",
    "vega",
}
REQUIRED_COLUMNS = {
    "optionSymbol",
    "expiration",
    "side",
    "strike",
    "dte",
    "openInterest",
    "underlyingPrice",
    "iv",
    "delta",
    "gamma",
}


class MarketDataError(RuntimeError):
    """Raised when MarketData.app cannot return a usable chain."""


class _LatestAvailableSession(MarketDataError):
    """Signals that an EOD-only plan must use an earlier closed session."""

    def __init__(self, message: str, latest_available: date) -> None:
        super().__init__(message)
        self.latest_available = latest_available


@dataclass(frozen=True)
class ChainResult:
    data: pd.DataFrame
    requested_date: date
    snapshot_date: date


class MarketDataClient:
    BASE_URL = "https://api.marketdata.app/v1"

    def __init__(self, token: str, timeout: int = 45) -> None:
        if not token:
            raise ValueError("A MarketData.app token is required.")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        )

    def fetch_chain(
        self,
        symbol: str,
        analysis_date: date,
        min_dte: int = 7,
        max_dte: int = 365,
        min_open_interest: int = 1,
    ) -> ChainResult:
        symbol = symbol.strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Enter a valid US ticker symbol.")
        if min_dte < 0 or max_dte <= min_dte:
            raise ValueError("Maximum DTE must be greater than minimum DTE.")

        candidate_date = analysis_date
        last_error: str | None = None
        for _ in range(5):
            try:
                payload = self._request_chain(
                    symbol,
                    candidate_date,
                    min_dte,
                    max_dte,
                    min_open_interest,
                )
            except _LatestAvailableSession as exc:
                last_error = str(exc)
                if exc.latest_available >= candidate_date:
                    break
                candidate_date = exc.latest_available
                continue
            status = payload.get("s")
            if status == "ok":
                frame = self._payload_to_frame(payload)
                if frame.empty:
                    raise MarketDataError("The chain response contained no usable contracts.")
                snapshot_date = self._snapshot_date(frame, candidate_date)
                return ChainResult(frame, analysis_date, snapshot_date)

            last_error = payload.get("errmsg") or "No chain data was found."
            previous = payload.get("prevTime")
            if previous is not None:
                candidate_date = datetime.fromtimestamp(int(previous), tz=EASTERN).date()
            else:
                candidate_date -= timedelta(days=1)

        raise MarketDataError(
            f"No usable end-of-day chain was found near {analysis_date.isoformat()}. "
            f"Provider response: {last_error}"
        )

    def _request_chain(
        self,
        symbol: str,
        snapshot_date: date,
        min_dte: int,
        max_dte: int,
        min_open_interest: int,
    ) -> dict[str, Any]:
        params = {
            "date": snapshot_date.isoformat(),
            "from": (snapshot_date + timedelta(days=min_dte)).isoformat(),
            "to": (snapshot_date + timedelta(days=max_dte)).isoformat(),
            "minOpenInterest": min_open_interest,
            "nonstandard": "false",
        }
        response = self.session.get(
            f"{self.BASE_URL}/options/chain/{symbol}/",
            params=params,
            timeout=self.timeout,
        )
        if response.status_code not in {200, 203}:
            detail = self._error_detail(response)
            latest_available = self._latest_available_date(detail)
            if response.status_code == 402 and latest_available is not None:
                raise _LatestAvailableSession(detail, latest_available)
            raise MarketDataError(f"MarketData.app returned HTTP {response.status_code}: {detail}")
        try:
            return response.json()
        except requests.JSONDecodeError as exc:
            raise MarketDataError("MarketData.app returned an invalid JSON response.") from exc

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
            return str(payload.get("errmsg") or payload.get("message") or response.reason)
        except requests.JSONDecodeError:
            return response.text[:240] or response.reason

    @staticmethod
    def _latest_available_date(message: str) -> date | None:
        match = re.search(
            r"latest\s+available(?:\s+session|\s+date)?\s+is\s+(\d{4}-\d{2}-\d{2})",
            message,
            flags=re.IGNORECASE,
        )
        if match is None:
            return None
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
        arrays = {
            key: value
            for key, value in payload.items()
            if isinstance(value, list)
        }
        if not arrays:
            return pd.DataFrame()

        row_count = max(len(values) for values in arrays.values())
        aligned = {
            key: values
            for key, values in arrays.items()
            if len(values) == row_count
        }
        frame = pd.DataFrame(aligned)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise MarketDataError(
                "The chain response is missing required fields: " + ", ".join(sorted(missing))
            )

        for column in NUMERIC_COLUMNS.intersection(frame.columns):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        frame["side"] = frame["side"].astype(str).str.lower()
        frame["expiration"] = MarketDataClient._parse_expiration(frame["expiration"])
        frame = frame[frame["side"].isin(["call", "put"])]
        frame = frame.dropna(
            subset=["strike", "openInterest", "underlyingPrice", "expiration"]
        )
        frame = frame[frame["openInterest"] > 0]
        return frame.reset_index(drop=True)

    @staticmethod
    def _parse_expiration(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            timestamps = pd.to_datetime(numeric[numeric_mask], unit="s", utc=True)
            parsed.loc[numeric_mask] = timestamps.dt.tz_convert(EASTERN).dt.tz_localize(None)
        text_mask = ~numeric_mask
        if text_mask.any():
            parsed.loc[text_mask] = pd.to_datetime(values[text_mask], errors="coerce")
        return parsed

    @staticmethod
    def _snapshot_date(frame: pd.DataFrame, fallback: date) -> date:
        if "updated" not in frame.columns:
            return fallback
        updated = pd.to_numeric(frame["updated"], errors="coerce").dropna()
        if updated.empty:
            return fallback
        return datetime.fromtimestamp(int(updated.max()), tz=EASTERN).date()
