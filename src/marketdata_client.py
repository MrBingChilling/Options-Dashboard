from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from src.expiration_filters import (
    ExpirationSelection,
    custom_expiration_selection,
    resolve_expiration_filter,
)


EASTERN = ZoneInfo("America/New_York")
SKEW_MIN_DTE = 0
SKEW_MAX_DTE = 45
SKEW_STRIKE_LIMIT = 30
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


@dataclass(frozen=True)
class LatestPrice:
    price: float
    updated: datetime


@dataclass(frozen=True)
class ApiUsage:
    endpoint: str
    consumed: int | None
    remaining: int | None
    limit: int | None


class MarketDataClient:
    BASE_URL = "https://api.marketdata.app/v1"

    def __init__(self, token: str, timeout: int = 45) -> None:
        if not token:
            raise ValueError("A MarketData.app token is required.")
        self.timeout = timeout
        self.session = requests.Session()
        self.usage_events: list[ApiUsage] = []
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
        expiration_filter: str | None = None,
        delta_filter: str | None = None,
        range_filter: str | None = None,
        strike_limit: int | None = None,
    ) -> ChainResult:
        symbol = symbol.strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Enter a valid US ticker symbol.")

        # The IV & Skew page historically called fetch_chain(..., delta_filter="0.25")
        # directly. Preserve that manual-request call site while routing it through
        # the same bounded local-25D path as the automatic collector.
        if (
            delta_filter == "0.25"
            and expiration_filter is None
            and min_dte == SKEW_MIN_DTE
            and max_dte == SKEW_MAX_DTE
            and range_filter is None
            and strike_limit is None
        ):
            delta_filter = None
            range_filter = "otm"
            strike_limit = SKEW_STRIKE_LIMIT

        selection = (
            resolve_expiration_filter(expiration_filter)
            if expiration_filter
            else custom_expiration_selection(min_dte, max_dte)
        )

        candidate_date = analysis_date
        last_error: str | None = None
        for _ in range(5):
            try:
                payload = self._request_chain(
                    symbol,
                    candidate_date,
                    selection,
                    min_open_interest,
                    delta_filter=delta_filter,
                    range_filter=range_filter,
                    strike_limit=strike_limit,
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

    def fetch_skew_chain(
        self,
        symbol: str,
        analysis_date: date,
    ) -> ChainResult:
        """Fetch the bounded historical chain used for 1W/1M 25D skew.

        Do not use MarketData's delta filter here. Historical rows can have null
        vendor IV/Greeks, so the 25-delta contract is selected locally after IV
        and delta are reconstructed from quote prices.
        """
        return self.fetch_chain(
            symbol,
            analysis_date,
            min_dte=SKEW_MIN_DTE,
            max_dte=SKEW_MAX_DTE,
            min_open_interest=0,
            range_filter="otm",
            strike_limit=SKEW_STRIKE_LIMIT,
        )

    def _request_chain(
        self,
        symbol: str,
        snapshot_date: date,
        expiration_selection: ExpirationSelection,
        min_open_interest: int,
        delta_filter: str | None = None,
        range_filter: str | None = None,
        strike_limit: int | None = None,
    ) -> dict[str, Any]:
        params = {
            "date": snapshot_date.isoformat(),
            "minOpenInterest": min_open_interest,
            "nonstandard": "false",
        }
        params.update(expiration_selection.request_params(snapshot_date))
        if delta_filter:
            params["delta"] = delta_filter
        if range_filter:
            params["range"] = range_filter
        if strike_limit is not None:
            if strike_limit <= 0:
                raise ValueError("strike_limit must be positive.")
            params["strikeLimit"] = int(strike_limit)
        response = self.session.get(
            f"{self.BASE_URL}/options/chain/{symbol}/",
            params=params,
            timeout=self.timeout,
        )
        self._record_usage(response, "options chain")
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

    def fetch_candles(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        resolution: str = "D",
    ) -> pd.DataFrame:
        symbol = symbol.strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Enter a valid US ticker symbol.")
        if from_date > to_date:
            raise ValueError("Price-history start date must not be after its end date.")
        response = self.session.get(
            f"{self.BASE_URL}/stocks/candles/{resolution}/{symbol}/",
            params={
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "adjustsplits": "true",
            },
            timeout=self.timeout,
        )
        self._record_usage(response, "daily candles")
        if response.status_code not in {200, 203}:
            raise MarketDataError(
                f"MarketData.app price history returned HTTP {response.status_code}: "
                f"{self._error_detail(response)}"
            )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise MarketDataError("MarketData.app returned invalid price-history JSON.") from exc
        if payload.get("s") != "ok":
            raise MarketDataError(payload.get("errmsg") or "No price candles were found.")
        return self._payload_to_candles(payload)

    def fetch_latest_price(self, symbol: str) -> LatestPrice:
        """Return MarketData.app's real-time SmartMid stock price.

        This is deliberately separate from daily candles: the candle endpoint
        cannot provide a real-time in-progress OHLC bar.
        """
        symbol = symbol.strip().upper()
        if not symbol or not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Enter a valid US ticker symbol.")
        response = self.session.get(
            f"{self.BASE_URL}/stocks/prices/{symbol}/",
            params={"extended": "true"},
            timeout=self.timeout,
        )
        self._record_usage(response, "real-time stock price")
        if response.status_code not in {200, 203}:
            raise MarketDataError(
                f"MarketData.app real-time price returned HTTP {response.status_code}: "
                f"{self._error_detail(response)}"
            )
        try:
            payload = response.json()
        except requests.JSONDecodeError as exc:
            raise MarketDataError("MarketData.app returned invalid real-time price JSON.") from exc
        if payload.get("s") != "ok" or not payload.get("mid") or not payload.get("updated"):
            raise MarketDataError(payload.get("errmsg") or "No real-time stock price was found.")
        try:
            price = float(payload["mid"][0])
            updated = datetime.fromtimestamp(int(payload["updated"][0]), tz=EASTERN)
        except (IndexError, TypeError, ValueError) as exc:
            raise MarketDataError("The real-time stock-price response was malformed.") from exc
        return LatestPrice(price=price, updated=updated)

    def usage_summary(self) -> dict[str, Any]:
        consumed_values = [event.consumed for event in self.usage_events if event.consumed is not None]
        last_remaining = next(
            (event.remaining for event in reversed(self.usage_events) if event.remaining is not None),
            None,
        )
        last_limit = next(
            (event.limit for event in reversed(self.usage_events) if event.limit is not None),
            None,
        )
        return {
            "consumed": sum(consumed_values) if consumed_values else None,
            "remaining": last_remaining,
            "limit": last_limit,
            "events": [event.__dict__.copy() for event in self.usage_events],
        }

    @staticmethod
    def _payload_to_candles(payload: dict[str, Any]) -> pd.DataFrame:
        required = ("t", "o", "h", "l", "c", "v")
        if any(not isinstance(payload.get(column), list) for column in required):
            raise MarketDataError("The price-history response is missing OHLCV fields.")
        lengths = {len(payload[column]) for column in required}
        if len(lengths) != 1:
            raise MarketDataError("The price-history response contains misaligned fields.")
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(payload["t"], unit="s", utc=True)
                .tz_convert(EASTERN)
                .tz_localize(None),
                "open": pd.to_numeric(payload["o"], errors="coerce"),
                "high": pd.to_numeric(payload["h"], errors="coerce"),
                "low": pd.to_numeric(payload["l"], errors="coerce"),
                "close": pd.to_numeric(payload["c"], errors="coerce"),
                "volume": pd.to_numeric(payload["v"], errors="coerce"),
            }
        )
        frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
        return frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
            return str(payload.get("errmsg") or payload.get("message") or response.reason)
        except requests.JSONDecodeError:
            return response.text[:240] or response.reason

    def _record_usage(self, response: requests.Response, endpoint: str) -> None:
        headers = getattr(response, "headers", {})
        if not isinstance(headers, Mapping):
            return

        def header_int(name: str) -> int | None:
            raw = headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None

        consumed = header_int("X-Api-Ratelimit-Consumed")
        remaining = header_int("X-Api-Ratelimit-Remaining")
        limit = header_int("X-Api-Ratelimit-Limit")
        if consumed is not None or remaining is not None or limit is not None:
            self.usage_events.append(ApiUsage(endpoint, consumed, remaining, limit))

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
        frame = frame[frame["openInterest"] >= 0]
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
