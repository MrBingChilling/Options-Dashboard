from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd

from src.marketdata_client import MarketDataClient
from src.expiration_filters import (
    FILTER_MONTHLY,
    FILTER_OVER_ONE_YEAR,
    custom_expiration_selection,
    resolve_expiration_filter,
)


EASTERN = ZoneInfo("America/New_York")


def test_parallel_api_payload_is_normalized() -> None:
    payload = {
        "s": "ok",
        "optionSymbol": ["XYZ260821C00100000", "XYZ260821P00100000"],
        "expiration": [1787342400, 1787342400],
        "side": ["call", "put"],
        "strike": [100, 100],
        "dte": [21, 21],
        "openInterest": [100, 200],
        "volume": [5, 6],
        "underlyingPrice": [101.25, 101.25],
        "iv": [0.25, 0.27],
        "delta": [0.55, -0.45],
        "gamma": [0.04, 0.04],
        "updated": [1785538800, 1785538800],
    }
    frame = MarketDataClient._payload_to_frame(payload)
    assert len(frame) == 2
    assert frame["strike"].dtype.kind in "fi"
    assert pd.api.types.is_datetime64_any_dtype(frame["expiration"])
    assert frame["openInterest"].sum() == 300


def test_http_402_retries_provider_latest_closed_session() -> None:
    latest_available = date(2026, 7, 30)
    expiration = int(datetime(2026, 8, 21, 16, tzinfo=EASTERN).timestamp())
    updated = int(datetime(2026, 7, 30, 20, tzinfo=EASTERN).timestamp())
    unavailable = Mock(status_code=402, reason="Payment Required", text="")
    unavailable.json.return_value = {
        "errmsg": (
            "Your plan can only access fully-closed sessions; "
            "the latest available is 2026-07-30."
        )
    }
    success = Mock(status_code=200, reason="OK", text="")
    success.json.return_value = {
        "s": "ok",
        "optionSymbol": ["SPY260821C00630000"],
        "expiration": [expiration],
        "side": ["call"],
        "strike": [630],
        "dte": [22],
        "openInterest": [100],
        "underlyingPrice": [632.08],
        "iv": [0.2],
        "delta": [0.52],
        "gamma": [0.03],
        "updated": [updated],
    }

    client = MarketDataClient("test-token")
    client.session.get = Mock(side_effect=[unavailable, success])

    result = client.fetch_chain("SPY", date(2026, 7, 31))

    assert result.requested_date == date(2026, 7, 31)
    assert result.snapshot_date == latest_available
    assert len(result.data) == 1
    assert client.session.get.call_count == 2
    assert client.session.get.call_args_list[0].kwargs["params"]["date"] == "2026-07-31"
    assert client.session.get.call_args_list[1].kwargs["params"]["date"] == "2026-07-30"


def test_monthly_filter_uses_provider_monthly_parameter() -> None:
    selection = resolve_expiration_filter(FILTER_MONTHLY)
    assert selection.request_params(date(2026, 7, 30)) == {
        "expiration": "all",
        "monthly": "true",
    }


def test_over_one_year_filter_has_open_ended_start_date() -> None:
    selection = resolve_expiration_filter(FILTER_OVER_ONE_YEAR)
    assert selection.request_params(date(2026, 7, 30)) == {"from": "2027-07-31"}


def test_custom_slider_can_select_one_exact_dte() -> None:
    selection = custom_expiration_selection(90, 90)
    assert selection.label == "Custom 90 DTE"
    assert selection.request_params(date(2026, 7, 30)) == {
        "from": "2026-10-28",
        "to": "2026-10-28",
    }


def test_stock_candles_payload_is_normalized() -> None:
    payload = {
        "s": "ok",
        "t": [1785369600, 1785456000],
        "o": [100.0, 101.0],
        "h": [102.0, 103.0],
        "l": [99.0, 100.0],
        "c": [101.0, 102.0],
        "v": [1000, 1100],
    }
    candles = MarketDataClient._payload_to_candles(payload)
    assert candles.columns.tolist() == ["time", "open", "high", "low", "close", "volume"]
    assert candles["close"].tolist() == [101.0, 102.0]


def test_latest_stock_price_and_credit_headers_are_normalized() -> None:
    updated = int(datetime(2026, 7, 31, 13, 15, tzinfo=EASTERN).timestamp())
    response = Mock(
        status_code=200,
        reason="OK",
        text="",
        headers={
            "X-Api-Ratelimit-Consumed": "1",
            "X-Api-Ratelimit-Remaining": "999",
            "X-Api-Ratelimit-Limit": "1000",
        },
    )
    response.json.return_value = {
        "s": "ok",
        "symbol": ["SPY"],
        "mid": [636.25],
        "updated": [updated],
    }
    client = MarketDataClient("test-token")
    client.session.get = Mock(return_value=response)

    latest = client.fetch_latest_price("SPY")

    assert latest.price == 636.25
    assert latest.updated == datetime(2026, 7, 31, 13, 15, tzinfo=EASTERN)
    assert client.usage_summary() == {
        "consumed": 1,
        "remaining": 999,
        "limit": 1000,
        "events": [
            {
                "endpoint": "real-time stock price",
                "consumed": 1,
                "remaining": 999,
                "limit": 1000,
            }
        ],
    }
