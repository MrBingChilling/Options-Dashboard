from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pandas as pd

from src.marketdata_client import MarketDataClient


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
