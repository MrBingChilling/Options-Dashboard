from datetime import date, datetime
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from src.marketdata_client import MarketDataClient


EASTERN = ZoneInfo("America/New_York")


def test_fetch_chain_passes_delta_filter_to_marketdata():
    expiration = int(datetime(2026, 9, 11, 16, tzinfo=EASTERN).timestamp())
    updated = int(datetime(2026, 8, 12, 20, tzinfo=EASTERN).timestamp())
    response = Mock(status_code=200, reason="OK", text="", headers={})
    response.json.return_value = {
        "s": "ok",
        "optionSymbol": ["XYZ260911C00100000", "XYZ260911P00100000"],
        "expiration": [expiration, expiration],
        "side": ["call", "put"],
        "strike": [100, 100],
        "dte": [30, 30],
        "openInterest": [10, 10],
        "underlyingPrice": [100, 100],
        "iv": [0.30, 0.31],
        "delta": [0.50, -0.50],
        "gamma": [0.04, 0.04],
        "updated": [updated, updated],
    }
    client = MarketDataClient("test-token")
    client.session.get = Mock(return_value=response)

    client.fetch_chain(
        "XYZ",
        date(2026, 8, 12),
        min_dte=0,
        max_dte=210,
        min_open_interest=0,
        delta_filter="0.50,0.25",
    )

    params = client.session.get.call_args.kwargs["params"]
    assert params["delta"] == "0.50,0.25"
