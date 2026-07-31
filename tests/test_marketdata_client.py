import pandas as pd

from src.marketdata_client import MarketDataClient


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
