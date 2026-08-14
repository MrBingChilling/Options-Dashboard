from datetime import date

import pandas as pd

from src.chain_archive import CALCULATION_VERSION, prepare_chain_for_archive


def test_prepare_chain_for_archive_preserves_quotes_and_local_surface_fields():
    frame = pd.DataFrame(
        [
            {
                "optionSymbol": "XYZ260911C00100000",
                "expiration": pd.Timestamp("2026-09-11"),
                "dte": 28,
                "side": "call",
                "strike": 100.0,
                "underlyingPrice": 100.0,
                "bid": 3.9,
                "ask": 4.1,
                "mid": 4.0,
                "last": 4.0,
                "openInterest": 120,
                "volume": 15,
                "iv": 0.30,
                "delta": 0.52,
                "gamma": 0.05,
                "theta": -0.08,
                "vega": 0.11,
            },
            {
                "optionSymbol": "XYZ260911P00100000",
                "expiration": pd.Timestamp("2026-09-11"),
                "dte": 28,
                "side": "put",
                "strike": 100.0,
                "underlyingPrice": 100.0,
                "bid": 3.7,
                "ask": 3.9,
                "mid": 3.8,
                "last": 3.8,
                "openInterest": 140,
                "volume": 18,
                "iv": 0.31,
                "delta": -0.48,
                "gamma": None,
                "theta": -0.08,
                "vega": 0.11,
            },
        ]
    )

    archived = prepare_chain_for_archive("xyz", date(2026, 8, 14), frame)

    assert len(archived) == 2
    assert set(archived["symbol"]) == {"XYZ"}
    assert set(archived["calculation_version"]) == {CALCULATION_VERSION}
    assert archived["bid"].tolist() == [3.9, 3.7]
    assert archived["openInterest"].tolist() == [120, 140]
    assert archived["iv_used"].notna().all()
    assert archived["delta_used"].notna().all()
    assert archived["gamma_used"].notna().all()
    assert (archived["gamma_used"] > 0).all()
    assert set(archived["iv_source"]) == {"vendor"}
    assert set(archived["delta_source"]) == {"vendor"}
    assert set(archived["gamma_source"]) == {"black_scholes_from_iv"}
