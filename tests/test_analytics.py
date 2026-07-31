from datetime import date

import pandas as pd
import pytest

from src.analytics import (
    DEALERS_SHORT_ALL,
    STANDARD,
    enrich_chain,
    find_gamma_flip,
    gamma_curve,
    strike_profile,
    summarize,
)


def sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "optionSymbol": ["XYZC90", "XYZC110", "XYZP90", "XYZP110"],
            "expiration": pd.to_datetime(["2026-12-18"] * 4),
            "side": ["call", "call", "put", "put"],
            "strike": [90.0, 110.0, 90.0, 110.0],
            "dte": [140, 140, 140, 140],
            "bid": [12.0, 3.0, 3.0, 12.0],
            "ask": [12.5, 3.5, 3.5, 12.5],
            "volume": [10, 20, 30, 40],
            "openInterest": [1000, 800, 1200, 900],
            "underlyingPrice": [100.0] * 4,
            "iv": [0.30, 0.28, 0.32, 0.31],
            "delta": [0.70, 0.35, -0.30, -0.65],
            "gamma": [0.020, 0.025, 0.021, 0.024],
        }
    )


def test_standard_signs_calls_positive_puts_negative() -> None:
    enriched = enrich_chain(sample_chain(), STANDARD)
    assert (enriched.loc[enriched["side"] == "call", "gex"] > 0).all()
    assert (enriched.loc[enriched["side"] == "put", "gex"] < 0).all()


def test_zero_vendor_gamma_is_recalculated_from_iv() -> None:
    chain = sample_chain()
    chain["gamma"] = 0.0
    enriched = enrich_chain(chain, STANDARD)
    assert (enriched["gamma_used"] > 0).all()
    assert (enriched["gamma_source"] == "modelled from IV").all()


def test_vendor_gamma_is_fallback_when_iv_is_missing() -> None:
    chain = sample_chain()
    chain["iv"] = float("nan")
    enriched = enrich_chain(chain, STANDARD)
    assert enriched["gamma_used"].tolist() == chain["gamma"].tolist()
    assert (enriched["gamma_source"] == "vendor fallback").all()


def test_dealers_short_all_has_negative_gamma_curve() -> None:
    chain = sample_chain()
    curve = gamma_curve(chain, DEALERS_SHORT_ALL)
    assert (curve["net_gex"] < 0).all()
    assert find_gamma_flip(curve, 100.0) is None


def test_flip_interpolates_nearest_zero_crossing() -> None:
    curve = pd.DataFrame({"spot": [80.0, 90.0, 100.0, 110.0], "net_gex": [-3.0, -1.0, 1.0, 4.0]})
    assert find_gamma_flip(curve, 97.0) == 95.0


def test_profile_and_summary_are_consistent() -> None:
    enriched = enrich_chain(sample_chain(), STANDARD)
    curve = gamma_curve(enriched, STANDARD)
    profile = strike_profile(enriched)
    summary = summarize("xyz", date(2026, 7, 30), enriched, curve, STANDARD, 7, 365)
    assert profile["net_gex"].sum() == pytest.approx(enriched["gex"].sum())
    assert summary.symbol == "XYZ"
    assert summary.call_wall in {90.0, 110.0}
    assert summary.put_wall in {90.0, 110.0}
    assert summary.put_call_oi_ratio == 2100 / 1800
