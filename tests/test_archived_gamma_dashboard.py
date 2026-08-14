import pandas as pd
import pytest

from src.archived_gamma_dashboard import (
    gamma_exposure_chart,
    profile_from_archive,
    volume_by_strike_chart,
)


def test_profile_from_archive_uses_standard_call_positive_put_negative_gex():
    chain = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "side": "call",
                "underlyingPrice": 100.0,
                "openInterest": 10,
                "volume": 7,
                "gamma_used": 0.02,
            },
            {
                "strike": 100.0,
                "side": "put",
                "underlyingPrice": 100.0,
                "openInterest": 20,
                "volume": 9,
                "gamma_used": 0.01,
            },
            {
                "strike": 105.0,
                "side": "call",
                "underlyingPrice": 100.0,
                "openInterest": 5,
                "volume": 3,
                "gamma_used": 0.01,
            },
        ]
    )

    profile, spot = profile_from_archive(chain)

    assert spot == pytest.approx(100.0)
    row_100 = profile.loc[profile["strike"] == 100.0].iloc[0]
    expected_call = 0.02 * 10 * 100 * 100.0**2 * 0.01
    expected_put = -(0.01 * 20 * 100 * 100.0**2 * 0.01)
    assert row_100["call_gex"] == pytest.approx(expected_call)
    assert row_100["put_gex"] == pytest.approx(expected_put)
    assert row_100["call_volume"] == pytest.approx(7)
    assert row_100["put_volume"] == pytest.approx(9)


def test_profile_aggregate_gex_is_cumulative_net_by_strike():
    chain = pd.DataFrame(
        [
            {"strike": 95.0, "side": "call", "underlyingPrice": 100.0, "openInterest": 10, "volume": 0, "gamma_used": 0.01},
            {"strike": 100.0, "side": "put", "underlyingPrice": 100.0, "openInterest": 5, "volume": 0, "gamma_used": 0.01},
        ]
    )

    profile, _ = profile_from_archive(chain)

    assert profile["aggregate_gex"].iloc[0] == pytest.approx(profile["net_gex"].iloc[0])
    assert profile["aggregate_gex"].iloc[1] == pytest.approx(profile["net_gex"].sum())


def test_archived_gamma_charts_serialize_with_reference_lines():
    profile = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "call_gex": [1_000_000.0, 2_000_000.0, 500_000.0],
            "put_gex": [-250_000.0, -1_250_000.0, -400_000.0],
            "aggregate_gex": [750_000.0, 1_500_000.0, 1_600_000.0],
            "call_volume": [10.0, 20.0, 5.0],
            "put_volume": [4.0, 12.0, 9.0],
        }
    )

    gamma_spec = gamma_exposure_chart(
        profile,
        spot=100.0,
        call_wall=105.0,
        put_wall=95.0,
        gamma_flip=101.0,
    ).to_dict()
    volume_spec = volume_by_strike_chart(profile).to_dict()

    assert "layer" in gamma_spec
    assert len(gamma_spec["layer"]) == 6
    assert volume_spec["mark"]["type"] == "bar"
