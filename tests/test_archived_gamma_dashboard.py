import pandas as pd
import pytest

from src.archived_gamma_dashboard import (
    focused_strike_window,
    gamma_exposure_spec,
    profile_from_archive,
    volume_by_strike_spec,
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


def test_gamma_spec_uses_dual_interactive_scales_and_reference_style_wall_markers():
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

    spec = gamma_exposure_spec(profile, call_wall=100.0, put_wall=95.0)

    assert spec["leftScale"] is True
    assert spec["rightScale"] is True
    assert [item["type"] for item in spec["series"]] == ["histogram", "histogram", "line"]
    assert spec["series"][0]["options"]["priceScaleId"] == "left"
    assert spec["series"][2]["options"]["priceScaleId"] == "right"
    call_marker = spec["series"][0]["options"]["markers"][0]
    put_marker = spec["series"][1]["options"]["markers"][0]
    assert call_marker["shape"] == "arrowDown"
    assert call_marker["position"] == "aboveBar"
    assert call_marker["text"] == "Call Wall 100"
    assert put_marker["shape"] == "arrowUp"
    assert put_marker["position"] == "belowBar"
    assert put_marker["text"] == "Put Wall 95"


def test_volume_spec_puts_are_negative_and_chart_uses_one_left_scale():
    profile = pd.DataFrame(
        {
            "strike": [95.0, 100.0],
            "call_volume": [10.0, 20.0],
            "put_volume": [4.0, 12.0],
        }
    )

    spec = volume_by_strike_spec(profile)

    assert spec["leftScale"] is True
    assert spec["rightScale"] is False
    assert spec["series"][0]["data"][0]["value"] == pytest.approx(10.0)
    assert spec["series"][1]["data"][0]["value"] == pytest.approx(-4.0)


def test_focus_window_keeps_spot_and_walls_visible_without_using_full_tail_range():
    profile = pd.DataFrame({"strike": [50.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0]})

    low, high = focused_strike_window(profile, spot=100.0, call_wall=110.0, put_wall=90.0)

    assert low <= 90.0
    assert high >= 110.0
    assert low > 50.0
    assert high < 150.0
