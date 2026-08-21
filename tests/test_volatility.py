from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analytics import black_scholes_price
from src.volatility import CALCULATION_VERSION, snapshot_from_chain


def sample_chain() -> pd.DataFrame:
    rows = []
    expiration = pd.Timestamp("2026-09-11")
    spot = 100.0
    for side, deltas, ivs in (
        ("call", [0.70, 0.50, 0.25, 0.10], [0.34, 0.30, 0.36, 0.43]),
        ("put", [-0.10, -0.25, -0.50, -0.70], [0.38, 0.32, 0.31, 0.37]),
    ):
        for strike, delta, iv in zip([90, 100, 110, 120], deltas, ivs):
            rows.append(
                {
                    "side": side,
                    "strike": strike,
                    "delta": delta,
                    "iv": iv,
                    "dte": 30,
                    "expiration": expiration,
                    "underlyingPrice": spot,
                }
            )
    return pd.DataFrame(rows)


def bracketing_chain() -> pd.DataFrame:
    rows = []
    spot = 100.0
    surfaces = (
        (
            pd.Timestamp("2026-08-16"),
            4,
            [0.34, 0.30, 0.40, 0.45],
            [0.48, 0.50, 0.31, 0.37],
        ),
        (
            pd.Timestamp("2026-08-22"),
            10,
            [0.44, 0.42, 0.60, 0.65],
            [0.68, 0.70, 0.43, 0.49],
        ),
    )
    for expiration, dte, call_ivs, put_ivs in surfaces:
        for side, deltas, ivs in (
            ("call", [0.70, 0.50, 0.25, 0.10], call_ivs),
            ("put", [-0.10, -0.25, -0.50, -0.70], put_ivs),
        ):
            for strike, delta, iv in zip([90, 100, 110, 120], deltas, ivs):
                rows.append(
                    {
                        "side": side,
                        "strike": strike,
                        "delta": delta,
                        "iv": iv,
                        "dte": dte,
                        "expiration": expiration,
                        "underlyingPrice": spot,
                    }
                )
    return pd.DataFrame(rows)


def expected_constant_maturity_iv(
    lower_iv: float,
    upper_iv: float,
    lower_dte: int,
    upper_dte: int,
    target_dte: int,
) -> float:
    weight = (target_dte - lower_dte) / (upper_dte - lower_dte)
    total_variance = (
        (1.0 - weight) * lower_iv**2 * lower_dte / 365.0
        + weight * upper_iv**2 * upper_dte / 365.0
    )
    return float(np.sqrt(total_variance / (target_dte / 365.0)))


def zero_vendor_iv_chain() -> pd.DataFrame:
    chain = sample_chain().copy()
    true_ivs = chain["iv"].to_numpy(float)
    prices = []
    for row, true_iv in zip(chain.itertuples(index=False), true_ivs):
        prices.append(
            black_scholes_price(
                spot=float(row.underlyingPrice),
                strike=float(row.strike),
                time_years=float(row.dte) / 365.0,
                volatility=float(true_iv),
                side=str(row.side),
                risk_free_rate=0.04,
                dividend_yield=0.0,
            )
        )
    chain["bid"] = np.asarray(prices) * 0.995
    chain["ask"] = np.asarray(prices) * 1.005
    chain["mid"] = prices
    chain["last"] = prices
    chain["iv"] = 0.0
    chain["delta"] = 0.0
    return chain


def test_snapshot_uses_constant_tenor_and_surface_metrics():
    snapshot = snapshot_from_chain("XYZ", sample_chain(), date(2026, 8, 12), "1M")

    assert snapshot.actual_dte == 30
    assert snapshot.call_10d_iv == pytest.approx(0.43)
    assert snapshot.put_10d_iv == pytest.approx(0.38)
    assert snapshot.skew_10d == pytest.approx(0.05)
    assert snapshot.call_25d_iv == pytest.approx(0.36)
    assert snapshot.put_25d_iv == pytest.approx(0.32)
    assert snapshot.skew_25d == pytest.approx(0.04)
    assert snapshot.atm_iv == pytest.approx((0.30 + 0.32) / 2)
    assert snapshot.calculation_version == CALCULATION_VERSION


def test_snapshot_interpolates_total_variance_between_bracketing_expiries():
    snapshot = snapshot_from_chain(
        "XYZ", bracketing_chain(), date(2026, 8, 12), "1W"
    )

    expected_call = expected_constant_maturity_iv(0.40, 0.60, 4, 10, 7)
    expected_put = expected_constant_maturity_iv(0.50, 0.70, 4, 10, 7)

    assert snapshot.actual_dte == 7
    assert snapshot.expiration == date(2026, 8, 19)
    assert snapshot.call_25d_iv == pytest.approx(expected_call)
    assert snapshot.put_25d_iv == pytest.approx(expected_put)
    assert snapshot.skew_25d == pytest.approx(expected_call - expected_put)
    # Regression guard: the old implementation snapped to either 4D or 10D.
    assert snapshot.call_25d_iv != pytest.approx(0.40)
    assert snapshot.call_25d_iv != pytest.approx(0.60)


def test_snapshot_derives_iv_and_delta_when_vendor_greeks_are_zero():
    snapshot = snapshot_from_chain(
        "XYZ", zero_vendor_iv_chain(), date(2026, 8, 12), "1M"
    )

    assert snapshot.actual_dte == 30
    assert snapshot.atm_iv is not None and snapshot.atm_iv > 0
    assert snapshot.call_25d_iv is not None and snapshot.call_25d_iv > 0
    assert snapshot.put_25d_iv is not None and snapshot.put_25d_iv > 0
    assert snapshot.skew_25d is not None and np.isfinite(snapshot.skew_25d)
    assert snapshot.call_10d_iv is not None and snapshot.call_10d_iv > 0
    assert snapshot.put_10d_iv is not None and snapshot.put_10d_iv > 0


def test_snapshot_record_serializes_dates_and_surface_fields():
    record = snapshot_from_chain("XYZ", sample_chain(), date(2026, 8, 12), "1M").record()

    assert record["snapshot_date"] == "2026-08-12"
    assert record["expiration"] == "2026-09-11"
    assert record["tenor"] == "1M"
    assert record["atm_iv"] is not None
    assert record["call_10d_iv"] == pytest.approx(0.43)
    assert record["calculation_version"] == CALCULATION_VERSION
