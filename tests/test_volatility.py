from datetime import date

import pandas as pd
import pytest

from src.volatility import snapshot_from_chain


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


def test_snapshot_uses_constant_tenor_and_call_minus_put_skew():
    snapshot = snapshot_from_chain("XYZ", sample_chain(), date(2026, 8, 12), "1M")

    assert snapshot.actual_dte == 30
    assert snapshot.call_25d_iv == pytest.approx(0.36)
    assert snapshot.put_25d_iv == pytest.approx(0.32)
    assert snapshot.skew_25d == pytest.approx(0.04)
    assert snapshot.atm_iv == pytest.approx((0.30 + 0.32) / 2)


def test_snapshot_record_serializes_dates():
    record = snapshot_from_chain("XYZ", sample_chain(), date(2026, 8, 12), "1M").record()

    assert record["snapshot_date"] == "2026-08-12"
    assert record["expiration"] == "2026-09-11"
    assert record["tenor"] == "1M"
