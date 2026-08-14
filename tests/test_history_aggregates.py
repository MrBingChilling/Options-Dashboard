import pandas as pd
import pytest

from src.history_aggregates import (
    EQUAL_WEIGHT,
    TRIMMED_MEAN,
    aggregate_history,
    apply_change_mode,
    metric_values,
)


def test_equal_weight_and_trimmed_mean_are_explicit_and_distinct_with_outlier():
    rows = []
    for i in range(20):
        rows.append(
            {
                "snapshot_date": "2026-08-12",
                "symbol": f"S{i:02d}",
                "atm_iv": 2.00 if i == 19 else 0.20 + i * 0.001,
            }
        )
    frame = pd.DataFrame(rows)
    members = [f"S{i:02d}" for i in range(20)]

    equal = aggregate_history(frame, members, "ATM IV", EQUAL_WEIGHT)
    trimmed = aggregate_history(frame, members, "ATM IV", TRIMMED_MEAN)

    assert len(equal) == len(trimmed) == 1
    assert equal.iloc[0]["valid_count"] == 20
    assert equal.iloc[0]["coverage"] == pytest.approx(1.0)
    assert trimmed.iloc[0]["value"] < equal.iloc[0]["value"]


def test_aggregate_requires_sixty_percent_constituent_coverage():
    members = [f"S{i}" for i in range(10)]
    frame = pd.DataFrame(
        [
            {"snapshot_date": "2026-08-12", "symbol": symbol, "atm_iv": 0.25}
            for symbol in members[:5]
        ]
        + [
            {"snapshot_date": "2026-08-13", "symbol": symbol, "atm_iv": 0.25}
            for symbol in members[:6]
        ]
    )

    result = aggregate_history(frame, members, "ATM IV", EQUAL_WEIGHT)

    assert result["snapshot_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-08-13"]
    assert result.iloc[0]["coverage"] == pytest.approx(0.60)


def test_surface_metrics_are_derived_without_new_data_fields():
    frame = pd.DataFrame(
        {
            "atm_iv": [0.20],
            "put_25d_iv": [0.24],
            "call_25d_iv": [0.22],
            "put_10d_iv": [0.30],
            "call_10d_iv": [0.26],
        }
    )

    convexity = metric_values(frame, "25Δ Smile Convexity")
    steepness = metric_values(frame, "Tail Steepness (10Δ−25Δ)")

    assert convexity.iloc[0] == pytest.approx(0.03)
    assert steepness.iloc[0] == pytest.approx(0.05)


def test_change_is_applied_after_aggregate_level_is_constructed():
    series = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2026-08-10", "2026-08-11", "2026-08-12"]),
            "value": [0.20, 0.22, 0.19],
        }
    )

    changed = apply_change_mode(series, "1D Δ")

    assert changed["value"].tolist() == pytest.approx([0.02, -0.03])
