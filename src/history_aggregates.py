from __future__ import annotations

from math import ceil, floor
from typing import Iterable

import pandas as pd


EQUAL_WEIGHT = "equal"
TRIMMED_MEAN = "trimmed_10"
MIN_COVERAGE = 0.60
MIN_NAMES = 3
TRIM_FRACTION = 0.10

METRIC_COLUMNS = {
    "ATM IV": ("atm_iv",),
    "25Δ Skew": ("skew_25d",),
    "10Δ Skew": ("skew_10d",),
    "25Δ Put IV": ("put_25d_iv",),
    "25Δ Call IV": ("call_25d_iv",),
    "10Δ Put IV": ("put_10d_iv",),
    "10Δ Call IV": ("call_10d_iv",),
    "25Δ Smile Convexity": ("put_25d_iv", "call_25d_iv", "atm_iv"),
    "Tail Steepness (10Δ−25Δ)": (
        "put_10d_iv",
        "call_10d_iv",
        "put_25d_iv",
        "call_25d_iv",
    ),
}


def metric_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    """Return a decimal-volatility series for one stored/derived surface metric."""
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"Unsupported historical metric: {metric}")

    def numeric(column: str) -> pd.Series:
        if column not in frame.columns:
            return pd.Series(float("nan"), index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    if metric == "25Δ Smile Convexity":
        return (numeric("put_25d_iv") + numeric("call_25d_iv")) / 2.0 - numeric("atm_iv")
    if metric == "Tail Steepness (10Δ−25Δ)":
        wing_10 = (numeric("put_10d_iv") + numeric("call_10d_iv")) / 2.0
        wing_25 = (numeric("put_25d_iv") + numeric("call_25d_iv")) / 2.0
        return wing_10 - wing_25
    return numeric(METRIC_COLUMNS[metric][0])


def individual_history(history: pd.DataFrame, symbol: str, metric: str) -> pd.DataFrame:
    """Return snapshot_date/value for one ticker, de-duplicated by date."""
    if history.empty:
        return pd.DataFrame(columns=["snapshot_date", "value"])
    work = history[history["symbol"].astype(str).str.upper() == symbol.upper()].copy()
    if work.empty:
        return pd.DataFrame(columns=["snapshot_date", "value"])
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    work["value"] = metric_values(work, metric)
    return (
        work.dropna(subset=["snapshot_date", "value"])
        .sort_values("snapshot_date")
        .drop_duplicates("snapshot_date", keep="last")[["snapshot_date", "value"]]
        .reset_index(drop=True)
    )


def _trimmed_mean(values: pd.Series, trim_fraction: float = TRIM_FRACTION) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if ordered.empty:
        return float("nan")
    cut = floor(len(ordered) * trim_fraction)
    if cut > 0 and len(ordered) - 2 * cut > 0:
        ordered = ordered.iloc[cut : len(ordered) - cut]
    return float(ordered.mean())


def aggregate_history(
    history: pd.DataFrame,
    members: Iterable[str],
    metric: str,
    method: str,
    *,
    min_coverage: float = MIN_COVERAGE,
    min_names: int = MIN_NAMES,
) -> pd.DataFrame:
    """Build a representative constituent aggregate from saved rows.

    A date is kept only when enough basket members have a usable value. The
    equal-weight mean averages all valid constituents. The trimmed method drops
    floor(10% * N) observations from each tail before averaging; for small
    baskets where that floor is zero, it naturally equals the ordinary mean.
    """
    members_clean = list(dict.fromkeys(str(x).strip().upper() for x in members if str(x).strip()))
    if history.empty or not members_clean:
        return pd.DataFrame(columns=["snapshot_date", "value", "valid_count", "coverage"])
    if method not in {EQUAL_WEIGHT, TRIMMED_MEAN}:
        raise ValueError(f"Unsupported aggregation method: {method}")

    work = history[history["symbol"].astype(str).str.upper().isin(members_clean)].copy()
    if work.empty:
        return pd.DataFrame(columns=["snapshot_date", "value", "valid_count", "coverage"])
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    work["value"] = metric_values(work, metric)
    work = work.dropna(subset=["snapshot_date", "value"])
    if work.empty:
        return pd.DataFrame(columns=["snapshot_date", "value", "valid_count", "coverage"])

    work = (
        work.sort_values(["snapshot_date", "symbol"])
        .drop_duplicates(["snapshot_date", "symbol"], keep="last")
    )
    threshold = max(int(min_names), ceil(len(members_clean) * float(min_coverage)))
    threshold = min(threshold, len(members_clean))

    rows: list[dict[str, object]] = []
    for snapshot_date, group in work.groupby("snapshot_date", sort=True):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        valid_count = int(values.size)
        if valid_count < threshold:
            continue
        value = float(values.mean()) if method == EQUAL_WEIGHT else _trimmed_mean(values)
        rows.append(
            {
                "snapshot_date": pd.Timestamp(snapshot_date),
                "value": value,
                "valid_count": valid_count,
                "coverage": valid_count / len(members_clean),
            }
        )
    return pd.DataFrame(rows, columns=["snapshot_date", "value", "valid_count", "coverage"])


def apply_change_mode(series: pd.DataFrame, change_mode: str) -> pd.DataFrame:
    """Apply level or lagged change after a ticker/aggregate level is built."""
    if series.empty:
        return series.copy()
    if change_mode == "Level":
        return series.copy()
    lag = {"1D Δ": 1, "1W Δ": 5, "1M Δ": 21}.get(change_mode)
    if lag is None:
        raise ValueError(f"Unsupported historical change view: {change_mode}")
    out = series.copy().sort_values("snapshot_date")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").diff(lag)
    return out.dropna(subset=["value"]).reset_index(drop=True)
