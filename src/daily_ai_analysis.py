from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from math import ceil, floor
from typing import Any, Iterable, Sequence

import pandas as pd

from src.daily_ai_summary import DailySummary, SummaryNotReady
from src.skew_collector import (
    AI_FABLESS_SEMI_SYMBOLS,
    AI_FABS_SYMBOLS,
    AI_MEMORY_SYMBOLS,
    AI_PHOTONICS_SYMBOLS,
    AI_POOL_SYMBOLS,
    AUTO_SYMBOLS,
    INDEX_SYMBOLS,
    MAG7_SYMBOLS,
    NEOCLOUD_SYMBOLS,
    POWER_SYMBOLS,
    SOFTWARE_SYMBOLS,
)


REQUIRED_TENORS = ("1W", "1M")
REQUIRED_25D_COLUMNS = ("call_25d_iv", "put_25d_iv", "skew_25d")
VOLATILITY_COLUMNS = (
    "atm_iv",
    "call_25d_iv",
    "put_25d_iv",
    "skew_25d",
)
TRIM_FRACTION = 0.10
PERCENTILE_LOOKBACK_DAYS = 60
MAX_PRIOR_SUMMARIES = 5

SUMMARY_GROUPS: dict[str, list[str]] = {
    "AI Infra": AI_POOL_SYMBOLS,
    "Dashboard ex-index": [symbol for symbol in AUTO_SYMBOLS if symbol not in INDEX_SYMBOLS],
    "Neoclouds": NEOCLOUD_SYMBOLS,
    "Mag 7": MAG7_SYMBOLS,
    "Software": SOFTWARE_SYMBOLS,
    "Power": POWER_SYMBOLS,
    "AI Photonics": AI_PHOTONICS_SYMBOLS,
    "AI Fabless Semis": AI_FABLESS_SEMI_SYMBOLS,
    "AI Memory": AI_MEMORY_SYMBOLS,
    "AI Fabs": AI_FABS_SYMBOLS,
}


@dataclass(frozen=True)
class AnalysisPacket:
    snapshot_date: date
    comparison_date: date
    week_comparison_date: date
    month_comparison_date: date
    symbol_count: int
    expected_symbol_count: int
    input_signature: str
    payload: dict[str, Any]


def _trimmed_mean(values: pd.Series) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if ordered.empty:
        return float("nan")
    cut = floor(len(ordered) * TRIM_FRACTION)
    if cut > 0 and len(ordered) - 2 * cut > 0:
        ordered = ordered.iloc[cut : len(ordered) - cut]
    return float(ordered.mean())


def _round(value: Any, digits: int = 2) -> float | None:
    return round(float(value), digits) if pd.notna(value) else None


def _prepared_history(history: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    expected = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    if history.empty or not expected:
        return pd.DataFrame()
    work = history.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["tenor"] = work["tenor"].astype(str)
    work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce")
    work = work[
        work["symbol"].isin(expected)
        & work["tenor"].isin(REQUIRED_TENORS)
        & work["snapshot_date"].notna()
    ].copy()
    for column in ("spot", *VOLATILITY_COLUMNS):
        if column not in work.columns:
            work[column] = float("nan")
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return (
        work.sort_values(["snapshot_date", "symbol", "tenor"])
        .drop_duplicates(["snapshot_date", "symbol", "tenor"], keep="last")
        .reset_index(drop=True)
    )


def complete_summary_sessions(history: pd.DataFrame, symbols: Iterable[str]) -> list[date]:
    expected = {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    work = _prepared_history(history, expected)
    if work.empty:
        return []
    complete: list[date] = []
    for stamp, dated in work.groupby("snapshot_date", sort=True):
        session_ok = True
        for tenor in REQUIRED_TENORS:
            rows = dated[dated["tenor"] == tenor].dropna(subset=list(REQUIRED_25D_COLUMNS))
            if not expected.issubset(set(rows["symbol"])):
                session_ok = False
                break
        if session_ok:
            complete.append(pd.Timestamp(stamp).date())
    return complete


def _paired_sessions(
    history: pd.DataFrame,
    current_date: date,
    comparison_date: date,
) -> pd.DataFrame:
    columns = ["symbol", "tenor", "spot", *VOLATILITY_COLUMNS]
    current = history[history["snapshot_date"].dt.date == current_date][columns].copy()
    prior = history[history["snapshot_date"].dt.date == comparison_date][columns].copy()
    return current.set_index(["symbol", "tenor"]).join(
        prior.set_index(["symbol", "tenor"]),
        how="inner",
        lsuffix="_current",
        rsuffix="_prior",
    )


def _nearest_session(sessions: Iterable[date], target: date) -> date:
    choices = list(sessions)
    if not choices:
        raise SummaryNotReady("No prior complete session is available for comparison.")
    return min(choices, key=lambda stamp: (abs((stamp - target).days), -stamp.toordinal()))


def _percentile(values: Iterable[float], current: float) -> float:
    clean = [float(value) for value in values if pd.notna(value)]
    if len(clean) <= 1:
        return 50.0
    below = sum(value < current for value in clean)
    equal = sum(abs(value - current) < 1e-10 for value in clean)
    rank = below + max(equal - 1, 0) / 2.0
    return 100.0 * rank / (len(clean) - 1)


def _aggregate(values: pd.Series, *, trimmed: bool) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna() * 100.0
    return _trimmed_mean(numeric) if trimmed else float(numeric.mean())


def _current_percentile(
    history: pd.DataFrame,
    current_date: date,
    members: Iterable[str],
    tenor: str,
    metric: str,
    *,
    trimmed: bool,
) -> tuple[float | None, int]:
    wanted = {str(symbol).upper() for symbol in members}
    start = current_date - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    work = history[
        (history["snapshot_date"].dt.date >= start)
        & (history["snapshot_date"].dt.date <= current_date)
        & history["symbol"].isin(wanted)
        & (history["tenor"] == tenor)
    ]
    minimum = max(1, min(len(wanted), max(3, ceil(len(wanted) * 0.60))))
    observations: list[tuple[date, float]] = []
    for stamp, dated in work.groupby("snapshot_date", sort=True):
        values = pd.to_numeric(dated[metric], errors="coerce").dropna()
        if len(values) < minimum:
            continue
        observations.append(
            (pd.Timestamp(stamp).date(), _aggregate(values, trimmed=trimmed))
        )
    current_values = [value for stamp, value in observations if stamp == current_date]
    if not current_values:
        return None, len(observations)
    return _percentile((value for _, value in observations), current_values[-1]), len(observations)


def _symbol_percentile(
    history: pd.DataFrame,
    current_date: date,
    symbol: str,
    tenor: str,
    metric: str,
) -> tuple[float | None, int]:
    start = current_date - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    work = history[
        (history["snapshot_date"].dt.date >= start)
        & (history["snapshot_date"].dt.date <= current_date)
        & (history["symbol"] == symbol)
        & (history["tenor"] == tenor)
    ].sort_values("snapshot_date")
    values = pd.to_numeric(work[metric], errors="coerce").dropna() * 100.0
    current = work[work["snapshot_date"].dt.date == current_date]
    if current.empty or values.empty:
        return None, len(values)
    current_value = float(pd.to_numeric(current[metric], errors="coerce").iloc[-1]) * 100.0
    return _percentile(values, current_value), len(values)


def _point_in_time(
    pair: pd.DataFrame,
    symbol: str,
    tenor: str,
) -> dict[str, float | None]:
    row = pair.loc[(symbol, tenor)]
    spot_current = float(row["spot_current"])
    spot_prior = float(row["spot_prior"])
    result: dict[str, float | None] = {
        "spot_return_pct": _round(
            (spot_current / spot_prior - 1.0) * 100.0 if spot_prior else float("nan")
        )
    }
    for metric in VOLATILITY_COLUMNS:
        current = float(row[f"{metric}_current"]) * 100.0
        prior = float(row[f"{metric}_prior"]) * 100.0
        result[f"{metric}_change_vol_points"] = _round(current - prior)
    return result


def _basket_horizon_stats(
    pair: pd.DataFrame,
    members: Iterable[str],
    tenor: str,
    metric: str,
) -> dict[str, Any] | None:
    wanted = {str(symbol).upper() for symbol in members}
    work = pair.reset_index()
    work = work[(work["symbol"].isin(wanted)) & (work["tenor"] == tenor)].copy()
    current_col = f"{metric}_current"
    prior_col = f"{metric}_prior"
    work = work.dropna(subset=[current_col, prior_col])
    if work.empty:
        return None
    current = pd.to_numeric(work[current_col], errors="coerce") * 100.0
    prior = pd.to_numeric(work[prior_col], errors="coerce") * 100.0
    changes = current - prior
    return {
        "count": int(len(work)),
        "current_equal": _round(current.mean()),
        "change_equal": _round(current.mean() - prior.mean()),
        "current_trimmed_10pct": _round(_trimmed_mean(current)),
        "change_trimmed_10pct": _round(_trimmed_mean(current) - _trimmed_mean(prior)),
        "breadth_higher_pct": _round((changes > 0).mean() * 100.0, 1),
        "breadth_lower_pct": _round((changes < 0).mean() * 100.0, 1),
    }


def _basket_spot_stats(
    pair: pd.DataFrame,
    members: Iterable[str],
) -> dict[str, Any] | None:
    wanted = {str(symbol).upper() for symbol in members}
    work = pair.reset_index()
    work = work[(work["symbol"].isin(wanted)) & (work["tenor"] == "1W")].copy()
    work = work.dropna(subset=["spot_current", "spot_prior"])
    work = work[work["spot_prior"] != 0]
    if work.empty:
        return None
    returns = (work["spot_current"] / work["spot_prior"] - 1.0) * 100.0
    return {
        "count": int(len(work)),
        "equal_weight_return_pct": _round(returns.mean()),
        "trimmed_10pct_return_pct": _round(_trimmed_mean(returns)),
        "breadth_higher_pct": _round((returns > 0).mean() * 100.0, 1),
        "breadth_lower_pct": _round((returns < 0).mean() * 100.0, 1),
    }


def _prior_summary_context(
    prior_summaries: Sequence[DailySummary],
    current_date: date,
) -> list[dict[str, Any]]:
    eligible = sorted(
        (summary for summary in prior_summaries if summary.snapshot_date < current_date),
        key=lambda summary: summary.snapshot_date,
        reverse=True,
    )[:MAX_PRIOR_SUMMARIES]
    return [
        {
            "snapshot_date": summary.snapshot_date.isoformat(),
            "bullets": [bullet.record() for bullet in summary.bullets],
            "bottom_line": summary.bottom_line,
        }
        for summary in eligible
    ]


def _input_signature(history: pd.DataFrame) -> str:
    columns = ["snapshot_date", "symbol", "tenor", "spot", *VOLATILITY_COLUMNS]
    records: list[dict[str, Any]] = []
    for row in history[columns].sort_values(["snapshot_date", "symbol", "tenor"]).itertuples(
        index=False
    ):
        records.append(
            {
                "snapshot_date": pd.Timestamp(row.snapshot_date).date().isoformat(),
                "symbol": row.symbol,
                "tenor": row.tenor,
                "spot": _round(row.spot, 8),
                **{
                    metric: _round(getattr(row, metric), 8)
                    for metric in VOLATILITY_COLUMNS
                },
            }
        )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_analysis_packet(
    history: pd.DataFrame,
    symbols: Iterable[str] = AUTO_SYMBOLS,
    prior_summaries: Sequence[DailySummary] = (),
) -> AnalysisPacket:
    expected = list(
        dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip())
    )
    work = _prepared_history(history, expected)
    sessions = complete_summary_sessions(work, expected)
    if len(sessions) < 2:
        raise SummaryNotReady(
            "Two complete sessions with 1W + 1M 25D rows for every configured ticker are required."
        )
    current_date = sessions[-1]
    comparison_date = sessions[-2]
    week_comparison_date = _nearest_session(sessions[:-1], current_date - timedelta(days=7))
    month_comparison_date = _nearest_session(sessions[:-1], current_date - timedelta(days=30))
    comparison_dates = {
        "1D": comparison_date,
        "1W": week_comparison_date,
        "1M": month_comparison_date,
    }
    pairs = {
        horizon: _paired_sessions(work, current_date, comparison)
        for horizon, comparison in comparison_dates.items()
    }

    current_rows = work[work["snapshot_date"].dt.date == current_date].set_index(
        ["symbol", "tenor"]
    )
    symbol_rows: list[dict[str, Any]] = []
    for symbol in expected:
        current_spot = current_rows.loc[(symbol, "1W"), "spot"]
        record: dict[str, Any] = {
            "symbol": symbol,
            "spot": _round(current_spot),
            "spot_returns_pct": {
                horizon: _point_in_time(pair, symbol, "1W")["spot_return_pct"]
                for horizon, pair in pairs.items()
            },
            "tenors": {},
        }
        for tenor in REQUIRED_TENORS:
            current = current_rows.loc[(symbol, tenor)]
            metrics: dict[str, Any] = {}
            for metric in VOLATILITY_COLUMNS:
                percentile, observations = _symbol_percentile(
                    work, current_date, symbol, tenor, metric
                )
                metrics[metric] = {
                    "current_vol_points": _round(float(current[metric]) * 100.0),
                    "changes_vol_points": {
                        horizon: _point_in_time(pair, symbol, tenor)[
                            f"{metric}_change_vol_points"
                        ]
                        for horizon, pair in pairs.items()
                    },
                    "trailing_60d_percentile": _round(percentile, 1),
                    "percentile_observations": observations,
                }
            record["tenors"][tenor] = metrics
        symbol_rows.append(record)

    basket_rows: dict[str, Any] = {}
    for name, configured_members in SUMMARY_GROUPS.items():
        members = [symbol for symbol in configured_members if symbol in expected]
        if not members:
            continue
        basket: dict[str, Any] = {
            "members": members,
            "spot_returns_pct": {
                horizon: _basket_spot_stats(pair, members)
                for horizon, pair in pairs.items()
            },
            "tenors": {},
        }
        for tenor in REQUIRED_TENORS:
            metrics: dict[str, Any] = {}
            for metric in VOLATILITY_COLUMNS:
                equal_percentile, observations = _current_percentile(
                    work, current_date, members, tenor, metric, trimmed=False
                )
                trimmed_percentile, _ = _current_percentile(
                    work, current_date, members, tenor, metric, trimmed=True
                )
                metrics[metric] = {
                    "horizons": {
                        horizon: _basket_horizon_stats(pair, members, tenor, metric)
                        for horizon, pair in pairs.items()
                    },
                    "current_equal_trailing_60d_percentile": _round(equal_percentile, 1),
                    "current_trimmed_trailing_60d_percentile": _round(
                        trimmed_percentile, 1
                    ),
                    "percentile_observations": observations,
                }
            basket["tenors"][tenor] = metrics
        basket_rows[name] = basket

    input_signature = _input_signature(work)
    payload = {
        "units": {
            "spot_returns": "percent",
            "iv_and_skew_levels_and_changes": "volatility points",
            "breadth": "percent of basket members",
            "percentiles": "0=lowest and 100=highest within available trailing 60 calendar days",
        },
        "session": {
            "snapshot_date": current_date.isoformat(),
            "comparison_dates": {
                horizon: comparison.isoformat()
                for horizon, comparison in comparison_dates.items()
            },
            "coverage": f"{len(expected)}/{len(expected)}",
            "available_complete_sessions": len(sessions),
            "input_signature": input_signature,
        },
        "symbols": symbol_rows,
        "baskets": basket_rows,
        "prior_reports_for_continuity_only": _prior_summary_context(
            prior_summaries, current_date
        ),
    }
    return AnalysisPacket(
        snapshot_date=current_date,
        comparison_date=comparison_date,
        week_comparison_date=week_comparison_date,
        month_comparison_date=month_comparison_date,
        symbol_count=len(expected),
        expected_symbol_count=len(expected),
        input_signature=input_signature,
        payload=payload,
    )

