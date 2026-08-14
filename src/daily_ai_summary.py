from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from math import ceil, floor
from typing import Any, Iterable

import pandas as pd
import requests

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
from src.storage import SnapshotStore, SnapshotStoreError


SUMMARY_TABLE = "daily_ai_summaries"
GENERATOR_VERSION = "daily_ai_summary_v2"
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
MAX_BULLETS = 9
MAX_SUMMARY_WORDS = 520

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


class SummaryNotReady(ValueError):
    """Raised when two fully comparable daily sessions are not available."""


@dataclass(frozen=True)
class SummaryBullet:
    title: str
    body: str

    def record(self) -> dict[str, str]:
        return {"title": self.title, "body": self.body}


@dataclass(frozen=True)
class DailySummary:
    snapshot_date: date
    comparison_date: date
    symbol_count: int
    expected_symbol_count: int
    bullets: tuple[SummaryBullet, ...]
    bottom_line: str
    week_comparison_date: date | None = None
    month_comparison_date: date | None = None
    generator_version: str = GENERATOR_VERSION

    def record(self) -> dict[str, Any]:
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "comparison_date": self.comparison_date.isoformat(),
            "symbol_count": self.symbol_count,
            "expected_symbol_count": self.expected_symbol_count,
            "generator_version": self.generator_version,
            "summary": {
                "bullets": [bullet.record() for bullet in self.bullets],
                "bottom_line": self.bottom_line,
                "comparison_dates": {
                    "1D": self.comparison_date.isoformat(),
                    **(
                        {"1W": self.week_comparison_date.isoformat()}
                        if self.week_comparison_date
                        else {}
                    ),
                    **(
                        {"1M": self.month_comparison_date.isoformat()}
                        if self.month_comparison_date
                        else {}
                    ),
                },
                "data_note": (
                    "Generated only from saved spot and volatility-surface data. "
                    "It does not use news, event calendars or observed option order flow."
                ),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "DailySummary":
        payload = record.get("summary") or {}
        if isinstance(payload, str):
            payload = json.loads(payload)
        bullets = tuple(
            SummaryBullet(str(item.get("title", "")), str(item.get("body", "")))
            for item in payload.get("bullets", [])
            if isinstance(item, dict)
        )
        comparison_dates = payload.get("comparison_dates") or {}

        def optional_date(key: str) -> date | None:
            value = comparison_dates.get(key)
            return date.fromisoformat(str(value)[:10]) if value else None

        return cls(
            snapshot_date=date.fromisoformat(str(record["snapshot_date"])[:10]),
            comparison_date=date.fromisoformat(str(record["comparison_date"])[:10]),
            symbol_count=int(record.get("symbol_count") or 0),
            expected_symbol_count=int(record.get("expected_symbol_count") or 0),
            bullets=bullets,
            bottom_line=str(payload.get("bottom_line", "")),
            week_comparison_date=optional_date("1W"),
            month_comparison_date=optional_date("1M"),
            generator_version=str(record.get("generator_version") or GENERATOR_VERSION),
        )


@dataclass(frozen=True)
class MetricStats:
    count: int
    current_equal: float
    delta_equal: float
    current_trimmed: float
    delta_trimmed: float


def summary_endpoint(store: SnapshotStore) -> str:
    return f"{store.url}/rest/v1/{SUMMARY_TABLE}"


def save_daily_summary(store: SnapshotStore, summary: DailySummary) -> None:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    response = requests.post(
        summary_endpoint(store),
        params={"on_conflict": "snapshot_date"},
        headers={**store.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
        json=summary.record(),
        timeout=store.timeout,
    )
    if response.status_code not in {200, 201, 204}:
        raise SnapshotStoreError(
            f"Supabase daily-summary save failed ({response.status_code}): "
            f"{response.text[:300]}"
        )


def load_daily_summaries(store: SnapshotStore, limit: int = 90) -> list[DailySummary]:
    if not store.enabled:
        return []
    response = requests.get(
        summary_endpoint(store),
        params={
            "select": (
                "snapshot_date,comparison_date,symbol_count,expected_symbol_count,"
                "generator_version,summary,generated_at"
            ),
            "order": "snapshot_date.desc",
            "limit": str(max(1, int(limit))),
        },
        headers=store.headers,
        timeout=store.timeout,
    )
    if response.status_code != 200:
        raise SnapshotStoreError(
            f"Supabase daily-summary load failed ({response.status_code}): "
            f"{response.text[:300]}"
        )
    try:
        rows = response.json()
    except requests.JSONDecodeError as exc:
        raise SnapshotStoreError("Supabase daily-summary load returned invalid JSON.") from exc
    if not isinstance(rows, list):
        raise SnapshotStoreError("Supabase daily-summary load returned an invalid payload.")
    return [DailySummary.from_record(row) for row in rows if isinstance(row, dict)]


def _trimmed_mean(values: pd.Series) -> float:
    ordered = pd.to_numeric(values, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if ordered.empty:
        return float("nan")
    cut = floor(len(ordered) * TRIM_FRACTION)
    if cut > 0 and len(ordered) - 2 * cut > 0:
        ordered = ordered.iloc[cut : len(ordered) - cut]
    return float(ordered.mean())


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


def _metric_stats(
    paired: pd.DataFrame,
    members: Iterable[str],
    tenor: str,
    metric: str,
) -> MetricStats | None:
    wanted = {str(symbol).upper() for symbol in members}
    work = paired.reset_index()
    work = work[(work["symbol"].isin(wanted)) & (work["tenor"] == tenor)].copy()
    current_col = f"{metric}_current"
    prior_col = f"{metric}_prior"
    work = work.dropna(subset=[current_col, prior_col])
    if work.empty:
        return None
    current = pd.to_numeric(work[current_col], errors="coerce") * 100.0
    prior = pd.to_numeric(work[prior_col], errors="coerce") * 100.0
    return MetricStats(
        count=len(work),
        current_equal=float(current.mean()),
        delta_equal=float(current.mean() - prior.mean()),
        current_trimmed=_trimmed_mean(current),
        delta_trimmed=float(_trimmed_mean(current) - _trimmed_mean(prior)),
    )


def _single_snapshot(paired: pd.DataFrame, symbol: str, tenor: str) -> dict[str, float]:
    row = paired.loc[(symbol.upper(), tenor)]
    spot_current = float(row["spot_current"])
    spot_prior = float(row["spot_prior"])
    output = {
        "spot_pct": (spot_current / spot_prior - 1.0) * 100.0 if spot_prior else float("nan")
    }
    for metric in VOLATILITY_COLUMNS:
        current = float(row[f"{metric}_current"]) * 100.0
        prior = float(row[f"{metric}_prior"]) * 100.0
        output[metric] = current
        output[f"{metric}_delta"] = current - prior
    return output


def _member_skew_changes(
    paired: pd.DataFrame,
    members: Iterable[str],
    tenor: str,
) -> dict[str, float]:
    changes: dict[str, float] = {}
    for symbol in members:
        try:
            row = _single_snapshot(paired, symbol, tenor)
        except KeyError:
            continue
        value = row.get("skew_25d_delta")
        if value is not None and pd.notna(value):
            changes[str(symbol).upper()] = float(value)
    return changes


def _signed(value: float) -> str:
    return f"{value:+.2f}"


def _absolute_change(value: float) -> str:
    return f"{abs(value):.2f}"


def _rose_or_fell(value: float) -> str:
    if value > 0.005:
        return f"rose {_absolute_change(value)}"
    if value < -0.005:
        return f"fell {_absolute_change(value)}"
    return "was effectively unchanged"


def _toward(value: float) -> str:
    return "calls" if value >= 0 else "puts"


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


def _percentile_text(value: float) -> str:
    rounded = int(round(value))
    suffix = "th" if 10 <= rounded % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(rounded % 10, "th")
    return f"{rounded}{suffix} percentile"


def _group_percentile(
    history: pd.DataFrame,
    current_date: date,
    members: Iterable[str],
    tenor: str,
    metric: str,
    *,
    trimmed: bool = False,
) -> tuple[float, int]:
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
        values = pd.to_numeric(dated[metric], errors="coerce").dropna() * 100.0
        if len(values) < minimum:
            continue
        aggregate = _trimmed_mean(values) if trimmed else float(values.mean())
        observations.append((pd.Timestamp(stamp).date(), aggregate))
    current_values = [value for stamp, value in observations if stamp == current_date]
    if not current_values:
        return 50.0, len(observations)
    return _percentile((value for _, value in observations), current_values[-1]), len(observations)


def _single_percentile(
    history: pd.DataFrame,
    current_date: date,
    symbol: str,
    tenor: str,
    metric: str,
) -> tuple[float, int]:
    start = current_date - timedelta(days=PERCENTILE_LOOKBACK_DAYS)
    work = history[
        (history["snapshot_date"].dt.date >= start)
        & (history["snapshot_date"].dt.date <= current_date)
        & (history["symbol"] == symbol.upper())
        & (history["tenor"] == tenor)
    ].dropna(subset=[metric])
    observations = pd.to_numeric(work[metric], errors="coerce").dropna() * 100.0
    current = work[work["snapshot_date"].dt.date == current_date]
    if current.empty or observations.empty:
        return 50.0, len(observations)
    current_value = float(pd.to_numeric(current[metric], errors="coerce").iloc[-1]) * 100.0
    return _percentile(observations, current_value), len(observations)


def _group_spot_change(paired: pd.DataFrame, members: Iterable[str]) -> float:
    wanted = {str(symbol).upper() for symbol in members}
    work = paired.reset_index()
    work = work[(work["symbol"].isin(wanted)) & (work["tenor"] == "1W")].copy()
    work = work.dropna(subset=["spot_current", "spot_prior"])
    work = work[work["spot_prior"] != 0]
    if work.empty:
        return float("nan")
    return float(((work["spot_current"] / work["spot_prior"] - 1.0) * 100.0).mean())


def _multi_horizon_broad_bullet(pairs: dict[str, pd.DataFrame]) -> SummaryBullet | None:
    dashboard = {
        horizon: _metric_stats(pair, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "atm_iv")
        for horizon, pair in pairs.items()
    }
    ai_infra = {
        horizon: _metric_stats(pair, AI_POOL_SYMBOLS, "1W", "atm_iv")
        for horizon, pair in pairs.items()
    }
    if any(value is None for value in (*dashboard.values(), *ai_infra.values())):
        return None
    d1, w1, m1 = dashboard["1D"], dashboard["1W"], dashboard["1M"]
    ai_d1, ai_w1, ai_m1 = ai_infra["1D"], ai_infra["1W"], ai_infra["1M"]
    assert d1 and w1 and m1 and ai_d1 and ai_w1 and ai_m1
    if max(abs(d1.delta_trimmed), abs(w1.delta_trimmed), abs(m1.delta_trimmed)) < 1.0:
        return None
    if m1.delta_trimmed <= -5.0 and ai_m1.delta_trimmed <= -5.0:
        title = "Volatility compression is a multi-week regime, not just a one-day move."
    elif m1.delta_trimmed >= 5.0 and ai_m1.delta_trimmed >= 5.0:
        title = "Volatility expansion is a multi-week regime, not just a one-day move."
    else:
        title = "The broad volatility regime changed materially across multiple horizons."
    return SummaryBullet(
        title,
        f"Dashboard ex-index 1W ATM IV is {d1.current_trimmed:.2f} on a 10% trimmed basis, "
        f"changing {_signed(d1.delta_trimmed)} over 1D, {_signed(w1.delta_trimmed)} over 1W "
        f"and {_signed(m1.delta_trimmed)} over 1M. AI Infra is {ai_d1.current_trimmed:.2f}, "
        f"with corresponding moves of {_signed(ai_d1.delta_trimmed)}, "
        f"{_signed(ai_w1.delta_trimmed)} and {_signed(ai_m1.delta_trimmed)} vol points.",
    )


def _index_context_bullet(
    history: pd.DataFrame,
    current_date: date,
    pairs: dict[str, pd.DataFrame],
) -> SummaryBullet:
    snapshots = {
        symbol: {horizon: _single_snapshot(pair, symbol, "1W") for horizon, pair in pairs.items()}
        for symbol in ("SPY", "QQQ")
    }
    skew_percentiles = {
        symbol: _single_percentile(history, current_date, symbol, "1W", "skew_25d")[0]
        for symbol in snapshots
    }
    atm_percentiles = {
        symbol: _single_percentile(history, current_date, symbol, "1W", "atm_iv")[0]
        for symbol in snapshots
    }
    daily_mean = sum(snapshots[symbol]["1D"]["skew_25d_delta"] for symbol in snapshots) / 2.0
    still_call_richer = sum(skew_percentiles.values()) / 2.0 >= 65.0
    compressed_atm = sum(atm_percentiles.values()) / 2.0 <= 25.0
    if daily_mean <= -0.50 and still_call_richer and compressed_atm:
        title = "Today's index put bid is cautionary, but not yet broad risk-off."
    elif daily_mean <= -0.50:
        title = "Short-term index skew shifted materially toward puts."
    elif daily_mean >= 0.50:
        title = "Short-term index skew shifted materially toward calls."
    else:
        title = "Index skew did not register a material one-day regime change."
    spy, qqq = snapshots["SPY"], snapshots["QQQ"]
    return SummaryBullet(
        title,
        f"SPY 1W skew moved {_signed(spy['1D']['skew_25d_delta'])} today to "
        f"{_signed(spy['1D']['skew_25d'])}, but remains {_signed(spy['1M']['skew_25d_delta'])} "
        f"versus one month ago and at the {_percentile_text(skew_percentiles['SPY'])} of its "
        f"recent range. QQQ shows the same pattern: {_signed(qqq['1D']['skew_25d_delta'])} "
        f"today, {_signed(qqq['1M']['skew_25d_delta'])} over 1M, and the "
        f"{_percentile_text(skew_percentiles['QQQ'])}; both indexes' 1W ATM IV readings sit "
        f"near the {_percentile_text(sum(atm_percentiles.values()) / 2.0)}.",
    )


def _persistent_basket_bullets(
    history: pd.DataFrame,
    current_date: date,
    pairs: dict[str, pd.DataFrame],
) -> tuple[list[SummaryBullet], list[str]]:
    candidates: list[tuple[float, str, dict[str, MetricStats], dict[str, MetricStats], float]] = []
    for name in ("Neoclouds", "Mag 7", "Software", "Power", "AI Photonics", "AI Fabless Semis", "AI Memory", "AI Fabs"):
        one_week = {h: _metric_stats(pair, SUMMARY_GROUPS[name], "1W", "skew_25d") for h, pair in pairs.items()}
        one_month = {h: _metric_stats(pair, SUMMARY_GROUPS[name], "1M", "skew_25d") for h, pair in pairs.items()}
        if any(value is None for value in (*one_week.values(), *one_month.values())):
            continue
        assert all(one_week.values()) and all(one_month.values())
        pctl, _ = _group_percentile(history, current_date, SUMMARY_GROUPS[name], "1W", "skew_25d")
        month_signal = max(abs(one_week["1M"].delta_equal), abs(one_month["1M"].delta_equal))
        direction = 1.0 if one_week["1M"].delta_equal >= 0 else -1.0
        extreme = (direction > 0 and pctl >= 85.0) or (direction < 0 and pctl <= 15.0)
        same_tenor_direction = one_week["1M"].delta_equal * one_month["1M"].delta_equal > 0
        violent_reversal = (
            one_week["1W"].delta_equal * one_week["1M"].delta_equal < 0
            and abs(one_week["1W"].delta_equal) > max(4.0, 2.0 * abs(one_week["1M"].delta_equal))
        )
        if month_signal < 3.0 or not extreme or not same_tenor_direction or violent_reversal:
            continue
        score = month_signal + abs(pctl - 50.0) / 10.0
        if one_week["1W"].delta_equal * one_week["1M"].delta_equal > 0:
            score += 2.0
        candidates.append((score, name, one_week, one_month, pctl))
    candidates.sort(reverse=True, key=lambda item: item[0])

    bullets: list[SummaryBullet] = []
    names: list[str] = []
    for _, name, one_week, one_month, pctl in candidates[:2]:
        names.append(name)
        direction = _toward(one_week["1M"].delta_equal)
        title = f"{name} shows a persistent, historically elevated shift toward {direction}."
        body = (
            f"Its equal-weight 1W skew changed {_signed(one_week['1W'].delta_equal)} over 1W "
            f"and {_signed(one_week['1M'].delta_equal)} over 1M to "
            f"{_signed(one_week['1D'].current_equal)}, now the {_percentile_text(pctl)} of the "
            f"recent range. The 1M-tenor skew moved {_signed(one_month['1M'].delta_equal)} "
            f"over the month to {_signed(one_month['1D'].current_equal)}."
        )
        if name == "AI Memory":
            spot_change = _group_spot_change(pairs["1W"], SUMMARY_GROUPS[name])
            atm_change = _metric_stats(pairs["1W"], SUMMARY_GROUPS[name], "1W", "atm_iv")
            if atm_change is not None:
                body += (
                    f" Spot rose {spot_change:.2f}% over 1W while 1W ATM IV moved "
                    f"{_signed(atm_change.delta_equal)}, consistent with upside crowding rather "
                    "than a broad volatility shock."
                )
        bullets.append(SummaryBullet(title, body))
    return bullets, names


def _localized_downside_bullet(
    history: pd.DataFrame,
    current_date: date,
    pairs: dict[str, pd.DataFrame],
    symbols: Iterable[str],
) -> tuple[SummaryBullet | None, list[str]]:
    candidates: list[tuple[float, str, dict[str, dict[str, float]], float]] = []
    for symbol in symbols:
        snapshots = {h: _single_snapshot(pair, str(symbol), "1W") for h, pair in pairs.items()}
        deltas = [snapshots[h]["skew_25d_delta"] for h in ("1D", "1W", "1M")]
        pctl, _ = _single_percentile(history, current_date, str(symbol), "1W", "skew_25d")
        severity = max(-value for value in deltas)
        confirmed = sum(value <= -1.0 for value in deltas) >= 2
        if snapshots["1D"]["skew_25d"] > -2.5 or severity < 5.0 or not confirmed:
            continue
        atm_pctl, _ = _single_percentile(history, current_date, str(symbol), "1W", "atm_iv")
        score = severity + max(20.0 - pctl, 0.0) / 5.0 + max(atm_pctl - 80.0, 0.0) / 10.0
        candidates.append((score, str(symbol).upper(), snapshots, atm_pctl))
    candidates.sort(reverse=True, key=lambda item: item[0])
    selected = candidates[:2]
    if not selected:
        return None, []
    names = [item[1] for item in selected]
    descriptions: list[str] = []
    for _, symbol, snapshots, atm_pctl in selected:
        descriptions.append(
            f"{symbol} 1W skew is {_signed(snapshots['1D']['skew_25d'])}, changing "
            f"{_signed(snapshots['1D']['skew_25d_delta'])} over 1D, "
            f"{_signed(snapshots['1W']['skew_25d_delta'])} over 1W and "
            f"{_signed(snapshots['1M']['skew_25d_delta'])} over 1M; its 1W ATM IV is "
            f"{snapshots['1D']['atm_iv']:.2f} at the {_percentile_text(atm_pctl)}"
        )
    return SummaryBullet(
        f"{' and '.join(names)} show the clearest sustained ticker-level downside signals.",
        ". ".join(descriptions) + ". These are localized exceptions, not evidence of basket-wide risk-off.",
    ), names


def _mag7_exception_bullet(
    history: pd.DataFrame,
    current_date: date,
    pairs: dict[str, pd.DataFrame],
) -> tuple[SummaryBullet | None, str | None]:
    basket = _metric_stats(pairs["1M"], MAG7_SYMBOLS, "1M", "skew_25d")
    if basket is None or abs(basket.delta_equal) > 1.0:
        return None, None
    candidates: list[tuple[float, str, dict[str, dict[str, float]], float]] = []
    for symbol in MAG7_SYMBOLS:
        snapshots = {h: _single_snapshot(pair, symbol, "1M") for h, pair in pairs.items()}
        pctl, _ = _single_percentile(history, current_date, symbol, "1M", "skew_25d")
        if snapshots["1M"]["skew_25d_delta"] <= -3.0 and snapshots["1D"]["skew_25d"] < 0 and pctl <= 15.0:
            candidates.append((-snapshots["1M"]["skew_25d_delta"] + (15.0 - pctl) / 5.0, symbol, snapshots, pctl))
    if not candidates:
        return None, None
    _, symbol, snapshots, pctl = max(candidates, key=lambda item: item[0])
    return SummaryBullet(
        f"A calm Mag 7 basket masks a {symbol}-specific downside-skew regime.",
        f"Mag 7 equal-weight 1M skew changed only {_signed(basket.delta_equal)} over the month, "
        f"but {symbol} 1M skew changed {_signed(snapshots['1M']['skew_25d_delta'])} to "
        f"{_signed(snapshots['1D']['skew_25d'])}, the {_percentile_text(pctl)} of its recent "
        f"range. Its daily and weekly changes were {_signed(snapshots['1D']['skew_25d_delta'])} "
        f"and {_signed(snapshots['1W']['skew_25d_delta'])}, respectively.",
    ), symbol


def _no_regime_change_bullet(pairs: dict[str, pd.DataFrame]) -> SummaryBullet:
    atm = {h: _metric_stats(pair, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "atm_iv") for h, pair in pairs.items()}
    skew = {h: _metric_stats(pair, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "skew_25d") for h, pair in pairs.items()}
    assert all(atm.values()) and all(skew.values())
    return SummaryBullet(
        "No additional broad regime change cleared the materiality threshold.",
        f"Dashboard ex-index 1W ATM IV changed {_signed(atm['1D'].delta_trimmed)}, "
        f"{_signed(atm['1W'].delta_trimmed)} and {_signed(atm['1M'].delta_trimmed)} over "
        f"1D/1W/1M; trimmed skew changed {_signed(skew['1D'].delta_trimmed)}, "
        f"{_signed(skew['1W'].delta_trimmed)} and {_signed(skew['1M'].delta_trimmed)}. "
        "The remaining basket moves were either small, short-lived or too concentrated in outliers.",
    )


def build_daily_summary(
    history: pd.DataFrame,
    symbols: Iterable[str] = AUTO_SYMBOLS,
) -> DailySummary:
    expected = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
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
    pairs = {
        "1D": _paired_sessions(work, current_date, comparison_date),
        "1W": _paired_sessions(work, current_date, week_comparison_date),
        "1M": _paired_sessions(work, current_date, month_comparison_date),
    }

    bullets: list[SummaryBullet] = []
    broad = _multi_horizon_broad_bullet(pairs)
    if broad is not None:
        bullets.append(broad)
    bullets.append(_index_context_bullet(work, current_date, pairs))
    basket_bullets, active_baskets = _persistent_basket_bullets(work, current_date, pairs)
    bullets.extend(basket_bullets)
    downside, downside_symbols = _localized_downside_bullet(work, current_date, pairs, expected)
    if downside is not None:
        bullets.append(downside)
    mag7, mag7_symbol = _mag7_exception_bullet(work, current_date, pairs)
    if mag7 is not None:
        bullets.append(mag7)
    if len(bullets) == 1:
        bullets.append(_no_regime_change_bullet(pairs))
    bullets = bullets[:MAX_BULLETS]

    dashboard_month = _metric_stats(pairs["1M"], SUMMARY_GROUPS["Dashboard ex-index"], "1W", "atm_iv")
    index_daily = sum(_single_snapshot(pairs["1D"], symbol, "1W")["skew_25d_delta"] for symbol in ("SPY", "QQQ")) / 2.0
    conclusions: list[str] = []
    if dashboard_month is not None:
        if dashboard_month.delta_trimmed <= -5.0:
            conclusions.append("the dominant regime is multi-week IV compression, not broad fear")
        elif dashboard_month.delta_trimmed >= 5.0:
            conclusions.append("the dominant regime is multi-week IV expansion")
        else:
            conclusions.append("broad IV has not made a decisive monthly regime shift")
    if index_daily <= -0.50:
        conclusions.append("today's index put demand is an early caution signal")
    elif index_daily >= 0.50:
        conclusions.append("today's index skew leaned toward calls")
    exceptions = active_baskets + downside_symbols + ([mag7_symbol] if mag7_symbol else [])
    if exceptions:
        conclusions.append("the material exceptions are " + ", ".join(exceptions))
    bottom_line = "; ".join(conclusions).rstrip(".") + "."
    bottom_line = bottom_line[:1].upper() + bottom_line[1:]
    while len(" ".join([*(f"{bullet.title} {bullet.body}" for bullet in bullets), bottom_line]).split()) > MAX_SUMMARY_WORDS and len(bullets) > 2:
        bullets.pop()

    return DailySummary(
        snapshot_date=current_date,
        comparison_date=comparison_date,
        symbol_count=len(expected),
        expected_symbol_count=len(expected),
        bullets=tuple(bullets),
        bottom_line=bottom_line,
        week_comparison_date=week_comparison_date,
        month_comparison_date=month_comparison_date,
    )
