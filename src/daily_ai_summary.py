from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from math import floor
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
GENERATOR_VERSION = "daily_ai_summary_v1"
REQUIRED_TENORS = ("1W", "1M")
REQUIRED_25D_COLUMNS = ("call_25d_iv", "put_25d_iv", "skew_25d")
VOLATILITY_COLUMNS = (
    "atm_iv",
    "call_25d_iv",
    "put_25d_iv",
    "skew_25d",
)
TRIM_FRACTION = 0.10

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
        return cls(
            snapshot_date=date.fromisoformat(str(record["snapshot_date"])[:10]),
            comparison_date=date.fromisoformat(str(record["comparison_date"])[:10]),
            symbol_count=int(record.get("symbol_count") or 0),
            expected_symbol_count=int(record.get("expected_symbol_count") or 0),
            bullets=bullets,
            bottom_line=str(payload.get("bottom_line", "")),
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


def _index_bullet(paired: pd.DataFrame) -> SummaryBullet:
    spy = _single_snapshot(paired, "SPY", "1W")
    qqq = _single_snapshot(paired, "QQQ", "1W")
    mean_skew_delta = (spy["skew_25d_delta"] + qqq["skew_25d_delta"]) / 2.0
    both_up = spy["spot_pct"] > 0 and qqq["spot_pct"] > 0
    if both_up and mean_skew_delta <= -0.50:
        title = "Indexes rose, but short-term downside protection became relatively richer."
    elif mean_skew_delta >= 0.50:
        title = "Short-term index skew shifted toward calls."
    elif mean_skew_delta <= -0.50:
        title = "Short-term index skew shifted toward puts."
    else:
        title = "Short-term index skew was broadly stable."
    body = (
        f"SPY {_rose_or_fell(spy['spot_pct'])}% while its 1W 25Δ skew moved "
        f"{_signed(spy['skew_25d_delta'])} vol points to {_signed(spy['skew_25d'])}; "
        f"QQQ {_rose_or_fell(qqq['spot_pct'])}% while skew moved "
        f"{_signed(qqq['skew_25d_delta'])} to {_signed(qqq['skew_25d'])}."
    )
    if (
        spy["put_25d_iv_delta"] > 0 > spy["call_25d_iv_delta"]
        and qqq["put_25d_iv_delta"] > 0 > qqq["call_25d_iv_delta"]
    ):
        body += " Put IV rose slightly while call IV declined in both indexes."
    return SummaryBullet(title, body)


def _broad_bullet(paired: pd.DataFrame) -> SummaryBullet | None:
    atm = _metric_stats(
        paired, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "atm_iv"
    )
    skew = _metric_stats(
        paired, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "skew_25d"
    )
    if atm is None or skew is None:
        return None
    if atm.delta_trimmed <= -1.0:
        title = "Volatility cooled across the broader dashboard."
    elif atm.delta_trimmed >= 1.0:
        title = "Volatility expanded across the broader dashboard."
    else:
        title = "Broad dashboard volatility was relatively stable."
    body = (
        f"Dashboard ex-index 1W ATM IV {_rose_or_fell(atm.delta_trimmed)} vol points "
        f"to {atm.current_trimmed:.2f} using the 10% trimmed mean, and "
        f"{_rose_or_fell(atm.delta_equal)} to {atm.current_equal:.2f} using the "
        f"equal-weight mean. 1W skew moved {_signed(skew.delta_trimmed)} to "
        f"{_signed(skew.current_trimmed)} trimmed and {_signed(skew.delta_equal)} to "
        f"{_signed(skew.current_equal)} equal-weight."
    )
    return SummaryBullet(title, body)


def _ai_infra_bullet(paired: pd.DataFrame) -> SummaryBullet | None:
    atm = _metric_stats(paired, AI_POOL_SYMBOLS, "1W", "atm_iv")
    skew = _metric_stats(paired, AI_POOL_SYMBOLS, "1W", "skew_25d")
    if atm is None or skew is None or abs(atm.delta_trimmed) < 1.5:
        return None
    if atm.delta_trimmed < 0:
        title = "AI infrastructure experienced strong short-term IV compression."
    else:
        title = "AI infrastructure experienced strong short-term IV expansion."
    body = (
        f"The basket's 1W ATM IV {_rose_or_fell(atm.delta_trimmed)} vol points "
        f"to {atm.current_trimmed:.2f} trimmed and {_rose_or_fell(atm.delta_equal)} "
        f"to {atm.current_equal:.2f} equal-weight. Its 1W skew moved only "
        f"{_signed(skew.delta_trimmed)} to {_signed(skew.current_trimmed)} trimmed, "
        "which points more to broad volatility repricing than a uniform directional shift."
    )
    return SummaryBullet(title, body)


def _material_basket_bullets(paired: pd.DataFrame) -> tuple[list[SummaryBullet], list[str]]:
    candidates: list[tuple[float, str, MetricStats, dict[str, float]]] = []
    for name in (
        "Neoclouds",
        "Mag 7",
        "Software",
        "Power",
        "AI Photonics",
        "AI Fabless Semis",
        "AI Memory",
        "AI Fabs",
    ):
        stats = _metric_stats(paired, SUMMARY_GROUPS[name], "1M", "skew_25d")
        if stats is None or abs(stats.delta_equal) < 1.50:
            continue
        changes = _member_skew_changes(paired, SUMMARY_GROUPS[name], "1M")
        candidates.append((abs(stats.delta_equal), name, stats, changes))
    candidates.sort(reverse=True, key=lambda item: item[0])

    bullets: list[SummaryBullet] = []
    selected_names: list[str] = []
    for _, name, stats, changes in candidates[:3]:
        selected_names.append(name)
        direction = _toward(stats.delta_equal)
        same_direction = sum(
            value >= 0 if stats.delta_equal >= 0 else value < 0
            for value in changes.values()
        )
        leader, leader_change = max(changes.items(), key=lambda item: abs(item[1]))
        leader_snapshot = _single_snapshot(paired, leader, "1M")
        title = f"{name} 1M positioning moved sharply toward {direction}."
        body = (
            f"Equal-weight basket skew moved {_signed(stats.delta_equal)} vol points "
            f"to {_signed(stats.current_equal)}; {same_direction}/{len(changes)} members "
            f"moved in the same direction. {leader} was the largest contributor, with "
            f"skew moving {_signed(leader_change)} to "
            f"{_signed(leader_snapshot['skew_25d'])}."
        )
        if stats.count >= 10 and abs(stats.delta_trimmed - stats.delta_equal) >= 0.50:
            body += (
                f" The 10% trimmed move was smaller at {_signed(stats.delta_trimmed)}, "
                "showing that outliers drove part of the equal-weight change."
            )
        bullets.append(SummaryBullet(title, body))
    return bullets, selected_names


def _downside_alert(paired: pd.DataFrame, symbols: Iterable[str]) -> tuple[SummaryBullet | None, str | None]:
    scored: list[tuple[float, str, str, dict[str, float]]] = []
    for symbol in symbols:
        one_week = _single_snapshot(paired, symbol, "1W")
        one_month = _single_snapshot(paired, symbol, "1M")
        values = sorted(
            [
                (one_week["skew_25d_delta"], "1W", one_week),
                (one_month["skew_25d_delta"], "1M", one_month),
            ],
            key=lambda item: item[0],
        )
        worst_delta, tenor, snapshot = values[0]
        other_delta = values[1][0]
        if worst_delta > -5.0:
            continue
        score = abs(worst_delta) + 0.5 * max(-other_delta, 0.0) + 0.15 * max(-snapshot["skew_25d"], 0.0)
        scored.append((score, str(symbol).upper(), tenor, snapshot))
    if not scored:
        return None, None
    _, symbol, tenor, snapshot = max(scored, key=lambda item: item[0])
    title = f"{symbol} produced the clearest downside-skew alert."
    body = (
        f"Its {tenor} skew {_rose_or_fell(snapshot['skew_25d_delta'])} vol points to "
        f"{_signed(snapshot['skew_25d'])}. Put IV moved "
        f"{_signed(snapshot['put_25d_iv_delta'])} points versus "
        f"{_signed(snapshot['call_25d_iv_delta'])} for call IV, making downside "
        "options substantially richer relative to calls."
    )
    return SummaryBullet(title, body), symbol


def _fragmentation_bullet(paired: pd.DataFrame) -> SummaryBullet | None:
    candidates: list[tuple[float, str, MetricStats, dict[str, float]]] = []
    for name in ("Neoclouds", "Mag 7", "Software", "Power", "AI Photonics"):
        changes = _member_skew_changes(paired, SUMMARY_GROUPS[name], "1W")
        stats = _metric_stats(paired, SUMMARY_GROUPS[name], "1W", "skew_25d")
        if not changes or stats is None:
            continue
        dispersion = max(changes.values()) - min(changes.values())
        if dispersion >= 10.0 and abs(stats.delta_equal) <= 2.0:
            candidates.append((dispersion, name, stats, changes))
    if not candidates:
        return None
    _, name, stats, changes = max(candidates, key=lambda item: item[0])
    strongest = max(changes.items(), key=lambda item: item[1])
    weakest = min(changes.items(), key=lambda item: item[1])
    return SummaryBullet(
        f"{name} positioning was highly fragmented.",
        f"{strongest[0]} shifted {_signed(strongest[1])} vol points toward calls, "
        f"while {weakest[0]} shifted {_signed(weakest[1])} toward puts. The basket's "
        f"equal-weight 1W skew moved only {_signed(stats.delta_equal)}, so its average "
        "hides substantial disagreement between members.",
    )


def _stable_bullet(paired: pd.DataFrame) -> SummaryBullet | None:
    stable: list[tuple[float, str, MetricStats, MetricStats, MetricStats, MetricStats]] = []
    for name in ("Mag 7", "Neoclouds", "Software", "Power", "AI Photonics"):
        skew_1w = _metric_stats(paired, SUMMARY_GROUPS[name], "1W", "skew_25d")
        skew_1m = _metric_stats(paired, SUMMARY_GROUPS[name], "1M", "skew_25d")
        atm_1w = _metric_stats(paired, SUMMARY_GROUPS[name], "1W", "atm_iv")
        atm_1m = _metric_stats(paired, SUMMARY_GROUPS[name], "1M", "atm_iv")
        if None in (skew_1w, skew_1m, atm_1w, atm_1m):
            continue
        assert skew_1w and skew_1m and atm_1w and atm_1m
        score = (
            abs(skew_1w.delta_equal)
            + abs(skew_1m.delta_equal)
            + 0.25 * abs(atm_1w.delta_equal)
            + 0.25 * abs(atm_1m.delta_equal)
        )
        if (
            abs(skew_1w.delta_equal) <= 0.50
            and abs(skew_1m.delta_equal) <= 0.50
            and abs(atm_1w.delta_equal) <= 1.50
            and abs(atm_1m.delta_equal) <= 1.50
        ):
            stable.append((score, name, skew_1w, skew_1m, atm_1w, atm_1m))
    if not stable:
        return None
    _, name, skew_1w, skew_1m, atm_1w, _ = min(stable, key=lambda item: item[0])
    return SummaryBullet(
        f"{name} was comparatively quiet.",
        f"Its 1W skew moved only {_signed(skew_1w.delta_equal)} to "
        f"{_signed(skew_1w.current_equal)}, 1M skew moved "
        f"{_signed(skew_1m.delta_equal)} to {_signed(skew_1m.current_equal)}, and "
        f"1W ATM IV moved {_signed(atm_1w.delta_equal)} vol points. There was no "
        "broad positioning change comparable with the day's more active baskets.",
    )


def _volatility_reset_bullet(paired: pd.DataFrame, symbols: Iterable[str]) -> SummaryBullet | None:
    resets: list[tuple[float, str, dict[str, float]]] = []
    for symbol in symbols:
        snapshot = _single_snapshot(paired, symbol, "1W")
        if abs(snapshot["spot_pct"]) >= 5.0 and snapshot["atm_iv_delta"] <= -10.0:
            resets.append((snapshot["atm_iv_delta"], str(symbol).upper(), snapshot))
    if not resets:
        return None
    resets.sort(key=lambda item: item[0])
    descriptions = [
        f"{symbol} ({snapshot['spot_pct']:+.2f}% spot; "
        f"{snapshot['atm_iv_delta']:+.2f} 1W ATM-IV points)"
        for _, symbol, snapshot in resets[:3]
    ]
    return SummaryBullet(
        "Unusual volatility resets deserve an individual review.",
        "; ".join(descriptions)
        + ". These look like event-type volatility resets or other sharp surface "
        "repricing, but the dashboard does not include a news or event calendar, so "
        "they should not be interpreted automatically as bullish signals.",
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
    comparison_date, current_date = sessions[-2], sessions[-1]
    paired = _paired_sessions(work, current_date, comparison_date)

    bullets: list[SummaryBullet] = [_index_bullet(paired)]
    for optional in (_broad_bullet(paired), _ai_infra_bullet(paired)):
        if optional is not None:
            bullets.append(optional)

    basket_bullets, active_baskets = _material_basket_bullets(paired)
    bullets.extend(basket_bullets)

    downside, downside_symbol = _downside_alert(paired, expected)
    if downside is not None:
        bullets.append(downside)
    fragmentation = _fragmentation_bullet(paired)
    if fragmentation is not None:
        bullets.append(fragmentation)
    stable = _stable_bullet(paired)
    if stable is not None:
        bullets.append(stable)
    reset = _volatility_reset_bullet(paired, expected)
    if reset is not None:
        bullets.append(reset)

    dashboard_atm = _metric_stats(
        paired, SUMMARY_GROUPS["Dashboard ex-index"], "1W", "atm_iv"
    )
    spy = _single_snapshot(paired, "SPY", "1W")
    qqq = _single_snapshot(paired, "QQQ", "1W")
    conclusions: list[str] = []
    if dashboard_atm is not None:
        conclusions.append(
            "Broad volatility cooled"
            if dashboard_atm.delta_trimmed < -1.0
            else "Broad volatility expanded"
            if dashboard_atm.delta_trimmed > 1.0
            else "Broad volatility was stable"
        )
    if (spy["skew_25d_delta"] + qqq["skew_25d_delta"]) / 2.0 < -0.50:
        conclusions.append("index downside options became richer relative to calls")
    elif (spy["skew_25d_delta"] + qqq["skew_25d_delta"]) / 2.0 > 0.50:
        conclusions.append("index skew moved toward calls")
    if active_baskets:
        conclusions.append(
            "the largest basket changes were concentrated in "
            + ", ".join(active_baskets[:-1])
            + (f" and {active_baskets[-1]}" if len(active_baskets) > 1 else active_baskets[0])
        )
    if downside_symbol:
        conclusions.append(f"{downside_symbol} was the clearest downside-skew alert")
    bottom_line = "; ".join(conclusions).rstrip(".") + "."
    bottom_line = bottom_line[:1].upper() + bottom_line[1:]

    return DailySummary(
        snapshot_date=current_date,
        comparison_date=comparison_date,
        symbol_count=len(expected),
        expected_symbol_count=len(expected),
        bullets=tuple(bullets),
        bottom_line=bottom_line,
    )
