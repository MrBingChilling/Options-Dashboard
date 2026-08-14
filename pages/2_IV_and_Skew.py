from __future__ import annotations

import html
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from lightweight_charts_v5 import lightweight_charts_v5_component

from src import skew_metric_bar_chart
from src.compare_charts import skew_metric_compare_bar_chart
from src.config import get_setting
from src.history_aggregates import (
    EQUAL_WEIGHT,
    MIN_COVERAGE,
    MIN_NAMES,
    TRIMMED_MEAN,
    aggregate_history,
    apply_change_mode,
    individual_history,
)
from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import (
    AI_FABLESS_SEMI_SYMBOLS,
    AI_FABS_SYMBOLS,
    AI_MEMORY_SYMBOLS,
    AI_PHOTONICS_SYMBOLS,
    AI_POOL_SYMBOLS,
    AUTO_SYMBOLS,
    DAILY_TENORS,
    INDEX_SYMBOLS,
    MAG7_SYMBOLS,
    NEOCLOUD_SYMBOLS,
    POWER_SYMBOLS,
    SOFTWARE_SYMBOLS,
    previous_weekday,
    skew_snapshots_from_chain,
)
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import latest_volatility, save_volatility_snapshots, volatility_history

EASTERN = ZoneInfo("America/New_York")
PRESET_STATE_VERSION = "2026-08-14-history-aggregates-v1"
PRESET_DEFAULTS = {
    "Dashboard": AUTO_SYMBOLS,
    "Neoclouds": NEOCLOUD_SYMBOLS,
    "Mag 7": MAG7_SYMBOLS,
    "Software": SOFTWARE_SYMBOLS,
    "Powers": POWER_SYMBOLS,
    "Index": INDEX_SYMBOLS,
    "AI Photonics": AI_PHOTONICS_SYMBOLS,
    "AI Fabless Semis": AI_FABLESS_SEMI_SYMBOLS,
    "AI Memory": AI_MEMORY_SYMBOLS,
    "AI Fabs": AI_FABS_SYMBOLS,
    "Custom": [],
}
PRESET_COLOR_ORDER = [
    "Neoclouds", "Mag 7", "Software", "Powers", "Index",
    "AI Photonics", "AI Fabless Semis", "AI Memory", "AI Fabs", "Custom",
]
PRESET_COLORS = {
    "Neoclouds": "#9B8AFB", "Mag 7": "#69A9F8", "Software": "#57C5A5",
    "Powers": "#E7B65A", "Index": "#A9B1C3", "AI Photonics": "#F09A6B",
    "AI Fabless Semis": "#E87891", "AI Memory": "#C58AE8", "AI Fabs": "#67C2D4",
    "Custom": "#8992A8",
}
REFERENCE_COLORS = {"SPY": "#7AB8FF", "QQQ": "#70D39A", "Pool avg": "#F4C45E"}
HISTORY_COLOR_PALETTE = [
    "#60A5FA", "#F472B6", "#34D399", "#FBBF24", "#A78BFA", "#FB7185",
    "#22D3EE", "#F97316", "#84CC16", "#E879F9", "#2DD4BF", "#818CF8",
    "#FACC15", "#4ADE80", "#38BDF8", "#C084FC", "#FB923C", "#F43F5E",
    "#14B8A6", "#A3E635", "#EAB308", "#D946EF", "#06B6D4", "#8B5CF6",
]
HISTORY_METRIC_ORDER = [
    "ATM IV",
    "25Δ Skew",
    "10Δ Skew",
    "25Δ Put IV",
    "25Δ Call IV",
    "10Δ Put IV",
    "10Δ Call IV",
    "25Δ Smile Convexity",
    "Tail Steepness (10Δ−25Δ)",
]
AGGREGATION_GROUPS = {
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
AGGREGATION_METHOD_LABELS = {
    EQUAL_WEIGHT: "Equal-weight mean",
    TRIMMED_MEAN: "10% trimmed mean",
}
DEFAULT_HISTORY_SOURCE = f"agg|AI Infra|{EQUAL_WEIGHT}"
PRIMARY_PRESET_BY_SYMBOL: dict[str, str] = {}
for group in PRESET_COLOR_ORDER:
    if group != "Custom":
        for symbol in PRESET_DEFAULTS[group]:
            PRIMARY_PRESET_BY_SYMBOL.setdefault(symbol, group)


def normalize_symbols(text: str) -> list[str]:
    return [x.strip().upper() for x in text.replace("\n", ",").replace(" ", ",").split(",") if x.strip()]


def history_start(period: str, end_date: date) -> date | None:
    days = {"1M": 35, "3M": 100, "6M": 190, "1Y": 375}
    return end_date - timedelta(days=days[period]) if period in days else None


def history_source_options() -> list[str]:
    options: list[str] = []
    for group_name in AGGREGATION_GROUPS:
        for method in (EQUAL_WEIGHT, TRIMMED_MEAN):
            options.append(f"agg|{group_name}|{method}")
    options.extend(f"index|{symbol}" for symbol in INDEX_SYMBOLS)
    individual = sorted(set(AUTO_SYMBOLS) - set(INDEX_SYMBOLS))
    options.extend(f"ticker|{symbol}" for symbol in individual)
    return options


def history_source_label(key: str) -> str:
    parts = key.split("|")
    if parts[0] == "agg" and len(parts) == 3:
        return f"{parts[1]} — {AGGREGATION_METHOD_LABELS.get(parts[2], parts[2])}"
    if parts[0] == "index" and len(parts) == 2:
        return f"{parts[1]} · Index"
    if parts[0] == "ticker" and len(parts) == 2:
        return parts[1]
    return key


def history_source_symbols(keys: list[str]) -> list[str]:
    symbols: list[str] = []
    for key in keys:
        parts = key.split("|")
        if parts[0] == "agg" and len(parts) == 3:
            symbols.extend(AGGREGATION_GROUPS.get(parts[1], []))
        elif parts[0] in {"index", "ticker"} and len(parts) == 2:
            symbols.append(parts[1])
    return list(dict.fromkeys(symbol.upper() for symbol in symbols))


def historical_chart_series(
    history: pd.DataFrame,
    source_keys: list[str],
    metric: str,
    change_mode: str,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    columns: list[pd.Series] = []
    coverage: dict[str, tuple[float, float]] = {}
    for key in source_keys:
        parts = key.split("|")
        label = history_source_label(key)
        if parts[0] == "agg" and len(parts) == 3:
            source = aggregate_history(
                history,
                AGGREGATION_GROUPS.get(parts[1], []),
                metric,
                parts[2],
            )
            if not source.empty and "coverage" in source.columns:
                coverage[label] = (
                    float(pd.to_numeric(source["coverage"], errors="coerce").min()),
                    float(pd.to_numeric(source["coverage"], errors="coerce").median()),
                )
        elif parts[0] in {"index", "ticker"} and len(parts) == 2:
            source = individual_history(history, parts[1], metric)
        else:
            continue
        source = apply_change_mode(source, change_mode)
        if source.empty:
            continue
        values = pd.to_numeric(source["value"], errors="coerce") * 100.0
        indexed = pd.Series(values.to_numpy(), index=pd.to_datetime(source["snapshot_date"]), name=label)
        columns.append(indexed)

    if not columns:
        return pd.DataFrame(), coverage
    return pd.concat(columns, axis=1).sort_index(), coverage


def historical_series_colors(series: pd.DataFrame) -> dict[str, str]:
    colors: dict[str, str] = {}
    palette_size = len(HISTORY_COLOR_PALETTE)
    for label in series.columns:
        text = str(label)
        seed = sum((i + 1) * ord(char) for i, char in enumerate(text.upper()))
        colors[text] = HISTORY_COLOR_PALETTE[seed % palette_size]
    return colors


def historical_tradingview_config(series: pd.DataFrame) -> list[dict[str, object]]:
    dense = len(series.columns) > 10
    show_price_labels = len(series.columns) <= 8
    colors = historical_series_colors(series)
    series_config: list[dict[str, object]] = []
    dates = pd.to_datetime(series.index, errors="coerce")
    for label in series.columns:
        values = pd.to_numeric(series[label], errors="coerce")
        data = [
            {"time": stamp.strftime("%Y-%m-%d"), "value": float(value)}
            for stamp, value in zip(dates, values)
            if not pd.isna(stamp) and not pd.isna(value)
        ]
        if not data:
            continue
        single_point = len(data) == 1
        series_config.append(
            {
                "type": "Line",
                "data": data,
                "options": {
                    "color": colors[str(label)],
                    "lineWidth": 1 if dense else 2,
                    "lineVisible": not single_point,
                    "pointMarkersVisible": True,
                    "pointMarkersRadius": 2.5,
                    "title": str(label) if show_price_labels else "",
                    "lastValueVisible": show_price_labels,
                    "priceLineVisible": False,
                    "crosshairMarkerVisible": True,
                    "priceFormat": {"type": "price", "precision": 2, "minMove": 0.01},
                },
            }
        )

    return [
        {
            "chart": {
                "layout": {
                    "background": {"color": "#081225"},
                    "textColor": "#D1D7E3",
                    "fontSize": 11,
                },
                "grid": {
                    "vertLines": {"color": "rgba(255,255,255,0.07)"},
                    "horzLines": {"color": "rgba(255,255,255,0.07)"},
                },
                "crosshair": {"mode": 0},
                "handleScale": {
                    "mouseWheel": True,
                    "pinch": True,
                    "axisPressedMouseMove": {"time": True, "price": True},
                    "axisDoubleClickReset": {"time": True, "price": True},
                },
                "handleScroll": {
                    "mouseWheel": True,
                    "pressedMouseMove": True,
                    "horzTouchDrag": True,
                    "vertTouchDrag": True,
                },
                "timeScale": {"visible": True},
                "priceScaleBorderColor": "rgba(255,255,255,0.18)",
                "timeScaleBorderColor": "rgba(255,255,255,0.18)",
            },
            "series": series_config,
            "height": 570,
        }
    ]


def historical_color_key(series: pd.DataFrame) -> str:
    colors = historical_series_colors(series)
    chips: list[str] = []
    for label in series.columns:
        values = pd.to_numeric(series[label], errors="coerce").dropna()
        if values.empty:
            continue
        color = colors[str(label)]
        safe_label = html.escape(str(label))
        latest = float(values.iloc[-1])
        chips.append(
            f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:6px;padding:4px 7px;'
            f'border:1px solid rgba(255,255,255,.12);border-radius:7px;font-size:11px;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{color};display:inline-block;"></span>'
            f'<b>{safe_label}</b><span style="color:#AEB8C9;">{latest:.2f}</span></span>'
        )
    return (
        '<div style="overflow-x:auto;white-space:nowrap;padding:2px 0 7px;scrollbar-width:thin;">'
        + "".join(chips)
        + "</div>"
    )


def preset_token(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_").replace("+", "plus")


def members_key(name: str) -> str:
    return f"iv_preset_members_{preset_token(name)}"


def selection_key(name: str) -> str:
    return f"iv_preset_selection_{preset_token(name)}"


def reset_preset_state_once() -> None:
    if st.session_state.get("iv_preset_state_version") == PRESET_STATE_VERSION:
        return
    for name in PRESET_DEFAULTS:
        st.session_state.pop(members_key(name), None)
        st.session_state.pop(selection_key(name), None)
    for old_key in (
        "iv_history_symbols",
        "iv_history_ticker_source",
        "iv_history_metrics",
        "iv_history_series",
        "iv_history_metric",
        "iv_history_tenor",
    ):
        st.session_state.pop(old_key, None)
    st.session_state["iv_preset_state_version"] = PRESET_STATE_VERSION


def ensure_preset(name: str) -> list[str]:
    mk, sk = members_key(name), selection_key(name)
    if mk not in st.session_state:
        st.session_state[mk] = list(PRESET_DEFAULTS[name])
    if sk not in st.session_state:
        st.session_state[sk] = list(st.session_state[mk])
    return list(st.session_state[mk])


def symbols_for_presets(presets: list[str]) -> list[str]:
    symbols: list[str] = []
    for name in presets:
        symbols.extend(ensure_preset(name))
    return list(dict.fromkeys(symbols))


def chart_group(symbol: str, display_presets: list[str]) -> str:
    symbol = symbol.upper()
    if display_presets == ["Custom"]:
        return "Custom"
    chosen = [x for x in display_presets if x not in {"Dashboard", "Custom"}]
    if len(chosen) == 1 and symbol in st.session_state.get(members_key(chosen[0]), PRESET_DEFAULTS[chosen[0]]):
        return chosen[0]
    for group in PRESET_COLOR_ORDER:
        if group == "Custom":
            continue
        if "Dashboard" not in display_presets and group not in display_presets:
            continue
        if symbol in st.session_state.get(members_key(group), PRESET_DEFAULTS[group]):
            return group
    return PRIMARY_PRESET_BY_SYMBOL.get(symbol, "Custom")


def attach_groups(cross: pd.DataFrame, display_presets: list[str]) -> pd.DataFrame:
    out = cross.copy()
    out["preset_group"] = [chart_group(str(symbol), display_presets) for symbol in out["symbol"]]
    return out


def sort_cross(cross: pd.DataFrame, mode: str, metric_column: str) -> pd.DataFrame:
    if mode == "Rank (High → low)":
        return cross.sort_values([metric_column, "symbol"], ascending=[False, True])
    if mode == "Alphabetical":
        return cross.sort_values("symbol")
    if mode == "Preset":
        rank = {group: i for i, group in enumerate(PRESET_COLOR_ORDER)}
        out = cross.assign(_rank=cross["preset_group"].map(rank).fillna(len(rank)))
        return out.sort_values(["_rank", metric_column, "symbol"], ascending=[True, True, True]).drop(columns="_rank")
    return cross.sort_values([metric_column, "symbol"], ascending=[True, True])


def completed_symbols(store: SnapshotStore, symbols: list[str], session_date: date) -> set[str]:
    complete = {symbol: set() for symbol in symbols}
    for tenor_name in DAILY_TENORS:
        frame = latest_volatility(store, symbols, tenor_name)
        if frame.empty:
            continue
        work = frame.copy()
        work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce").dt.date
        for symbol in work.loc[work["snapshot_date"] == session_date, "symbol"].astype(str):
            complete.setdefault(symbol.upper(), set()).add(tenor_name)
    needed = set(DAILY_TENORS)
    return {symbol for symbol, tenors in complete.items() if needed.issubset(tenors)}


def chart(
    cross: pd.DataFrame,
    metric_column: str,
    axis_title: str,
    label_title: str,
    signed: bool,
    comparison: pd.DataFrame | None = None,
):
    if comparison is not None and not comparison.empty:
        return skew_metric_compare_bar_chart(
            cross, comparison, metric_column, axis_title, label_title, signed,
            PRESET_COLOR_ORDER, PRESET_COLORS, REFERENCE_COLORS, INDEX_SYMBOLS,
        )
    return skew_metric_bar_chart(
        cross, metric_column, axis_title, label_title, signed,
        PRESET_COLOR_ORDER, PRESET_COLORS, REFERENCE_COLORS, INDEX_SYMBOLS,
    )


st.set_page_config(page_title="IV & Skew", page_icon="↕", layout="wide")
st.markdown(
    """<style>.block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1500px}.muted{color:#A8B3C7}</style>""",
    unsafe_allow_html=True,
)
reset_preset_state_once()
marketdata_token = get_setting("MARKETDATA_TOKEN", "")
store = SnapshotStore(
    get_setting("SUPABASE_URL", ""),
    get_setting("SUPABASE_SECRET_KEY", get_setting("SUPABASE_SERVICE_ROLE_KEY", "")),
)

st.title("IV & Skew")
st.caption("Saved daily volatility-surface history. Preset filters and historical aggregates read Supabase only and consume 0 MarketData credits.")

filter_col, edit_col = st.columns([2.4, 1.2])
display_presets = filter_col.multiselect(
    "Preset filter",
    list(PRESET_DEFAULTS),
    default=["Dashboard"],
    key="iv_display_presets",
    help="Select one or multiple presets. This only filters saved Supabase rows and never sends a MarketData request.",
)
edit_preset = edit_col.selectbox("Edit preset", list(PRESET_DEFAULTS), help="Choose a preset to customize for this app session.")

members = ensure_preset(edit_preset)
mk, sk = members_key(edit_preset), selection_key(edit_preset)
add_col, add_button_col, reset_col = st.columns([5.0, 1.15, 1.1])
custom_text = add_col.text_input(
    "Add tickers to edited preset",
    placeholder="e.g. ORCL, CRWV, NBIS",
    key=f"iv_add_tickers_{preset_token(edit_preset)}",
)
if add_button_col.button("Add", width="stretch", key=f"iv_add_button_{preset_token(edit_preset)}"):
    additions = normalize_symbols(custom_text)
    if additions:
        updated = list(dict.fromkeys(members + additions))
        st.session_state[mk] = updated
        st.session_state[sk] = updated
        st.rerun()
    st.warning("Enter at least one ticker to add.")
if reset_col.button("Reset preset", width="stretch", key=f"iv_reset_button_{preset_token(edit_preset)}"):
    st.session_state[mk] = list(PRESET_DEFAULTS[edit_preset])
    st.session_state[sk] = list(PRESET_DEFAULTS[edit_preset])
    st.rerun()

members = list(st.session_state[mk])
edited = st.multiselect(
    "Preset members",
    members,
    key=sk,
    placeholder="Add tickers above",
    help="Removing chips changes only this app session, not the automatic daily basket.",
)
if edited != members:
    st.session_state[mk] = list(edited)

selected = symbols_for_presets(display_presets)
controls = st.columns([0.9, 1.25, 1.2, 1.8])
tenor = controls[0].segmented_control("Tenor", list(DAILY_TENORS), default="1M") or "1M"
sort_mode = controls[1].selectbox(
    "Cross-section sort",
    ["Rank (low → high)", "Alphabetical", "Preset", "Rank (High → low)"],
    help="Applies independently to the skew, put-IV and call-IV tables using each table's own displayed metric.",
)
compare_date = controls[2].date_input(
    "Compare",
    value=None,
    max_value=datetime.now(EASTERN).date() - timedelta(days=1),
    format="YYYY-MM-DD",
    help=(
        "Optional past saved session. When selected, all three bar charts show that session as a wider faded bar "
        "behind the thinner solid current bar. This reads Supabase only and uses 0 MarketData credits."
    ),
)
manual_request = controls[3].button(
    "Request missing displayed from MarketData",
    type="primary",
    width="stretch",
    help="This explicit button is the only control on this page that can call MarketData.",
)

if manual_request:
    if not marketdata_token:
        st.error("MARKETDATA_TOKEN is not configured.")
    elif not store.enabled:
        st.error("Supabase is not configured.")
    elif not selected:
        st.warning("Select at least one preset with members.")
    else:
        requested_date = previous_weekday(datetime.now(EASTERN).date())
        try:
            complete = completed_symbols(store, selected, requested_date)
        except SnapshotStoreError as exc:
            complete = set()
            st.error(f"Could not check Supabase: {exc}")
        missing = [symbol for symbol in selected if symbol not in complete]
        if not missing:
            st.success(f"All displayed tickers already have 1W + 1M data for {requested_date}. MarketData requests: 0.")
        else:
            client = MarketDataClient(marketdata_token)
            progress = st.progress(0.0, text=f"{len(missing)} ticker(s) need MarketData…")
            failures, saved = [], 0
            for i, symbol in enumerate(missing, start=1):
                try:
                    result = client.fetch_skew_chain(symbol, requested_date)
                    save_volatility_snapshots(store, skew_snapshots_from_chain(symbol, result.snapshot_date, result.data))
                    saved += 1
                except (MarketDataError, SnapshotStoreError, ValueError) as exc:
                    failures.append(f"{symbol}: {exc}")
                progress.progress(i / len(missing), text=f"Processed {i}/{len(missing)}")
            progress.empty()
            if saved:
                st.success(f"Saved {saved} ticker(s). Skipped {len(complete)} already-complete ticker(s).")
            if failures:
                st.warning("Some tickers failed:\n\n" + "\n\n".join(failures))

preset_label = ", ".join(display_presets) if display_presets else "No preset"
st.subheader(f"25Δ Put/Call Skew — {preset_label}")
cross = pd.DataFrame()
compare_cross = pd.DataFrame()
if not store.enabled:
    st.info("Configure Supabase to load saved IV/skew history.")
elif not selected:
    st.info("Select at least one preset with members above.")
else:
    try:
        latest = latest_volatility(store, selected, tenor)
    except SnapshotStoreError as exc:
        latest = pd.DataFrame()
        st.error(str(exc))
    if latest.empty:
        st.info("No saved snapshots exist for this selection yet.")
    else:
        cross = latest.dropna(subset=["skew_25d", "call_25d_iv", "put_25d_iv"]).copy()
        cross = attach_groups(cross, display_presets)
        skew_cross = sort_cross(cross, sort_mode, "skew_25d")
        newest = cross["snapshot_date"].max() if not cross.empty else None
        newest_date = pd.Timestamp(newest).date() if newest is not None else None
        if compare_date is not None and newest_date is not None:
            if compare_date >= newest_date:
                st.warning(f"Compare must be earlier than the current saved session ({newest_date}).")
            else:
                try:
                    compare_rows = volatility_history(
                        store,
                        selected,
                        tenor,
                        start_date=compare_date,
                        end_date=compare_date,
                        limit=max(1000, len(selected) * 4),
                    )
                except SnapshotStoreError as exc:
                    compare_rows = pd.DataFrame()
                    st.error(str(exc))
                if compare_rows.empty:
                    st.info(f"No saved {tenor} snapshot exists for the displayed tickers on {compare_date}.")
                else:
                    compare_cross = compare_rows.dropna(
                        subset=["skew_25d", "call_25d_iv", "put_25d_iv"]
                    ).copy()
                    compare_cross = attach_groups(compare_cross, display_presets)
        if newest is not None:
            st.caption(f"Latest saved session shown: {newest}. SPY, QQQ and the displayed non-index average use only the filtered rows.")
        if not compare_cross.empty:
            matched = cross["symbol"].isin(compare_cross["symbol"]).sum()
            st.caption(
                f"Compare: {compare_date}. Wide faded bar = {compare_date}; thin solid bar = current. "
                f"Historical matches available for {matched}/{len(cross)} displayed ticker(s)."
            )
        st.altair_chart(
            chart(
                skew_cross,
                "skew_25d",
                "25Δ call IV − 25Δ put IV (vol points)",
                "Skew (vol pts)",
                True,
                compare_cross,
            ),
            use_container_width=True,
        )
        st.caption("Bar colors identify preset groups. The edge touching the white zero line is square; only the outer value edge is rounded.")
        with st.expander("Cross-section details"):
            details = skew_cross[["symbol", "preset_group", "snapshot_date", "actual_dte", "expiration", "spot", "call_25d_iv", "put_25d_iv", "skew_25d"]].copy()
            for column in ("call_25d_iv", "put_25d_iv", "skew_25d"):
                details[column] *= 100.0
            details = details.rename(columns={
                "symbol": "Ticker", "preset_group": "Preset", "snapshot_date": "Date", "actual_dte": "DTE",
                "expiration": "Expiration", "spot": "Spot", "call_25d_iv": "25Δ Call IV %",
                "put_25d_iv": "25Δ Put IV %", "skew_25d": "25Δ Skew vol pts",
            })
            st.dataframe(details, hide_index=True, width="stretch")

st.divider()
st.subheader("Historical Trend")
st.caption(
    "Choose representative basket aggregates, indexes, or individual tickers in one selector. "
    "Aggregations are calculated locally from saved constituent rows and use 0 MarketData credits."
)
source_options = history_source_options()
history_sources = st.multiselect(
    "Ticker / aggregation",
    options=source_options,
    default=[DEFAULT_HISTORY_SOURCE],
    format_func=history_source_label,
    key="iv_history_series",
    help=(
        "Ordered as aggregations first, then SPY/QQQ/IWM, then individual tickers. "
        "Select multiple entries to overlay multiple lines. Every basket is available as both an equal-weight mean "
        "and a 10% trimmed mean."
    ),
)

metric_col, tenor_col = st.columns([2.2, 1.2])
history_metric = metric_col.selectbox(
    "Metric",
    HISTORY_METRIC_ORDER,
    index=0,
    key="iv_history_metric",
    help=(
        "ATM, 10Δ and 25Δ metrics use saved surface fields. Smile Convexity and Tail Steepness are derived locally "
        "from those saved fields and do not make API requests."
    ),
)
history_tenor = tenor_col.segmented_control(
    "Tenor",
    list(DAILY_TENORS),
    default="1W",
    key="iv_history_tenor",
) or "1W"

view_col, period_col = st.columns([1.15, 1.45])
change_mode = view_col.segmented_control(
    "View",
    ["Level", "1D Δ", "1W Δ", "1M Δ"],
    default="Level",
    help="Changes are calculated after each ticker or aggregate level series is constructed.",
) or "Level"
history_period = period_col.segmented_control(
    "History",
    ["1M", "3M", "6M", "1Y", "Max"],
    default="6M",
    key="iv_history_period",
    help="Controls only the historical chart below.",
) or "6M"

if not history_sources:
    st.info("Choose at least one ticker or aggregation.")
elif store.enabled:
    history_symbols = history_source_symbols(history_sources)
    end_date = datetime.now(EASTERN).date()
    try:
        hist = volatility_history(
            store,
            history_symbols,
            history_tenor,
            start_date=history_start(history_period, end_date),
            end_date=end_date,
            limit=max(50000, len(history_symbols) * 400),
        )
    except SnapshotStoreError as exc:
        hist = pd.DataFrame()
        st.error(str(exc))
    if hist.empty:
        st.info("No saved history is available for this selection yet.")
    else:
        chart_data, coverage = historical_chart_series(
            hist,
            history_sources,
            history_metric,
            change_mode,
        )
        if chart_data.dropna(how="all").empty:
            st.info(
                "No usable observations are available for this metric/view. Older legacy rows can have ATM/10Δ fields blank; "
                "25Δ history remains available where it was previously stored."
            )
        else:
            st.markdown(historical_color_key(chart_data), unsafe_allow_html=True)
            lightweight_charts_v5_component(
                name="Historical IV & skew",
                charts=historical_tradingview_config(chart_data),
                height=570,
                zoom_level=max(10000, len(chart_data.index) + 50),
                configure_time_scale=False,
                key="iv_history_tradingview",
            )
            first_date = pd.to_datetime(chart_data.index.min()).date()
            last_date = pd.to_datetime(chart_data.index.max()).date()
            st.caption(
                f"{history_metric} · {history_tenor} · {change_mode}. Showing {first_date} → {last_date}. "
                "Small dots mark stored-session observations; single-observation series render as a dot only."
            )
            if coverage:
                coverage_text = "; ".join(
                    f"{label}: median {median:.0%}, min {minimum:.0%}"
                    for label, (minimum, median) in coverage.items()
                )
                st.caption("Aggregate constituent coverage — " + coverage_text)
            st.caption(
                f"Aggregate dates require at least {MIN_COVERAGE:.0%} constituent coverage and at least {MIN_NAMES} usable names. "
                "Equal-weight mean averages all valid constituents. 10% trimmed mean removes floor(10% × N) values from each tail before averaging; "
                "for small baskets where that is zero, the two methods are naturally identical."
            )
elif not store.enabled:
    st.info("Configure Supabase to load saved historical data.")

if not cross.empty:
    st.divider()
    st.subheader("25Δ Put IV")
    put_cross = sort_cross(cross, sort_mode, "put_25d_iv")
    st.altair_chart(
        chart(
            put_cross,
            "put_25d_iv",
            "25Δ put implied volatility (%)",
            "Put IV (%)",
            False,
            compare_cross,
        ),
        use_container_width=True,
    )
    st.divider()
    st.subheader("25Δ Call IV")
    call_cross = sort_cross(cross, sort_mode, "call_25d_iv")
    st.altair_chart(
        chart(
            call_cross,
            "call_25d_iv",
            "25Δ call implied volatility (%)",
            "Call IV (%)",
            False,
            compare_cross,
        ),
        use_container_width=True,
    )

st.divider()
with st.expander("Method & API-credit behavior"):
    st.markdown(f"""
- **25Δ skew:** `25Δ call IV − 25Δ put IV`, in volatility points. Negative values mean the put wing is richer than the call wing.
- **ATM IV:** stored near-the-money implied volatility for the expiration closest to the selected 1W or 1M target tenor.
- **10Δ skew:** `10Δ call IV − 10Δ put IV`; it measures the farther-tail asymmetry and can be blank on legacy rows or when the bounded chain does not contain a reliable 10Δ pair.
- **25Δ Smile Convexity:** `(25Δ put IV + 25Δ call IV) / 2 − ATM IV`. Positive means the 25Δ wings trade richer than ATM.
- **Tail Steepness (10Δ−25Δ):** average 10Δ wing IV minus average 25Δ wing IV. Positive means the far tails are richer than the nearer wings.
- **Historical selector:** one multi-select replaces the old Filtered/Custom switch. Options are ordered **aggregations → indexes → individual tickers** and one **AI Infra — Equal-weight mean** line is selected by default.
- **Aggregation method 1 — Equal-weight mean:** arithmetic mean of all usable constituent observations on that saved session.
- **Aggregation method 2 — 10% trimmed mean:** sorts usable constituent values and removes `floor(10% × N)` from each tail before averaging. This is more resistant to one-stock IV spikes.
- **Aggregate coverage:** a date is plotted only when at least **{MIN_COVERAGE:.0%}** of basket members and at least **{MIN_NAMES}** names have usable values, reducing composition jumps caused by sparse data.
- **Representative basket caveat:** these are constituent aggregates, not a tradable index-option implied volatility. They intentionally do not pretend to include cross-stock correlation like a true index IV would.
- **Historical changes:** Level is the stored/derived level. 1D Δ, 1W Δ and 1M Δ use lags of 1, 5 and 21 stored observations **after** the aggregate level is constructed.
- **Daily basket:** Dashboard contains all **{len(AUTO_SYMBOLS)}** symbols run by the automatic daily task.
- **Preset filter:** selecting one or multiple presets changes presentation only and consumes **0 MarketData credits**.
- **Compare:** an earlier exact saved session overlays the three cross-section bar charts; no date substitution and no MarketData request.
- **Historical range:** 1M/3M/6M/1Y/Max affects only the historical chart.
- **Daily tenors:** 1W and 1M target 7 and 30 DTE and use the available expiration closest to each target.
- **Automatic data:** GitHub Actions writes daily rows to Supabase. Opening, refreshing, filtering, changing metrics, or changing aggregation methods does **not** call MarketData.
- **Manual button:** the explicit request button remains the only page control that can call MarketData and it requests only missing displayed tickers.
- **Storage:** optimized rows store ATM, 10Δ and 25Δ surface metrics plus archive metadata; newly fetched bounded chains are preserved in private compressed storage. Older legacy rows can have newer fields blank and are not silently refetched just to populate the chart.
""")
