from __future__ import annotations

import html
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from lightweight_charts_v5 import lightweight_charts_v5_component

from src import skew_metric_bar_chart
from src.archived_gamma_dashboard import render_archived_gamma_dashboard
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
)
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import latest_volatility, volatility_history

EASTERN = ZoneInfo("America/New_York")
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
}
PRESET_COLOR_ORDER = [
    "Neoclouds", "Mag 7", "Software", "Powers", "Index",
    "AI Photonics", "AI Fabless Semis", "AI Memory", "AI Fabs",
]
PRESET_COLORS = {
    "Neoclouds": "#9B8AFB",
    "Mag 7": "#69A9F8",
    "Software": "#57C5A5",
    "Powers": "#E7B65A",
    "Index": "#A9B1C3",
    "AI Photonics": "#F09A6B",
    "AI Fabless Semis": "#E87891",
    "AI Memory": "#C58AE8",
    "AI Fabs": "#67C2D4",
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
    for symbol in PRESET_DEFAULTS[group]:
        PRIMARY_PRESET_BY_SYMBOL.setdefault(symbol, group)


def symbols_for_presets(presets: list[str]) -> list[str]:
    symbols: list[str] = []
    for name in presets:
        symbols.extend(PRESET_DEFAULTS.get(name, []))
    return list(dict.fromkeys(symbols))


def chart_group(symbol: str, display_presets: list[str]) -> str:
    symbol = symbol.upper()
    chosen = [name for name in display_presets if name != "Dashboard"]
    if len(chosen) == 1 and symbol in PRESET_DEFAULTS.get(chosen[0], []):
        return chosen[0]
    for group in PRESET_COLOR_ORDER:
        if "Dashboard" not in display_presets and group not in display_presets:
            continue
        if symbol in PRESET_DEFAULTS[group]:
            return group
    return PRIMARY_PRESET_BY_SYMBOL.get(symbol, "Index")


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


def cross_section_chart(
    cross: pd.DataFrame,
    metric_column: str,
    axis_title: str,
    label_title: str,
    signed: bool,
    comparison: pd.DataFrame | None = None,
):
    if comparison is not None and not comparison.empty:
        return skew_metric_compare_bar_chart(
            cross,
            comparison,
            metric_column,
            axis_title,
            label_title,
            signed,
            PRESET_COLOR_ORDER,
            PRESET_COLORS,
            REFERENCE_COLORS,
            INDEX_SYMBOLS,
        )
    return skew_metric_bar_chart(
        cross,
        metric_column,
        axis_title,
        label_title,
        signed,
        PRESET_COLOR_ORDER,
        PRESET_COLORS,
        REFERENCE_COLORS,
        INDEX_SYMBOLS,
    )


def history_start(period: str, end_date: date) -> date | None:
    days = {"1M": 35, "3M": 100, "6M": 190, "1Y": 375}
    return end_date - timedelta(days=days[period]) if period in days else None


def history_source_options() -> list[str]:
    options: list[str] = []
    for group_name in AGGREGATION_GROUPS:
        for method in (EQUAL_WEIGHT, TRIMMED_MEAN):
            options.append(f"agg|{group_name}|{method}")
    options.extend(f"index|{symbol}" for symbol in INDEX_SYMBOLS)
    options.extend(f"ticker|{symbol}" for symbol in sorted(set(AUTO_SYMBOLS) - set(INDEX_SYMBOLS)))
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
        columns.append(
            pd.Series(
                values.to_numpy(),
                index=pd.to_datetime(source["snapshot_date"]),
                name=label,
            )
        )
    if not columns:
        return pd.DataFrame(), coverage
    return pd.concat(columns, axis=1).sort_index(), coverage


def historical_series_colors(series: pd.DataFrame) -> dict[str, str]:
    colors: dict[str, str] = {}
    for label in series.columns:
        text = str(label)
        seed = sum((i + 1) * ord(char) for i, char in enumerate(text.upper()))
        colors[text] = HISTORY_COLOR_PALETTE[seed % len(HISTORY_COLOR_PALETTE)]
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


st.set_page_config(page_title="IV & Skew", page_icon="↕", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top:1.0rem; padding-bottom:3rem; max-width:1500px;}
      .muted {color:#A8B3C7;}
      [data-testid="stMetric"] {background:#141B2D; border:1px solid #25304A; padding:.75rem; border-radius:.8rem;}
      @media (max-width:700px) {
        .block-container {padding-left:.7rem; padding-right:.7rem; padding-top:.7rem;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)
store = SnapshotStore(
    get_setting("SUPABASE_URL", ""),
    get_setting("SUPABASE_SECRET_KEY", get_setting("SUPABASE_SERVICE_ROLE_KEY", "")),
)

st.title("IV & Skew")
view = st.segmented_control(
    "View",
    ["IV & Skew", "Gamma & Volume"],
    default="IV & Skew",
    key="iv_skew_page_view",
    help="Switch between the volatility/skew dashboard and the single-ticker gamma/volume dashboard.",
) or "IV & Skew"

if view == "Gamma & Volume":
    render_archived_gamma_dashboard(store)
    st.stop()

st.caption(
    "Saved daily volatility-surface history. Filters, comparisons and historical aggregates read Supabase only and consume 0 MarketData credits."
)

display_presets = st.multiselect(
    "Preset filter",
    list(PRESET_DEFAULTS),
    default=["Dashboard"],
    key="iv_display_presets",
    help="Select one or multiple fixed presets. Preset composition is managed in the daily data-request configuration, not from this page.",
)
selected = symbols_for_presets(display_presets)

controls = st.columns([0.9, 1.35, 1.3])
tenor = controls[0].segmented_control("Tenor", list(DAILY_TENORS), default="1M") or "1M"
sort_mode = controls[1].selectbox(
    "Cross-section sort",
    ["Rank (low → high)", "Alphabetical", "Preset", "Rank (High → low)"],
    help="Applies independently to skew, put-IV and call-IV using each chart's own metric.",
)
compare_date = controls[2].date_input(
    "Compare",
    value=None,
    max_value=datetime.now(EASTERN).date() - timedelta(days=1),
    format="YYYY-MM-DD",
    help="Optional exact prior saved session. Current bars remain thin/solid; comparison bars are wider/faded.",
)

preset_label = ", ".join(display_presets) if display_presets else "No preset"
st.subheader(f"25Δ Put/Call Skew — {preset_label}")
cross = pd.DataFrame()
compare_cross = pd.DataFrame()
if not store.enabled:
    st.info("Configure Supabase to load saved IV/skew history.")
elif not selected:
    st.info("Select at least one preset.")
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
            st.caption(f"Latest saved session shown: {pd.Timestamp(newest).date()}.")
        if not compare_cross.empty:
            matched = cross["symbol"].isin(compare_cross["symbol"]).sum()
            st.caption(
                f"Compare: {compare_date}. Wide faded bar = prior; thin solid bar = current. "
                f"Historical matches: {matched}/{len(cross)} displayed ticker(s)."
            )
        st.altair_chart(
            cross_section_chart(
                skew_cross,
                "skew_25d",
                "25Δ call IV − 25Δ put IV (vol points)",
                "Skew (vol pts)",
                True,
                compare_cross,
            ),
            use_container_width=True,
        )
        with st.expander("Cross-section details"):
            detail_columns = [
                "symbol", "preset_group", "snapshot_date", "actual_dte", "expiration",
                "spot", "atm_iv", "call_10d_iv", "put_10d_iv", "skew_10d",
                "call_25d_iv", "put_25d_iv", "skew_25d",
            ]
            details = cross[[column for column in detail_columns if column in cross.columns]].copy()
            for column in (
                "atm_iv", "call_10d_iv", "put_10d_iv", "skew_10d",
                "call_25d_iv", "put_25d_iv", "skew_25d",
            ):
                if column in details.columns:
                    details[column] *= 100.0
            st.dataframe(details, hide_index=True, width="stretch")

st.divider()
st.subheader("Historical Trend")
st.caption(
    "One selector contains representative aggregates first, then indexes, then individual tickers. "
    "Select multiple entries only when you want to compare lines."
)
history_sources = st.multiselect(
    "Ticker / aggregation",
    options=history_source_options(),
    default=[DEFAULT_HISTORY_SOURCE],
    format_func=history_source_label,
    key="iv_history_series",
)
metric_col, tenor_col = st.columns([2.2, 1.2])
history_metric = metric_col.selectbox(
    "Metric",
    HISTORY_METRIC_ORDER,
    index=0,
    key="iv_history_metric",
)
history_tenor = tenor_col.segmented_control(
    "Tenor",
    list(DAILY_TENORS),
    default="1W",
    key="iv_history_tenor",
) or "1W"
view_col, period_col = st.columns([1.15, 1.45])
change_mode = view_col.segmented_control(
    "Change",
    ["Level", "1D Δ", "1W Δ", "1M Δ"],
    default="Level",
) or "Level"
history_period = period_col.segmented_control(
    "History",
    ["1M", "3M", "6M", "1Y", "Max"],
    default="6M",
    key="iv_history_period",
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
                "No usable observations are available for this metric/view. Older legacy rows can have ATM/10Δ fields blank."
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
                f"{history_metric} · {history_tenor} · {change_mode}. {first_date} → {last_date}. "
                "Drag/pinch to pan or zoom; drag the right price scale to stretch/compress Y."
            )
            if coverage:
                coverage_text = "; ".join(
                    f"{label}: median {median:.0%}, min {minimum:.0%}"
                    for label, (minimum, median) in coverage.items()
                )
                st.caption("Aggregate constituent coverage — " + coverage_text)
            st.caption(
                f"Aggregate dates require at least {MIN_COVERAGE:.0%} constituent coverage and at least {MIN_NAMES} usable names. "
                "Equal-weight mean uses all valid names; the 10% trimmed mean removes floor(10% × N) observations from each tail."
            )

if not cross.empty:
    st.divider()
    st.subheader("25Δ Put IV")
    put_cross = sort_cross(cross, sort_mode, "put_25d_iv")
    st.altair_chart(
        cross_section_chart(
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
        cross_section_chart(
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
with st.expander("Method & data behavior"):
    st.markdown(f"""
- **Preset composition is fixed in code/data collection.** The page no longer has edit-preset controls, member chips, add/reset controls, or an API request button.
- **25Δ skew:** `25Δ call IV − 25Δ put IV`.
- **10Δ skew:** `10Δ call IV − 10Δ put IV`.
- **25Δ Smile Convexity:** `(25Δ put IV + 25Δ call IV) / 2 − ATM IV`.
- **Tail Steepness:** average 10Δ wing IV minus average 25Δ wing IV.
- **Historical aggregates:** both equal-weight and 10% trimmed-mean versions are available. Dates require at least **{MIN_COVERAGE:.0%}** constituent coverage and **{MIN_NAMES}** usable names.
- **Gamma & Volume view:** loads the latest private archived bounded chain for one ticker, lets you choose one expiration and a strike range, and reconstructs gamma exposure plus volume by strike without calling MarketData.
- **Gamma display convention:** calls positive, puts negative. Aggregate GEX is cumulative net call-minus-put GEX across strikes. Gamma flip is modelled from the archived IV surface.
- **API credits:** opening the page, switching views, changing presets, changing chart settings, comparisons, historical aggregates, gamma, volume, expiration and strike range all consume **0 MarketData credits**.
- **Storage:** newer daily rows include ATM/10Δ/25Δ surface fields and a private compressed archived chain; older legacy rows can have newer fields blank.
""")
