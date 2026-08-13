from __future__ import annotations

import html
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from lightweight_charts_v5 import lightweight_charts_v5_component

from src import skew_metric_bar_chart
from src.config import get_setting
from src.marketdata_client import MarketDataClient, MarketDataError
from src.skew_collector import (
    AI_FABLESS_SEMI_SYMBOLS,
    AI_FABS_SYMBOLS,
    AI_MEMORY_SYMBOLS,
    AI_PHOTONICS_SYMBOLS,
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
PRESET_STATE_VERSION = "2026-08-13-history-tv-v6"
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
HISTORY_METRIC_COLUMNS = {
    "25Δ Skew": "skew_25d",
    "25Δ Call IV": "call_25d_iv",
    "25Δ Put IV": "put_25d_iv",
}
HISTORY_METRIC_SHORT = {
    "25Δ Skew": "Skew",
    "25Δ Call IV": "Call IV",
    "25Δ Put IV": "Put IV",
}
HISTORY_METRIC_PALETTES = {
    "Skew": ["#C084FC", "#E879F9", "#A78BFA", "#F472B6", "#D946EF", "#8B5CF6", "#FB7185", "#D8B4FE"],
    "Call IV": ["#60A5FA", "#38BDF8", "#22D3EE", "#818CF8", "#3B82F6", "#06B6D4", "#93C5FD", "#67E8F9"],
    "Put IV": ["#34D399", "#2DD4BF", "#A3E635", "#4ADE80", "#14B8A6", "#84CC16", "#6EE7B7", "#5EEAD4"],
}
SERIES_SEPARATOR = " · "
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


def display_series(history: pd.DataFrame, metrics: list[str], change_mode: str) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    for metric in metrics:
        column = HISTORY_METRIC_COLUMNS[metric]
        work = history[["snapshot_date", "symbol", column]].dropna().copy()
        if work.empty:
            continue
        work["value"] = work[column] * 100.0
        work = work.sort_values(["symbol", "snapshot_date"])
        if change_mode != "Level":
            work["value"] = work.groupby("symbol")["value"].diff({"1D Δ": 1, "1W Δ": 5, "1M Δ": 21}[change_mode])
        work["series_label"] = work["symbol"].astype(str) + SERIES_SEPARATOR + HISTORY_METRIC_SHORT[metric]
        frames.append(work[["snapshot_date", "series_label", "value"]])

    if not frames:
        return pd.DataFrame(), "Change (vol points)" if change_mode != "Level" else "IV / skew (vol points)"

    combined = pd.concat(frames, ignore_index=True)
    if change_mode != "Level":
        ylabel = "Change (vol points)"
    elif len(metrics) > 1:
        ylabel = "IV / skew (vol points)"
    elif metrics[0] == "25Δ Skew":
        ylabel = "25Δ call IV − 25Δ put IV (vol points)"
    else:
        ylabel = "Implied volatility (%)"
    return combined.pivot(index="snapshot_date", columns="series_label", values="value").sort_index(), ylabel


def historical_series_colors(series: pd.DataFrame) -> dict[str, str]:
    tickers: list[str] = []
    for label in series.columns:
        ticker = str(label).split(SERIES_SEPARATOR, 1)[0]
        if ticker not in tickers:
            tickers.append(ticker)
    ticker_rank = {ticker: i for i, ticker in enumerate(tickers)}

    colors: dict[str, str] = {}
    for label in series.columns:
        text = str(label)
        ticker, metric_short = text.split(SERIES_SEPARATOR, 1) if SERIES_SEPARATOR in text else (text, "Call IV")
        palette = HISTORY_METRIC_PALETTES.get(metric_short, HISTORY_METRIC_PALETTES["Call IV"])
        colors[text] = palette[ticker_rank.get(ticker, 0) % len(palette)]
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
    st.session_state.pop("iv_history_symbols", None)
    st.session_state.pop("iv_history_ticker_source", None)
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


def all_preset_symbols() -> list[str]:
    symbols: list[str] = []
    for name in PRESET_DEFAULTS:
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


def chart(cross: pd.DataFrame, metric_column: str, axis_title: str, label_title: str, signed: bool):
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
st.caption("Saved daily 25Δ call/put skew and IV. Preset filters only change the Supabase presentation and consume 0 MarketData credits.")

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
controls = st.columns([1, 1.35, 1.8])
tenor = controls[0].segmented_control("Tenor", list(DAILY_TENORS), default="1M") or "1M"
sort_mode = controls[1].selectbox(
    "Cross-section sort",
    ["Rank (low → high)", "Alphabetical", "Preset", "Rank (High → low)"],
    help="Applies independently to the skew, put-IV and call-IV tables using each table's own displayed metric.",
)
manual_request = controls[2].button(
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
        if newest is not None:
            st.caption(f"Latest saved session shown: {newest}. SPY, QQQ and the displayed non-index average use only the filtered rows.")
        if newest is not None and not cross[cross["snapshot_date"] < newest].empty:
            st.caption("Some tickers use an older available session; hover a bar or open details to see each snapshot date.")
        st.altair_chart(chart(skew_cross, "skew_25d", "25Δ call IV − 25Δ put IV (vol points)", "Skew (vol pts)", True), use_container_width=True)
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
st.subheader("Historical 25Δ IV & skew")
history_controls = st.columns([1.55, 1.15, 1.45])
metrics = history_controls[0].multiselect(
    "Metric",
    list(HISTORY_METRIC_COLUMNS),
    default=["25Δ Call IV"],
    key="iv_history_metrics",
    help="Select one or multiple metrics. Each ticker/metric combination is plotted as a separate colored series.",
)
change_mode = history_controls[1].segmented_control(
    "View",
    ["Level", "1D Δ", "1W Δ", "1M Δ"],
    default="Level",
    help="Level shows the stored metric. 1D Δ is the change from the prior stored session; 1W Δ uses 5 stored observations; 1M Δ uses 21 stored observations.",
) or "Level"
history_period = history_controls[2].segmented_control(
    "History",
    ["1M", "3M", "6M", "1Y", "Max"],
    default="6M",
    key="iv_history_period",
    help="Controls only the historical chart below.",
) or "6M"

history_ticker_controls = st.columns([1.2, 3.8])
history_ticker_source = history_ticker_controls[0].segmented_control(
    "Tickers",
    ["Filtered", "Custom"],
    default="Filtered",
    key="iv_history_ticker_source",
    help="Filtered mirrors the main Preset filter. Custom lets you select any saved dashboard ticker without changing the other graphs.",
) or "Filtered"
all_history_options = all_preset_symbols()

if history_ticker_source == "Filtered":
    history_symbols = list(selected)
    history_ticker_controls[1].markdown(
        f"**Historical tickers**  \nUsing all **{len(history_symbols)}** ticker(s) from the current Preset filter."
    )
else:
    current_history = [
        symbol
        for symbol in st.session_state.get("iv_history_symbols", selected)
        if symbol in all_history_options
    ]
    if "iv_history_symbols" not in st.session_state:
        st.session_state["iv_history_symbols"] = current_history or list(selected)
    elif st.session_state["iv_history_symbols"] != current_history:
        st.session_state["iv_history_symbols"] = current_history
    history_symbols = history_ticker_controls[1].multiselect(
        "Historical tickers",
        options=all_history_options,
        key="iv_history_symbols",
        help="Choose any ticker available in the dashboard presets. This only reads saved Supabase history and does not request MarketData.",
    )

if not metrics:
    st.info("Select at least one historical metric.")
elif store.enabled and history_symbols:
    end_date = datetime.now(EASTERN).date()
    try:
        hist = volatility_history(store, history_symbols, tenor, start_date=history_start(history_period, end_date), end_date=end_date)
    except SnapshotStoreError as exc:
        hist = pd.DataFrame()
        st.error(str(exc))
    if hist.empty:
        st.info("No history is stored for this selection yet. Daily scheduled snapshots will build it automatically.")
    else:
        chart_data, ylabel = display_series(hist, metrics, change_mode)
        if chart_data.dropna(how="all").empty:
            st.info("There are not enough saved observations for this change view yet.")
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
                f"Showing all {len(chart_data.index)} stored session(s) returned for the selected History range: {first_date} → {last_date}. "
                "Small dots mark each stored daily observation. Single-observation series render as a dot instead of a misleading horizontal line. "
                "Drag the chart to pan; drag the right price scale vertically to stretch/compress Y; pinch or mouse-wheel to zoom; double-click/double-tap the scale to reset."
            )
            st.caption(
                "Colors identify both ticker and metric: Call IV uses blue/cyan shades, Put IV uses green/teal shades, and skew uses purple/pink shades. "
                "The color key labels every series as TICKER · metric so the same ticker can be compared across multiple metrics."
            )
            if len(chart_data.columns) > 8:
                st.caption(
                    "The color key above is horizontally scrollable so a large ticker/metric basket does not cover the chart. "
                    "For 8 or fewer series, TradingView also shows ticker/metric labels directly on the price scale."
                )
            st.caption(
                "View: Level = the stored IV/skew value; 1D Δ = change versus the previous stored session; "
                "1W Δ = change versus 5 stored sessions; 1M Δ = change versus 21 stored sessions."
            )
elif store.enabled and history_ticker_source == "Filtered" and not selected:
    st.info("Select at least one preset above, or switch Historical tickers to Custom.")
elif store.enabled and history_ticker_source == "Custom" and not history_symbols:
    st.info("Choose at least one historical ticker.")

if not cross.empty:
    st.divider()
    st.subheader("25Δ Put IV")
    put_cross = sort_cross(cross, sort_mode, "put_25d_iv")
    st.altair_chart(chart(put_cross, "put_25d_iv", "25Δ put implied volatility (%)", "Put IV (%)", False), use_container_width=True)
    st.divider()
    st.subheader("25Δ Call IV")
    call_cross = sort_cross(cross, sort_mode, "call_25d_iv")
    st.altair_chart(chart(call_cross, "call_25d_iv", "25Δ call implied volatility (%)", "Call IV (%)", False), use_container_width=True)

st.divider()
with st.expander("Method & API-credit behavior"):
    st.markdown(f"""
- **25Δ skew:** `25Δ call IV − 25Δ put IV`, in volatility points.
- **Daily basket:** Dashboard contains all **{len(AUTO_SYMBOLS)}** symbols run by the automatic daily task.
- **Preset filter:** selecting one or multiple presets changes presentation only and consumes **0 MarketData credits**.
- **Historical ticker source:** Filtered mirrors the main preset filter; Custom can display any saved dashboard ticker(s) independently.
- **Historical metrics:** select one or several of 25Δ skew, call IV and put IV. Every ticker/metric combination is a separate series and uses a metric-specific color family.
- **Historical range:** the 1M/3M/6M/1Y/Max selector sits with the historical chart because it does not affect the cross-section tables.
- **Historical chart:** uses TradingView Lightweight Charts v5. Small point markers show every stored observation; one-observation series are dots only instead of fake horizontal lines. Large baskets use a one-row scrollable color key; small baskets also show price-scale ticker/metric labels. Native price/time scale dragging and pinch/mouse-wheel zoom are enabled.
- **Historical View:** Level shows the stored value. 1D Δ compares with the prior stored session, 1W Δ with 5 stored sessions earlier, and 1M Δ with 21 stored sessions earlier.
- **Cross-section sort:** the same control applies to all three bar tables; rank modes use each table's own metric.
- **Daily tenors:** 1W and 1M target 7 and 30 DTE and use the available expiration closest to each target.
- **Pool average:** equal-weight average of displayed non-index stocks; SPY, QQQ and IWM are excluded.
- **Automatic data:** GitHub Actions writes daily rows to Supabase. Opening, refreshing or filtering this page does **not** call MarketData.
- **Manual button:** the explicit request button is the only page control that can call MarketData and it requests only missing displayed tickers.
- **Storage:** Supabase stores spot, DTE, expiration, 25Δ call IV, 25Δ put IV and 25Δ skew; raw chains are not stored.
""")