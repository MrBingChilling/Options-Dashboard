from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

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
PRESET_STATE_VERSION = "2026-08-13-filter-v2"
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


def display_series(history: pd.DataFrame, metric: str, change_mode: str) -> tuple[pd.DataFrame, str]:
    column = {"25Δ Call IV": "call_25d_iv", "25Δ Put IV": "put_25d_iv", "25Δ Skew": "skew_25d"}[metric]
    work = history[["snapshot_date", "symbol", column]].dropna().copy()
    work["value"] = work[column] * 100.0
    work = work.sort_values(["symbol", "snapshot_date"])
    if change_mode != "Level":
        work["value"] = work.groupby("symbol")["value"].diff({"1D Δ": 1, "1W Δ": 5, "1M Δ": 21}[change_mode])
        ylabel = "Change (vol points)"
    elif metric == "25Δ Skew":
        ylabel = "25Δ call IV − 25Δ put IV (vol points)"
    else:
        ylabel = "Implied volatility (%)"
    return work.pivot(index="snapshot_date", columns="symbol", values="value").sort_index(), ylabel


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


def sort_cross(cross: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Most positive first":
        return cross.sort_values(["skew_25d", "symbol"], ascending=[False, True])
    if mode == "Alphabetical":
        return cross.sort_values("symbol")
    if mode == "Preset":
        rank = {group: i for i, group in enumerate(PRESET_COLOR_ORDER)}
        out = cross.assign(_rank=cross["preset_group"].map(rank).fillna(len(rank)))
        return out.sort_values(["_rank", "skew_25d", "symbol"]).drop(columns="_rank")
    return cross.sort_values(["skew_25d", "symbol"])


def completed_symbols(store: SnapshotStore, symbols: list[str], session_date: date) -> set[str]:
    complete = {symbol: set() for symbol in symbols}
    for tenor in DAILY_TENORS:
        frame = latest_volatility(store, symbols, tenor)
        if frame.empty:
            continue
        work = frame.copy()
        work["snapshot_date"] = pd.to_datetime(work["snapshot_date"], errors="coerce").dt.date
        for symbol in work.loc[work["snapshot_date"] == session_date, "symbol"].astype(str):
            complete.setdefault(symbol.upper(), set()).add(tenor)
    needed = set(DAILY_TENORS)
    return {symbol for symbol, tenors in complete.items() if needed.issubset(tenors)}


def chart(cross: pd.DataFrame, metric_column: str, axis_title: str, label_title: str, signed: bool):
    return skew_metric_bar_chart(
        cross, metric_column, axis_title, label_title, signed,
        PRESET_COLOR_ORDER, PRESET_COLORS, REFERENCE_COLORS, INDEX_SYMBOLS,
    )


def sanitize_history(options: list[str]) -> None:
    current = [x for x in st.session_state.get("iv_history_symbols", []) if x in options]
    st.session_state["iv_history_symbols"] = current or options[: min(5, len(options))]


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
controls = st.columns([1, 1.25, 1.25, 1.6])
tenor = controls[0].segmented_control("Tenor", list(DAILY_TENORS), default="1M") or "1M"
sort_mode = controls[1].selectbox("Cross-section sort", ["Rank (low → high)", "Alphabetical", "Preset", "Most positive first"])
history_period = controls[2].segmented_control("History", ["1M", "3M", "6M", "1Y", "Max"], default="6M") or "6M"
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
        cross = sort_cross(attach_groups(cross, display_presets), sort_mode)
        newest = cross["snapshot_date"].max() if not cross.empty else None
        if newest is not None:
            st.caption(f"Latest saved session shown: {newest}. SPY, QQQ and the displayed non-index average use only the filtered rows.")
        if newest is not None and not cross[cross["snapshot_date"] < newest].empty:
            st.caption("Some tickers use an older available session; hover a bar or open details to see each snapshot date.")
        st.altair_chart(chart(cross, "skew_25d", "25Δ call IV − 25Δ put IV (vol points)", "Skew (vol pts)", True), use_container_width=True)
        st.caption("Bar colors identify preset groups. The edge touching the white zero line is square; only the outer value edge is rounded.")
        with st.expander("Cross-section details"):
            details = cross[["symbol", "preset_group", "snapshot_date", "actual_dte", "expiration", "spot", "call_25d_iv", "put_25d_iv", "skew_25d"]].copy()
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
history_controls = st.columns([1.4, 1, 2.2])
metric = history_controls[0].selectbox("Metric", ["25Δ Skew", "25Δ Call IV", "25Δ Put IV"])
change_mode = history_controls[1].segmented_control("View", ["Level", "1D Δ", "1W Δ", "1M Δ"], default="Level") or "Level"
if selected:
    sanitize_history(selected)
history_symbols = history_controls[2].multiselect("Historical tickers", options=selected, key="iv_history_symbols")
if store.enabled and history_symbols:
    end_date = datetime.now(EASTERN).date()
    try:
        hist = volatility_history(store, history_symbols, tenor, start_date=history_start(history_period, end_date), end_date=end_date)
    except SnapshotStoreError as exc:
        hist = pd.DataFrame()
        st.error(str(exc))
    if hist.empty:
        st.info("No history is stored for this selection yet. Daily scheduled snapshots will build it automatically.")
    else:
        chart_data, ylabel = display_series(hist, metric, change_mode)
        st.line_chart(chart_data, height=460, use_container_width=True)
        st.caption(ylabel + f" · target tenor: {tenor} ({DAILY_TENORS[tenor]} DTE).")

if not cross.empty:
    st.divider()
    st.subheader("25Δ Put IV")
    st.altair_chart(chart(cross, "put_25d_iv", "25Δ put implied volatility (%)", "Put IV (%)", False), use_container_width=True)
    st.divider()
    st.subheader("25Δ Call IV")
    st.altair_chart(chart(cross, "call_25d_iv", "25Δ call implied volatility (%)", "Call IV (%)", False), use_container_width=True)

st.divider()
with st.expander("Method & API-credit behavior"):
    st.markdown(f"""
- **25Δ skew:** `25Δ call IV − 25Δ put IV`, in volatility points.
- **Daily basket:** Dashboard contains all **{len(AUTO_SYMBOLS)}** symbols run by the automatic daily task.
- **Preset filter:** selecting one or multiple presets changes presentation only and consumes **0 MarketData credits**.
- **Daily tenors:** 1W and 1M target 7 and 30 DTE and use the available expiration closest to each target.
- **Pool average:** equal-weight average of displayed non-index stocks; SPY, QQQ and IWM are excluded.
- **Automatic data:** GitHub Actions writes daily rows to Supabase. Opening, refreshing or filtering this page does **not** call MarketData.
- **Manual button:** the explicit request button is the only page control that can call MarketData and it requests only missing displayed tickers.
- **Storage:** Supabase stores spot, DTE, expiration, 25Δ call IV, 25Δ put IV and 25Δ skew; raw chains are not stored.
""")
