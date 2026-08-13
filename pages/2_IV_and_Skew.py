from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from src.config import get_setting
from src.marketdata_client import MarketDataClient, MarketDataError
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility import TENORS, snapshot_from_chain
from src.volatility_storage import latest_volatility, save_volatility_snapshot, volatility_history


EASTERN = ZoneInfo("America/New_York")

# Match the Herman Jin reference pool supplied by the user.
SEMIS_PRESET = [
    "AEHR",
    "STM",
    "AXTI",
    "ANET",
    "NVDA",
    "ON",
    "SNDK",
    "KLAC",
    "AVGO",
    "CIEN",
    "AAOI",
    "COHR",
    "GFS",
    "LITE",
    "NOK",
    "AMD",
    "LRCX",
    "AMAT",
    "WOLF",
    "MU",
    "INTC",
    "SKHY",
    "CBRS",
    "ASML",
]
AI_PRESET = ["NVDA", "AVGO", "AMD", "MU", "MSFT", "META", "AMZN", "GOOG", "ORCL", "CRWV", "NBIS"]
INDEX_PRESET = ["SPY", "QQQ", "IWM"]
PRESET_DEFAULTS = {
    "AI / Semis": SEMIS_PRESET,
    "AI MegaCap": AI_PRESET,
    "Indexes": INDEX_PRESET,
    "Custom": [],
}


def previous_weekday(value: date) -> date:
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def normalize_symbols(text: str) -> list[str]:
    return [
        token.strip().upper()
        for token in text.replace("\n", ",").replace(" ", ",").split(",")
        if token.strip()
    ]


def history_start(period: str, end_date: date) -> date | None:
    days = {"1M": 35, "3M": 100, "6M": 190, "1Y": 375}
    return end_date - timedelta(days=days[period]) if period in days else None


def display_series(history: pd.DataFrame, metric: str, change_mode: str) -> tuple[pd.DataFrame, str]:
    metric_map = {
        "ATM IV": "atm_iv",
        "25Δ Call IV": "call_25d_iv",
        "25Δ Put IV": "put_25d_iv",
        "25Δ Skew": "skew_25d",
    }
    column = metric_map[metric]
    work = history[["snapshot_date", "symbol", column]].dropna().copy()
    work["value"] = work[column] * 100.0
    work = work.sort_values(["symbol", "snapshot_date"])

    if change_mode != "Level":
        periods = {"1D Δ": 1, "1W Δ": 5, "1M Δ": 21}[change_mode]
        work["value"] = work.groupby("symbol")["value"].diff(periods)
        ylabel = "Change (vol points)"
    elif metric == "25Δ Skew":
        ylabel = "Call IV − put IV (vol points)"
    else:
        ylabel = "Implied volatility (%)"

    pivot = work.pivot(index="snapshot_date", columns="symbol", values="value").sort_index()
    return pivot, ylabel


def preset_token(name: str) -> str:
    return name.lower().replace("/", "_").replace(" ", "_")


def preset_members_key(name: str) -> str:
    return f"iv_preset_members_{preset_token(name)}"


def preset_selection_key(name: str) -> str:
    return f"iv_preset_selection_{preset_token(name)}"


def ensure_preset_state(name: str) -> list[str]:
    members_key = preset_members_key(name)
    selection_key = preset_selection_key(name)
    if members_key not in st.session_state:
        st.session_state[members_key] = list(PRESET_DEFAULTS[name])
    if selection_key not in st.session_state:
        st.session_state[selection_key] = list(st.session_state[members_key])
    return list(st.session_state[members_key])


def skew_cross_section_chart(cross: pd.DataFrame) -> alt.Chart:
    chart_data = cross.copy()
    chart_data["skew_vol_pts"] = chart_data["skew_25d"] * 100.0
    chart_data["call_25d_iv_pct"] = chart_data["call_25d_iv"] * 100.0
    chart_data["put_25d_iv_pct"] = chart_data["put_25d_iv"] * 100.0
    chart_data["atm_iv_pct"] = chart_data["atm_iv"] * 100.0

    symbol_order = chart_data["symbol"].tolist()
    y = alt.Y(
        "symbol:N",
        sort=symbol_order,
        title=None,
        axis=alt.Axis(labelFontSize=13, labelLimit=110),
    )
    x = alt.X(
        "skew_vol_pts:Q",
        title="25Δ call IV − 25Δ put IV (vol points)",
        axis=alt.Axis(format=".1f"),
    )
    tooltips = [
        alt.Tooltip("symbol:N", title="Ticker"),
        alt.Tooltip("skew_vol_pts:Q", title="25Δ skew (vol pts)", format="+.2f"),
        alt.Tooltip("call_25d_iv_pct:Q", title="25Δ call IV (%)", format=".2f"),
        alt.Tooltip("put_25d_iv_pct:Q", title="25Δ put IV (%)", format=".2f"),
        alt.Tooltip("atm_iv_pct:Q", title="ATM IV (%)", format=".2f"),
        alt.Tooltip("actual_dte:Q", title="Actual DTE", format=".0f"),
        alt.Tooltip("expiration:T", title="Expiration", format="%Y-%m-%d"),
        alt.Tooltip("snapshot_date:T", title="Snapshot", format="%Y-%m-%d"),
        alt.Tooltip("spot:Q", title="Spot", format="$.2f"),
    ]

    base = alt.Chart(chart_data).encode(y=y, x=x)
    bars = base.mark_bar(cornerRadiusEnd=4).encode(tooltip=tooltips)
    zero = (
        alt.Chart(pd.DataFrame({"zero": [0.0]}))
        .mark_rule(strokeWidth=1.2, opacity=0.65)
        .encode(x=alt.X("zero:Q"))
    )
    positive_labels = (
        base.transform_filter(alt.datum.skew_vol_pts >= 0)
        .mark_text(align="left", dx=6, fontSize=12)
        .encode(text=alt.Text("skew_vol_pts:Q", format="+.1f"))
    )
    negative_labels = (
        base.transform_filter(alt.datum.skew_vol_pts < 0)
        .mark_text(align="right", dx=-6, fontSize=12)
        .encode(text=alt.Text("skew_vol_pts:Q", format="+.1f"))
    )
    height = min(920, max(430, 31 * len(chart_data)))
    return (
        (bars + zero + positive_labels + negative_labels)
        .properties(height=height)
        .configure_view(stroke=None)
    )


st.set_page_config(page_title="IV & Skew", page_icon="↕", layout="wide")
st.markdown(
    '''
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: #141B2D; border: 1px solid #25304A; padding: .8rem; border-radius: .8rem;}
      .muted {color: #A8B3C7;}
    </style>
    ''',
    unsafe_allow_html=True,
)

marketdata_token = get_setting("MARKETDATA_TOKEN", "")
store = SnapshotStore(
    get_setting("SUPABASE_URL", ""),
    get_setting("SUPABASE_SECRET_KEY", get_setting("SUPABASE_SERVICE_ROLE_KEY", "")),
)

st.title("IV & Skew")
st.caption(
    "Herman Jin-style cross-sectional 25Δ call/put skew plus constant-tenor IV and skew history. "
    "Positive skew means 25Δ upside calls carry higher IV than comparable 25Δ puts."
)

if "iv_active_preset" not in st.session_state:
    st.session_state["iv_active_preset"] = "AI / Semis"

preset = st.segmented_control(
    "Ticker preset",
    list(PRESET_DEFAULTS),
    key="iv_active_preset",
) or "AI / Semis"
members = ensure_preset_state(preset)
members_key = preset_members_key(preset)
selection_key = preset_selection_key(preset)

add_col, add_button_col, reset_col = st.columns([5.0, 1.15, 1.1])
custom_text = add_col.text_input(
    "Add tickers to this preset",
    value="",
    placeholder="e.g. CBRS, AEHR, AXTI",
    help="Comma- or space-separated. Added tickers become members of the preset currently selected above.",
    key=f"iv_add_tickers_{preset_token(preset)}",
)
add_clicked = add_button_col.button(
    "Add",
    width="stretch",
    key=f"iv_add_button_{preset_token(preset)}",
)
reset_clicked = reset_col.button(
    "Reset preset",
    width="stretch",
    key=f"iv_reset_button_{preset_token(preset)}",
)

if add_clicked:
    additions = normalize_symbols(custom_text)
    if not additions:
        st.warning("Enter at least one ticker to add.")
    else:
        updated = list(dict.fromkeys(members + additions))
        st.session_state[members_key] = updated
        st.session_state[selection_key] = updated
        st.rerun()

if reset_clicked:
    defaults = list(PRESET_DEFAULTS[preset])
    st.session_state[members_key] = defaults
    st.session_state[selection_key] = defaults
    st.rerun()

members = list(st.session_state[members_key])
selected = st.multiselect(
    "Preset members",
    options=members,
    key=selection_key,
    placeholder="Add tickers above",
    help="Remove any chip to delete that ticker from this preset. Additions and deletions apply directly to the active preset for this app session.",
)
if selected != members:
    st.session_state[members_key] = list(selected)
    members = list(selected)

controls = st.columns([1, 1, 1.25, 1.2])
tenor = controls[0].segmented_control("Tenor", list(TENORS), default="1M") or "1M"
sort_mode = controls[1].selectbox(
    "Cross-section sort",
    ["Skew low → high", "Skew high → low", "Ticker"],
)
history_period = controls[2].segmented_control(
    "History", ["1M", "3M", "6M", "1Y", "Max"], default="6M"
) or "6M"
refresh = controls[3].button(
    "Refresh selected from MarketData",
    type="primary",
    width="stretch",
)

if refresh:
    if not marketdata_token:
        st.error("MARKETDATA_TOKEN is not configured.")
    elif not store.enabled:
        st.error("Supabase is not configured, so refreshed volatility rows cannot be saved.")
    elif not selected:
        st.warning("Select at least one ticker.")
    else:
        client = MarketDataClient(marketdata_token)
        target_dte = TENORS[tenor]
        requested_date = previous_weekday(datetime.now(EASTERN).date())
        progress = st.progress(0.0, text="Refreshing IV/skew…")
        failures: list[str] = []
        for index, symbol in enumerate(selected, start=1):
            try:
                result = client.fetch_chain(
                    symbol,
                    requested_date,
                    min_dte=max(0, target_dte - 10),
                    max_dte=target_dte + 20,
                    min_open_interest=0,
                    delta_filter="0.50,0.25",
                )
                snap = snapshot_from_chain(symbol, result.data, result.snapshot_date, tenor)
                save_volatility_snapshot(store, snap)
            except (MarketDataError, SnapshotStoreError, ValueError) as exc:
                failures.append(f"{symbol}: {exc}")
            progress.progress(index / len(selected), text=f"Refreshed {index}/{len(selected)}")
        progress.empty()
        if failures:
            st.warning("Some tickers failed:\n\n" + "\n\n".join(failures))
        else:
            st.toast("IV/skew snapshots refreshed")

st.subheader("25Δ Put/Call Skew — cross section")
if not store.enabled:
    st.info("Configure Supabase to load saved IV/skew history.")
elif not selected:
    st.info("Select tickers above.")
else:
    try:
        latest = latest_volatility(store, selected, tenor)
    except SnapshotStoreError as exc:
        latest = pd.DataFrame()
        st.error(str(exc))

    if latest.empty:
        st.info(
            "No saved volatility snapshots yet. Use **Refresh selected from MarketData** "
            "or run the daily snapshot workflow after applying the schema update."
        )
    else:
        cross = latest.dropna(subset=["skew_25d"]).copy()
        if sort_mode == "Skew high → low":
            cross = cross.sort_values("skew_25d", ascending=False)
        elif sort_mode == "Skew low → high":
            cross = cross.sort_values("skew_25d")
        else:
            cross = cross.sort_values("symbol")

        newest = cross["snapshot_date"].max() if not cross.empty else None
        stale = cross[cross["snapshot_date"] < newest] if newest is not None else pd.DataFrame()
        if not stale.empty:
            st.caption(
                "Some tickers use an older available options session; hover or check the table below "
                "for each snapshot date."
            )

        st.altair_chart(skew_cross_section_chart(cross), use_container_width=True)
        st.caption(
            "Above zero = upside 25Δ calls are richer than downside 25Δ puts. "
            "Below zero = downside puts are richer. Hover any bar for call IV, put IV, ATM IV, DTE, "
            "expiration, spot, and snapshot date."
        )

        with st.expander("Cross-section details"):
            details = cross[
                [
                    "symbol",
                    "snapshot_date",
                    "actual_dte",
                    "expiration",
                    "spot",
                    "atm_iv",
                    "call_25d_iv",
                    "put_25d_iv",
                    "skew_25d",
                ]
            ].copy()
            for column in ("atm_iv", "call_25d_iv", "put_25d_iv", "skew_25d"):
                details[column] = details[column] * 100.0
            st.dataframe(details, hide_index=True, width="stretch")

st.divider()
st.subheader("Historical implied volatility & skew")
history_controls = st.columns([1.4, 1, 2.2])
metric = history_controls[0].selectbox(
    "Metric", ["ATM IV", "25Δ Call IV", "25Δ Put IV", "25Δ Skew"], index=0
)
change_mode = history_controls[1].segmented_control(
    "View", ["Level", "1D Δ", "1W Δ", "1M Δ"], default="Level"
) or "Level"
history_symbols = history_controls[2].multiselect(
    "Historical tickers",
    options=selected,
    default=selected[: min(5, len(selected))],
    key=f"iv_history_symbols_{preset_token(preset)}",
)

if store.enabled and history_symbols:
    end_date = datetime.now(EASTERN).date()
    start_date = history_start(history_period, end_date)
    try:
        hist = volatility_history(
            store,
            history_symbols,
            tenor,
            start_date=start_date,
            end_date=end_date,
        )
    except SnapshotStoreError as exc:
        hist = pd.DataFrame()
        st.error(str(exc))
    if hist.empty:
        st.info(
            "No history is stored for this selection yet. Daily snapshots will populate it, "
            "or run the historical backfill workflow."
        )
    else:
        chart_data, ylabel = display_series(hist, metric, change_mode)
        st.line_chart(chart_data, height=460, use_container_width=True)
        st.caption(ylabel + f" · constant-tenor target: {tenor} ({TENORS[tenor]} DTE).")

        latest_hist = (
            hist.sort_values("snapshot_date").groupby("symbol", as_index=False).tail(1)
        )
        summary_cols = st.columns(min(4, len(latest_hist))) if len(latest_hist) else []
        metric_col = {
            "ATM IV": "atm_iv",
            "25Δ Call IV": "call_25d_iv",
            "25Δ Put IV": "put_25d_iv",
            "25Δ Skew": "skew_25d",
        }[metric]
        for col, row in zip(summary_cols, latest_hist.itertuples(index=False)):
            value = getattr(row, metric_col)
            label_value = "—" if pd.isna(value) else f"{value * 100:.1f}"
            suffix = " vol pts" if metric == "25Δ Skew" else "%"
            col.metric(row.symbol, label_value + suffix)

st.divider()
with st.expander("Method"):
    st.markdown(
        '''
- **Constant tenor:** 1W/1M/3M/6M targets 7/30/90/180 DTE and uses the available expiration closest to that target.
- **ATM IV:** average IV of the nearest-to-50Δ call and put for the selected expiration when using the credit-efficient filtered request.
- **25Δ call IV:** IV of the call whose delta is closest to +0.25.
- **25Δ put IV:** IV of the put whose delta is closest to −0.25.
- **25Δ skew:** `25Δ call IV − 25Δ put IV`, shown in volatility points.
- **Historical changes:** 1D/1W/1M changes use 1/5/21 stored trading observations respectively.

Supabase stores the derived constant-tenor snapshot fields needed by this page: spot, target/actual DTE, expiration, ATM IV, 25Δ call IV, 25Δ put IV, and 25Δ skew. The full raw option-chain payload is not stored.
        '''
    )
