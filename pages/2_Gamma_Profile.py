from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.analytics import (
    ASSUMPTIONS,
    CUSTOM_WEIGHTS,
    STANDARD,
    enrich_chain,
    expiration_profile,
    gamma_curve,
    snapshot_record,
    strike_profile,
    summarize,
)
from src.charts import (
    expiration_net_chart,
    expiration_total_chart,
    gamma_curve_chart,
    price_gamma_chart,
    render_chart,
    strike_gex_chart,
)
from src.config import as_bool, get_setting
from src.expiration_filters import (
    CUSTOM_DTE_MAX,
    EXPIRATION_CHOICES,
    FILTER_ALL,
    FILTER_CUSTOM,
    FILTER_MONTHLY,
    FILTER_OVER_ONE_YEAR,
    FILTER_61_120,
    PRESET_DTE_RANGES,
    custom_expiration_selection,
    resolve_expiration_filter,
)
from src.marketdata_client import MarketDataClient, MarketDataError
from src.storage import SnapshotStore, SnapshotStoreError


EASTERN = ZoneInfo("America/New_York")
PRICE_LOOKBACK_DAYS = 365 * 5 + 10


def previous_weekday(value: date) -> date:
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def compact_dollars(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    sign = "-" if value < 0 else ""
    amount = abs(float(value))
    if amount >= 1e9:
        return f"{sign}${amount / 1e9:,.2f}B"
    if amount >= 1e6:
        return f"{sign}${amount / 1e6:,.1f}M"
    if amount >= 1e3:
        return f"{sign}${amount / 1e3:,.1f}K"
    return f"{sign}${amount:,.0f}"


def save_snapshot(store: SnapshotStore, summary, profile: pd.DataFrame) -> None:
    store.save(snapshot_record(summary, profile))


def load_price_candles(
    client: MarketDataClient,
    store: SnapshotStore,
    symbol: str,
    end_date: date,
) -> tuple[pd.DataFrame, str | None]:
    start_date = end_date - timedelta(days=PRICE_LOOKBACK_DAYS)
    cached = pd.DataFrame()
    warning = None
    if store.enabled:
        try:
            cached = store.price_history(symbol, start_date, end_date)
        except SnapshotStoreError as exc:
            warning = str(exc)

    enough_history = (
        not cached.empty
        and cached["time"].min().date() <= start_date + timedelta(days=14)
        and cached["time"].max().date() >= end_date - timedelta(days=7)
    )
    if enough_history:
        return cached, warning

    fetched = client.fetch_candles(symbol, start_date, end_date)
    combined = pd.concat([cached, fetched], ignore_index=True) if not cached.empty else fetched
    combined = combined.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if store.enabled:
        try:
            store.save_candles(symbol, fetched)
        except SnapshotStoreError as exc:
            warning = str(exc)
    return combined, warning


def current_history_row(summary) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "snapshot_date": pd.Timestamp(summary.snapshot_date),
                "spot": summary.spot,
                "net_gex": summary.net_gex,
                "gross_gex": summary.gross_gex,
                "gamma_flip": summary.gamma_flip,
                "call_wall": summary.call_wall,
                "put_wall": summary.put_wall,
                "put_call_oi_ratio": summary.put_call_oi_ratio,
                "net_delta_exposure": summary.net_delta_exposure,
                "contract_count": summary.contract_count,
            }
        ]
    )


def merge_current_history(history: pd.DataFrame, summary) -> pd.DataFrame:
    current = current_history_row(summary)
    if history.empty:
        return current
    return (
        pd.concat([history, current], ignore_index=True)
        .sort_values("snapshot_date")
        .drop_duplicates("snapshot_date", keep="last")
        .reset_index(drop=True)
    )


def filtered_period(
    candles: pd.DataFrame,
    history: pd.DataFrame,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = {"6M": 183, "1Y": 366, "2Y": 732, "5Y": PRICE_LOOKBACK_DAYS}[period]
    end = candles["time"].max() if not candles.empty else pd.Timestamp.today()
    start = end - pd.Timedelta(int(days), unit="D")
    visible_candles = candles[candles["time"] >= start].copy() if not candles.empty else candles
    visible_history = history[history["snapshot_date"] >= start].copy() if not history.empty else history
    return visible_candles, visible_history


def calculate_positioning(
    result_state: dict,
    assumption: str,
    risk_free_pct: float,
    dividend_yield_pct: float,
    dealer_call_weight: float,
    dealer_put_weight: float,
):
    enriched = enrich_chain(
        result_state["raw_chain"],
        assumption,
        risk_free_rate=risk_free_pct / 100.0,
        dividend_yield=dividend_yield_pct / 100.0,
        call_weight=dealer_call_weight,
        put_weight=dealer_put_weight,
    )
    curve = gamma_curve(
        enriched,
        assumption,
        risk_free_rate=risk_free_pct / 100.0,
        dividend_yield=dividend_yield_pct / 100.0,
        call_weight=dealer_call_weight,
        put_weight=dealer_put_weight,
    )
    profile = strike_profile(enriched)
    expiry_profile = expiration_profile(enriched)
    summary = summarize(
        result_state["symbol"],
        result_state["snapshot_date"],
        enriched,
        curve,
        assumption,
        result_state["min_dte"],
        result_state["max_dte"],
        expiration_filter=result_state["expiration_filter"],
        call_weight=dealer_call_weight,
        put_weight=dealer_put_weight,
    )
    return enriched, curve, profile, expiry_profile, summary


def apply_expiration_preset() -> None:
    preset = st.session_state.get("expiration_preset")
    if preset in PRESET_DTE_RANGES:
        st.session_state["expiration_dte_range"] = PRESET_DTE_RANGES[preset]


def mark_expiration_range_custom() -> None:
    st.session_state["expiration_preset"] = FILTER_CUSTOM


st.set_page_config(
    page_title="Options Positioning Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {background: #141B2D; border: 1px solid #25304A; padding: 0.85rem; border-radius: 0.8rem;}
      [data-testid="stMetricLabel"] {color: #A8B3C7;}
      .dashboard-kicker {color: #7EA6FF; font-size: .78rem; letter-spacing: .14em; font-weight: 700; text-transform: uppercase;}
      .dashboard-subtitle {color: #A8B3C7; margin-top: -.6rem; margin-bottom: 1.2rem;}
      .assumption-note {background: rgba(126,166,255,.08); border-left: 3px solid #7EA6FF; padding: .75rem 1rem; border-radius: .35rem; color: #C7D0E0;}
      iframe {border-radius: 12px;}
      @media (max-width: 700px) {
        .block-container {padding-left: .7rem; padding-right: .7rem; padding-top: .8rem;}
        h1 {font-size: 1.7rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

marketdata_token = get_setting("MARKETDATA_TOKEN", "")
store = SnapshotStore(
    get_setting("SUPABASE_URL", ""),
    get_setting("SUPABASE_SECRET_KEY", get_setting("SUPABASE_SERVICE_ROLE_KEY", "")),
)
auto_save = as_bool(get_setting("AUTO_SAVE_SNAPSHOTS", True), True)

st.markdown('<div class="dashboard-kicker">Long-horizon positioning</div>', unsafe_allow_html=True)
st.title("Options Positioning Dashboard")
st.markdown(
    '<div class="dashboard-subtitle">Price action, gamma structure, and positioning history in one touch-friendly workspace.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Analysis")
    symbol = st.text_input("Ticker", value="SPY", max_chars=12).strip().upper()
    analysis_date = st.date_input(
        "End-of-day snapshot",
        value=previous_weekday(datetime.now(EASTERN).date()),
        max_value=datetime.now(EASTERN).date(),
    )
    if "expiration_preset" not in st.session_state:
        st.session_state["expiration_preset"] = FILTER_61_120
    if "expiration_dte_range" not in st.session_state:
        st.session_state["expiration_dte_range"] = PRESET_DTE_RANGES[FILTER_61_120]

    expiration_preset = st.selectbox(
        "Quick expiration bucket",
        options=EXPIRATION_CHOICES,
        key="expiration_preset",
        on_change=apply_expiration_preset,
        help="Choose a preset, or move the DTE slider below for a precise custom range.",
    )
    range_disabled = expiration_preset in {FILTER_ALL, FILTER_MONTHLY, FILTER_OVER_ONE_YEAR}
    dte_range = st.slider(
        "Fine-tune DTE range",
        min_value=0,
        max_value=CUSTOM_DTE_MAX,
        step=1,
        key="expiration_dte_range",
        disabled=range_disabled,
        on_change=mark_expiration_range_custom,
        help="Moving either handle switches the request to a custom DTE range. Open-ended presets bypass the slider.",
    )
    selection = (
        custom_expiration_selection(int(dte_range[0]), int(dte_range[1]))
        if expiration_preset == FILTER_CUSTOM
        else resolve_expiration_filter(expiration_preset)
    )
    expiration_filter = selection.label
    if range_disabled:
        st.caption("This preset is open-ended; choose Custom DTE range to use the slider.")
    min_open_interest = st.number_input(
        "Minimum open interest",
        min_value=0,
        max_value=100000,
        value=10,
        step=10,
    )
    assumption = st.selectbox(
        "Dealer-position assumption",
        options=ASSUMPTIONS,
        index=ASSUMPTIONS.index(STANDARD),
    )
    if assumption == CUSTOM_WEIGHTS:
        st.caption("Scenario weights are sensitivity assumptions—not observed dealer positions.")
        dealer_call_weight = st.slider(
            "Dealer call weight",
            min_value=-1.0,
            max_value=1.0,
            value=-0.40,
            step=0.05,
            help="−1 entirely short · 0 neutral · +1 entirely long",
        )
        dealer_put_weight = st.slider(
            "Dealer put weight",
            min_value=-1.0,
            max_value=1.0,
            value=-0.70,
            step=0.05,
            help="−1 entirely short · 0 neutral · +1 entirely long",
        )
    else:
        dealer_call_weight = -0.40
        dealer_put_weight = -0.70

    with st.expander("Model settings"):
        risk_free_pct = st.number_input(
            "Risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.25
        )
        dividend_yield_pct = st.number_input(
            "Dividend yield (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.25
        )
        visible_range_pct = st.slider("Strike chart range around spot (%)", 10, 100, 25, 5)

    load = st.button("Load positioning", type="primary", width="stretch")
    current_state = st.session_state.get("dashboard_result")
    if current_state and current_state["expiration_filter"] != expiration_filter:
        st.caption(f"Loaded: {current_state['expiration_filter']}. Click Load to apply the new filter.")
    st.caption("Market data: configured" if marketdata_token else "Market data: token needed")
    st.caption("History: configured" if store.enabled else "History: Supabase setup needed")

if load:
    if not marketdata_token:
        st.error("Add MARKETDATA_TOKEN to Streamlit secrets before loading data.")
    else:
        try:
            with st.spinner(f"Loading {symbol} options and cached price history…"):
                client = MarketDataClient(marketdata_token)
                chain_result = client.fetch_chain(
                    symbol,
                    analysis_date,
                    min_dte=selection.min_dte,
                    max_dte=selection.max_dte,
                    min_open_interest=int(min_open_interest),
                    expiration_filter=(
                        None if expiration_preset == FILTER_CUSTOM else expiration_preset
                    ),
                )
                try:
                    price_candles, price_warning = load_price_candles(
                        client,
                        store,
                        symbol,
                        chain_result.snapshot_date,
                    )
                except MarketDataError as exc:
                    price_candles = pd.DataFrame()
                    price_warning = f"Options loaded, but price history could not be loaded: {exc}"
                latest_price = None
                latest_price_updated = None
                latest_price_warning = None
                try:
                    latest = client.fetch_latest_price(symbol)
                    latest_price = latest.price
                    latest_price_updated = latest.updated
                except MarketDataError as exc:
                    latest_price_warning = f"Latest stock price could not be loaded: {exc}"
                st.session_state["dashboard_result"] = {
                    "symbol": symbol,
                    "raw_chain": chain_result.data,
                    "requested_date": chain_result.requested_date,
                    "snapshot_date": chain_result.snapshot_date,
                    "expiration_filter": expiration_filter,
                    "min_dte": selection.min_dte,
                    "max_dte": selection.max_dte,
                    "price_candles": price_candles,
                    "price_warning": price_warning,
                    "latest_price": latest_price,
                    "latest_price_updated": latest_price_updated,
                    "latest_price_warning": latest_price_warning,
                    "api_usage": client.usage_summary(),
                }
                st.session_state["save_after_recalculation"] = True
        except (MarketDataError, ValueError) as exc:
            st.error(str(exc))

result_state = st.session_state.get("dashboard_result")
if not result_state:
    st.info(
        "Choose a ticker and click **Load positioning**. Price candles are cached separately, and all chart toggles and dealer scenarios then run locally."
    )
    st.markdown(
        """
        <div class="assumption-note"><strong>Important:</strong> open interest does not identify who owns each contract. Every dealer-position model here is a scenario, not observed dealer inventory.</div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    enriched, curve, profile, expiry_profile, summary = calculate_positioning(
        result_state,
        assumption,
        risk_free_pct,
        dividend_yield_pct,
        dealer_call_weight,
        dealer_put_weight,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if st.session_state.pop("save_after_recalculation", False) and store.enabled and auto_save:
    try:
        save_snapshot(store, summary, profile)
        st.toast("Daily snapshot saved")
    except SnapshotStoreError as exc:
        st.warning(f"Positioning loaded, but the snapshot was not saved: {exc}")

if result_state["snapshot_date"] != result_state["requested_date"]:
    st.caption(
        f"No chain was available for {result_state['requested_date']:%Y-%m-%d}; showing {result_state['snapshot_date']:%Y-%m-%d}."
    )
else:
    st.caption(
        f"Snapshot: {result_state['snapshot_date']:%Y-%m-%d} · {summary.contract_count:,} contracts · {summary.expiration_filter}"
    )
if result_state.get("price_warning"):
    st.warning(result_state["price_warning"])
if result_state.get("latest_price_warning"):
    st.warning(result_state["latest_price_warning"])
api_usage = result_state.get("api_usage") or {}
if api_usage.get("consumed") is not None:
    usage_text = f"MarketData credits reported for this load: {api_usage['consumed']}"
    if api_usage.get("remaining") is not None:
        usage_text += f" · {api_usage['remaining']} remaining"
    st.caption(usage_text)

history = pd.DataFrame()
history_error = None
if store.enabled:
    try:
        history = store.history(
            summary.symbol,
            summary.assumption,
            summary.min_dte,
            summary.max_dte,
            expiration_filter=summary.expiration_filter,
            dealer_call_weight=summary.dealer_call_weight,
            dealer_put_weight=summary.dealer_put_weight,
        )
    except SnapshotStoreError as exc:
        history_error = str(exc)
history_with_current = merge_current_history(history, summary)
price_candles = result_state.get("price_candles", pd.DataFrame())
if price_candles.empty:
    snapshot_time = pd.Timestamp(summary.snapshot_date)
    price_candles = pd.DataFrame(
        [{"time": snapshot_time, "open": summary.spot, "high": summary.spot, "low": summary.spot, "close": summary.spot, "volume": 0}]
    )

overview_tab, expiry_tab, history_tab, data_tab, method_tab = st.tabs(
    ["Overview", "Expirations", "History", "Data", "Method"]
)

with overview_tab:
    latest_price = result_state.get("latest_price")
    first_row = st.columns(5)
    first_row[0].metric("Latest price", f"${latest_price:,.2f}" if latest_price is not None else "—")
    first_row[1].metric("Options snapshot spot", f"${summary.spot:,.2f}")
    first_row[2].metric("Gamma flip", f"${summary.gamma_flip:,.2f}" if summary.gamma_flip else "No flip")
    first_row[3].metric("Call wall", f"${summary.call_wall:,.2f}")
    first_row[4].metric("Put wall", f"${summary.put_wall:,.2f}")
    if result_state.get("latest_price_updated") is not None:
        st.caption(
            f"Latest stock midpoint: {result_state['latest_price_updated']:%Y-%m-%d %H:%M %Z}. "
            f"Options levels remain based on the {summary.snapshot_date} end-of-day chain."
        )

    second_row = st.columns(4)
    second_row[0].metric("Net GEX / 1%", compact_dollars(summary.net_gex))
    second_row[1].metric("Absolute total GEX / 1%", compact_dollars(summary.gross_gex))
    second_row[2].metric("Put/call OI ratio", f"{summary.put_call_oi_ratio:.2f}" if summary.put_call_oi_ratio is not None else "—")
    second_row[3].metric("Net delta exposure", compact_dollars(summary.net_delta_exposure))
    st.caption(
        f"Put/call OI = total put open interest ÷ total call open interest within this filter: "
        f"{summary.put_open_interest:,} ÷ {summary.call_open_interest:,}."
    )

    if abs(summary.net_gex) <= max(summary.gross_gex, 1.0) * 1e-12:
        regime = "neutral gamma under the selected weights"
    elif summary.net_gex > 0:
        regime = "positive gamma: hedging may dampen moves"
    else:
        regime = "negative gamma: hedging may amplify moves"
    st.markdown(
        f'<div class="assumption-note"><strong>Scenario reading:</strong> {regime}. Call weight {summary.dealer_call_weight:+.2f}; put weight {summary.dealer_put_weight:+.2f}.</div>',
        unsafe_allow_html=True,
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        render_chart(strike_gex_chart(profile, summary.spot, visible_range_pct / 100), height=455)
    with chart_right:
        render_chart(gamma_curve_chart(curve, summary.spot, summary.gamma_flip), height=455)

    st.subheader("Price and gamma map")
    controls = st.columns([1, 1.2, 0.8, 0.8])
    price_style = controls[0].segmented_control(
        "Price style", ["Candlestick", "Line"], default="Candlestick", key="overview_price_style"
    ) or "Candlestick"
    gamma_mode = controls[1].segmented_control(
        "Gamma profile", ["Calls left / puts right", "Stacked together"], default="Calls left / puts right"
    ) or "Calls left / puts right"
    overview_period = controls[2].selectbox("Visible period", ["6M", "1Y", "2Y", "5Y"], index=2)
    show_regime = controls[3].toggle("GEX + put/call", value=True)
    visible_candles, visible_history = filtered_period(price_candles, history_with_current, overview_period)
    render_chart(
        price_gamma_chart(
            visible_candles,
            visible_history,
            profile=profile,
            price_style=price_style,
            gamma_mode=gamma_mode,
            show_levels=True,
            show_regime=show_regime,
            latest_price=result_state.get("latest_price"),
            latest_price_updated=result_state.get("latest_price_updated"),
        ),
        height=680,
    )
    st.caption(
        "The price scale is fixed on the left and the gamma profile is drawn on the right from the loaded option chain. "
        "Chart controls do not request the options API again."
    )

with expiry_tab:
    net_column, total_column = st.columns(2)
    with net_column:
        render_chart(expiration_net_chart(expiry_profile), height=440)
    with total_column:
        render_chart(expiration_total_chart(expiry_profile), height=440)

    display_expiry = expiry_profile.copy()
    for column in ("call_gex", "put_gex", "net_gex", "total_gex"):
        values_mm = display_expiry[column] / 1e6
        display_expiry[column] = values_mm.mask(values_mm.abs() < 0.00005, 0.0).round(4)
    if not expiry_profile.empty and expiry_profile["total_gex"].sum() > 0:
        dominant_index = expiry_profile["total_gex"].idxmax()
        dominant_date = expiry_profile.loc[dominant_index, "expiration_date"]
        dominant_share = expiry_profile.loc[dominant_index, "total_gex"] / expiry_profile["total_gex"].sum()
        st.caption(f"Largest expiration: {dominant_date:%Y-%m-%d} ({dominant_share:.1%} of absolute total expiration gamma).")
    st.dataframe(
        display_expiry.rename(
            columns={
                "expiration_date": "Expiration",
                "call_gex": "Call GEX ($mm)",
                "put_gex": "Put GEX ($mm)",
                "net_gex": "Net GEX ($mm)",
                "total_gex": "Absolute total GEX ($mm)",
            }
        ),
        column_config={
            "Expiration": st.column_config.DateColumn(format="YYYY-MM-DD"),
            "Call GEX ($mm)": st.column_config.NumberColumn(format="%.4f"),
            "Put GEX ($mm)": st.column_config.NumberColumn(format="%.4f"),
            "Net GEX ($mm)": st.column_config.NumberColumn(format="%.4f"),
            "Absolute total GEX ($mm)": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
        width="stretch",
    )

with history_tab:
    if history_error:
        st.warning(history_error)
    elif history.empty:
        st.info("No prior snapshots match this ticker, expiration filter, and dealer-weight scenario yet. The current snapshot is shown below.")
    else:
        if len(history) >= 2:
            latest, prior = history.iloc[-1], history.iloc[-2]
            st.caption(f"Latest saved change: {prior['snapshot_date']:%Y-%m-%d} to {latest['snapshot_date']:%Y-%m-%d}")
            changes = st.columns(5)
            changes[0].metric("Net GEX", compact_dollars(latest["net_gex"]), compact_dollars(latest["net_gex"] - prior["net_gex"]))
            changes[1].metric(
                "Gamma flip",
                f"${latest['gamma_flip']:,.2f}" if pd.notna(latest["gamma_flip"]) else "—",
                f"${latest['gamma_flip'] - prior['gamma_flip']:+,.2f}" if pd.notna(latest["gamma_flip"]) and pd.notna(prior["gamma_flip"]) else None,
            )
            changes[2].metric("Call wall", f"${latest['call_wall']:,.2f}", f"${latest['call_wall'] - prior['call_wall']:+,.2f}")
            changes[3].metric("Put wall", f"${latest['put_wall']:,.2f}", f"${latest['put_wall'] - prior['put_wall']:+,.2f}")
            changes[4].metric("Put/call OI ratio", f"{latest['put_call_oi_ratio']:.2f}", f"{latest['put_call_oi_ratio'] - prior['put_call_oi_ratio']:+.2f}")

    history_controls = st.columns([1, 1, 2])
    history_style = history_controls[0].segmented_control(
        "Price style", ["Candlestick", "Line"], default="Candlestick", key="history_price_style"
    ) or "Candlestick"
    history_period = history_controls[1].selectbox("Visible period", ["6M", "1Y", "2Y", "5Y"], index=2, key="history_period")
    history_controls[2].caption("Price, flip and walls share the right axis; net GEX and the put/call OI ratio occupy the lower indicator band.")
    visible_candles, visible_history = filtered_period(price_candles, history_with_current, history_period)
    render_chart(
        price_gamma_chart(
            visible_candles,
            visible_history,
            profile=None,
            price_style=history_style,
            show_levels=True,
            show_regime=True,
            latest_price=result_state.get("latest_price"),
            latest_price_updated=result_state.get("latest_price_updated"),
            title="Price with positioning history",
        ),
        height=650,
    )
    if not history.empty:
        st.dataframe(history.sort_values("snapshot_date", ascending=False), hide_index=True, width="stretch")
    if store.enabled and st.button("Save or update this snapshot", width="stretch"):
        try:
            save_snapshot(store, summary, profile)
            st.success("Snapshot saved.")
        except SnapshotStoreError as exc:
            st.error(str(exc))

with data_tab:
    table_columns = [
        "optionSymbol", "expiration", "side", "strike", "dte", "bid", "ask", "volume",
        "openInterest", "iv", "iv_used", "iv_source", "delta", "delta_used", "delta_source",
        "gamma", "gamma_used", "gamma_source", "dealer_weight", "base_gex", "gex",
    ]
    table = enriched[[column for column in table_columns if column in enriched.columns]].copy()
    table["expiration"] = pd.to_datetime(table["expiration"]).dt.date
    table["base_gex"] = table["base_gex"] / 1e6
    table["gex"] = table["gex"] / 1e6
    table = table.rename(
        columns={
            "gamma": "vendor_gamma",
            "iv": "vendor_iv",
            "gamma_used": "gamma_used_for_gex",
            "base_gex": "unweighted_gex_usd_mm_per_1pct",
            "gex": "scenario_gex_usd_mm_per_1pct",
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", height=560)
    st.download_button(
        "Download filtered chain as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{summary.symbol}_{summary.snapshot_date}_options_positioning.csv",
        mime="text/csv",
        width="stretch",
    )

with method_tab:
    st.subheader("What the dashboard is estimating")
    st.markdown(
        f"""
        - **Scenario GEX:** `unweighted gamma exposure × dealer weight`, where call and put weights range from −1 to +1.
        - **Net gamma:** call GEX + put GEX. **Absolute total gamma:** |call GEX| + |put GEX|.
        - **Gamma flip:** the nearest zero crossing after repricing contract gamma across a 70%–130% spot range.
        - **Call/put walls:** strikes with the greatest unweighted call-side and put-side gamma concentration. Their locations remain defined even when a scenario weight is zero.
        - **Price history:** split-adjusted daily MarketData.app candles cached in Supabase and reused by Overview and History.
        - **Latest price:** MarketData.app's real-time stock midpoint is drawn separately from the completed daily candles. It does not make the one-day-old option positions current.
        - **Put/call OI ratio:** total open interest of the selected put contracts divided by total open interest of the selected call contracts. It is not the day's put/call trading volume.

        Current scenario weights: **calls {summary.dealer_call_weight:+.2f}**, **puts {summary.dealer_put_weight:+.2f}**. These are assumptions, not measured dealer inventory. Open interest identifies outstanding contracts but not their owner or trade direction. The charts are a positioning lens, not a prediction or trading signal.
        """
    )
