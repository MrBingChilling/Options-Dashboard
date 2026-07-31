from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.analytics import (
    ASSUMPTIONS,
    STANDARD,
    enrich_chain,
    expiration_profile,
    gamma_curve,
    snapshot_record,
    strike_profile,
    summarize,
)
from src.charts import (
    expiration_gex_chart,
    gamma_curve_chart,
    strike_gex_chart,
    trend_levels_chart,
    trend_regime_chart,
)
from src.config import as_bool, get_setting
from src.marketdata_client import MarketDataClient, MarketDataError
from src.storage import SnapshotStore, SnapshotStoreError


EASTERN = ZoneInfo("America/New_York")


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


st.set_page_config(
    page_title="Options Positioning Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}
      [data-testid="stMetric"] {background: #141B2D; border: 1px solid #25304A; padding: 0.85rem; border-radius: 0.8rem;}
      [data-testid="stMetricLabel"] {color: #A8B3C7;}
      .dashboard-kicker {color: #7EA6FF; font-size: .78rem; letter-spacing: .14em; font-weight: 700; text-transform: uppercase;}
      .dashboard-subtitle {color: #A8B3C7; margin-top: -.6rem; margin-bottom: 1.2rem;}
      .assumption-note {background: rgba(126,166,255,.08); border-left: 3px solid #7EA6FF; padding: .75rem 1rem; border-radius: .35rem; color: #C7D0E0;}
      @media (max-width: 700px) {
        .block-container {padding-left: .8rem; padding-right: .8rem; padding-top: .8rem;}
        h1 {font-size: 1.75rem !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

marketdata_token = get_setting("MARKETDATA_TOKEN", "")
store = SnapshotStore(
    get_setting("SUPABASE_URL", ""),
    get_setting(
        "SUPABASE_SECRET_KEY",
        get_setting("SUPABASE_SERVICE_ROLE_KEY", ""),
    ),
)
auto_save = as_bool(get_setting("AUTO_SAVE_SNAPSHOTS", True), True)

st.markdown('<div class="dashboard-kicker">Long-horizon positioning</div>', unsafe_allow_html=True)
st.title("Options Positioning Dashboard")
st.markdown(
    '<div class="dashboard-subtitle">Track potential support, resistance, and volatility regimes from end-of-day options positioning.</div>',
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
    min_dte, max_dte = st.slider(
        "Expiration horizon (DTE)",
        min_value=0,
        max_value=730,
        value=(7, 365),
        step=7,
        help="The default excludes 0DTE and focuses on positioning over the next year.",
    )
    min_open_interest = st.number_input(
        "Minimum open interest",
        min_value=0,
        max_value=100000,
        value=1,
        step=10,
    )
    assumption = st.selectbox(
        "Dealer-position assumption",
        options=ASSUMPTIONS,
        index=ASSUMPTIONS.index(STANDARD),
    )
    with st.expander("Model settings"):
        risk_free_pct = st.number_input(
            "Risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.0, step=0.25
        )
        dividend_yield_pct = st.number_input(
            "Dividend yield (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.25
        )
        visible_range_pct = st.slider(
            "Strike chart range around spot (%)", 10, 100, 25, 5
        )

    load = st.button("Load positioning", type="primary", use_container_width=True)
    if marketdata_token:
        st.caption("Market data: configured")
    else:
        st.caption("Market data: token needed")
    st.caption("History: configured" if store.enabled else "History: Supabase setup needed")

if load:
    if not marketdata_token:
        st.error(
            "Add MARKETDATA_TOKEN to Streamlit secrets before loading data. "
            "The README included with this project shows exactly where it goes."
        )
    else:
        try:
            with st.spinner(f"Loading {symbol} end-of-day option chain…"):
                client = MarketDataClient(marketdata_token)
                result = client.fetch_chain(
                    symbol,
                    analysis_date,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    min_open_interest=int(min_open_interest),
                )
                enriched = enrich_chain(
                    result.data,
                    assumption,
                    risk_free_rate=risk_free_pct / 100.0,
                    dividend_yield=dividend_yield_pct / 100.0,
                )
                curve = gamma_curve(
                    enriched,
                    assumption,
                    risk_free_rate=risk_free_pct / 100.0,
                    dividend_yield=dividend_yield_pct / 100.0,
                )
                profile = strike_profile(enriched)
                expiry_profile = expiration_profile(enriched)
                summary = summarize(
                    symbol,
                    result.snapshot_date,
                    enriched,
                    curve,
                    assumption,
                    min_dte,
                    max_dte,
                )
                st.session_state["dashboard_result"] = {
                    "symbol": symbol,
                    "enriched": enriched,
                    "curve": curve,
                    "profile": profile,
                    "expiry_profile": expiry_profile,
                    "summary": summary,
                    "requested_date": result.requested_date,
                    "snapshot_date": result.snapshot_date,
                    "visible_range_pct": visible_range_pct,
                }

            if store.enabled and auto_save:
                try:
                    save_snapshot(store, summary, profile)
                    st.toast("Daily snapshot saved")
                except SnapshotStoreError as exc:
                    st.warning(f"Positioning loaded, but the history snapshot was not saved: {exc}")
        except (MarketDataError, ValueError) as exc:
            st.error(str(exc))

result_state = st.session_state.get("dashboard_result")
if not result_state:
    st.info(
        "Choose a ticker and click **Load positioning**. The app deliberately requests an "
        "end-of-day chain so it fits longer-term analysis and uses API credits more efficiently."
    )
    st.markdown(
        """
        <div class="assumption-note">
        <strong>Important:</strong> open interest does not identify who owns each contract. Every dealer-position model here is a heuristic, not observed dealer inventory.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

summary = result_state["summary"]
enriched = result_state["enriched"]
curve = result_state["curve"]
profile = result_state["profile"]
expiry_profile = result_state["expiry_profile"]

if result_state["snapshot_date"] != result_state["requested_date"]:
    st.caption(
        f"No chain was available for {result_state['requested_date']:%Y-%m-%d}; "
        f"showing the prior available session, {result_state['snapshot_date']:%Y-%m-%d}."
    )
else:
    st.caption(f"Snapshot: {result_state['snapshot_date']:%Y-%m-%d} · {summary.contract_count:,} contracts")

overview_tab, expiry_tab, history_tab, data_tab, method_tab = st.tabs(
    ["Overview", "Expirations", "History", "Data", "Method"]
)

with overview_tab:
    first_row = st.columns(4)
    first_row[0].metric("Spot", f"${summary.spot:,.2f}")
    first_row[1].metric("Gamma flip", f"${summary.gamma_flip:,.2f}" if summary.gamma_flip else "No flip")
    first_row[2].metric("Call wall", f"${summary.call_wall:,.2f}")
    first_row[3].metric("Put wall", f"${summary.put_wall:,.2f}")

    second_row = st.columns(4)
    second_row[0].metric("Net GEX / 1%", compact_dollars(summary.net_gex))
    second_row[1].metric("Gross GEX / 1%", compact_dollars(summary.gross_gex))
    second_row[2].metric(
        "Put/call OI",
        f"{summary.put_call_oi_ratio:.2f}" if summary.put_call_oi_ratio is not None else "—",
    )
    second_row[3].metric("Net delta exposure", compact_dollars(summary.net_delta_exposure))

    regime = "positive gamma: hedging may dampen moves" if summary.net_gex >= 0 else "negative gamma: hedging may amplify moves"
    st.markdown(
        f'<div class="assumption-note"><strong>Model reading:</strong> {regime}. Assumption: {summary.assumption}.</div>',
        unsafe_allow_html=True,
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            strike_gex_chart(profile, summary.spot, result_state["visible_range_pct"] / 100),
            use_container_width=True,
            config={"displaylogo": False},
        )
    with chart_right:
        st.plotly_chart(
            gamma_curve_chart(curve, summary.spot, summary.gamma_flip),
            use_container_width=True,
            config={"displaylogo": False},
        )

with expiry_tab:
    st.plotly_chart(
        expiration_gex_chart(expiry_profile),
        use_container_width=True,
        config={"displaylogo": False},
    )
    display_expiry = expiry_profile.copy()
    display_expiry["call_gex"] /= 1e6
    display_expiry["put_gex"] /= 1e6
    display_expiry["net_gex"] /= 1e6
    st.dataframe(
        display_expiry.rename(
            columns={
                "expiration_date": "Expiration",
                "call_gex": "Call GEX ($mm)",
                "put_gex": "Put GEX ($mm)",
                "net_gex": "Net GEX ($mm)",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

with history_tab:
    if not store.enabled:
        st.info("Add the Supabase secrets and run supabase_schema.sql to begin daily history tracking.")
    else:
        try:
            history = store.history(
                summary.symbol,
                summary.assumption,
                summary.min_dte,
                summary.max_dte,
            )
            if history.empty:
                st.info("No saved snapshots match this ticker, DTE range, and assumption yet.")
            else:
                if len(history) >= 2:
                    latest = history.iloc[-1]
                    prior = history.iloc[-2]
                    st.caption(
                        f"Latest saved change: {prior['snapshot_date']:%Y-%m-%d} to "
                        f"{latest['snapshot_date']:%Y-%m-%d}"
                    )
                    changes = st.columns(5)
                    changes[0].metric(
                        "Net GEX",
                        compact_dollars(latest["net_gex"]),
                        compact_dollars(latest["net_gex"] - prior["net_gex"]),
                    )
                    changes[1].metric(
                        "Gamma flip",
                        f"${latest['gamma_flip']:,.2f}" if pd.notna(latest["gamma_flip"]) else "—",
                        f"${latest['gamma_flip'] - prior['gamma_flip']:+,.2f}"
                        if pd.notna(latest["gamma_flip"]) and pd.notna(prior["gamma_flip"])
                        else None,
                    )
                    changes[2].metric(
                        "Call wall",
                        f"${latest['call_wall']:,.2f}",
                        f"${latest['call_wall'] - prior['call_wall']:+,.2f}",
                    )
                    changes[3].metric(
                        "Put wall",
                        f"${latest['put_wall']:,.2f}",
                        f"${latest['put_wall'] - prior['put_wall']:+,.2f}",
                    )
                    changes[4].metric(
                        "Put/call OI",
                        f"{latest['put_call_oi_ratio']:.2f}",
                        f"{latest['put_call_oi_ratio'] - prior['put_call_oi_ratio']:+.2f}",
                    )
                st.plotly_chart(trend_levels_chart(history), use_container_width=True, config={"displaylogo": False})
                st.plotly_chart(trend_regime_chart(history), use_container_width=True, config={"displaylogo": False})
                st.dataframe(history.sort_values("snapshot_date", ascending=False), hide_index=True, use_container_width=True)
        except SnapshotStoreError as exc:
            st.warning(str(exc))

    if store.enabled:
        if st.button("Save or update this snapshot", use_container_width=True):
            try:
                save_snapshot(store, summary, profile)
                st.success("Snapshot saved.")
            except SnapshotStoreError as exc:
                st.error(str(exc))

with data_tab:
    table = enriched[
        [
            "optionSymbol",
            "expiration",
            "side",
            "strike",
            "dte",
            "bid",
            "ask",
            "volume",
            "openInterest",
            "iv",
            "delta",
            "gamma",
            "gex",
        ]
    ].copy()
    table["expiration"] = pd.to_datetime(table["expiration"]).dt.date
    table["gex"] = table["gex"] / 1e6
    table = table.rename(columns={"gex": "gex_usd_mm_per_1pct"})
    st.dataframe(table, hide_index=True, use_container_width=True, height=560)
    st.download_button(
        "Download filtered chain as CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name=f"{summary.symbol}_{summary.snapshot_date}_options_positioning.csv",
        mime="text/csv",
        use_container_width=True,
    )

with method_tab:
    st.subheader("What the dashboard is estimating")
    st.markdown(
        """
        - **Gamma exposure (GEX):** `gamma × open interest × 100 × spot² × 1% × assumed dealer sign`.
        - **Gamma flip:** the closest zero crossing after repricing contract gamma over a 70%–130% spot range with Black–Scholes gamma and each contract's implied volatility.
        - **Call/put walls:** the strikes with the largest absolute call-side and put-side gamma exposure inside the selected DTE horizon.
        - **Positive net gamma:** a heuristic for dealer hedging that may oppose price moves. **Negative net gamma:** a heuristic for hedging that may reinforce price moves.

        Open interest shows outstanding contracts, but not whether dealers are long or short them. The standard mode assigns calls a positive sign and puts a negative sign, a common public-dashboard convention. The two comparison modes show how dependent the result is on that assumption. This is a positioning lens, not a prediction or trading signal.
        """
    )
