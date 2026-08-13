from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

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
from src.volatility_storage import (
    latest_volatility,
    save_volatility_snapshots,
    volatility_history,
)


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
    "Custom": [],
}

PRESET_COLOR_ORDER = [
    "Neoclouds",
    "Mag 7",
    "Software",
    "Powers",
    "Index",
    "AI Photonics",
    "AI Fabless Semis",
    "AI Memory",
    "AI Fabs",
    "Custom",
]

PRESET_COLORS = {
    "Neoclouds": "#7A5AF8",
    "Mag 7": "#2E90FA",
    "Software": "#12B76A",
    "Powers": "#36BFFA",
    "Index": "#98A2B3",
    "AI Photonics": "#F79009",
    "AI Fabless Semis": "#F04438",
    "AI Memory": "#EE46BC",
    "AI Fabs": "#6172F3",
    "Custom": "#667085",
}

PRIMARY_PRESET_BY_SYMBOL: dict[str, str] = {}
for group_name in PRESET_COLOR_ORDER:
    if group_name == "Custom":
        continue
    for group_symbol in PRESET_DEFAULTS[group_name]:
        PRIMARY_PRESET_BY_SYMBOL.setdefault(group_symbol, group_name)

REFERENCE_COLORS = {
    "SPY": "#6CB6FF",
    "QQQ": "#62C370",
    "Pool avg": "#F5B942",
}


def normalize_symbols(text: str) -> list[str]:
    return [
        token.strip().upper()
        for token in text.replace("\n", ",").replace(" ", ",").split(",")
        if token.strip()
    ]


def history_start(period: str, end_date: date) -> date | None:
    days = {"1M": 35, "3M": 100, "6M": 190, "1Y": 375}
    return end_date - timedelta(days=days[period]) if period in days else None


def display_series(
    history: pd.DataFrame,
    metric: str,
    change_mode: str,
) -> tuple[pd.DataFrame, str]:
    metric_map = {
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
        ylabel = "25Δ call IV − 25Δ put IV (vol points)"
    else:
        ylabel = "Implied volatility (%)"

    pivot = work.pivot(
        index="snapshot_date",
        columns="symbol",
        values="value",
    ).sort_index()
    return pivot, ylabel


def preset_token(name: str) -> str:
    return (
        name.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("+", "plus")
    )


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


def chart_group_for_symbol(symbol: str, active_preset: str) -> str:
    if active_preset in PRESET_COLORS:
        return active_preset
    return PRIMARY_PRESET_BY_SYMBOL.get(symbol, "Custom")


def attach_chart_groups(
    cross: pd.DataFrame,
    active_preset: str,
) -> pd.DataFrame:
    work = cross.copy()
    work["preset_group"] = [
        chart_group_for_symbol(str(symbol).upper(), active_preset)
        for symbol in work["symbol"]
    ]
    return work


def sort_cross_section(
    cross: pd.DataFrame,
    sort_mode: str,
) -> pd.DataFrame:
    if sort_mode == "Most positive first":
        return cross.sort_values(
            ["skew_25d", "symbol"],
            ascending=[False, True],
        )
    if sort_mode == "Alphabetical":
        return cross.sort_values("symbol")
    if sort_mode == "Preset":
        preset_rank = {
            group: index
            for index, group in enumerate(PRESET_COLOR_ORDER)
        }
        work = cross.copy()
        work["_preset_rank"] = (
            work["preset_group"]
            .map(preset_rank)
            .fillna(len(PRESET_COLOR_ORDER))
        )
        return (
            work.sort_values(
                ["_preset_rank", "skew_25d", "symbol"],
                ascending=[True, True, True],
            )
            .drop(columns="_preset_rank")
        )

    return cross.sort_values(
        ["skew_25d", "symbol"],
        ascending=[True, True],
    )


def complete_for_requested_date(
    store: SnapshotStore,
    symbols: list[str],
    session_date: date,
) -> set[str]:
    complete: dict[str, set[str]] = {symbol: set() for symbol in symbols}

    for check_tenor in DAILY_TENORS:
        frame = latest_volatility(store, symbols, check_tenor)
        if frame.empty:
            continue

        work = frame.copy()
        work["snapshot_date"] = pd.to_datetime(
            work["snapshot_date"], errors="coerce"
        ).dt.date

        for symbol in work.loc[
            work["snapshot_date"] == session_date, "symbol"
        ].astype(str):
            complete.setdefault(symbol.upper(), set()).add(check_tenor)

    needed = set(DAILY_TENORS)
    return {
        symbol
        for symbol, saved_tenors in complete.items()
        if needed.issubset(saved_tenors)
    }


def skew_cross_section_chart(cross: pd.DataFrame) -> alt.Chart:
    chart_data = cross.copy()
    chart_data["skew_vol_pts"] = chart_data["skew_25d"] * 100.0
    chart_data["call_25d_iv_pct"] = chart_data["call_25d_iv"] * 100.0
    chart_data["put_25d_iv_pct"] = chart_data["put_25d_iv"] * 100.0
    chart_data["zero"] = 0.0

    label_width = max(
        5,
        int(chart_data["symbol"].astype(str).str.len().max()),
    )
    chart_data["symbol_label"] = [
        f"{symbol:<{label_width}}  {skew:+.1f}"
        for symbol, skew in zip(
            chart_data["symbol"].astype(str),
            chart_data["skew_vol_pts"],
        )
    ]

    symbol_order = chart_data["symbol_label"].tolist()
    height = min(1180, max(470, 30 * len(chart_data)))

    present_groups = [
        group
        for group in PRESET_COLOR_ORDER
        if group in set(chart_data["preset_group"])
    ]
    color_scale = alt.Scale(
        domain=present_groups,
        range=[PRESET_COLORS[group] for group in present_groups],
    )

    x = alt.X(
        "zero:Q",
        title="25Δ call IV − 25Δ put IV (vol points)",
        axis=alt.Axis(format=".1f"),
    )
    y = alt.Y(
        "symbol_label:N",
        sort=symbol_order,
        title=None,
        axis=alt.Axis(
            labelFont="monospace",
            labelFontSize=13,
            labelLimit=190,
            labelColor="#E8EDF7",
        ),
    )

    tooltips = [
        alt.Tooltip("symbol:N", title="Ticker"),
        alt.Tooltip("preset_group:N", title="Preset"),
        alt.Tooltip(
            "skew_vol_pts:Q",
            title="25Δ skew (vol pts)",
            format="+.2f",
        ),
        alt.Tooltip(
            "call_25d_iv_pct:Q",
            title="25Δ call IV (%)",
            format=".2f",
        ),
        alt.Tooltip(
            "put_25d_iv_pct:Q",
            title="25Δ put IV (%)",
            format=".2f",
        ),
        alt.Tooltip("actual_dte:Q", title="Actual DTE", format=".0f"),
        alt.Tooltip("expiration:T", title="Expiration", format="%Y-%m-%d"),
        alt.Tooltip(
            "snapshot_date:T",
            title="Snapshot",
            format="%Y-%m-%d",
        ),
        alt.Tooltip("spot:Q", title="Spot", format="$.2f"),
    ]

    bars = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=x,
            x2=alt.X2("skew_vol_pts:Q"),
            y=y,
            color=alt.Color(
                "preset_group:N",
                scale=color_scale,
                legend=alt.Legend(
                    title="Preset color",
                    orient="right",
                ),
            ),
            tooltip=tooltips,
        )
    )

    zero = (
        alt.Chart(pd.DataFrame({"value": [0.0]}))
        .mark_rule(
            color="#FFFFFF",
            strokeWidth=1.7,
            opacity=0.95,
        )
        .encode(x=alt.X("value:Q"))
    )

    reference_rows: list[dict[str, object]] = []

    for symbol in ("SPY", "QQQ"):
        match = chart_data[chart_data["symbol"] == symbol]
        if not match.empty:
            value = float(match.iloc[0]["skew_vol_pts"])
            reference_rows.append(
                {
                    "kind": symbol,
                    "value": value,
                    "label": f"{symbol} {value:+.1f}",
                }
            )

    pool = chart_data[~chart_data["symbol"].isin(INDEX_SYMBOLS)]
    if not pool.empty:
        pool_value = float(pool["skew_vol_pts"].mean())
        reference_rows.append(
            {
                "kind": "Pool avg",
                "value": pool_value,
                "label": f"Pool avg {pool_value:+.1f}",
            }
        )

    chart = bars + zero

    for row_number, row in enumerate(reference_rows):
        label_frame = pd.DataFrame([row])
        ref_color = REFERENCE_COLORS[str(row["kind"])]

        rule = (
            alt.Chart(label_frame)
            .mark_rule(
                color=ref_color,
                strokeDash=[6, 5],
                strokeWidth=2,
            )
            .encode(
                x=alt.X("value:Q"),
                tooltip=[
                    alt.Tooltip("kind:N", title="Reference"),
                    alt.Tooltip(
                        "value:Q",
                        title="Skew (vol pts)",
                        format="+.2f",
                    ),
                ],
            )
        )
        label_layer = (
            alt.Chart(label_frame)
            .mark_text(
                align="left",
                dx=5,
                fontSize=12,
                fontWeight="bold",
                color=ref_color,
            )
            .encode(
                x=alt.X("value:Q"),
                y=alt.value(14 + 18 * row_number),
                text=alt.Text("label:N"),
            )
        )
        chart = chart + rule + label_layer

    return (
        chart.properties(height=height)
        .configure_view(stroke=None)
    )


st.set_page_config(
    page_title="IV & Skew",
    page_icon="↕",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
      }
      .muted {color: #A8B3C7;}
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

st.title("IV & Skew")
st.caption(
    "Saved daily 25Δ call/put skew. Opening or refreshing this page reads "
    "Supabase only and does not consume MarketData credits."
)

if (
    "iv_active_preset" not in st.session_state
    or st.session_state["iv_active_preset"] not in PRESET_DEFAULTS
):
    st.session_state["iv_active_preset"] = "Dashboard"

preset = (
    st.segmented_control(
        "Ticker preset",
        list(PRESET_DEFAULTS),
        key="iv_active_preset",
    )
    or "Dashboard"
)

members = ensure_preset_state(preset)
members_key = preset_members_key(preset)
selection_key = preset_selection_key(preset)

add_col, add_button_col, reset_col = st.columns([5.0, 1.15, 1.1])
custom_text = add_col.text_input(
    "Add tickers to this preset",
    value="",
    placeholder="e.g. ORCL, CRWV, NBIS",
    help=(
        "Comma- or space-separated. Added tickers are inserted directly into "
        "the active preset for this app session."
    ),
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
    help=(
        "Remove any chip to remove that ticker from the active preset. "
        "Dashboard is the complete automatic daily basket; editing here only "
        "changes this app session."
    ),
)

if selected != members:
    st.session_state[members_key] = list(selected)
    members = list(selected)

controls = st.columns([1, 1.25, 1.25, 1.6])
tenor = (
    controls[0].segmented_control(
        "Tenor",
        list(DAILY_TENORS),
        default="1M",
    )
    or "1M"
)
sort_mode = controls[1].selectbox(
    "Cross-section sort",
    [
        "Rank (low → high)",
        "Alphabetical",
        "Preset",
        "Most positive first",
    ],
)
history_period = (
    controls[2].segmented_control(
        "History",
        ["1M", "3M", "6M", "1Y", "Max"],
        default="6M",
    )
    or "6M"
)
manual_request = controls[3].button(
    "Request missing selected from MarketData",
    type="primary",
    width="stretch",
    help=(
        "Checks Supabase first. Already-saved tickers use 0 MarketData "
        "credits. Missing tickers use the same bounded historical-chain path "
        "as the daily task and save both 1W and 1M."
    ),
)

if manual_request:
    if not marketdata_token:
        st.error("MARKETDATA_TOKEN is not configured.")
    elif not store.enabled:
        st.error("Supabase is not configured.")
    elif not selected:
        st.warning("Select at least one ticker.")
    else:
        requested_date = previous_weekday(datetime.now(EASTERN).date())

        try:
            complete = complete_for_requested_date(
                store,
                selected,
                requested_date,
            )
        except SnapshotStoreError as exc:
            complete = set()
            st.error(f"Could not check Supabase: {exc}")

        missing = [symbol for symbol in selected if symbol not in complete]

        if not missing:
            st.success(
                f"All selected tickers already have 1W + 1M data for "
                f"{requested_date}. MarketData requests: 0."
            )
        else:
            client = MarketDataClient(marketdata_token)
            progress = st.progress(
                0.0,
                text=f"{len(missing)} ticker(s) need MarketData…",
            )
            failures: list[str] = []
            saved = 0

            for index, symbol in enumerate(missing, start=1):
                try:
                    result = client.fetch_skew_chain(
                        symbol,
                        requested_date,
                    )
                    snapshots = skew_snapshots_from_chain(
                        symbol,
                        result.snapshot_date,
                        result.data,
                    )
                    save_volatility_snapshots(store, snapshots)
                    saved += 1
                except (
                    MarketDataError,
                    SnapshotStoreError,
                    ValueError,
                ) as exc:
                    failures.append(f"{symbol}: {exc}")

                progress.progress(
                    index / len(missing),
                    text=f"Processed {index}/{len(missing)}",
                )

            progress.empty()

            if saved:
                st.success(
                    f"Saved {saved} ticker(s). "
                    f"Skipped {len(complete)} already-complete ticker(s)."
                )
            if failures:
                st.warning(
                    "Some tickers failed:\n\n" + "\n\n".join(failures)
                )

st.subheader(f"25Δ Put/Call Skew — {preset}")

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
            "No saved skew snapshots yet. The scheduled GitHub workflow will "
            "populate this automatically, or use the manual request button."
        )
    else:
        cross = latest.dropna(
            subset=["skew_25d", "call_25d_iv", "put_25d_iv"]
        ).copy()
        cross = attach_chart_groups(cross, preset)
        cross = sort_cross_section(cross, sort_mode)

        newest = cross["snapshot_date"].max() if not cross.empty else None
        if newest is not None:
            st.caption(
                f"Latest saved session shown: {newest}. "
                "SPY and QQQ dashed references plus the equal-weight displayed "
                "non-index average are calculated from the displayed rows."
            )

        stale = (
            cross[cross["snapshot_date"] < newest]
            if newest is not None
            else pd.DataFrame()
        )
        if not stale.empty:
            st.caption(
                "Some tickers use an older available session; hover a bar or "
                "open details to see each snapshot date."
            )

        st.altair_chart(
            skew_cross_section_chart(cross),
            use_container_width=True,
        )

        st.caption(
            "Bar colors identify presets; the legend shows the color mapping. "
            "Ticker and skew value are aligned together on the left. Dashed "
            "lines mark SPY, QQQ and the equal-weight average of displayed "
            "non-index stocks. The white vertical line is zero skew."
        )

        with st.expander("Cross-section details"):
            details = cross[
                [
                    "symbol",
                    "preset_group",
                    "snapshot_date",
                    "actual_dte",
                    "expiration",
                    "spot",
                    "call_25d_iv",
                    "put_25d_iv",
                    "skew_25d",
                ]
            ].copy()

            for column in (
                "call_25d_iv",
                "put_25d_iv",
                "skew_25d",
            ):
                details[column] = details[column] * 100.0

            details = details.rename(
                columns={
                    "symbol": "Ticker",
                    "preset_group": "Preset",
                    "snapshot_date": "Date",
                    "actual_dte": "DTE",
                    "expiration": "Expiration",
                    "spot": "Spot",
                    "call_25d_iv": "25Δ Call IV %",
                    "put_25d_iv": "25Δ Put IV %",
                    "skew_25d": "25Δ Skew vol pts",
                }
            )
            st.dataframe(
                details,
                hide_index=True,
                width="stretch",
            )

st.divider()
st.subheader("Historical 25Δ IV & skew")

history_controls = st.columns([1.4, 1, 2.2])
metric = history_controls[0].selectbox(
    "Metric",
    ["25Δ Skew", "25Δ Call IV", "25Δ Put IV"],
    index=0,
)
change_mode = (
    history_controls[1].segmented_control(
        "View",
        ["Level", "1D Δ", "1W Δ", "1M Δ"],
        default="Level",
    )
    or "Level"
)
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
            "No history is stored for this selection yet. Daily scheduled "
            "snapshots will build it automatically."
        )
    else:
        chart_data, ylabel = display_series(
            hist,
            metric,
            change_mode,
        )
        st.line_chart(
            chart_data,
            height=460,
            use_container_width=True,
        )
        st.caption(
            ylabel
            + f" · target tenor: {tenor} "
            + f"({DAILY_TENORS[tenor]} DTE)."
        )

st.divider()

with st.expander("Method & API-credit behavior"):
    st.markdown(
        f"""
- **25Δ skew:** `25Δ call IV − 25Δ put IV`, in volatility points.
- **Daily basket:** Dashboard contains all **{len(AUTO_SYMBOLS)}** symbols run by the automatic daily task.
- **Daily tenors:** 1W and 1M target 7 and 30 DTE and use the available expiration closest to each target.
- **Pool average:** equal-weight average of the displayed non-index stocks. SPY, QQQ and IWM are excluded from the pool average.
- **Reference lines:** SPY and QQQ are displayed both as normal bars and as dashed benchmark lines when they are in the selection.
- **Automatic data:** GitHub Actions writes the saved daily rows to Supabase. Opening this Streamlit page does **not** call MarketData.
- **Manual button:** checks Supabase first. A ticker already saved for the requested session consumes **0 MarketData credits**. A missing ticker uses the same bounded historical-chain request as the automatic task, then derives missing IV/delta locally and produces both 1W and 1M from that one chain.
- **Storage:** Supabase stores the compact daily rows needed for the chart: spot, target/actual DTE, expiration, 25Δ call IV, 25Δ put IV and 25Δ skew. Raw option chains are not stored.
        """
    )
