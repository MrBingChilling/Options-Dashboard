from __future__ import annotations

from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.analytics import STANDARD, find_gamma_flip, gamma_curve
from src.skew_collector import AUTO_SYMBOLS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import volatility_history

CALL_COLOR = "#F43F5E"
PUT_COLOR = "#10B981"
AGG_COLOR = "#6384FF"
SPOT_COLOR = "#AAB3C5"
FLIP_COLOR = "#F97316"
CONTRACT_MULTIPLIER = 100.0


def latest_archive_rows(store: SnapshotStore) -> pd.DataFrame:
    """Return the newest archived daily chain metadata for each automatic ticker."""
    history = volatility_history(
        store,
        AUTO_SYMBOLS,
        "1W",
        limit=max(5000, len(AUTO_SYMBOLS) * 120),
        newest_first=True,
    )
    if history.empty or "archive_path" not in history.columns:
        return pd.DataFrame()
    history = history.dropna(subset=["archive_path"]).copy()
    if history.empty:
        return history
    return (
        history.sort_values("snapshot_date")
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
        .reset_index(drop=True)
    )


def download_archived_chain(store: SnapshotStore, archive_path: str) -> pd.DataFrame:
    """Download one private Parquet chain from Supabase Storage."""
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")
    cleaned = str(archive_path).strip().lstrip("/")
    if "/" not in cleaned:
        raise SnapshotStoreError(f"Invalid archive path: {archive_path}")
    bucket, object_path = cleaned.split("/", 1)
    headers = {"apikey": store.key, "Accept": "application/vnd.apache.parquet"}
    if store.key and not store.key.startswith(("sb_secret_", "sb_publishable_")):
        headers["Authorization"] = f"Bearer {store.key}"

    responses = []
    for endpoint in (
        f"{store.url}/storage/v1/object/authenticated/{bucket}/{object_path}",
        f"{store.url}/storage/v1/object/{bucket}/{object_path}",
    ):
        response = requests.get(endpoint, headers=headers, timeout=max(store.timeout, 60))
        responses.append(response)
        if response.status_code == 200:
            try:
                frame = pd.read_parquet(BytesIO(response.content))
            except Exception as exc:  # pragma: no cover - defensive decode error
                raise SnapshotStoreError(f"Archived chain could not be decoded: {exc}") from exc
            if "expiration" in frame.columns:
                frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce")
            return frame
        if response.status_code not in {400, 401, 403, 404}:
            break

    response = responses[-1]
    raise SnapshotStoreError(
        f"Supabase archived-chain download failed ({response.status_code}): {response.text[:300]}"
    )


def profile_from_archive(chain: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Build standard call-positive / put-negative GEX and volume by strike."""
    if chain.empty:
        raise ValueError("The archived option chain is empty.")
    frame = chain.copy()
    for column in ("strike", "underlyingPrice", "openInterest", "volume", "gamma_used", "gamma"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["side"] = frame["side"].astype(str).str.lower()
    spot_values = pd.to_numeric(frame.get("underlyingPrice"), errors="coerce").dropna()
    if spot_values.empty:
        raise ValueError("The archived chain has no usable underlying price.")
    spot = float(spot_values.median())

    gamma_used = pd.to_numeric(frame.get("gamma_used"), errors="coerce")
    if gamma_used.isna().all() and "gamma" in frame.columns:
        gamma_used = pd.to_numeric(frame["gamma"], errors="coerce")
    frame["_gamma"] = gamma_used.fillna(0.0).clip(lower=0.0)
    frame["openInterest"] = pd.to_numeric(frame.get("openInterest"), errors="coerce").fillna(0.0)
    frame["volume"] = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0)
    frame["base_gex"] = (
        frame["_gamma"]
        * frame["openInterest"]
        * CONTRACT_MULTIPLIER
        * spot**2
        * 0.01
    )
    frame["gex"] = np.where(frame["side"].eq("call"), frame["base_gex"], -frame["base_gex"])

    grouped = (
        frame.groupby(["strike", "side"], as_index=False)
        .agg(gex=("gex", "sum"), base_gex=("base_gex", "sum"), volume=("volume", "sum"))
    )
    pieces: list[pd.DataFrame] = []
    for metric in ("gex", "base_gex", "volume"):
        pivot = grouped.pivot(index="strike", columns="side", values=metric).fillna(0.0)
        pivot = pivot.rename(columns={"call": f"call_{metric}", "put": f"put_{metric}"})
        pieces.append(pivot)
    profile = pd.concat(pieces, axis=1).fillna(0.0).reset_index().sort_values("strike")
    for column in (
        "call_gex", "put_gex", "call_base_gex", "put_base_gex", "call_volume", "put_volume"
    ):
        if column not in profile.columns:
            profile[column] = 0.0
    profile["net_gex"] = profile["call_gex"] + profile["put_gex"]
    profile["aggregate_gex"] = profile["net_gex"].cumsum()
    return profile.reset_index(drop=True), spot


def _wall_levels(profile: pd.DataFrame) -> tuple[float | None, float | None]:
    call_wall = None
    put_wall = None
    if not profile.empty and profile["call_base_gex"].abs().max() > 0:
        call_wall = float(profile.loc[profile["call_base_gex"].abs().idxmax(), "strike"])
    if not profile.empty and profile["put_base_gex"].abs().max() > 0:
        put_wall = float(profile.loc[profile["put_base_gex"].abs().idxmax(), "strike"])
    return call_wall, put_wall


def _gamma_flip(chain: pd.DataFrame, spot: float) -> float | None:
    work = chain.copy()
    if "iv_used" not in work.columns:
        return None
    work["iv_used"] = pd.to_numeric(work["iv_used"], errors="coerce")
    work["openInterest"] = pd.to_numeric(work.get("openInterest"), errors="coerce").fillna(0.0)
    if work["iv_used"].notna().sum() < 2 or work["openInterest"].sum() <= 0:
        return None
    try:
        curve = gamma_curve(work, STANDARD)
    except (ValueError, KeyError, TypeError):
        return None
    return find_gamma_flip(curve, spot)


def gamma_exposure_chart(
    profile: pd.DataFrame,
    spot: float,
    call_wall: float | None,
    put_wall: float | None,
    gamma_flip: float | None,
) -> alt.Chart:
    data = profile.copy()
    data["call_gex_mm"] = data["call_gex"] / 1e6
    data["put_gex_mm"] = data["put_gex"] / 1e6
    data["aggregate_gex_mm"] = data["aggregate_gex"] / 1e6
    bars = pd.concat(
        [
            data[["strike", "call_gex_mm"]].rename(columns={"call_gex_mm": "value"}).assign(side="Call"),
            data[["strike", "put_gex_mm"]].rename(columns={"put_gex_mm": "value"}).assign(side="Put"),
        ],
        ignore_index=True,
    )
    base = alt.Chart(bars).encode(
        x=alt.X("strike:Q", title="Strike", axis=alt.Axis(format="~g", labelOverlap=True)),
        y=alt.Y("value:Q", title="GEX ($mm / 1% move)"),
        color=alt.Color(
            "side:N",
            scale=alt.Scale(domain=["Call", "Put"], range=[CALL_COLOR, PUT_COLOR]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=[
            alt.Tooltip("side:N", title="Side"),
            alt.Tooltip("strike:Q", title="Strike", format=".2f"),
            alt.Tooltip("value:Q", title="GEX ($mm)", format=",.2f"),
        ],
    )
    bar_layer = base.mark_bar(size=8, opacity=0.95)
    aggregate = (
        alt.Chart(data)
        .mark_line(color=AGG_COLOR, strokeWidth=2.4)
        .encode(
            x=alt.X("strike:Q"),
            y=alt.Y("aggregate_gex_mm:Q", title="Aggregate GEX ($mm)", axis=alt.Axis(orient="right")),
            tooltip=[
                alt.Tooltip("strike:Q", title="Strike", format=".2f"),
                alt.Tooltip("aggregate_gex_mm:Q", title="Aggregate GEX ($mm)", format=",.2f"),
            ],
        )
    )
    rules = [
        alt.Chart(pd.DataFrame({"x": [spot]}))
        .mark_rule(color=SPOT_COLOR, strokeDash=[5, 5], strokeWidth=1.5)
        .encode(x="x:Q")
    ]
    if gamma_flip is not None:
        rules.append(
            alt.Chart(pd.DataFrame({"x": [gamma_flip]}))
            .mark_rule(color=FLIP_COLOR, strokeDash=[3, 4], strokeWidth=1.5)
            .encode(x="x:Q")
        )

    labels: list[dict[str, object]] = []
    if call_wall is not None:
        call_value = float(data.loc[(data["strike"] - call_wall).abs().idxmin(), "call_gex_mm"])
        labels.append({"strike": call_wall, "value": call_value, "label": f"Call Wall {call_wall:g}", "kind": "call"})
    if put_wall is not None:
        put_value = float(data.loc[(data["strike"] - put_wall).abs().idxmin(), "put_gex_mm"])
        labels.append({"strike": put_wall, "value": put_value, "label": f"Put Wall {put_wall:g}", "kind": "put"})
    label_layer: alt.Chart | None = None
    if labels:
        label_frame = pd.DataFrame(labels)
        label_layer = (
            alt.Chart(label_frame)
            .mark_text(dy=-12, fontSize=12, fontWeight="bold")
            .encode(
                x="strike:Q",
                y="value:Q",
                text="label:N",
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["call", "put"], range=[CALL_COLOR, PUT_COLOR]),
                    legend=None,
                ),
            )
        )

    chart = alt.layer(bar_layer, aggregate, *rules)
    if label_layer is not None:
        chart = alt.layer(chart, label_layer)
    return (
        chart.resolve_scale(y="independent")
        .properties(height=430, title="Gamma Exposure")
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#293144", domainColor="#4A556C", tickColor="#4A556C")
    )


def volume_by_strike_chart(profile: pd.DataFrame) -> alt.Chart:
    data = profile.copy()
    calls = data[["strike", "call_volume"]].rename(columns={"call_volume": "value"}).assign(side="Call")
    puts = data[["strike", "put_volume"]].rename(columns={"put_volume": "value"}).assign(side="Put")
    puts["value"] = -puts["value"]
    bars = pd.concat([calls, puts], ignore_index=True)
    return (
        alt.Chart(bars)
        .mark_bar(size=8, opacity=0.95)
        .encode(
            x=alt.X("strike:Q", title="Strike", axis=alt.Axis(format="~g", labelOverlap=True)),
            y=alt.Y("value:Q", title="Contracts (puts shown below zero)"),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["Call", "Put"], range=[CALL_COLOR, PUT_COLOR]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("side:N", title="Side"),
                alt.Tooltip("strike:Q", title="Strike", format=".2f"),
                alt.Tooltip("value:Q", title="Signed volume", format=",.0f"),
            ],
        )
        .properties(height=340, title="Volume by Strike Price")
        .configure_view(strokeOpacity=0)
        .configure_axis(gridColor="#293144", domainColor="#4A556C", tickColor="#4A556C")
    )


def render_archived_gamma_dashboard(store: SnapshotStore) -> None:
    st.caption(
        "Gamma exposure and option volume reconstructed from the latest archived daily bounded chain. "
        "Changing ticker, expiration, or strike range reads Supabase only and uses 0 MarketData credits."
    )
    if not store.enabled:
        st.info("Configure Supabase to load archived option chains.")
        return
    try:
        archives = latest_archive_rows(store)
    except SnapshotStoreError as exc:
        st.error(str(exc))
        return
    if archives.empty:
        st.info("No archived option chains are available yet.")
        return

    options = archives["symbol"].astype(str).tolist()
    default_index = options.index("SPY") if "SPY" in options else 0
    symbol = st.selectbox(
        "Ticker",
        options,
        index=default_index,
        key="iv_gamma_symbol",
        help="One ticker at a time. The list includes automatic-daily symbols with an archived chain.",
    )
    row = archives.loc[archives["symbol"] == symbol].iloc[-1]
    archive_path = str(row["archive_path"])
    archive_key = f"{symbol}|{archive_path}"
    if st.session_state.get("iv_gamma_archive_key") != archive_key:
        try:
            with st.spinner(f"Loading saved {symbol} option chain…"):
                chain = download_archived_chain(store, archive_path)
        except SnapshotStoreError as exc:
            st.error(str(exc))
            return
        st.session_state["iv_gamma_archive_key"] = archive_key
        st.session_state["iv_gamma_archive_frame"] = chain
    chain = st.session_state.get("iv_gamma_archive_frame", pd.DataFrame()).copy()
    if chain.empty:
        st.info("The saved chain is empty.")
        return

    chain["expiration"] = pd.to_datetime(chain["expiration"], errors="coerce")
    expirations = sorted(chain["expiration"].dropna().dt.date.unique().tolist())
    if not expirations:
        st.info("The archived chain has no expiration dates.")
        return

    expiration = st.selectbox(
        "Expiration",
        expirations,
        index=0,
        format_func=lambda value: pd.Timestamp(value).strftime("%b %d, %Y"),
        key=f"iv_gamma_expiration_{symbol}",
    )
    expiry_chain = chain[chain["expiration"].dt.date == expiration].copy()
    if expiry_chain.empty:
        st.info("No contracts are available for that expiration.")
        return
    try:
        profile, spot = profile_from_archive(expiry_chain)
    except ValueError as exc:
        st.error(str(exc))
        return

    strikes = profile["strike"].dropna().astype(float).sort_values().unique()
    if len(strikes) == 0:
        st.info("No strikes are available for that expiration.")
        return
    min_strike, max_strike = float(strikes[0]), float(strikes[-1])
    if len(strikes) > 1:
        diffs = np.diff(strikes)
        positive = diffs[diffs > 0]
        step = float(np.min(positive)) if len(positive) else 1.0
    else:
        step = max(abs(min_strike) * 0.01, 0.5)
    strike_range = st.slider(
        "Strike range",
        min_value=min_strike,
        max_value=max_strike,
        value=(min_strike, max_strike),
        step=step,
        key=f"iv_gamma_strike_range_{symbol}_{expiration}",
    )
    visible = profile[profile["strike"].between(strike_range[0], strike_range[1])].copy()
    if visible.empty:
        st.info("No strikes fall inside the selected range.")
        return

    call_wall, put_wall = _wall_levels(profile)
    flip = _gamma_flip(expiry_chain, spot)
    latest_date = pd.Timestamp(row["snapshot_date"]).date()
    metrics = st.columns(4)
    metrics[0].metric("Snapshot spot", f"${spot:,.2f}")
    metrics[1].metric("Call wall", f"${call_wall:,.2f}" if call_wall is not None else "—")
    metrics[2].metric("Put wall", f"${put_wall:,.2f}" if put_wall is not None else "—")
    metrics[3].metric("Gamma flip", f"${flip:,.2f}" if flip is not None else "—")
    st.caption(
        f"Saved chain: {latest_date:%Y-%m-%d} · expiration {expiration:%Y-%m-%d} · "
        f"{len(expiry_chain):,} contracts in this expiration. Standard display convention: calls positive, puts negative."
    )
    st.altair_chart(
        gamma_exposure_chart(visible, spot, call_wall, put_wall, flip),
        use_container_width=True,
    )
    st.altair_chart(volume_by_strike_chart(visible), use_container_width=True)
    st.caption(
        "Blue Aggregate GEX is the cumulative net call-minus-put gamma exposure across the displayed strikes. "
        "The dashed gray line is the saved underlying spot; the orange dashed line is the modelled gamma flip when it falls on the strike axis."
    )
