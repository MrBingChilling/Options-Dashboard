from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.analytics import STANDARD, find_gamma_flip, gamma_curve
from src.charts import render_chart
from src.skew_collector import AUTO_SYMBOLS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import volatility_history

CALL_COLOR = "#F43F5E"
PUT_COLOR = "#10B981"
AGG_COLOR = "#6384FF"
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


def focused_strike_window(
    profile: pd.DataFrame,
    spot: float,
    call_wall: float | None,
    put_wall: float | None,
) -> tuple[float, float]:
    """Choose a compact mobile-first window that keeps spot and both walls visible."""
    strikes = pd.to_numeric(profile["strike"], errors="coerce").dropna().sort_values().unique()
    if len(strikes) == 0:
        return spot * 0.95, spot * 1.05
    full_min, full_max = float(strikes[0]), float(strikes[-1])
    anchors = [float(spot)]
    for value in (call_wall, put_wall):
        if value is not None and np.isfinite(value):
            anchors.append(float(value))
    anchor_min, anchor_max = min(anchors), max(anchors)
    min_span = max(abs(spot) * 0.08, 1.0)
    span = max(anchor_max - anchor_min, min_span)
    padding = max(span * 0.18, abs(spot) * 0.012)
    low = max(full_min, anchor_min - padding)
    high = min(full_max, anchor_max + padding)
    if low >= high:
        return full_min, full_max
    return float(low), float(high)


def _numeric_strike_axis(profile: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    origin = date(2000, 1, 1)
    times = [(origin + timedelta(days=index)).isoformat() for index in range(len(profile))]
    labels = {
        time: f"{float(strike):g}"
        for time, strike in zip(times, pd.to_numeric(profile["strike"], errors="coerce"))
    }
    return times, labels


def _marker_time(profile: pd.DataFrame, times: list[str], strike: float | None) -> str | None:
    if strike is None or not np.isfinite(strike) or profile.empty:
        return None
    strikes = pd.to_numeric(profile["strike"], errors="coerce").to_numpy(float)
    index = int(np.nanargmin(np.abs(strikes - float(strike))))
    return times[index]


def gamma_exposure_spec(
    profile: pd.DataFrame,
    call_wall: float | None,
    put_wall: float | None,
) -> dict[str, object]:
    """TradingView-style, touch-friendly GEX-by-strike chart specification."""
    data = profile.reset_index(drop=True).copy()
    times, labels = _numeric_strike_axis(data)
    call_markers = []
    put_markers = []
    call_time = _marker_time(data, times, call_wall)
    put_time = _marker_time(data, times, put_wall)
    if call_time is not None:
        call_markers.append(
            {
                "time": call_time,
                "position": "aboveBar",
                "color": CALL_COLOR,
                "shape": "arrowDown",
                "text": f"Call Wall {call_wall:g}",
            }
        )
    if put_time is not None:
        put_markers.append(
            {
                "time": put_time,
                "position": "belowBar",
                "color": PUT_COLOR,
                "shape": "arrowUp",
                "text": f"Put Wall {put_wall:g}",
            }
        )

    call_data = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, data["call_gex"])
    ]
    put_data = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, data["put_gex"])
    ]
    aggregate_data = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, data["aggregate_gex"])
    ]
    return {
        "title": "Gamma Exposure",
        "subtitle": "Pinch/scroll to zoom · drag to pan · drag either price axis to rescale",
        "numericLabels": labels,
        "leftScale": True,
        "rightScale": True,
        "series": [
            {
                "name": "Call",
                "color": CALL_COLOR,
                "type": "histogram",
                "data": call_data,
                "options": {
                    "priceScaleId": "left",
                    "priceFormat": {"type": "custom", "formatter": "compact"},
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                    "markers": call_markers,
                },
            },
            {
                "name": "Put",
                "color": PUT_COLOR,
                "type": "histogram",
                "data": put_data,
                "options": {
                    "priceScaleId": "left",
                    "priceFormat": {"type": "custom", "formatter": "compact"},
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                    "markers": put_markers,
                },
            },
            {
                "name": "Aggregate GEX",
                "color": AGG_COLOR,
                "type": "line",
                "data": aggregate_data,
                "options": {
                    "priceScaleId": "right",
                    "priceFormat": {"type": "custom", "formatter": "compact"},
                    "lineWidth": 3,
                    "lastValueVisible": True,
                    "priceLineVisible": False,
                },
            },
        ],
    }


def volume_by_strike_spec(profile: pd.DataFrame) -> dict[str, object]:
    """Touch-friendly call-positive / put-negative volume chart."""
    data = profile.reset_index(drop=True).copy()
    times, labels = _numeric_strike_axis(data)
    calls = [
        {"time": time, "value": float(value)}
        for time, value in zip(times, data["call_volume"])
    ]
    puts = [
        {"time": time, "value": -float(value)}
        for time, value in zip(times, data["put_volume"])
    ]
    return {
        "title": "Volume by Strike Price",
        "subtitle": "Calls above zero · puts below zero · pinch/scroll to zoom · drag to pan",
        "numericLabels": labels,
        "leftScale": True,
        "rightScale": False,
        "series": [
            {
                "name": "Call",
                "color": CALL_COLOR,
                "type": "histogram",
                "data": calls,
                "options": {
                    "priceScaleId": "left",
                    "priceFormat": {"type": "custom", "formatter": "compact"},
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                },
            },
            {
                "name": "Put",
                "color": PUT_COLOR,
                "type": "histogram",
                "data": puts,
                "options": {
                    "priceScaleId": "left",
                    "priceFormat": {"type": "custom", "formatter": "compact"},
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                },
            },
        ],
    }


def render_archived_gamma_dashboard(store: SnapshotStore) -> None:
    st.caption(
        "Gamma exposure and option volume reconstructed from the latest archived daily bounded chain. "
        "Changing ticker, expiration, strike view, zoom, or axes reads Supabase only and uses 0 MarketData credits."
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

    control_left, control_right = st.columns([1.4, 1.0])
    expiration = control_left.selectbox(
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

    call_wall, put_wall = _wall_levels(profile)
    flip = _gamma_flip(expiry_chain, spot)
    strike_view = control_right.selectbox(
        "Strike view",
        ["Focus", "Full", "Custom"],
        index=0,
        help="Focus keeps spot and both walls in view. Use the chart itself to pinch/zoom and pan.",
    )

    strikes = profile["strike"].dropna().astype(float).sort_values().unique()
    if len(strikes) == 0:
        st.info("No strikes are available for that expiration.")
        return
    min_strike, max_strike = float(strikes[0]), float(strikes[-1])
    if strike_view == "Focus":
        low, high = focused_strike_window(profile, spot, call_wall, put_wall)
    elif strike_view == "Full":
        low, high = min_strike, max_strike
    else:
        if len(strikes) > 1:
            positive = np.diff(strikes)
            positive = positive[positive > 0]
            step = float(np.min(positive)) if len(positive) else 1.0
        else:
            step = max(abs(min_strike) * 0.01, 0.5)
        low, high = st.slider(
            "Custom strike range",
            min_value=min_strike,
            max_value=max_strike,
            value=(min_strike, max_strike),
            step=step,
            key=f"iv_gamma_strike_range_{symbol}_{expiration}",
        )

    visible = profile[profile["strike"].between(low, high)].copy()
    if visible.empty:
        st.info("No strikes fall inside the selected range.")
        return

    latest_date = pd.Timestamp(row["snapshot_date"]).date()
    metrics = st.columns(4)
    metrics[0].metric("Spot", f"${spot:,.2f}")
    metrics[1].metric("Call wall", f"${call_wall:,.2f}" if call_wall is not None else "—")
    metrics[2].metric("Put wall", f"${put_wall:,.2f}" if put_wall is not None else "—")
    metrics[3].metric("Gamma flip", f"${flip:,.2f}" if flip is not None else "—")
    st.caption(
        f"Saved chain {latest_date:%Y-%m-%d} · expiration {expiration:%Y-%m-%d} · "
        f"{len(expiry_chain):,} contracts. Standard display convention: calls positive, puts negative."
    )

    render_chart(gamma_exposure_spec(visible, call_wall, put_wall), height=455)
    st.caption(
        "Wall labels now match the reference style: Call Wall is marked above the red bar with a downward arrow; "
        "Put Wall is marked below the green bar with an upward arrow. Tap legend chips to hide/show a series."
    )
    render_chart(volume_by_strike_spec(visible), height=380)
    st.caption(
        "Both charts use the same touch-friendly Lightweight Charts engine as the TradingView-style history view. "
        "On mobile: pinch to zoom, drag horizontally to pan, and drag a price axis vertically to expand/compress that scale."
    )
