from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CALL_COLOR = "#56D39B"
PUT_COLOR = "#FF6B7A"
NET_COLOR = "#7EA6FF"
MUTED = "#8B97AD"


def _layout(fig: go.Figure, title: str, height: int = 440) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.04, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(139,151,173,0.14)", zerolinecolor=MUTED)
    return fig


def strike_gex_chart(profile: pd.DataFrame, spot: float, range_pct: float = 0.25) -> go.Figure:
    visible = profile[
        profile["strike"].between(spot * (1 - range_pct), spot * (1 + range_pct))
    ]
    if visible.empty:
        visible = profile
    fig = go.Figure()
    fig.add_bar(x=visible["strike"], y=visible["call_gex"] / 1e6, name="Calls", marker_color=CALL_COLOR)
    fig.add_bar(x=visible["strike"], y=visible["put_gex"] / 1e6, name="Puts", marker_color=PUT_COLOR)
    fig.add_vline(x=spot, line_dash="dash", line_color="#F4F7FB", annotation_text="Spot")
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="GEX ($mm per 1% move)")
    fig.update_xaxes(title="Strike")
    return _layout(fig, "Gamma exposure by strike")


def gamma_curve_chart(curve: pd.DataFrame, spot: float, flip: float | None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=curve["spot"],
        y=curve["net_gex"] / 1e6,
        mode="lines",
        name="Net GEX",
        line=dict(color=NET_COLOR, width=3),
        fill="tozeroy",
        fillcolor="rgba(126,166,255,0.12)",
    )
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_vline(x=spot, line_dash="dash", line_color="#F4F7FB", annotation_text="Spot")
    if flip is not None:
        fig.add_vline(x=flip, line_dash="dot", line_color="#F6C85F", annotation_text="Flip")
    fig.update_yaxes(title="Modelled net GEX ($mm per 1% move)")
    fig.update_xaxes(title="Simulated underlying price")
    return _layout(fig, "Gamma regime across underlying prices")


def expiration_gex_chart(profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=profile["expiration_date"], y=profile["call_gex"] / 1e6, name="Calls", marker_color=CALL_COLOR)
    fig.add_bar(x=profile["expiration_date"], y=profile["put_gex"] / 1e6, name="Puts", marker_color=PUT_COLOR)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="GEX ($mm per 1% move)")
    return _layout(fig, "Gamma exposure by expiration")


def trend_levels_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color, dash in (
        ("spot", "Spot", "#F4F7FB", "solid"),
        ("gamma_flip", "Gamma flip", "#F6C85F", "dot"),
        ("call_wall", "Call wall", CALL_COLOR, "dash"),
        ("put_wall", "Put wall", PUT_COLOR, "dash"),
    ):
        fig.add_scatter(
            x=history["snapshot_date"],
            y=history[column],
            mode="lines+markers",
            name=label,
            line=dict(color=color, dash=dash),
        )
    fig.update_yaxes(title="Price level")
    return _layout(fig, "Positioning levels over time")


def trend_regime_chart(history: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(
        x=history["snapshot_date"],
        y=history["net_gex"] / 1e6,
        mode="lines+markers",
        name="Net GEX",
        line=dict(color=NET_COLOR, width=3),
    )
    fig.add_scatter(
        x=history["snapshot_date"],
        y=history["put_call_oi_ratio"],
        mode="lines+markers",
        name="Put/call OI",
        line=dict(color="#C792EA"),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Net GEX ($mm per 1%)", secondary_y=False)
    fig.update_yaxes(title_text="Put/call OI ratio", secondary_y=True, showgrid=False)
    return _layout(fig, "Regime and sentiment trend")
