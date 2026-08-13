"""Options positioning dashboard package."""

from __future__ import annotations

import altair as alt
import pandas as pd


def skew_metric_bar_chart(
    cross: pd.DataFrame,
    metric_column: str,
    axis_title: str,
    label_title: str,
    signed: bool,
    preset_color_order: list[str],
    preset_colors: dict[str, str],
    reference_colors: dict[str, str],
    index_symbols: list[str],
) -> alt.Chart:
    data = cross.copy()
    data["skew_vol_pts"] = data["skew_25d"] * 100.0
    data["call_25d_iv_pct"] = data["call_25d_iv"] * 100.0
    data["put_25d_iv_pct"] = data["put_25d_iv"] * 100.0
    data["display_value"] = data[metric_column] * 100.0
    data["zero"] = 0.0

    label_width = max(5, int(data["symbol"].astype(str).str.len().max()))
    value_format = "+.1f" if signed else ".1f"
    data["symbol_label"] = [
        f"{symbol:<{label_width}}  {value:{value_format}}"
        for symbol, value in zip(data["symbol"].astype(str), data["display_value"])
    ]
    symbol_order = data["symbol_label"].tolist()
    height = min(1500, max(470, 30 * len(data)))

    values = pd.to_numeric(data["display_value"], errors="coerce").dropna()
    if values.empty:
        domain_min, domain_max = -1.0, 1.0
    else:
        raw_min = min(0.0, float(values.min()))
        raw_max = max(0.0, float(values.max()))
        span = max(raw_max - raw_min, 1.0)
        edge_pad = span * 0.012
        domain_min = raw_min - (edge_pad if raw_min < 0 else 0.0)
        domain_max = raw_max + (edge_pad if raw_max > 0 else 0.0)

    present_groups = [
        group for group in preset_color_order if group in set(data["preset_group"])
    ]
    color_scale = alt.Scale(
        domain=present_groups,
        range=[preset_colors[group] for group in present_groups],
    )

    tooltips = [
        alt.Tooltip("symbol:N", title="Ticker"),
        alt.Tooltip("preset_group:N", title="Preset"),
        alt.Tooltip("skew_vol_pts:Q", title="25Δ skew (vol pts)", format="+.2f"),
        alt.Tooltip("call_25d_iv_pct:Q", title="25Δ call IV (%)", format=".2f"),
        alt.Tooltip("put_25d_iv_pct:Q", title="25Δ put IV (%)", format=".2f"),
        alt.Tooltip("actual_dte:Q", title="Actual DTE", format=".0f"),
        alt.Tooltip("expiration:T", title="Expiration", format="%Y-%m-%d"),
        alt.Tooltip("snapshot_date:T", title="Snapshot", format="%Y-%m-%d"),
        alt.Tooltip("spot:Q", title="Spot", format="$.2f"),
    ]

    base = alt.Chart(data).encode(
        x=alt.X(
            "zero:Q",
            title=axis_title,
            axis=alt.Axis(format=".1f"),
            scale=alt.Scale(domain=[domain_min, domain_max], nice=False, zero=False),
        ),
        x2=alt.X2("display_value:Q"),
        y=alt.Y(
            "symbol_label:N",
            sort=symbol_order,
            title=None,
            axis=alt.Axis(
                labelFont="monospace",
                labelFontSize=13,
                labelLimit=200,
                labelColor="#E8EDF7",
            ),
        ),
        color=alt.Color(
            "preset_group:N",
            scale=color_scale,
            legend=alt.Legend(
                title=None,
                orient="top",
                direction="horizontal",
                columns=2,
                labelFontSize=11,
                symbolSize=90,
                labelLimit=140,
                offset=8,
            ),
        ),
        tooltip=tooltips,
    )

    positive = base.transform_filter(alt.datum.display_value >= 0).mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
        cornerRadiusTopLeft=0,
        cornerRadiusBottomLeft=0,
    )
    negative = base.transform_filter(alt.datum.display_value < 0).mark_bar(
        cornerRadiusTopLeft=6,
        cornerRadiusBottomLeft=6,
        cornerRadiusTopRight=0,
        cornerRadiusBottomRight=0,
    )
    zero = (
        alt.Chart(pd.DataFrame({"value": [0.0]}))
        .mark_rule(color="#FFFFFF", strokeWidth=1.7, opacity=0.95)
        .encode(x=alt.X("value:Q"))
    )

    refs: list[dict[str, object]] = []
    for symbol in ("SPY", "QQQ"):
        match = data[data["symbol"] == symbol]
        if not match.empty:
            value = float(match.iloc[0]["display_value"])
            refs.append(
                {
                    "kind": symbol,
                    "value": value,
                    "label": f"{symbol} {value:+.1f}" if signed else f"{symbol} {value:.1f}",
                }
            )

    pool = data[~data["symbol"].isin(index_symbols)]
    if not pool.empty:
        value = float(pool["display_value"].mean())
        refs.append(
            {
                "kind": "Pool avg",
                "value": value,
                "label": f"Pool avg {value:+.1f}" if signed else f"Pool avg {value:.1f}",
            }
        )

    chart = positive + negative + zero
    for row_number, row in enumerate(refs):
        frame = pd.DataFrame([row])
        ref_color = reference_colors[str(row["kind"])]
        rule = (
            alt.Chart(frame)
            .mark_rule(color=ref_color, strokeDash=[6, 5], strokeWidth=2)
            .encode(
                x=alt.X("value:Q"),
                tooltip=[
                    alt.Tooltip("kind:N", title="Reference"),
                    alt.Tooltip(
                        "value:Q",
                        title=label_title,
                        format="+.2f" if signed else ".2f",
                    ),
                ],
            )
        )
        label = (
            alt.Chart(frame)
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
        chart = chart + rule + label

    return chart.properties(height=height).configure_view(stroke=None)
