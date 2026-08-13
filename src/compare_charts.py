from __future__ import annotations

import altair as alt
import pandas as pd


def skew_metric_compare_bar_chart(
    current: pd.DataFrame,
    comparison: pd.DataFrame,
    metric_column: str,
    axis_title: str,
    label_title: str,
    signed: bool,
    preset_color_order: list[str],
    preset_colors: dict[str, str],
    reference_colors: dict[str, str],
    index_symbols: list[str],
) -> alt.Chart:
    """Render a wide faded historical bar behind a thin solid current bar."""
    data = current.copy()
    for frame in (data, comparison):
        if frame.empty:
            continue
        frame["skew_vol_pts"] = frame["skew_25d"] * 100.0
        frame["call_25d_iv_pct"] = frame["call_25d_iv"] * 100.0
        frame["put_25d_iv_pct"] = frame["put_25d_iv"] * 100.0

    data["display_value"] = data[metric_column] * 100.0
    data["zero"] = 0.0

    compare_columns = [
        "symbol",
        "snapshot_date",
        "actual_dte",
        "expiration",
        "spot",
        "skew_vol_pts",
        "call_25d_iv_pct",
        "put_25d_iv_pct",
        metric_column,
    ]
    available_compare_columns = [column for column in compare_columns if column in comparison.columns]
    compare_data = comparison[available_compare_columns].copy() if available_compare_columns else pd.DataFrame()
    if not compare_data.empty:
        compare_data = compare_data.drop_duplicates(subset=["symbol"], keep="last")
        rename_map = {
            column: f"compare_{column}"
            for column in compare_data.columns
            if column != "symbol"
        }
        compare_data = compare_data.rename(columns=rename_map)
        data = data.merge(compare_data, on="symbol", how="left")
        data["compare_display_value"] = data[f"compare_{metric_column}"] * 100.0
    else:
        data["compare_display_value"] = pd.NA
        for column in (
            "snapshot_date",
            "actual_dte",
            "expiration",
            "spot",
            "skew_vol_pts",
            "call_25d_iv_pct",
            "put_25d_iv_pct",
        ):
            data[f"compare_{column}"] = pd.NA

    data["change_display_value"] = data["display_value"] - pd.to_numeric(
        data["compare_display_value"], errors="coerce"
    )

    label_width = max(5, int(data["symbol"].astype(str).str.len().max()))
    value_format = "+.1f" if signed else ".1f"
    data["symbol_label"] = [
        f"{symbol:<{label_width}}  {value:{value_format}}"
        for symbol, value in zip(data["symbol"].astype(str), data["display_value"])
    ]
    symbol_order = data["symbol_label"].tolist()
    height = min(1500, max(470, 30 * len(data)))

    current_values = pd.to_numeric(data["display_value"], errors="coerce").dropna()
    compare_values = pd.to_numeric(data["compare_display_value"], errors="coerce").dropna()
    values = pd.concat([current_values, compare_values], ignore_index=True)
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

    current_tooltips = [
        alt.Tooltip("symbol:N", title="Ticker"),
        alt.Tooltip("preset_group:N", title="Preset"),
        alt.Tooltip("display_value:Q", title=f"Current {label_title}", format="+.2f" if signed else ".2f"),
        alt.Tooltip("compare_display_value:Q", title=f"Compare {label_title}", format="+.2f" if signed else ".2f"),
        alt.Tooltip("change_display_value:Q", title="Current − compare", format="+.2f"),
        alt.Tooltip("snapshot_date:T", title="Current date", format="%Y-%m-%d"),
        alt.Tooltip("compare_snapshot_date:T", title="Compare date", format="%Y-%m-%d"),
        alt.Tooltip("skew_vol_pts:Q", title="Current 25Δ skew", format="+.2f"),
        alt.Tooltip("compare_skew_vol_pts:Q", title="Compare 25Δ skew", format="+.2f"),
        alt.Tooltip("call_25d_iv_pct:Q", title="Current 25Δ call IV", format=".2f"),
        alt.Tooltip("compare_call_25d_iv_pct:Q", title="Compare 25Δ call IV", format=".2f"),
        alt.Tooltip("put_25d_iv_pct:Q", title="Current 25Δ put IV", format=".2f"),
        alt.Tooltip("compare_put_25d_iv_pct:Q", title="Compare 25Δ put IV", format=".2f"),
        alt.Tooltip("actual_dte:Q", title="Current DTE", format=".0f"),
        alt.Tooltip("compare_actual_dte:Q", title="Compare DTE", format=".0f"),
        alt.Tooltip("expiration:T", title="Current expiration", format="%Y-%m-%d"),
        alt.Tooltip("compare_expiration:T", title="Compare expiration", format="%Y-%m-%d"),
        alt.Tooltip("spot:Q", title="Current spot", format="$.2f"),
        alt.Tooltip("compare_spot:Q", title="Compare spot", format="$.2f"),
    ]

    y_encoding = alt.Y(
        "symbol_label:N",
        sort=symbol_order,
        title=None,
        axis=alt.Axis(
            labelFont="monospace",
            labelFontSize=13,
            labelLimit=200,
            labelColor="#E8EDF7",
        ),
    )
    color_encoding = alt.Color(
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
    )
    x_scale = alt.Scale(domain=[domain_min, domain_max], nice=False, zero=False)

    compare_base = alt.Chart(data).encode(
        x=alt.X("zero:Q", title=axis_title, axis=alt.Axis(format=".1f"), scale=x_scale),
        x2=alt.X2("compare_display_value:Q"),
        y=y_encoding,
        color=color_encoding,
        tooltip=current_tooltips,
    )
    compare_positive = compare_base.transform_filter(
        alt.datum.compare_display_value >= 0
    ).mark_bar(
        size=24,
        opacity=0.32,
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
        cornerRadiusTopLeft=0,
        cornerRadiusBottomLeft=0,
    )
    compare_negative = compare_base.transform_filter(
        alt.datum.compare_display_value < 0
    ).mark_bar(
        size=24,
        opacity=0.32,
        cornerRadiusTopLeft=6,
        cornerRadiusBottomLeft=6,
        cornerRadiusTopRight=0,
        cornerRadiusBottomRight=0,
    )

    current_base = alt.Chart(data).encode(
        x=alt.X("zero:Q", title=axis_title, axis=alt.Axis(format=".1f"), scale=x_scale),
        x2=alt.X2("display_value:Q"),
        y=y_encoding,
        color=color_encoding,
        tooltip=current_tooltips,
    )
    current_positive = current_base.transform_filter(alt.datum.display_value >= 0).mark_bar(
        size=12,
        opacity=1.0,
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6,
        cornerRadiusTopLeft=0,
        cornerRadiusBottomLeft=0,
    )
    current_negative = current_base.transform_filter(alt.datum.display_value < 0).mark_bar(
        size=12,
        opacity=1.0,
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

    chart = compare_positive + compare_negative + current_positive + current_negative + zero
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
