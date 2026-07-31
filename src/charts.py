from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


CALL_COLOR = "#36D399"
PUT_COLOR = "#FB7185"
NET_COLOR = "#60A5FA"
TOTAL_COLOR = "#C084FC"
FLIP_COLOR = "#F6C85F"
PRICE_COLOR = "#F4F7FB"
MUTED = "#8B97AD"


@lru_cache(maxsize=1)
def _chart_library_source() -> str:
    """Load the bundled chart engine for direct embedding in each iframe."""
    path = Path(__file__).resolve().parent.parent / "static" / "lightweight-charts.standalone.production.js"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Bundled chart engine could not be loaded: {exc}") from exc


def _number(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if isfinite(output) else None


def _date_text(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def _time_points(frame: pd.DataFrame, time_column: str, value_column: str) -> list[dict[str, Any]]:
    points = []
    for time_value, raw_value in zip(frame[time_column], frame[value_column]):
        value = _number(raw_value)
        if value is not None and pd.notna(time_value):
            points.append({"time": _date_text(time_value), "value": value})
    return points


def _numeric_axis(values: list[float]) -> tuple[list[str], dict[str, str]]:
    origin = date(2000, 1, 1)
    times = [(origin + timedelta(days=index)).isoformat() for index in range(len(values))]
    labels = {time: f"{value:,.2f}" for time, value in zip(times, values)}
    return times, labels


def _series(
    name: str,
    color: str,
    data: list[dict[str, Any]],
    series_type: str = "line",
    **options: Any,
) -> dict[str, Any]:
    return {
        "name": name,
        "color": color,
        "type": series_type,
        "data": data,
        "options": options,
    }


def strike_gex_chart(profile: pd.DataFrame, spot: float, range_pct: float = 0.25) -> dict[str, Any]:
    visible = profile[
        profile["strike"].between(spot * (1 - range_pct), spot * (1 + range_pct))
    ].copy()
    if visible.empty:
        visible = profile.copy()
    strikes = visible["strike"].astype(float).tolist()
    times, labels = _numeric_axis(strikes)
    calls = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, visible["call_gex"])
    ]
    puts = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, visible["put_gex"])
    ]
    net = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, visible["net_gex"])
    ]
    closest_index = min(range(len(strikes)), key=lambda index: abs(strikes[index] - spot))
    return {
        "title": "Gamma exposure by strike",
        "subtitle": "GEX ($mm per 1% move)",
        "numericLabels": labels,
        "series": [
            _series("Calls", CALL_COLOR, calls, "histogram", priceFormat={"type": "custom", "formatter": "compact"}),
            _series("Puts", PUT_COLOR, puts, "histogram", priceFormat={"type": "custom", "formatter": "compact"}),
            _series(
                "Net GEX",
                NET_COLOR,
                net,
                "line",
                lineWidth=2,
                markers=[{"time": times[closest_index], "position": "aboveBar", "color": PRICE_COLOR, "shape": "circle", "text": f"Spot {spot:,.2f}"}],
            ),
        ],
    }


def gamma_curve_chart(curve: pd.DataFrame, spot: float, flip: float | None) -> dict[str, Any]:
    prices = curve["spot"].astype(float).tolist()
    times, labels = _numeric_axis(prices)
    data = [
        {"time": time, "value": float(value) / 1e6}
        for time, value in zip(times, curve["net_gex"])
    ]
    markers = []
    spot_index = min(range(len(prices)), key=lambda index: abs(prices[index] - spot))
    markers.append({"time": times[spot_index], "position": "aboveBar", "color": PRICE_COLOR, "shape": "circle", "text": "Spot"})
    if flip is not None:
        flip_index = min(range(len(prices)), key=lambda index: abs(prices[index] - flip))
        markers.append({"time": times[flip_index], "position": "belowBar", "color": FLIP_COLOR, "shape": "arrowUp", "text": f"Flip {flip:,.2f}"})
    return {
        "title": "Gamma regime across underlying prices",
        "subtitle": "Modelled net GEX ($mm per 1% move)",
        "numericLabels": labels,
        "series": [
            _series(
                "Net GEX",
                NET_COLOR,
                data,
                "area",
                lineWidth=3,
                topColor="rgba(96,165,250,.30)",
                bottomColor="rgba(96,165,250,.02)",
                markers=markers,
            )
        ],
    }


def expiration_net_chart(profile: pd.DataFrame) -> dict[str, Any]:
    return {
        "title": "Net gamma by expiration",
        "subtitle": "Call GEX + put GEX ($mm per 1% move)",
        "series": [
            _series(
                "Net gamma",
                NET_COLOR,
                [
                    {"time": _date_text(day), "value": float(value) / 1e6}
                    for day, value in zip(profile["expiration_date"], profile["net_gex"])
                ],
                "line",
                lineWidth=3,
            )
        ],
    }


def expiration_total_chart(profile: pd.DataFrame) -> dict[str, Any]:
    return {
        "title": "Absolute total gamma by expiration",
        "subtitle": "|Call GEX| + |Put GEX| ($mm per 1% move)",
        "series": [
            _series(
                "Absolute total gamma",
                TOTAL_COLOR,
                [
                    {"time": _date_text(day), "value": float(value) / 1e6}
                    for day, value in zip(profile["expiration_date"], profile["total_gex"])
                ],
                "line",
                lineWidth=3,
            )
        ],
    }


def trend_levels_chart(history: pd.DataFrame) -> dict[str, Any]:
    series = []
    for column, label, color, style in (
        ("spot", "Snapshot spot", PRICE_COLOR, 0),
        ("gamma_flip", "Gamma flip", FLIP_COLOR, 2),
        ("call_wall", "Call wall", CALL_COLOR, 1),
        ("put_wall", "Put wall", PUT_COLOR, 1),
    ):
        series.append(
            _series(
                label,
                color,
                _time_points(history, "snapshot_date", column),
                "line",
                lineWidth=2,
                lineStyle=style,
            )
        )
    return {
        "title": "Positioning levels over time",
        "subtitle": "Daily closing price scale",
        "series": series,
    }


def trend_regime_chart(history: pd.DataFrame) -> dict[str, Any]:
    gex = history.assign(net_gex_mm=history["net_gex"] / 1e6)
    return {
        "title": "Regime and sentiment trend",
        "subtitle": "Separate scales; hover for exact values",
        "series": [
            _series(
                "Net GEX ($mm)",
                NET_COLOR,
                _time_points(gex, "snapshot_date", "net_gex_mm"),
                "line",
                lineWidth=3,
                priceScaleId="right",
            ),
            _series(
                "Put/call OI",
                TOTAL_COLOR,
                _time_points(history, "snapshot_date", "put_call_oi_ratio"),
                "line",
                lineWidth=2,
                priceScaleId="left",
            ),
        ],
        "leftScale": True,
    }


def price_gamma_chart(
    candles: pd.DataFrame,
    history: pd.DataFrame,
    profile: pd.DataFrame | None = None,
    price_style: str = "Candlestick",
    gamma_mode: str = "Calls left / puts right",
    show_levels: bool = True,
    show_regime: bool = True,
    title: str = "Price action and options positioning",
) -> dict[str, Any]:
    candle_rows = []
    line_rows = []
    for row in candles.itertuples(index=False):
        day = _date_text(row.time)
        candle_rows.append(
            {
                "time": day,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
        )
        line_rows.append({"time": day, "value": float(row.close)})
    price_data = candle_rows if price_style == "Candlestick" else line_rows
    price_type = "candlestick" if price_style == "Candlestick" else "line"
    series = [
        _series(
            "Price",
            PRICE_COLOR,
            price_data,
            price_type,
            lineWidth=2,
            upColor=CALL_COLOR,
            downColor=PUT_COLOR,
            borderUpColor=CALL_COLOR,
            borderDownColor=PUT_COLOR,
            wickUpColor=CALL_COLOR,
            wickDownColor=PUT_COLOR,
            priceScaleId="right",
            role="price",
        )
    ]
    if show_levels and not history.empty:
        for column, label, color, style in (
            ("gamma_flip", "Gamma flip", FLIP_COLOR, 2),
            ("call_wall", "Call wall", CALL_COLOR, 1),
            ("put_wall", "Put wall", PUT_COLOR, 1),
        ):
            series.append(
                _series(
                    label,
                    color,
                    _time_points(history, "snapshot_date", column),
                    "line",
                    lineWidth=2,
                    lineStyle=style,
                    priceScaleId="right",
                )
            )
    if show_regime and not history.empty:
        regime = history.assign(net_gex_mm=history["net_gex"] / 1e6)
        series.extend(
            [
                _series(
                    "Net GEX ($mm)",
                    NET_COLOR,
                    _time_points(regime, "snapshot_date", "net_gex_mm"),
                    "line",
                    lineWidth=2,
                    priceScaleId="gex",
                    scaleMargins={"top": 0.76, "bottom": 0.03},
                ),
                _series(
                    "Put/call OI",
                    TOTAL_COLOR,
                    _time_points(history, "snapshot_date", "put_call_oi_ratio"),
                    "line",
                    lineWidth=2,
                    priceScaleId="ratio",
                    scaleMargins={"top": 0.76, "bottom": 0.03},
                ),
            ]
        )
    gamma_profile = []
    if profile is not None and not profile.empty:
        for row in profile.itertuples(index=False):
            gamma_profile.append(
                {
                    "strike": float(row.strike),
                    "call": abs(float(row.call_gex)) / 1e6,
                    "put": abs(float(row.put_gex)) / 1e6,
                }
            )
    return {
        "title": title,
        "subtitle": "Drag to pan · scroll/pinch to zoom · tap or hover for values",
        "series": series,
        "gammaProfile": gamma_profile,
        "gammaMode": gamma_mode,
        "priceScaleMargins": {"top": 0.05, "bottom": 0.30 if show_regime else 0.08},
    }


def render_chart(spec: dict[str, Any], height: int = 460) -> None:
    payload = json.dumps(spec, separators=(",", ":"), allow_nan=False)
    st.iframe(_chart_document(payload), height=height, width="stretch")


def _chart_document(payload: str) -> str:
    # The app previously referenced /app/static/... from inside a srcdoc iframe.
    # That URL is deployment/base-path dependent and can leave a fully styled but
    # empty chart shell. Embedding the vendored library makes every chart
    # self-contained and avoids any browser-side asset request.
    chart_library = _chart_library_source()
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <script>{chart_library}</script>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: transparent; color: #E5EAF2; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }}
    #wrap {{ position: relative; width: 100%; height: 100%; border: 1px solid #25304A; border-radius: 12px; overflow: hidden; background: #0E1525; }}
    #head {{ height: 58px; padding: 10px 12px 6px; display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; border-bottom: 1px solid rgba(37,48,74,.75); }}
    #title {{ font-size: 14px; font-weight: 700; line-height: 1.2; }}
    #subtitle {{ margin-top: 3px; color: #8B97AD; font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 60vw; }}
    #legend {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 5px; max-height: 42px; overflow: hidden; }}
    .legend {{ border: 1px solid #2B3752; border-radius: 999px; background: #141D31; color: #C7D0E0; padding: 4px 8px; font-size: 10px; cursor: pointer; touch-action: manipulation; }}
    .legend.off {{ opacity: .38; }}
    .dot {{ display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }}
    #chart {{ position: relative; width: 100%; height: calc(100% - 58px); }}
    #tip {{ position: absolute; left: 10px; top: 66px; z-index: 8; pointer-events: none; background: rgba(8,13,24,.91); border: 1px solid #2B3752; border-radius: 7px; padding: 6px 8px; font-size: 10px; line-height: 1.45; display: none; max-width: min(360px, calc(100% - 20px)); }}
    #gamma-label {{ position: absolute; right: 52px; top: 66px; z-index: 7; color: #A8B3C7; font-size: 10px; background: rgba(14,21,37,.72); padding: 3px 5px; border-radius: 4px; pointer-events: none; }}
    #chart-error {{ position: absolute; inset: 74px 18px 18px; z-index: 20; display: none; align-items: center; justify-content: center; padding: 18px; color: #FCA5A5; background: rgba(14,21,37,.96); border: 1px solid rgba(251,113,133,.55); border-radius: 8px; text-align: center; font-size: 12px; line-height: 1.5; }}
    .gamma-canvas {{ position: absolute; inset: 0; z-index: 5; pointer-events: none; }}
    @media (max-width: 620px) {{ #head {{ height: 72px; display: block; }} #legend {{ justify-content: flex-start; margin-top: 6px; flex-wrap: nowrap; overflow-x: auto; }} #chart {{ height: calc(100% - 72px); }} #tip, #gamma-label {{ top: 80px; }} #subtitle {{ max-width: 92vw; }} }}
  </style>
</head>
<body>
  <div id="wrap">
    <div id="head"><div><div id="title"></div><div id="subtitle"></div></div><div id="legend"></div></div>
    <div id="chart"></div><div id="tip"></div><div id="gamma-label"></div><div id="chart-error"></div>
  </div>
  <script>
    window.addEventListener('error', (event) => {{
      const panel = document.getElementById('chart-error');
      if (!panel) return;
      panel.textContent = `Chart rendering error: ${{event.message || 'unknown browser error'}}`;
      panel.style.display = 'flex';
    }});
  </script>
  <script>
  (() => {{
    const spec = {payload};
    const host = document.getElementById('chart');
    document.getElementById('title').textContent = spec.title || '';
    document.getElementById('subtitle').textContent = spec.subtitle || '';
    const labels = spec.numericLabels || {{}};
    const keyOf = (time) => {{
      if (typeof time === 'string') return time;
      if (typeof time === 'number') return new Date(time * 1000).toISOString().slice(0, 10);
      if (time && typeof time === 'object') return `${{time.year}}-${{String(time.month).padStart(2,'0')}}-${{String(time.day).padStart(2,'0')}}`;
      return '';
    }};
    const compact = (v) => {{ const a=Math.abs(v); return a>=1000 ? `${{(v/1000).toFixed(1)}}k` : a>=1 ? v.toFixed(2) : v.toFixed(4); }};
    const chart = LightweightCharts.createChart(host, {{
      width: host.clientWidth,
      height: host.clientHeight,
      layout: {{ background: {{ color: '#0E1525' }}, textColor: '#A8B3C7', fontSize: 11 }},
      grid: {{ vertLines: {{ color: 'rgba(139,151,173,.07)' }}, horzLines: {{ color: 'rgba(139,151,173,.10)' }} }},
      crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal, vertLine: {{ color: '#60708F', labelBackgroundColor: '#27344E' }}, horzLine: {{ color: '#60708F', labelBackgroundColor: '#27344E' }} }},
      rightPriceScale: {{ borderColor: '#2B3752', scaleMargins: spec.priceScaleMargins || {{ top: .08, bottom: .08 }} }},
      leftPriceScale: {{ visible: !!spec.leftScale, borderColor: '#2B3752' }},
      timeScale: {{ borderColor: '#2B3752', timeVisible: false, rightOffset: (spec.gammaProfile || []).length ? 18 : 3, barSpacing: 8, minBarSpacing: 2, tickMarkFormatter: (time) => labels[keyOf(time)] || keyOf(time).slice(5) }},
      handleScroll: {{ mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: true }},
      handleScale: {{ axisPressedMouseMove: true, mouseWheel: true, pinch: true }},
      kineticScroll: {{ mouse: true, touch: true }},
    }});
    const apis = [];
    let priceApi = null;
    const legend = document.getElementById('legend');
    const addSeries = (item) => {{
      const raw = item.options || {{}};
      const options = Object.assign({{ color: item.color, title: item.name, lastValueVisible: true, priceLineVisible: false }}, raw);
      if (options.priceFormat && options.priceFormat.formatter === 'compact') options.priceFormat = {{ type: 'custom', formatter: compact }};
      const markers = options.markers || []; delete options.markers;
      const role = options.role; delete options.role;
      const margins = options.scaleMargins; delete options.scaleMargins;
      let api;
      if (item.type === 'candlestick') api = chart.addCandlestickSeries(options);
      else if (item.type === 'histogram') api = chart.addHistogramSeries(Object.assign({{ base: 0 }}, options));
      else if (item.type === 'area') api = chart.addAreaSeries(options);
      else api = chart.addLineSeries(options);
      api.setData(item.data || []);
      if (markers.length) api.setMarkers(markers);
      if (margins && options.priceScaleId) chart.priceScale(options.priceScaleId).applyOptions({{ scaleMargins: margins }});
      if (role === 'price') priceApi = api;
      apis.push({{ api, item }});
      const button = document.createElement('button'); button.className = 'legend';
      button.innerHTML = `<span class="dot" style="background:${{item.color}}"></span>${{item.name}}`;
      let shown = true; button.onclick = () => {{ shown = !shown; api.applyOptions({{ visible: shown }}); button.classList.toggle('off', !shown); drawGamma(); }};
      legend.appendChild(button);
    }};
    (spec.series || []).forEach(addSeries);
    if (spec.priceScaleMargins) chart.priceScale('right').applyOptions({{ scaleMargins: spec.priceScaleMargins }});
    chart.timeScale().fitContent();

    const tip = document.getElementById('tip');
    chart.subscribeCrosshairMove((param) => {{
      if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {{ tip.style.display = 'none'; return; }}
      const rows = [`<strong>${{labels[keyOf(param.time)] || keyOf(param.time)}}</strong>`];
      apis.forEach(({{api,item}}) => {{ const datum = param.seriesData.get(api); if (!datum) return; const value = datum.value ?? datum.close; if (value !== undefined) rows.push(`<span style="color:${{item.color}}">●</span> ${{item.name}}: ${{Number(value).toLocaleString(undefined, {{maximumFractionDigits: 4}})}}`); }});
      tip.innerHTML = rows.join('<br>'); tip.style.display = 'block';
    }});

    const profile = spec.gammaProfile || [];
    const gammaLabel = document.getElementById('gamma-label');
    gammaLabel.textContent = profile.length ? (spec.gammaMode === 'Stacked together' ? 'Gamma: stacked →' : 'Calls ←  |  Puts →') : '';
    gammaLabel.style.display = profile.length ? 'block' : 'none';
    const overlay = document.createElement('canvas'); overlay.className = 'gamma-canvas'; host.appendChild(overlay);
    const drawGamma = () => {{
      if (!profile.length || !priceApi) return;
      const dpr = window.devicePixelRatio || 1, w = host.clientWidth, h = host.clientHeight;
      overlay.width = Math.round(w*dpr); overlay.height = Math.round(h*dpr); overlay.style.width = `${{w}}px`; overlay.style.height = `${{h}}px`;
      const ctx = overlay.getContext('2d'); ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
      const scaleWidth = chart.priceScale('right').width(); const plotRight = w - scaleWidth - 4;
      const profileWidth = Math.min(210, Math.max(100, w * .24)); const stacked = spec.gammaMode === 'Stacked together';
      const maxValue = Math.max(...profile.map(x => stacked ? x.call + x.put : Math.max(x.call, x.put)), 1e-9);
      const coords = profile.map(x => priceApi.priceToCoordinate(x.strike)).filter(x => x !== null && Number.isFinite(x)).sort((a,b)=>a-b);
      let barHeight = 6; if (coords.length > 1) {{ const gaps = coords.slice(1).map((x,i)=>x-coords[i]).filter(x=>x>0); if (gaps.length) barHeight = Math.max(2, Math.min(14, Math.min(...gaps)*.72)); }}
      profile.forEach(row => {{
        const y = priceApi.priceToCoordinate(row.strike); if (y === null || y < 0 || y > h) return;
        if (stacked) {{ const axis = plotRight-profileWidth; const callW=row.call/maxValue*profileWidth, putW=row.put/maxValue*profileWidth; ctx.fillStyle='rgba(54,211,153,.72)'; ctx.fillRect(axis,y-barHeight/2,callW,barHeight); ctx.fillStyle='rgba(251,113,133,.72)'; ctx.fillRect(axis+callW,y-barHeight/2,putW,barHeight); }}
        else {{ const half=profileWidth/2, axis=plotRight-half; const callW=row.call/maxValue*half, putW=row.put/maxValue*half; ctx.fillStyle='rgba(54,211,153,.72)'; ctx.fillRect(axis-callW,y-barHeight/2,callW,barHeight); ctx.fillStyle='rgba(251,113,133,.72)'; ctx.fillRect(axis,y-barHeight/2,putW,barHeight); ctx.strokeStyle='rgba(168,179,199,.30)'; ctx.beginPath(); ctx.moveTo(axis,y-barHeight/2); ctx.lineTo(axis,y+barHeight/2); ctx.stroke(); }}
      }});
    }};
    const redraw = () => requestAnimationFrame(drawGamma);
    chart.timeScale().subscribeVisibleTimeRangeChange(redraw);
    ['wheel','mousemove','touchmove','touchend'].forEach(event => host.addEventListener(event, redraw, {{passive:true}}));
    const ro = new ResizeObserver(() => {{ chart.applyOptions({{ width: host.clientWidth, height: host.clientHeight }}); redraw(); }}); ro.observe(host);
    setTimeout(drawGamma, 80);
  }})();
  </script>
</body>
</html>
"""
