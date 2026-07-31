import pandas as pd

from src.charts import _chart_document, price_gamma_chart


def test_price_gamma_chart_contains_candles_history_and_profile() -> None:
    candles = pd.DataFrame(
        {
            "time": pd.to_datetime(["2026-07-29", "2026-07-30"]),
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000, 1100],
        }
    )
    history = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2026-07-29", "2026-07-30"]),
            "gamma_flip": [99.0, 100.0],
            "call_wall": [105.0, 106.0],
            "put_wall": [95.0, 96.0],
            "net_gex": [1_000_000.0, 2_000_000.0],
            "put_call_oi_ratio": [1.1, 1.2],
        }
    )
    profile = pd.DataFrame(
        {
            "strike": [95.0, 100.0, 105.0],
            "call_gex": [1.0, 2.0, 3.0],
            "put_gex": [-3.0, -2.0, -1.0],
        }
    )
    spec = price_gamma_chart(candles, history, profile=profile)
    assert spec["series"][0]["type"] == "candlestick"
    assert len(spec["gammaProfile"]) == 3
    assert {series["name"] for series in spec["series"]} >= {
        "Price",
        "Gamma flip",
        "Call wall",
        "Put wall",
        "Net GEX ($mm)",
        "Put/call OI",
    }


def test_chart_document_uses_lightweight_charts_and_touch_controls() -> None:
    document = _chart_document('{"title":"Test","series":[]}')
    assert "app/static/lightweight-charts.standalone.production.js" in document
    assert "horzTouchDrag: true" in document
    assert "pinch: true" in document
