from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from src.daily_ai_analysis import build_analysis_packet, complete_summary_sessions
from src.daily_ai_model import SummaryGenerationError, generate_daily_summary
from src.daily_ai_summary import (
    DailySummary,
    SummaryBullet,
    load_daily_summaries,
    save_daily_summary,
)
from src.skew_collector import AUTO_SYMBOLS
from src.storage import SnapshotStore


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sessions = ["2026-07-15", "2026-08-06", "2026-08-12", "2026-08-13"]
    broad_atm = [0.72, 0.62, 0.54, 0.50]
    for session_index, session in enumerate(sessions):
        for tenor in ("1W", "1M"):
            for symbol in AUTO_SYMBOLS:
                spot = 100.0 * (1.0 + 0.005 * session_index)
                atm = broad_atm[session_index]
                put_iv = 0.52
                call_iv = 0.53
                skew = call_iv - put_iv

                if symbol in {"SPY", "QQQ"}:
                    index_skew = [-0.030, -0.015, -0.005, -0.015][session_index]
                    spot = [100.0, 102.0, 103.0, 104.0][session_index]
                    put_iv = 0.20
                    call_iv = put_iv + index_skew
                    skew = index_skew
                    atm = [0.24, 0.21, 0.19, 0.18][session_index]

                if symbol == "WOLF" and tenor == "1W":
                    wolf_skew = [0.020, 0.040, 0.020, -0.060][session_index]
                    spot = [100.0, 110.0, 114.0, 113.0][session_index]
                    put_iv = 1.60
                    call_iv = put_iv + wolf_skew
                    skew = wolf_skew
                    atm = [1.30, 1.40, 1.50, 1.60][session_index]

                rows.append(
                    {
                        "symbol": symbol,
                        "snapshot_date": session,
                        "tenor": tenor,
                        "spot": spot,
                        "atm_iv": atm,
                        "call_25d_iv": call_iv,
                        "put_25d_iv": put_iv,
                        "skew_25d": skew,
                    }
                )
    return pd.DataFrame(rows)


def _model_output() -> dict[str, object]:
    return {
        "bullets": [
            {
                "title": "Compression remains broad, but its pace is changing.",
                "body": "The cross-section shows a persistent fall in short-dated ATM IV across the monthly horizon, while today's smaller move says the regime is maturing rather than newly accelerating.",
            },
            {
                "title": "Index skew diverged from the single-name backdrop.",
                "body": "SPY and QQQ repriced short-dated downside relative to calls even as the broader basket remained calmer, making this a cross-asset relative-pricing change rather than a uniform volatility shock.",
            },
            {
                "title": "WOLF is the concentrated exception.",
                "body": "Its one-week skew changed much more sharply than the basket measures, and the gap between equal-weight and trimmed aggregates confirms that the move should not be generalized to all AI infrastructure names.",
            },
            {
                "title": "Tenor confirmation is selective, not universal.",
                "body": "Where one-week and one-month surfaces agree, the signal deserves more weight; elsewhere the mixed tenor evidence argues for treating the latest move as localized rather than a durable new regime.",
            },
        ],
        "bottom_line": "Broad volatility compression remains the governing backdrop, but the useful new information is the divergence between firmer index downside pricing and a small number of concentrated single-name exceptions.",
    }


def _openai_response(output: dict[str, object]):
    class Response:
        status_code = 200
        text = ""
        headers: dict[str, str] = {}

        @staticmethod
        def json():
            return {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(output)}
                        ],
                    }
                ],
            }

    return Response()


def test_analysis_packet_contains_full_cross_section_and_non_template_context():
    older = DailySummary(
        snapshot_date=date(2026, 8, 12),
        comparison_date=date(2026, 8, 6),
        symbol_count=49,
        expected_symbol_count=49,
        bullets=(SummaryBullet("Earlier observation", "Use this only as context."),),
        bottom_line="Earlier conclusion.",
    )
    same_day = DailySummary(
        snapshot_date=date(2026, 8, 13),
        comparison_date=date(2026, 8, 12),
        symbol_count=49,
        expected_symbol_count=49,
        bullets=(SummaryBullet("Stale same-day prose", "Exclude this from the prompt."),),
        bottom_line="Stale conclusion.",
    )

    packet = build_analysis_packet(_history(), AUTO_SYMBOLS, [same_day, older])

    assert packet.snapshot_date == date(2026, 8, 13)
    assert packet.comparison_date == date(2026, 8, 12)
    assert packet.week_comparison_date == date(2026, 8, 6)
    assert packet.month_comparison_date == date(2026, 7, 15)
    assert packet.symbol_count == packet.expected_symbol_count == 49
    assert len(packet.input_signature) == 64
    assert packet.payload["session"]["input_signature"] == packet.input_signature
    assert len(packet.payload["symbols"]) == 49
    wolf = next(row for row in packet.payload["symbols"] if row["symbol"] == "WOLF")
    assert set(wolf["tenors"]) == {"1W", "1M"}
    assert set(wolf["spot_returns_pct"]) == {"1D", "1W", "1M"}
    assert set(wolf["tenors"]["1W"]["skew_25d"]["changes_vol_points"]) == {
        "1D",
        "1W",
        "1M",
    }
    dashboard_atm = packet.payload["baskets"]["Dashboard ex-index"]["tenors"]["1W"]["atm_iv"]
    assert set(dashboard_atm["horizons"]) == {"1D", "1W", "1M"}
    assert "current_equal" in dashboard_atm["horizons"]["1D"]
    assert "current_trimmed_10pct" in dashboard_atm["horizons"]["1D"]
    assert "breadth_lower_pct" in dashboard_atm["horizons"]["1D"]
    assert set(packet.payload["baskets"]["Dashboard ex-index"]["spot_returns_pct"]) == {
        "1D",
        "1W",
        "1M",
    }
    assert packet.payload["prior_reports_for_continuity_only"] == [
        {
            "snapshot_date": "2026-08-12",
            "bullets": [
                {"title": "Earlier observation", "body": "Use this only as context."}
            ],
            "bottom_line": "Earlier conclusion.",
        }
    ]


def test_generation_calls_model_for_fresh_structured_analysis(monkeypatch):
    captured: dict[str, object] = {}

    def fake_post(url, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _openai_response(_model_output())

    monkeypatch.setattr("src.daily_ai_model.requests.post", fake_post)

    summary = generate_daily_summary(
        _history(),
        AUTO_SYMBOLS,
        api_key="sk-test",
        model="gpt-5.6",
    )

    assert summary.snapshot_date == date(2026, 8, 13)
    assert summary.comparison_date == date(2026, 8, 12)
    assert summary.generator_version == "daily_ai_summary_llm_v1:gpt-5.6"
    assert summary.input_signature
    assert [bullet.title for bullet in summary.bullets] == [
        item["title"] for item in _model_output()["bullets"]
    ]
    assert summary.bottom_line == _model_output()["bottom_line"]
    request_body = captured["json"]
    assert captured["url"] == "https://api.openai.com/v1/responses"
    assert request_body["model"] == "gpt-5.6"
    assert request_body["reasoning"] == {"effort": "high"}
    assert request_body["max_output_tokens"] == 8000
    assert request_body["store"] is False
    assert request_body["text"]["format"]["strict"] is True
    assert request_body["text"]["format"]["schema"]["additionalProperties"] is False
    assert "Do not fit the facts" in request_body["input"][0]["content"]
    assert "WOLF" in request_body["input"][1]["content"]
    assert "2026-08-13" in request_body["input"][1]["content"]


def test_missing_api_key_fails_without_a_template_fallback(monkeypatch):
    def unexpected_post(*args, **kwargs):
        raise AssertionError("The API must not be called without a key.")

    monkeypatch.setattr("src.daily_ai_model.requests.post", unexpected_post)

    with pytest.raises(SummaryGenerationError, match="no deterministic prose fallback"):
        generate_daily_summary(_history(), AUTO_SYMBOLS, api_key="")


def test_reused_prior_headline_is_rejected_instead_of_saved(monkeypatch):
    prior = DailySummary(
        snapshot_date=date(2026, 8, 12),
        comparison_date=date(2026, 8, 6),
        symbol_count=49,
        expected_symbol_count=49,
        bullets=(
            SummaryBullet(
                "Compression remains broad, but its pace is changing.",
                "Older body.",
            ),
        ),
        bottom_line="Older conclusion.",
    )

    monkeypatch.setattr(
        "src.daily_ai_model.requests.post",
        lambda *args, **kwargs: _openai_response(_model_output()),
    )
    monkeypatch.setattr("src.daily_ai_model.time.sleep", lambda *_: None)

    with pytest.raises(SummaryGenerationError, match="reused a prior headline"):
        generate_daily_summary(
            _history(),
            AUTO_SYMBOLS,
            api_key="sk-test",
            prior_summaries=[prior],
        )


def test_session_is_not_complete_when_one_required_tenor_is_missing():
    history = _history()
    missing = history[
        ~(
            (history["snapshot_date"] == "2026-08-13")
            & (history["symbol"] == "SPY")
            & (history["tenor"] == "1M")
        )
    ]

    assert complete_summary_sessions(missing, AUTO_SYMBOLS) == [
        date(2026, 7, 15),
        date(2026, 8, 6),
        date(2026, 8, 12),
    ]


def test_daily_summary_round_trips_through_supabase_payload(monkeypatch):
    stored: dict[str, object] = {}

    class Response:
        status_code = 201
        text = ""

    def fake_post(url, params, headers, json, timeout):
        stored.update(json)
        assert params == {"on_conflict": "snapshot_date"}
        assert "resolution=merge-duplicates" in headers["Prefer"]
        return Response()

    monkeypatch.setattr("src.daily_ai_summary.requests.post", fake_post)
    store = SnapshotStore("https://example.supabase.co", "sb_secret_test")
    summary = DailySummary(
        snapshot_date=date(2026, 8, 13),
        comparison_date=date(2026, 8, 12),
        symbol_count=49,
        expected_symbol_count=49,
        bullets=(SummaryBullet("Title", "Body"),),
        bottom_line="Bottom line.",
        week_comparison_date=date(2026, 8, 6),
        month_comparison_date=date(2026, 7, 15),
    )

    save_daily_summary(store, summary)

    assert stored["snapshot_date"] == "2026-08-13"
    assert stored["summary"]["bullets"] == [{"title": "Title", "body": "Body"}]
    assert stored["summary"]["comparison_dates"] == {
        "1D": "2026-08-12",
        "1W": "2026-08-06",
        "1M": "2026-07-15",
    }
    assert "Fresh model-written analysis" in stored["summary"]["data_note"]
    assert "input_signature" in stored["summary"]


def test_summary_history_loads_newest_first(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return [
                {
                    "snapshot_date": "2026-08-13",
                    "comparison_date": "2026-08-12",
                    "symbol_count": 49,
                    "expected_symbol_count": 49,
                    "generator_version": "daily_ai_summary_v1",
                    "summary": {
                        "bullets": [{"title": "Title", "body": "Body"}],
                        "bottom_line": "Bottom line.",
                    },
                }
            ]

    def fake_get(url, params, headers, timeout):
        assert params["order"] == "snapshot_date.desc"
        return Response()

    monkeypatch.setattr("src.daily_ai_summary.requests.get", fake_get)
    store = SnapshotStore("https://example.supabase.co", "sb_secret_test")

    reports = load_daily_summaries(store)

    assert len(reports) == 1
    assert reports[0].snapshot_date == date(2026, 8, 13)
    assert reports[0].bullets[0].body == "Body"
    assert reports[0].week_comparison_date is None
