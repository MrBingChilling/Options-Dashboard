from __future__ import annotations

from datetime import date

import pandas as pd

from src.daily_ai_summary import (
    DailySummary,
    SummaryBullet,
    build_daily_summary,
    complete_summary_sessions,
    load_daily_summaries,
    save_daily_summary,
)
from src.skew_collector import AUTO_SYMBOLS
from src.storage import SnapshotStore


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sessions = ["2026-08-12", "2026-08-13"]
    for session_index, session in enumerate(sessions):
        for tenor in ("1W", "1M"):
            for symbol in AUTO_SYMBOLS:
                spot = 100.0 * (1.01 if session_index else 1.0)
                atm = 0.50 - 0.02 * session_index
                put_iv = 0.52
                call_iv = 0.53
                skew = call_iv - put_iv

                if symbol in {"SPY", "QQQ"}:
                    spot = 100.0 * (1.01 if session_index else 1.0)
                    put_iv = 0.195 + 0.005 * session_index
                    call_iv = 0.190 - 0.005 * session_index
                    skew = call_iv - put_iv
                    atm = 0.19 - 0.005 * session_index

                if symbol == "WOLF" and tenor == "1W":
                    spot = 100.0 * (0.995 if session_index else 1.0)
                    put_iv = 1.55 + 0.15 * session_index
                    call_iv = 1.57 + 0.07 * session_index
                    skew = call_iv - put_iv
                    atm = 1.55

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


def test_summary_uses_latest_two_fully_complete_sessions():
    summary = build_daily_summary(_history(), AUTO_SYMBOLS)

    assert summary.snapshot_date == date(2026, 8, 13)
    assert summary.comparison_date == date(2026, 8, 12)
    assert summary.symbol_count == summary.expected_symbol_count == 49
    assert "Indexes rose" in summary.bullets[0].title
    assert any("WOLF" in bullet.title for bullet in summary.bullets)
    assert "49" not in summary.bottom_line  # Coverage belongs in page metadata, not prose.


def test_session_is_not_complete_when_one_required_tenor_is_missing():
    history = _history()
    missing = history[
        ~(
            (history["snapshot_date"] == "2026-08-13")
            & (history["symbol"] == "SPY")
            & (history["tenor"] == "1M")
        )
    ]

    assert complete_summary_sessions(missing, AUTO_SYMBOLS) == [date(2026, 8, 12)]


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
    )

    save_daily_summary(store, summary)

    assert stored["snapshot_date"] == "2026-08-13"
    assert stored["summary"]["bullets"] == [{"title": "Title", "body": "Body"}]


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
