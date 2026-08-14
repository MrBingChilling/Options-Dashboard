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


def test_summary_uses_latest_two_fully_complete_sessions():
    summary = build_daily_summary(_history(), AUTO_SYMBOLS)

    assert summary.snapshot_date == date(2026, 8, 13)
    assert summary.comparison_date == date(2026, 8, 12)
    assert summary.week_comparison_date == date(2026, 8, 6)
    assert summary.month_comparison_date == date(2026, 7, 15)
    assert summary.symbol_count == summary.expected_symbol_count == 49
    assert "multi-week regime" in summary.bullets[0].title
    assert any("index" in bullet.title.lower() for bullet in summary.bullets)
    assert any("WOLF" in bullet.title for bullet in summary.bullets)
    assert "49" not in summary.bottom_line  # Coverage belongs in page metadata, not prose.
    assert len(summary.bullets) <= 9
    assert len(" ".join(f"{item.title} {item.body}" for item in summary.bullets).split()) < 650


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
    assert reports[0].week_comparison_date is None  # v1 records remain readable.
