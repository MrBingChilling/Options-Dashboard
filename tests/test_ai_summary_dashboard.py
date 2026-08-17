from __future__ import annotations

from datetime import date

from src.ai_summary_dashboard import (
    LATEST_REPORT_STATE_KEY,
    REPORT_DATE_STATE_KEY,
    sync_report_selection,
)


def test_new_report_is_selected_when_it_first_appears():
    older = date(2026, 8, 13)
    latest = date(2026, 8, 14)
    state = {
        REPORT_DATE_STATE_KEY: older,
        LATEST_REPORT_STATE_KEY: older,
    }

    selected = sync_report_selection([latest, older], state)

    assert selected == latest
    assert state[REPORT_DATE_STATE_KEY] == latest
    assert state[LATEST_REPORT_STATE_KEY] == latest


def test_manual_history_selection_survives_periodic_refresh():
    older = date(2026, 8, 13)
    latest = date(2026, 8, 14)
    state = {
        REPORT_DATE_STATE_KEY: older,
        LATEST_REPORT_STATE_KEY: latest,
    }

    selected = sync_report_selection([latest, older], state)

    assert selected == older
    assert state[REPORT_DATE_STATE_KEY] == older


def test_removed_report_selection_falls_back_to_latest():
    unavailable = date(2026, 8, 12)
    latest = date(2026, 8, 14)
    state = {
        REPORT_DATE_STATE_KEY: unavailable,
        LATEST_REPORT_STATE_KEY: latest,
    }

    selected = sync_report_selection([latest], state)

    assert selected == latest
    assert state[REPORT_DATE_STATE_KEY] == latest


def test_rewritten_latest_report_is_selected_by_freshness_token():
    older = date(2026, 8, 13)
    latest = date(2026, 8, 14)
    previous_token = (latest, "old-write", "daily_ai_summary_v2")
    current_token = (latest, "new-write", "chatgpt_scheduled_analysis_v1")
    state = {
        REPORT_DATE_STATE_KEY: older,
        LATEST_REPORT_STATE_KEY: previous_token,
    }

    selected = sync_report_selection([latest, older], state, current_token)

    assert selected == latest
    assert state[REPORT_DATE_STATE_KEY] == latest
    assert state[LATEST_REPORT_STATE_KEY] == current_token
