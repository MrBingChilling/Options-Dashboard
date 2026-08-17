from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from datetime import date
from typing import Any

import streamlit as st

from src.daily_ai_summary import DailySummary, load_daily_summaries
from src.storage import SnapshotStore, SnapshotStoreError


SUMMARY_REFRESH_INTERVAL = "15s"
REPORT_DATE_STATE_KEY = "ai_summary_report_date"
LATEST_REPORT_STATE_KEY = "_ai_summary_latest_report"


def sync_report_selection(
    available_dates: Sequence[date],
    state: MutableMapping[str, Any],
    latest_report_token: Any | None = None,
) -> date:
    """Select newly arrived or rewritten data without disrupting history browsing."""
    if not available_dates:
        raise ValueError("At least one report date is required.")

    latest_date = available_dates[0]
    latest_marker = latest_report_token if latest_report_token is not None else latest_date
    selected_date = state.get(REPORT_DATE_STATE_KEY)
    latest_seen = state.get(LATEST_REPORT_STATE_KEY)
    if latest_seen != latest_marker or selected_date not in available_dates:
        selected_date = latest_date
        state[REPORT_DATE_STATE_KEY] = selected_date
    state[LATEST_REPORT_STATE_KEY] = latest_marker
    return selected_date


@st.fragment(run_every=SUMMARY_REFRESH_INTERVAL)
def render_ai_summary_dashboard(store: SnapshotStore) -> None:
    """Render saved daily market insights inside the IV & Skew page."""
    st.markdown(
        """
        <style>
          .summary-meta {
            color:#A8B3C7; font-size:.92rem; line-height:1.55;
            padding:.7rem .85rem; margin:.2rem 0 1.15rem;
            border:1px solid #25304A; border-radius:.75rem; background:#11192A;
          }
          .summary-bottom {
            padding:.9rem 1rem; margin-top:1rem; border-radius:.8rem;
            border-left:4px solid #69A9F8; background:#141B2D;
          }
          @media (max-width:700px) {
            .summary-meta {font-size:.84rem; padding:.62rem .7rem;}
            .summary-bottom {padding:.75rem .8rem;}
            ul {padding-left:1.25rem !important;}
            li {margin-bottom:.8rem !important; line-height:1.48 !important;}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Daily AI Summary")
    st.caption(
        "Fresh ChatGPT-written analysis of the newest complete 49-ticker daily snapshot. "
        "The scheduled analysis is saved after the morning collector and uses 0 additional MarketData credits. "
        "This view checks for a new saved report every 15 seconds."
    )
    st.button(
        "Refresh now",
        key="ai_summary_refresh_now",
        help="Fetch the newest saved report immediately.",
    )

    if not store.enabled:
        st.info("Configure Supabase to load saved daily summaries.")
        return

    try:
        reports = load_daily_summaries(store)
    except SnapshotStoreError as exc:
        st.error(str(exc))
        return

    if not reports:
        st.info(
            "No completed summary is stored yet. The first one will appear automatically "
            "after two complete daily sessions are available."
        )
        return

    report_by_date: dict[date, DailySummary] = {
        report.snapshot_date: report for report in reports
    }
    available_dates = list(report_by_date)
    latest_report = report_by_date[available_dates[0]]
    latest_report_token = (
        latest_report.snapshot_date,
        latest_report.generated_at,
        latest_report.generator_version,
    )
    selected_date = sync_report_selection(
        available_dates,
        st.session_state,
        latest_report_token,
    )
    if len(available_dates) > 1:
        selected_date = st.selectbox(
            "Report date",
            available_dates,
            key=REPORT_DATE_STATE_KEY,
            format_func=lambda value: value.strftime("%b %d, %Y"),
            help=(
                "A newly saved report is selected automatically. You can still choose an "
                "older report until the next daily report arrives."
            ),
        )
    report = report_by_date[selected_date]
    comparison_parts = [f"{report.comparison_date:%b %d} (1D)"]
    if report.week_comparison_date:
        comparison_parts.append(f"{report.week_comparison_date:%b %d} (1W)")
    if report.month_comparison_date:
        comparison_parts.append(f"{report.month_comparison_date:%b %d} (1M)")
    comparison_text = ", ".join(comparison_parts)

    st.markdown(
        f"""
        <div class="summary-meta">
          <b>{report.snapshot_date:%B %d, %Y}</b> · options data through that market close<br>
          Compared with {comparison_text} ·
          coverage {report.symbol_count}/{report.expected_symbol_count} tickers · 1W + 1M
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What changed")
    for bullet in report.bullets:
        st.markdown(f"- **{bullet.title}** {bullet.body}")

    st.markdown(
        f'<div class="summary-bottom"><b>Bottom line:</b> {report.bottom_line}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to read this summary"):
        st.markdown(
            """
- **25Δ skew** means `25Δ call IV − 25Δ put IV`. Positive values price calls richer relative to puts; negative values price puts richer relative to calls.
- **1D/1W/1M comparisons** use the nearest fully complete saved sessions to those horizons, never a partially collected basket.
- **Recent-range percentiles** use up to 60 calendar days of saved history and help distinguish a one-day move from an unusually stretched regime.
- **Basket statistics** state whether they use the equal-weight mean or the 10% trimmed mean. The trimmed version removes `floor(10% × N)` observations from each tail.
- **1W and 1M are constant-tenor targets.** The selected expiration can roll as the calendar advances, especially for indexes with daily expirations.
- **This is a pricing summary, not observed order flow.** Open interest and IV do not prove who bought or sold an option, and the report does not use news or an event calendar.
- Generating, opening and changing report dates read saved Supabase data only and consume **0 MarketData credits**.
            """
        )
