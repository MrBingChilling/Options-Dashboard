from __future__ import annotations

from datetime import date

import streamlit as st

from src.daily_ai_summary import DailySummary, load_daily_summaries
from src.storage import SnapshotStore, SnapshotStoreError


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
        "Automatically generated from the newest complete 49-ticker daily snapshot. "
        "The summary is saved after the morning collector and uses 0 additional MarketData credits."
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
    if len(available_dates) > 1:
        selected_date = st.selectbox(
            "Report date",
            available_dates,
            index=0,
            format_func=lambda value: value.strftime("%b %d, %Y"),
            help="The newest saved report is selected by default.",
        )
    else:
        selected_date = available_dates[0]
    report = report_by_date[selected_date]

    st.markdown(
        f"""
        <div class="summary-meta">
          <b>{report.snapshot_date:%B %d, %Y}</b> · options data through that market close<br>
          Compared with {report.comparison_date:%B %d, %Y} ·
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
- **Daily comparison** uses the immediately preceding fully complete saved session, never a partially collected basket.
- **Basket statistics** state whether they use the equal-weight mean or the 10% trimmed mean. The trimmed version removes `floor(10% × N)` observations from each tail.
- **1W and 1M are constant-tenor targets.** The selected expiration can roll as the calendar advances, especially for indexes with daily expirations.
- **This is a pricing summary, not observed order flow.** Open interest and IV do not prove who bought or sold an option, and the report does not use news or an event calendar.
- Generating, opening and changing report dates read saved Supabase data only and consume **0 MarketData credits**.
            """
        )
