from __future__ import annotations

import html
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from src.market_calendar import MarketEvent, upcoming_opex_events


EASTERN = ZoneInfo("America/New_York")


def _event_row(event: MarketEvent, is_next: bool) -> str:
    stamp = event.event_date
    note = f'<span class="calendar-note">{html.escape(event.note)}</span>' if event.note else ""
    next_class = " calendar-row-next" if is_next else ""
    return f"""
      <div class="calendar-row{next_class}">
        <div class="calendar-date-box">
          <span class="calendar-month">{stamp.strftime('%b').upper()}</span>
          <span class="calendar-day">{stamp.day}</span>
        </div>
        <div class="calendar-copy">
          <div class="calendar-title">{html.escape(event.title)}</div>
          <div class="calendar-meta">{stamp.strftime('%A')} · {stamp.isoformat()}</div>
          {note}
        </div>
        <span class="calendar-badge">OPEX</span>
      </div>
    """


def render_calendar_dashboard() -> None:
    today = datetime.now(EASTERN).date()
    events = upcoming_opex_events(today, years_ahead=2)

    st.markdown(
        """
        <style>
          .calendar-next {
            background:linear-gradient(135deg,rgba(60,130,246,.20),rgba(20,29,49,.92));
            border:1px solid rgba(96,165,250,.55); border-radius:14px;
            padding:14px 16px; margin:4px 0 18px;
          }
          .calendar-next-label {color:#93C5FD;font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;}
          .calendar-next-date {color:#F8FAFC;font-size:25px;font-weight:750;margin-top:2px;}
          .calendar-next-meta {color:#B8C3D7;font-size:13px;margin-top:2px;}
          .calendar-year {color:#E5EAF3;font-size:18px;font-weight:700;margin:18px 0 8px;}
          .calendar-list {display:flex;flex-direction:column;gap:7px;}
          .calendar-row {
            display:flex;align-items:center;gap:12px;background:#11192A;
            border:1px solid #26324A;border-radius:11px;padding:9px 11px;min-height:64px;
          }
          .calendar-row-next {border-color:rgba(96,165,250,.55);background:#13203A;}
          .calendar-date-box {width:48px;flex:0 0 48px;text-align:center;border-right:1px solid #33415C;padding-right:10px;}
          .calendar-month {display:block;color:#93C5FD;font-size:10px;font-weight:800;letter-spacing:.08em;}
          .calendar-day {display:block;color:#F8FAFC;font-size:22px;font-weight:750;line-height:1.05;}
          .calendar-copy {min-width:0;flex:1;}
          .calendar-title {color:#F1F5F9;font-size:14px;font-weight:650;}
          .calendar-meta {color:#9CAAC0;font-size:12px;margin-top:2px;}
          .calendar-note {display:block;color:#FBBF24;font-size:11px;margin-top:3px;}
          .calendar-badge {color:#BFDBFE;background:rgba(59,130,246,.15);border:1px solid rgba(96,165,250,.35);border-radius:999px;padding:3px 7px;font-size:10px;font-weight:800;}
          @media (max-width:700px) {
            .calendar-next-date {font-size:21px;}
            .calendar-row {gap:9px;padding:8px 9px;}
            .calendar-badge {font-size:9px;padding:3px 6px;}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Market Calendar")
    st.caption("Upcoming standard monthly U.S. options expirations. Dates are shown in New York market time and holiday-adjusted.")

    if not events:
        st.info("No upcoming calendar events are configured.")
        return

    next_event = events[0]
    days_away = (next_event.event_date - today).days
    timing = "Today" if days_away == 0 else f"In {days_away} day{'s' if days_away != 1 else ''}"
    st.markdown(
        f"""
        <div class="calendar-next">
          <div class="calendar-next-label">Next OPEX</div>
          <div class="calendar-next-date">{next_event.event_date.strftime('%B')} {next_event.event_date.day}, {next_event.event_date.year}</div>
          <div class="calendar-next-meta">{next_event.event_date.strftime('%A')} · {timing}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    years = sorted({event.event_date.year for event in events})
    selected_years = st.multiselect(
        "Years",
        years,
        default=years,
        key="market_calendar_years",
        help="Use this to condense the list. All upcoming dates through the same month two years ahead are included by default.",
    )
    visible = [event for event in events if event.event_date.year in selected_years]
    if not visible:
        st.info("Select at least one year.")
        return

    for year in selected_years:
        year_events = [event for event in visible if event.event_date.year == year]
        if not year_events:
            continue
        rows = "".join(_event_row(event, event == next_event) for event in year_events)
        st.markdown(
            f'<div class="calendar-year">{year}</div><div class="calendar-list">{rows}</div>',
            unsafe_allow_html=True,
        )

    st.caption("Calendar display only. It does not call MarketData or consume API credits.")
