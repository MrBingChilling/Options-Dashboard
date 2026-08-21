from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from src.config import get_setting
from src.market_calendar import upcoming_opex_events
from src.storage import SnapshotStore


EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class CalendarDisplayEvent:
    event_date: date
    title: str
    note: str
    badge: str
    source_url: str = ""


def _earnings_events(today: date) -> tuple[list[CalendarDisplayEvent], str | None]:
    store = SnapshotStore(
        get_setting("SUPABASE_URL", ""),
        get_setting("SUPABASE_SECRET_KEY", get_setting("SUPABASE_SERVICE_ROLE_KEY", "")),
    )
    if not store.enabled:
        return [], "Configure Supabase to load confirmed earnings dates."

    try:
        response = requests.get(
            f"{store.url}/rest/v1/earnings_calendar",
            params={
                "select": (
                    "symbol,company_name,earnings_date,session,fiscal_period,"
                    "source_url,source_title,last_verified_at"
                ),
                "earnings_date": f"gte.{today.isoformat()}",
                "confirmed": "eq.true",
                "order": "earnings_date.asc,symbol.asc",
                "limit": "500",
            },
            headers=store.headers,
            timeout=store.timeout,
        )
    except requests.RequestException as exc:
        return [], f"Confirmed earnings dates could not be loaded: {exc}"
    if response.status_code != 200:
        return [], f"Confirmed earnings dates could not be loaded ({response.status_code})."

    events: list[CalendarDisplayEvent] = []
    for row in response.json():
        try:
            event_date = date.fromisoformat(str(row.get("earnings_date", "")))
        except ValueError:
            continue
        symbol = str(row.get("symbol", "")).upper().strip()
        company = str(row.get("company_name", "")).strip()
        session = str(row.get("session", "Time not specified")).strip()
        fiscal_period = str(row.get("fiscal_period", "")).strip()
        note_parts = [part for part in (fiscal_period, session) if part]
        title = f"{symbol} · {company}" if company else symbol
        events.append(
            CalendarDisplayEvent(
                event_date=event_date,
                title=title,
                note=" · ".join(note_parts),
                badge="EARNINGS",
                source_url=str(row.get("source_url", "")).strip(),
            )
        )
    return events, None


def _event_row(event: CalendarDisplayEvent, is_next: bool) -> str:
    stamp = event.event_date
    note = f'<span class="calendar-note">{html.escape(event.note)}</span>' if event.note else ""
    next_class = " calendar-row-next" if is_next else ""
    badge_class = " calendar-badge-earnings" if event.badge == "EARNINGS" else ""
    safe_title = html.escape(event.title)
    if event.source_url:
        safe_url = html.escape(event.source_url, quote=True)
        title = f'<a class="calendar-source" href="{safe_url}" target="_blank">{safe_title}</a>'
    else:
        title = safe_title
    return (
        f'<div class="calendar-row{next_class}">'
        '<div class="calendar-date-box">'
        f'<span class="calendar-month">{stamp.strftime("%b").upper()}</span>'
        f'<span class="calendar-day">{stamp.day}</span>'
        '</div>'
        '<div class="calendar-copy">'
        f'<div class="calendar-title">{title}</div>'
        f'<div class="calendar-meta">{stamp.strftime("%A")} · {stamp.isoformat()}</div>'
        f'{note}'
        '</div>'
        f'<span class="calendar-badge{badge_class}">{event.badge}</span>'
        '</div>'
    )


def _next_card(label: str, event: CalendarDisplayEvent, today: date) -> str:
    days_away = (event.event_date - today).days
    timing = "Today" if days_away == 0 else f"In {days_away} day{'s' if days_away != 1 else ''}"
    return (
        '<div class="calendar-next">'
        f'<div class="calendar-next-label">{html.escape(label)}</div>'
        f'<div class="calendar-next-date">{event.event_date.strftime("%B")} '
        f'{event.event_date.day}, {event.event_date.year}</div>'
        f'<div class="calendar-next-meta">{html.escape(event.title)} · '
        f'{event.event_date.strftime("%A")} · {timing}</div>'
        '</div>'
    )


def render_calendar_dashboard() -> None:
    today = datetime.now(EASTERN).date()
    opex_events = [
        CalendarDisplayEvent(
            event_date=event.event_date,
            title=event.title,
            note=event.note,
            badge="OPEX",
        )
        for event in upcoming_opex_events(today, years_ahead=2)
    ]
    earnings_events, earnings_error = _earnings_events(today)
    events = sorted(opex_events + earnings_events, key=lambda event: (event.event_date, event.badge, event.title))

    st.markdown(
        """
        <style>
          .calendar-top-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:4px 0 18px;}
          .calendar-next {
            background:linear-gradient(135deg,rgba(60,130,246,.20),rgba(20,29,49,.92));
            border:1px solid rgba(96,165,250,.55);border-radius:14px;padding:14px 16px;
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
          .calendar-source {color:#F1F5F9;text-decoration:none;}
          .calendar-source:hover {color:#93C5FD;text-decoration:underline;}
          .calendar-meta {color:#9CAAC0;font-size:12px;margin-top:2px;}
          .calendar-note {display:block;color:#FBBF24;font-size:11px;margin-top:3px;}
          .calendar-badge {color:#BFDBFE;background:rgba(59,130,246,.15);border:1px solid rgba(96,165,250,.35);border-radius:999px;padding:3px 7px;font-size:10px;font-weight:800;}
          .calendar-badge-earnings {color:#BBF7D0;background:rgba(34,197,94,.13);border-color:rgba(74,222,128,.35);}
          @media (max-width:700px) {
            .calendar-top-grid {grid-template-columns:1fr;}
            .calendar-next-date {font-size:21px;}
            .calendar-row {gap:9px;padding:8px 9px;}
            .calendar-badge {font-size:9px;padding:3px 6px;}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Market Calendar")
    st.caption(
        "Upcoming standard monthly U.S. options expirations and confirmed earnings dates for individual stocks in the dashboard watchlist. "
        "Dates are shown in New York market time; estimated or unconfirmed earnings dates are excluded."
    )

    if earnings_error:
        st.warning(earnings_error)
    elif earnings_events:
        st.success(
            f"{len(earnings_events)} confirmed upcoming watchlist earnings "
            f"date{'s' if len(earnings_events) != 1 else ''} loaded."
        )
    else:
        st.info("No confirmed upcoming watchlist earnings dates are currently stored.")
    if not events:
        st.info("No upcoming calendar events are configured.")
        return

    cards: list[str] = []
    if opex_events:
        cards.append(_next_card("Next OPEX", opex_events[0], today))
    if earnings_events:
        cards.append(_next_card("Next confirmed earnings", earnings_events[0], today))
    if cards:
        st.markdown(f'<div class="calendar-top-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    years = sorted({event.event_date.year for event in events})
    controls = st.columns(2)
    selected_types = controls[0].multiselect(
        "Event type",
        ["EARNINGS", "OPEX"],
        default=["EARNINGS", "OPEX"],
        key="market_calendar_types",
    )
    selected_years = controls[1].multiselect(
        "Years",
        years,
        default=years,
        key="market_calendar_years",
        help="Use this to condense the list. All upcoming dates through the same month two years ahead are included by default.",
    )
    visible = [
        event for event in events
        if event.event_date.year in selected_years and event.badge in selected_types
    ]
    if not visible:
        st.info("Select at least one event type and year.")
        return

    next_visible = visible[0]
    for year in selected_years:
        year_events = [event for event in visible if event.event_date.year == year]
        if not year_events:
            continue
        rows = "".join(_event_row(event, event == next_visible) for event in year_events)
        st.markdown(
            f'<div class="calendar-year">{year}</div><div class="calendar-list">{rows}</div>',
            unsafe_allow_html=True,
        )

    st.caption("Calendar display only. It does not call MarketData or consume API credits. Earnings titles link to the confirming source.")
