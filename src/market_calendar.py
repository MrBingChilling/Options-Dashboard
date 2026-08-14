from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class MarketEvent:
    event_date: date
    title: str
    event_type: str
    note: str = ""


def _third_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    first_friday = first + timedelta(days=(4 - first.weekday()) % 7)
    return first_friday + timedelta(days=14)


def _easter_sunday(year: int) -> date:
    """Return Gregorian Easter using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = (h + ell - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _observed_juneteenth(year: int) -> date:
    holiday = date(year, 6, 19)
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def monthly_opex_date(year: int, month: int) -> date:
    """Return standard monthly U.S. options expiration, holiday-adjusted."""
    expiration = _third_friday(year, month)
    closed_dates = {
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _observed_juneteenth(year),
    }
    while expiration in closed_dates or expiration.weekday() >= 5:
        expiration -= timedelta(days=1)
    return expiration


def upcoming_opex_events(as_of: date, years_ahead: int = 2) -> list[MarketEvent]:
    """Return OPEX dates from the current month through the same month N years ahead."""
    events: list[MarketEvent] = []
    total_months = years_ahead * 12
    for offset in range(total_months + 1):
        month_index = as_of.month - 1 + offset
        year = as_of.year + month_index // 12
        month = month_index % 12 + 1
        standard_date = _third_friday(year, month)
        event_date = monthly_opex_date(year, month)
        if event_date < as_of:
            continue
        note = "Holiday-adjusted from Friday" if event_date != standard_date else ""
        events.append(
            MarketEvent(
                event_date=event_date,
                title="Monthly OPEX",
                event_type="opex",
                note=note,
            )
        )
    return events
