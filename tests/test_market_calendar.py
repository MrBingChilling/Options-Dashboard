from datetime import date

from src.market_calendar import monthly_opex_date, upcoming_opex_events


def test_standard_monthly_opex_is_third_friday():
    assert monthly_opex_date(2026, 8) == date(2026, 8, 21)


def test_juneteenth_observation_moves_opex_to_thursday():
    assert monthly_opex_date(2027, 6) == date(2027, 6, 17)


def test_two_year_window_matches_dashboard_schedule():
    events = upcoming_opex_events(date(2026, 8, 14), years_ahead=2)
    assert len(events) == 25
    assert events[0].event_date == date(2026, 8, 21)
    assert events[-1].event_date == date(2028, 8, 18)
