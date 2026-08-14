from scripts.daily_skew import (
    _choose_collection_tier,
    _daily_credit_limit,
    _priority_credit_reserve,
)


def test_automatic_credit_limit_is_hard_capped_below_100(monkeypatch):
    monkeypatch.setenv("DAILY_MARKETDATA_CREDIT_LIMIT", "500")
    assert _daily_credit_limit() == 99


def test_dense_index_chains_receive_larger_25d_reserve():
    assert _priority_credit_reserve("SPY") == 3
    assert _priority_credit_reserve("QQQ") == 3
    assert _priority_credit_reserve("AAPL") == 2
    assert _priority_credit_reserve("AEHR") == 1


def test_optional_gex_is_disabled_when_it_would_consume_later_25d_reserve():
    later = ["QQQ", "IWM", "AAPL", "AEHR"]
    reserve_for_later = sum(_priority_credit_reserve(symbol) for symbol in later)

    assert _choose_collection_tier(
        "SPY",
        later,
        effective_credits_left=6 + reserve_for_later,
    ) == "full_surface"
    assert _choose_collection_tier(
        "SPY",
        later,
        effective_credits_left=5 + reserve_for_later,
    ) == "priority_25d"
