from datetime import date

import pandas as pd

import scripts.backfill_surface_v3 as surface
from src.chain_archive import CALCULATION_VERSION


def test_surface_v3_resume_ignores_legacy_rows_and_keeps_verified_archives(monkeypatch):
    symbols = ["SPY", "QQQ", "NVDA"]
    legacy_date = date(2026, 1, 2)
    summary_date = date(2026, 1, 5)
    partial_date = date(2026, 1, 6)
    audit_date = date(2026, 1, 7)

    def fake_history(store, requested_symbols, tenor, **kwargs):
        rows = [
            {
                "symbol": "SPY",
                "snapshot_date": pd.Timestamp(legacy_date),
                "calculation_version": None,
                "archive_path": None,
            },
            {
                "symbol": "QQQ",
                "snapshot_date": pd.Timestamp(summary_date),
                "calculation_version": CALCULATION_VERSION,
                "archive_path": "options-chain-archive/v3/QQQ.parquet",
            },
        ]
        if tenor == "1M":
            rows.append(
                {
                    "symbol": "NVDA",
                    "snapshot_date": pd.Timestamp(partial_date),
                    "calculation_version": CALCULATION_VERSION,
                    "archive_path": "options-chain-archive/v3/NVDA.parquet",
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(surface, "volatility_history", fake_history)
    monkeypatch.setattr(
        surface,
        "_successful_full_surface_pairs",
        lambda *args, **kwargs: {("SPY", audit_date)},
    )

    attempted = surface._surface_v3_attempted_tenors_by_date(
        object(), symbols, legacy_date, audit_date
    )

    assert legacy_date not in attempted["SPY"]
    assert attempted["QQQ"][summary_date] == {"1W", "1M"}
    assert attempted["NVDA"][partial_date] == {"1M"}
    assert attempted["SPY"][audit_date] == {"1W", "1M"}


def test_main_installs_full_surface_fetch_and_v3_resume(monkeypatch):
    monkeypatch.setattr(surface.legacy, "main", lambda: 0)

    assert surface.main() == 0
    assert (
        surface.legacy.MarketDataClient.fetch_skew_chain
        is surface.MarketDataClient.fetch_surface_chain
    )
    assert (
        surface.legacy._attempted_tenors_by_date
        is surface._surface_v3_attempted_tenors_by_date
    )
