from datetime import date

from src import collection_storage


def test_weekend_stale_resolution_does_not_block_next_weekday_retry(monkeypatch):
    monkeypatch.setattr(
        collection_storage,
        "_collection_rows",
        lambda *args, **kwargs: [
            {
                "snapshot_date": "2026-08-13",
                "status": "saved",
                "created_at": "2026-08-15T15:25:06Z",
            }
        ],
    )

    assert (
        collection_storage.resolved_snapshot_date_for_requested_date(
            object(),
            "daily_iv_gex_surface",
            date(2026, 8, 14),
        )
        is None
    )


def test_weekday_holiday_resolution_is_reused_by_backup_runs(monkeypatch):
    monkeypatch.setattr(
        collection_storage,
        "_collection_rows",
        lambda *args, **kwargs: [
            {
                "snapshot_date": "2026-09-04",
                "status": "saved",
                "created_at": "2026-09-08T14:15:00Z",
            }
        ],
    )

    assert collection_storage.resolved_snapshot_date_for_requested_date(
        object(),
        "daily_iv_gex_surface",
        date(2026, 9, 7),
    ) == date(2026, 9, 4)
