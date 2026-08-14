from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.daily_ai_summary import (
    SummaryNotReady,
    build_daily_summary,
    save_daily_summary,
)
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import volatility_history


EASTERN = ZoneInfo("America/New_York")
LOOKBACK_DAYS = 21


def _store() -> SnapshotStore:
    return SnapshotStore(
        os.environ.get("SUPABASE_URL", ""),
        os.environ.get("SUPABASE_SECRET_KEY", "")
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
    )


def main() -> int:
    store = _store()
    if not store.enabled:
        raise SystemExit("Supabase is not configured.")

    end_date = datetime.now(EASTERN).date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    frames: list[pd.DataFrame] = []
    try:
        for tenor in DAILY_TENORS:
            frame = volatility_history(
                store,
                AUTO_SYMBOLS,
                tenor,
                start_date=start_date,
                end_date=end_date,
                limit=max(1000, len(AUTO_SYMBOLS) * 40),
            )
            if not frame.empty:
                frames.append(frame)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not load daily-summary inputs: {exc}")

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        summary = build_daily_summary(history, AUTO_SYMBOLS)
    except SummaryNotReady as exc:
        # A partial first run is expected to be completed by the scheduled
        # backups. Keep the workflow green so the collector can resume later.
        print(f"[summary-wait] {exc}", flush=True)
        return 0

    try:
        save_daily_summary(store, summary)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not save daily AI summary: {exc}")

    print(
        f"[summary-saved] session={summary.snapshot_date} "
        f"comparison={summary.comparison_date} coverage="
        f"{summary.symbol_count}/{summary.expected_symbol_count} "
        f"bullets={len(summary.bullets)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
