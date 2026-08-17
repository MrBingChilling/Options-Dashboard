from __future__ import annotations

import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.daily_ai_analysis import build_analysis_packet
from src.daily_ai_model import (
    DEFAULT_MODEL,
    SummaryGenerationError,
    generate_daily_summary,
)
from src.daily_ai_summary import (
    GENERATOR_VERSION,
    SummaryNotReady,
    load_daily_summaries,
    save_daily_summary,
)
from src.skew_collector import AUTO_SYMBOLS, DAILY_TENORS
from src.storage import SnapshotStore, SnapshotStoreError
from src.volatility_storage import volatility_history


EASTERN = ZoneInfo("America/New_York")
LOOKBACK_DAYS = 75


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
                limit=max(5000, len(AUTO_SYMBOLS) * 100),
            )
            if not frame.empty:
                frames.append(frame)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not load daily-summary inputs: {exc}")

    history = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        prior_summaries = load_daily_summaries(store, limit=6)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not load prior daily summaries: {exc}")
    model = os.environ.get("OPENAI_SUMMARY_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    expected_version = f"{GENERATOR_VERSION}:{model}"
    try:
        packet = build_analysis_packet(history, AUTO_SYMBOLS, prior_summaries)
    except SummaryNotReady as exc:
        print(f"[summary-wait] {exc}", flush=True)
        return 0
    latest_session = packet.snapshot_date
    force = os.environ.get("FORCE_DAILY_AI_SUMMARY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not force and any(
        report.snapshot_date == latest_session
        and report.generator_version == expected_version
        and report.input_signature == packet.input_signature
        for report in prior_summaries
    ):
        print(
            f"[summary-current] session={latest_session} generator={expected_version}",
            flush=True,
        )
        return 0

    try:
        summary = generate_daily_summary(
            history,
            AUTO_SYMBOLS,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=model,
            prior_summaries=prior_summaries,
            analysis_packet=packet,
        )
    except SummaryGenerationError as exc:
        # Do not save deterministic or stale prose. A failed run stays failed so
        # the scheduled backup run can retry genuine model analysis.
        raise SystemExit(f"Could not generate fresh daily AI summary: {exc}")

    try:
        save_daily_summary(store, summary)
    except SnapshotStoreError as exc:
        raise SystemExit(f"Could not save daily AI summary: {exc}")

    print(
        f"[summary-saved] session={summary.snapshot_date} "
        f"comparison={summary.comparison_date} coverage="
        f"{summary.symbol_count}/{summary.expected_symbol_count} "
        f"week_comparison={summary.week_comparison_date} "
        f"month_comparison={summary.month_comparison_date} "
        f"bullets={len(summary.bullets)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
