from __future__ import annotations

"""Resume the legacy 1W/1M backfill using surface_v3_gex all-strike chains.

The mature resume/holiday/marker logic remains in backfill_25d_history.py. This
wrapper changes only the request and audit metadata for newly fetched ticker-days,
so legacy completed dates are still skipped and never rewritten.
"""

from src.chain_archive import CALCULATION_VERSION
from src.marketdata_client import MarketDataClient

import scripts.backfill_25d_history as legacy


_original_audit_save = legacy.save_collection_run_best_effort


def _surface_v3_audit(store, record):
    upgraded = {
        **record,
        "collector": "backfill_surface_v3",
        "range_filter": "all",
        "strike_limit": 30,
        "calculation_version": CALCULATION_VERSION,
        "collection_tier": "full_surface",
    }
    _original_audit_save(store, upgraded)


def main() -> int:
    # The legacy collector calls fetch_skew_chain exactly once per missing
    # ticker-day. Redirect that call to the bounded all-moneyness surface. The
    # downstream local 1W/1M ATM/10D/25D calculations and resume state are kept.
    legacy.MarketDataClient.fetch_skew_chain = MarketDataClient.fetch_surface_chain
    legacy.save_collection_run_best_effort = _surface_v3_audit
    print(
        "surface_v3_gex backfill enabled: DTE=0..45, range=all, strikeLimit=30; "
        "newly fetched chains retain volume/OI and reconstructed IV/delta/gamma.",
        flush=True,
    )
    return legacy.main()


if __name__ == "__main__":
    raise SystemExit(main())
