from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd

from src.volatility import VolatilitySnapshot, snapshot_from_chain


DAILY_TENORS = {"1W": 7, "1M": 30}

AI_POOL_SYMBOLS = [
    "AEHR",
    "STM",
    "AXTI",
    "ANET",
    "NVDA",
    "ON",
    "SNDK",
    "KLAC",
    "AVGO",
    "CIEN",
    "AAOI",
    "COHR",
    "GFS",
    "LITE",
    "NOK",
    "AMD",
    "LRCX",
    "AMAT",
    "WOLF",
    "MU",
    "INTC",
    "SKHY",
    "CBRS",
    "ASML",
]

INDEX_SYMBOLS = ["SPY", "QQQ", "IWM"]
AUTO_SYMBOLS = list(dict.fromkeys(AI_POOL_SYMBOLS + INDEX_SYMBOLS))


def previous_weekday(value: date) -> date:
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def skew_snapshots_from_chain(
    symbol: str,
    snapshot_date: date,
    chain: pd.DataFrame,
    tenors: dict[str, int] | None = None,
) -> list[VolatilitySnapshot]:
    if chain is None or chain.empty:
        raise ValueError("The option chain is empty.")

    snapshots: list[VolatilitySnapshot] = []
    for tenor, target_dte in (tenors or DAILY_TENORS).items():
        snapshot = snapshot_from_chain(
            symbol,
            chain,
            snapshot_date,
            tenor,
            target_dte=int(target_dte),
        )
        if (
            snapshot.call_25d_iv is None
            or snapshot.put_25d_iv is None
            or snapshot.skew_25d is None
        ):
            raise ValueError(f"{tenor} has no usable 25D call/put IV pair.")
        snapshots.append(replace(snapshot, atm_iv=None))

    return snapshots
