from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.volatility import VolatilitySnapshot, snapshot_from_chain


DAILY_TENORS = {"1W": 7, "1M": 30}

# Existing 24-stock AI / semiconductor basket, kept in its original order for
# backward compatibility and for the Dashboard preset.
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

# The existing stock basket is intentionally divided into one primary display
# group per symbol. Some names have mixed business models; these buckets are
# meant for dashboard organization rather than strict industry classification.
AI_PHOTONICS_SYMBOLS = [
    "AXTI",
    "ANET",
    "CIEN",
    "AAOI",
    "COHR",
    "LITE",
    "NOK",
]

AI_FABLESS_SEMI_SYMBOLS = [
    "NVDA",
    "AVGO",
    "AMD",
    "CBRS",
]

AI_MEMORY_SYMBOLS = [
    "SNDK",
    "MU",
    "SKHY",
]

AI_FABS_SYMBOLS = [
    "AEHR",
    "STM",
    "ON",
    "KLAC",
    "GFS",
    "LRCX",
    "AMAT",
    "WOLF",
    "INTC",
    "ASML",
]

INDEX_SYMBOLS = ["SPY", "QQQ", "IWM"]

NEOCLOUD_SYMBOLS = [
    "CRWV",
    "NBIS",
    "IREN",
    "ORCL",
]

MAG7_SYMBOLS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
]

SOFTWARE_SYMBOLS = [
    "PANW",
    "PLTR",
    "CRWD",
    "CRM",
    "NOW",
    "NET",
]

POWER_SYMBOLS = [
    "BE",
    "GEV",
    "HUBB",
    "PWR",
    "AGX",
    "IESC",
]

# Automatic daily basket. dict.fromkeys preserves the display order while
# de-duplicating symbols that belong to more than one preset.
AUTO_SYMBOLS = list(
    dict.fromkeys(
        AI_POOL_SYMBOLS
        + INDEX_SYMBOLS
        + NEOCLOUD_SYMBOLS
        + MAG7_SYMBOLS
        + SOFTWARE_SYMBOLS
        + POWER_SYMBOLS
    )
)


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
        snapshots.append(snapshot)

    return snapshots
