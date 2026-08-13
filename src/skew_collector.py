from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.volatility import VolatilitySnapshot


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
    """Return the previous Monday-Friday date.

    MarketData may fall back to the most recent actual trading session on
    exchange holidays. The collector records result.snapshot_date, so holiday
    fallbacks remain idempotent after the first probe.
    """
    candidate = value - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _usable(chain: pd.DataFrame) -> pd.DataFrame:
    if chain is None or chain.empty:
        raise ValueError("The option chain is empty.")

    required = {"expiration", "side", "dte", "underlyingPrice", "iv", "delta"}
    missing = required.difference(chain.columns)
    if missing:
        raise ValueError(
            "The option chain is missing required columns: "
            + ", ".join(sorted(missing))
        )

    work = chain.copy()
    work["expiration"] = pd.to_datetime(work["expiration"], errors="coerce").dt.date
    work["side"] = work["side"].astype(str).str.lower().str.strip()

    for column in ("dte", "underlyingPrice", "iv", "delta"):
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work[
        work["expiration"].notna()
        & work["side"].isin(["call", "put"])
        & work["dte"].notna()
        & work["iv"].notna()
        & work["delta"].notna()
        & (work["iv"] > 0)
    ].copy()

    if work.empty:
        raise ValueError("The chain has no contracts with usable IV/delta data.")

    return work


def _nearest_expiration(frame: pd.DataFrame, target_dte: int) -> pd.DataFrame:
    expirations = (
        frame.groupby("expiration", as_index=False)["dte"]
        .median()
        .assign(distance=lambda x: (x["dte"] - target_dte).abs())
        .sort_values(["distance", "dte", "expiration"])
    )
    if expirations.empty:
        raise ValueError("No usable expiration was returned.")

    expiration = expirations.iloc[0]["expiration"]
    return frame[frame["expiration"] == expiration].copy()


def _nearest_25d_iv(frame: pd.DataFrame, side: str) -> float:
    target = 0.25 if side == "call" else -0.25
    candidates = frame[frame["side"] == side].copy()
    if candidates.empty:
        raise ValueError(f"No usable {side} contracts were returned.")

    candidates["delta_distance"] = (candidates["delta"] - target).abs()
    row = candidates.sort_values("delta_distance").iloc[0]
    return float(row["iv"])


def skew_snapshots_from_chain(
    symbol: str,
    snapshot_date: date,
    chain: pd.DataFrame,
    tenors: dict[str, int] | None = None,
) -> list[VolatilitySnapshot]:
    """Build the minimum daily records required for 25-delta skew.

    The MarketData request is intentionally filtered to ~25D contracts only.
    ATM IV is therefore stored as NULL rather than pretending a 25D contract is
    ATM. The existing Supabase column is nullable.
    """
    targets = tenors or DAILY_TENORS
    work = _usable(chain)
    snapshots: list[VolatilitySnapshot] = []

    for tenor, target_dte in targets.items():
        expiry_frame = _nearest_expiration(work, target_dte)

        call_iv = _nearest_25d_iv(expiry_frame, "call")
        put_iv = _nearest_25d_iv(expiry_frame, "put")
        actual_dte = int(round(float(expiry_frame["dte"].median())))

        spot_values = expiry_frame["underlyingPrice"].dropna()
        if spot_values.empty:
            raise ValueError("No usable underlying price was returned.")
        spot = float(spot_values.median())

        expiration = expiry_frame["expiration"].iloc[0]
        snapshots.append(
            VolatilitySnapshot(
                symbol=symbol.upper(),
                snapshot_date=snapshot_date,
                tenor=tenor,
                target_dte=int(target_dte),
                actual_dte=actual_dte,
                expiration=expiration,
                spot=spot,
                atm_iv=None,
                call_25d_iv=call_iv,
                put_25d_iv=put_iv,
                skew_25d=call_iv - put_iv,
            )
        )

    return snapshots
