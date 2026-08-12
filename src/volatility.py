from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd


TENORS = {"1W": 7, "1M": 30, "3M": 90, "6M": 180}


@dataclass(frozen=True)
class VolatilitySnapshot:
    symbol: str
    snapshot_date: date
    tenor: str
    target_dte: int
    actual_dte: int
    expiration: date
    spot: float
    atm_iv: float | None
    call_25d_iv: float | None
    put_25d_iv: float | None
    skew_25d: float | None

    def record(self) -> dict:
        row = asdict(self)
        row["snapshot_date"] = self.snapshot_date.isoformat()
        row["expiration"] = self.expiration.isoformat()
        return row


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _closest_row(frame: pd.DataFrame, column: str, target: float) -> pd.Series | None:
    if frame.empty or column not in frame.columns:
        return None
    work = frame.copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column, "iv"])
    work = work[pd.to_numeric(work["iv"], errors="coerce") > 0]
    if work.empty:
        return None
    index = (work[column] - target).abs().idxmin()
    return work.loc[index]


def _atm_iv(expiry_chain: pd.DataFrame, spot: float) -> float | None:
    values: list[float] = []
    for side in ("call", "put"):
        side_frame = expiry_chain[expiry_chain["side"] == side]
        row = _closest_row(side_frame, "strike", spot)
        if row is not None:
            value = _finite(row.get("iv"))
            if value is not None and value > 0:
                values.append(value)
    return float(np.mean(values)) if values else None


def _delta_iv(expiry_chain: pd.DataFrame, side: str, target_delta: float) -> float | None:
    side_frame = expiry_chain[expiry_chain["side"] == side]
    row = _closest_row(side_frame, "delta", target_delta)
    if row is None:
        return None
    value = _finite(row.get("iv"))
    return value if value is not None and value > 0 else None


def snapshot_from_chain(
    symbol: str,
    chain: pd.DataFrame,
    snapshot_date: date,
    tenor: str,
    target_dte: int | None = None,
) -> VolatilitySnapshot:
    if chain.empty:
        raise ValueError("Cannot calculate IV/skew from an empty option chain.")
    if tenor not in TENORS and target_dte is None:
        raise ValueError(f"Unsupported tenor: {tenor}")
    target = int(target_dte if target_dte is not None else TENORS[tenor])

    work = chain.copy()
    for column in ("dte", "strike", "iv", "delta", "underlyingPrice"):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["dte", "expiration", "strike", "iv", "underlyingPrice"])
    work = work[work["iv"] > 0]
    if work.empty:
        raise ValueError("The chain has no contracts with usable IV data.")

    dte_by_expiry = work.groupby("expiration", as_index=False)["dte"].median()
    expiry_row = dte_by_expiry.loc[(dte_by_expiry["dte"] - target).abs().idxmin()]
    expiration_ts = pd.Timestamp(expiry_row["expiration"])
    actual_dte = int(round(float(expiry_row["dte"])))
    expiry_chain = work[work["expiration"] == expiry_row["expiration"]].copy()
    spot = float(expiry_chain["underlyingPrice"].median())

    atm = _atm_iv(expiry_chain, spot)
    call_iv = _delta_iv(expiry_chain, "call", 0.25)
    put_iv = _delta_iv(expiry_chain, "put", -0.25)
    skew = call_iv - put_iv if call_iv is not None and put_iv is not None else None

    return VolatilitySnapshot(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date,
        tenor=tenor,
        target_dte=target,
        actual_dte=actual_dte,
        expiration=expiration_ts.date(),
        spot=spot,
        atm_iv=atm,
        call_25d_iv=call_iv,
        put_25d_iv=put_iv,
        skew_25d=skew,
    )


def snapshots_from_chain(
    symbol: str,
    chain: pd.DataFrame,
    snapshot_date: date,
    tenors: Iterable[str] = ("1W", "1M", "3M", "6M"),
) -> list[VolatilitySnapshot]:
    return [snapshot_from_chain(symbol, chain, snapshot_date, tenor) for tenor in tenors]
