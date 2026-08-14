from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from src.analytics import black_scholes_delta, derive_implied_volatility


TENORS = {"1W": 7, "1M": 30, "3M": 90, "6M": 180}
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_DIVIDEND_YIELD = 0.0
CALCULATION_VERSION = "surface_v2"


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
    call_10d_iv: float | None = None
    put_10d_iv: float | None = None
    skew_10d: float | None = None
    archive_path: str | None = None
    chain_contract_count: int | None = None
    calculation_version: str | None = None

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
    if frame.empty or column not in frame.columns or "iv_used" not in frame.columns:
        return None
    work = frame.copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work["iv_used"] = pd.to_numeric(work["iv_used"], errors="coerce")
    work = work.dropna(subset=[column, "iv_used"])
    work = work[work["iv_used"] > 0]
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
            value = _finite(row.get("iv_used"))
            if value is not None and value > 0:
                values.append(value)
    return float(np.mean(values)) if values else None


def _delta_iv(expiry_chain: pd.DataFrame, side: str, target_delta: float) -> float | None:
    side_frame = expiry_chain[expiry_chain["side"] == side]
    row = _closest_row(side_frame, "delta_used", target_delta)
    if row is None:
        return None
    value = _finite(row.get("iv_used"))
    return value if value is not None and value > 0 else None


def _prepare_expiry_chain(
    work: pd.DataFrame,
    target: int,
) -> tuple[pd.DataFrame, pd.Timestamp, int, float]:
    dte_by_expiry = work.groupby("expiration", as_index=False)["dte"].median()
    if dte_by_expiry.empty:
        raise ValueError("The chain has no usable expiration data.")

    expiry_row = dte_by_expiry.loc[(dte_by_expiry["dte"] - target).abs().idxmin()]
    expiration_ts = pd.Timestamp(expiry_row["expiration"])
    actual_dte = int(round(float(expiry_row["dte"])))
    expiry_chain = work[work["expiration"] == expiry_row["expiration"]].copy()
    if expiry_chain.empty:
        raise ValueError("The selected expiration has no usable contracts.")

    spot = float(pd.to_numeric(expiry_chain["underlyingPrice"], errors="coerce").median())
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("The chain has no usable underlying price.")

    vendor_iv = pd.to_numeric(expiry_chain.get("iv"), errors="coerce").to_numpy(float)
    valid_vendor_iv = np.isfinite(vendor_iv) & (vendor_iv > 0)
    iv_used = vendor_iv.copy()
    missing_iv = ~valid_vendor_iv
    if missing_iv.any():
        derived = derive_implied_volatility(
            expiry_chain.loc[missing_iv],
            spot,
            DEFAULT_RISK_FREE_RATE,
            DEFAULT_DIVIDEND_YIELD,
        )
        iv_used[missing_iv] = derived

    valid_iv = np.isfinite(iv_used) & (iv_used > 0)
    if not valid_iv.any():
        raise ValueError(
            "The chain has no usable IV data and IV could not be derived from option prices."
        )
    expiry_chain["iv_used"] = iv_used
    expiry_chain["iv_source"] = np.where(
        valid_vendor_iv,
        "vendor",
        np.where(valid_iv, "derived_from_option_price", "unavailable"),
    )

    model_delta = black_scholes_delta(
        spot,
        pd.to_numeric(expiry_chain["strike"], errors="coerce").to_numpy(float),
        pd.to_numeric(expiry_chain["dte"], errors="coerce").to_numpy(float) / 365.0,
        iv_used,
        expiry_chain["side"].astype(str).to_numpy(),
        DEFAULT_RISK_FREE_RATE,
        DEFAULT_DIVIDEND_YIELD,
    )
    vendor_delta = pd.to_numeric(expiry_chain.get("delta"), errors="coerce").to_numpy(float)
    side_values = expiry_chain["side"].astype(str).str.lower().to_numpy()
    valid_vendor_delta = (
        np.isfinite(vendor_delta)
        & (
            ((side_values == "call") & (vendor_delta > 0) & (vendor_delta <= 1))
            | ((side_values == "put") & (vendor_delta < 0) & (vendor_delta >= -1))
        )
    )
    valid_model_delta = valid_iv & np.isfinite(model_delta)
    expiry_chain["delta_used"] = np.where(
        valid_vendor_delta,
        vendor_delta,
        np.where(valid_model_delta, model_delta, np.nan),
    )
    expiry_chain["delta_source"] = np.where(
        valid_vendor_delta,
        "vendor",
        np.where(valid_model_delta, "black_scholes_from_iv", "unavailable"),
    )

    return expiry_chain, expiration_ts, actual_dte, spot


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
    for column in (
        "dte",
        "strike",
        "iv",
        "delta",
        "underlyingPrice",
        "bid",
        "ask",
        "mid",
        "last",
    ):
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["dte", "expiration", "strike", "underlyingPrice"])
    work = work[work["side"].astype(str).str.lower().isin(["call", "put"])]
    work["side"] = work["side"].astype(str).str.lower()
    if work.empty:
        raise ValueError("The chain has no usable contracts for IV/skew.")

    expiry_chain, expiration_ts, actual_dte, spot = _prepare_expiry_chain(work, target)

    atm = _atm_iv(expiry_chain, spot)
    call_25d = _delta_iv(expiry_chain, "call", 0.25)
    put_25d = _delta_iv(expiry_chain, "put", -0.25)
    skew_25d = call_25d - put_25d if call_25d is not None and put_25d is not None else None
    call_10d = _delta_iv(expiry_chain, "call", 0.10)
    put_10d = _delta_iv(expiry_chain, "put", -0.10)
    skew_10d = call_10d - put_10d if call_10d is not None and put_10d is not None else None

    return VolatilitySnapshot(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date,
        tenor=tenor,
        target_dte=target,
        actual_dte=actual_dte,
        expiration=expiration_ts.date(),
        spot=spot,
        atm_iv=atm,
        call_25d_iv=call_25d,
        put_25d_iv=put_25d,
        skew_25d=skew_25d,
        call_10d_iv=call_10d,
        put_10d_iv=put_10d,
        skew_10d=skew_10d,
        calculation_version=CALCULATION_VERSION,
    )


def snapshots_from_chain(
    symbol: str,
    chain: pd.DataFrame,
    snapshot_date: date,
    tenors: Iterable[str] = ("1W", "1M", "3M", "6M"),
) -> list[VolatilitySnapshot]:
    return [snapshot_from_chain(symbol, chain, snapshot_date, tenor) for tenor in tenors]
