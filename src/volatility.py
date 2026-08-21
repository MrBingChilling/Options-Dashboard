from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
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


def _spot_from_chain(work: pd.DataFrame) -> float:
    spot_values = pd.to_numeric(work["underlyingPrice"], errors="coerce").dropna()
    if spot_values.empty:
        raise ValueError("The chain has no usable underlying price.")
    spot = float(spot_values.median())
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("The chain has no usable underlying price.")
    return spot


def _expiry_rows(work: pd.DataFrame) -> pd.DataFrame:
    dte_by_expiry = (
        work.groupby("expiration", as_index=False)["dte"]
        .median()
        .dropna(subset=["expiration", "dte"])
        .sort_values(["dte", "expiration"])
        .reset_index(drop=True)
    )
    if dte_by_expiry.empty:
        raise ValueError("The chain has no usable expiration data.")
    return dte_by_expiry


def _bracketing_expiries(work: pd.DataFrame, target: int) -> tuple[pd.Series, pd.Series]:
    expiries = _expiry_rows(work)
    lower = expiries[expiries["dte"] <= target]
    upper = expiries[expiries["dte"] >= target]

    if lower.empty:
        lower_row = upper.iloc[0]
    else:
        lower_row = lower.iloc[-1]

    if upper.empty:
        upper_row = lower.iloc[-1]
    else:
        upper_row = upper.iloc[0]

    return lower_row, upper_row


def _prepare_expiry_chain(
    work: pd.DataFrame,
    expiration: pd.Timestamp,
    spot: float,
) -> tuple[pd.DataFrame, int]:
    expiry_chain = work[work["expiration"] == expiration].copy()
    if expiry_chain.empty:
        raise ValueError("The selected expiration has no usable contracts.")

    actual_dte = int(round(float(pd.to_numeric(expiry_chain["dte"], errors="coerce").median())))

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

    return expiry_chain, actual_dte


def _surface_metrics(expiry_chain: pd.DataFrame, spot: float) -> dict[str, float | None]:
    return {
        "atm_iv": _atm_iv(expiry_chain, spot),
        "call_25d_iv": _delta_iv(expiry_chain, "call", 0.25),
        "put_25d_iv": _delta_iv(expiry_chain, "put", -0.25),
        "call_10d_iv": _delta_iv(expiry_chain, "call", 0.10),
        "put_10d_iv": _delta_iv(expiry_chain, "put", -0.10),
    }


def _constant_maturity_iv(
    lower_iv: float | None,
    upper_iv: float | None,
    lower_dte: int,
    upper_dte: int,
    target_dte: int,
) -> float | None:
    """Interpolate implied variance in time to an exact target maturity."""
    lower = _finite(lower_iv)
    upper = _finite(upper_iv)
    if lower is None or lower <= 0:
        return upper if upper is not None and upper > 0 else None
    if upper is None or upper <= 0:
        return lower
    if target_dte <= 0:
        raise ValueError("Constant-maturity target DTE must be positive.")
    if lower_dte == upper_dte:
        return float((lower + upper) / 2.0)

    # Outside the available expiration range, keep the nearest expiry's IV
    # level rather than creating unstable variance extrapolation.
    if target_dte <= lower_dte:
        return lower
    if target_dte >= upper_dte:
        return upper

    weight = (target_dte - lower_dte) / float(upper_dte - lower_dte)
    lower_time = max(float(lower_dte), 0.0) / 365.0
    upper_time = max(float(upper_dte), 0.0) / 365.0
    target_time = float(target_dte) / 365.0
    target_variance = (
        (1.0 - weight) * (lower**2) * lower_time
        + weight * (upper**2) * upper_time
    )
    if not np.isfinite(target_variance) or target_variance <= 0:
        return None
    return float(np.sqrt(target_variance / target_time))


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
    if target <= 0:
        raise ValueError("Target DTE must be positive.")

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
    work["expiration"] = pd.to_datetime(work["expiration"], errors="coerce")
    work = work.dropna(subset=["dte", "expiration", "strike", "underlyingPrice"])
    work = work[work["side"].astype(str).str.lower().isin(["call", "put"])]
    work["side"] = work["side"].astype(str).str.lower()
    if work.empty:
        raise ValueError("The chain has no usable contracts for IV/skew.")

    spot = _spot_from_chain(work)
    lower_row, upper_row = _bracketing_expiries(work, target)
    lower_expiration = pd.Timestamp(lower_row["expiration"])
    upper_expiration = pd.Timestamp(upper_row["expiration"])

    lower_chain, lower_dte = _prepare_expiry_chain(work, lower_expiration, spot)
    lower_metrics = _surface_metrics(lower_chain, spot)

    if lower_expiration == upper_expiration:
        upper_dte = lower_dte
        upper_metrics = lower_metrics
    else:
        upper_chain, upper_dte = _prepare_expiry_chain(work, upper_expiration, spot)
        upper_metrics = _surface_metrics(upper_chain, spot)

    metrics = {
        key: _constant_maturity_iv(
            lower_metrics.get(key),
            upper_metrics.get(key),
            lower_dte,
            upper_dte,
            target,
        )
        for key in lower_metrics
    }

    call_25d = metrics["call_25d_iv"]
    put_25d = metrics["put_25d_iv"]
    call_10d = metrics["call_10d_iv"]
    put_10d = metrics["put_10d_iv"]

    return VolatilitySnapshot(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date,
        tenor=tenor,
        target_dte=target,
        # These fields now describe the synthetic constant-maturity point,
        # rather than whichever listed option happened to be nearest.
        actual_dte=target,
        expiration=snapshot_date + timedelta(days=target),
        spot=spot,
        atm_iv=metrics["atm_iv"],
        call_25d_iv=call_25d,
        put_25d_iv=put_25d,
        skew_25d=(
            call_25d - put_25d
            if call_25d is not None and put_25d is not None
            else None
        ),
        call_10d_iv=call_10d,
        put_10d_iv=put_10d,
        skew_10d=(
            call_10d - put_10d
            if call_10d is not None and put_10d is not None
            else None
        ),
        calculation_version=CALCULATION_VERSION,
    )


def snapshots_from_chain(
    symbol: str,
    chain: pd.DataFrame,
    snapshot_date: date,
    tenors: Iterable[str] = ("1W", "1M", "3M", "6M"),
) -> list[VolatilitySnapshot]:
    return [snapshot_from_chain(symbol, chain, snapshot_date, tenor) for tenor in tenors]
