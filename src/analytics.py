from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import pi
from typing import Any

import numpy as np
import pandas as pd


STANDARD = "Standard: calls + / puts -"
DEALERS_SHORT_ALL = "Dealers short all options"
DEALERS_LONG_ALL = "Dealers long all options"
ASSUMPTIONS = (STANDARD, DEALERS_SHORT_ALL, DEALERS_LONG_ALL)
CONTRACT_MULTIPLIER = 100.0


@dataclass(frozen=True)
class PositioningSummary:
    symbol: str
    snapshot_date: str
    spot: float
    net_gex: float
    gross_gex: float
    gamma_flip: float | None
    flip_distance_pct: float | None
    call_wall: float
    put_wall: float
    call_open_interest: int
    put_open_interest: int
    put_call_oi_ratio: float | None
    net_delta_exposure: float
    contract_count: int
    min_dte: int
    max_dte: int
    assumption: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assumption_signs(sides: pd.Series, assumption: str) -> np.ndarray:
    sides = sides.astype(str).str.lower().to_numpy()
    if assumption == STANDARD:
        return np.where(sides == "call", 1.0, -1.0)
    if assumption == DEALERS_SHORT_ALL:
        return np.full(len(sides), -1.0)
    if assumption == DEALERS_LONG_ALL:
        return np.full(len(sides), 1.0)
    raise ValueError(f"Unknown dealer assumption: {assumption}")


def black_scholes_gamma(
    spot: float | np.ndarray,
    strike: np.ndarray,
    time_years: np.ndarray,
    volatility: np.ndarray,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    spot_array = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_years = np.maximum(np.asarray(time_years, dtype=float), 1.0 / 3650.0)
    volatility = np.maximum(np.asarray(volatility, dtype=float), 1e-4)
    root_time = np.sqrt(time_years)
    d1 = (
        np.log(spot_array / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * root_time)
    normal_density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * pi)
    return (
        np.exp(-dividend_yield * time_years)
        * normal_density
        / (spot_array * volatility * root_time)
    )


def enrich_chain(
    chain: pd.DataFrame,
    assumption: str,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
) -> pd.DataFrame:
    if chain.empty:
        raise ValueError("The option chain is empty.")
    frame = chain.copy()
    spot = float(frame["underlyingPrice"].dropna().median())
    signs = assumption_signs(frame["side"], assumption)
    volatility = pd.to_numeric(frame["iv"], errors="coerce").to_numpy(float)
    model_gamma = black_scholes_gamma(
        spot,
        frame["strike"].to_numpy(float),
        frame["dte"].to_numpy(float) / 365.0,
        volatility,
        risk_free_rate,
        dividend_yield,
    )
    vendor_gamma = pd.to_numeric(frame["gamma"], errors="coerce").to_numpy(float)
    valid_model_gamma = (
        np.isfinite(volatility)
        & (volatility > 0)
        & np.isfinite(model_gamma)
        & (model_gamma >= 0)
    )
    valid_vendor_gamma = np.isfinite(vendor_gamma) & (vendor_gamma >= 0)

    # MarketData's reported gamma is intentionally low precision and frequently
    # arrives as zero. Recalculate it from the supplied IV whenever possible so
    # those rounded zeros do not erase otherwise meaningful aggregate exposure.
    gamma = np.where(
        valid_model_gamma,
        model_gamma,
        np.where(valid_vendor_gamma, vendor_gamma, 0.0),
    )
    frame["dealer_sign"] = signs
    frame["gamma_source"] = np.where(valid_model_gamma, "modelled from IV", "vendor fallback")
    frame["gamma_used"] = gamma
    frame["gex"] = (
        gamma
        * frame["openInterest"].to_numpy(float)
        * CONTRACT_MULTIPLIER
        * spot**2
        * 0.01
        * signs
    )
    frame["delta_exposure"] = (
        frame["delta"].fillna(0).to_numpy(float)
        * frame["openInterest"].to_numpy(float)
        * CONTRACT_MULTIPLIER
        * spot
        * signs
    )
    return frame


def gamma_curve(
    chain: pd.DataFrame,
    assumption: str,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    lower_pct: float = 0.70,
    upper_pct: float = 1.30,
    points: int = 121,
) -> pd.DataFrame:
    spot = float(chain["underlyingPrice"].dropna().median())
    price_grid = np.linspace(spot * lower_pct, spot * upper_pct, points)
    strikes = chain["strike"].to_numpy(float)
    times = chain["dte"].to_numpy(float) / 365.0
    iv = pd.to_numeric(chain["iv"], errors="coerce").to_numpy(float)
    valid_iv = np.isfinite(iv) & (iv > 0)
    weights = (
        chain["openInterest"].to_numpy(float)
        * CONTRACT_MULTIPLIER
        * assumption_signs(chain["side"], assumption)
    )
    weights = np.where(valid_iv, weights, 0.0)

    totals: list[float] = []
    for simulated_spot in price_grid:
        gamma = black_scholes_gamma(
            simulated_spot,
            strikes,
            times,
            iv,
            risk_free_rate,
            dividend_yield,
        )
        totals.append(float(np.nansum(gamma * weights * simulated_spot**2 * 0.01)))
    return pd.DataFrame({"spot": price_grid, "net_gex": totals})


def find_gamma_flip(curve: pd.DataFrame, current_spot: float) -> float | None:
    prices = curve["spot"].to_numpy(float)
    values = curve["net_gex"].to_numpy(float)
    finite = np.isfinite(prices) & np.isfinite(values)
    prices, values = prices[finite], values[finite]
    if len(prices) < 2:
        return None

    exact = np.flatnonzero(np.isclose(values, 0.0, atol=1e-9))
    candidates = [float(prices[index]) for index in exact]
    for index in np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:])):
        x1, x2 = prices[index], prices[index + 1]
        y1, y2 = values[index], values[index + 1]
        candidates.append(float(x1 - y1 * (x2 - x1) / (y2 - y1)))
    if not candidates:
        return None
    return min(candidates, key=lambda value: abs(value - current_spot))


def strike_profile(enriched: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        enriched.groupby(["strike", "side"], as_index=False)
        .agg(gex=("gex", "sum"), open_interest=("openInterest", "sum"), volume=("volume", "sum"))
    )
    pieces: list[pd.DataFrame] = []
    for metric in ("gex", "open_interest", "volume"):
        pivot = grouped.pivot(index="strike", columns="side", values=metric).fillna(0)
        pivot = pivot.rename(columns={"call": f"call_{metric}", "put": f"put_{metric}"})
        pieces.append(pivot)
    profile = pd.concat(pieces, axis=1).fillna(0).reset_index()
    for column in (
        "call_gex",
        "put_gex",
        "call_open_interest",
        "put_open_interest",
        "call_volume",
        "put_volume",
    ):
        if column not in profile:
            profile[column] = 0.0
    profile["net_gex"] = profile["call_gex"] + profile["put_gex"]
    return profile.sort_values("strike").reset_index(drop=True)


def expiration_profile(enriched: pd.DataFrame) -> pd.DataFrame:
    frame = enriched.copy()
    frame["expiration_date"] = pd.to_datetime(frame["expiration"]).dt.date
    grouped = frame.groupby(["expiration_date", "side"], as_index=False)["gex"].sum()
    pivot = grouped.pivot(index="expiration_date", columns="side", values="gex").fillna(0)
    pivot = pivot.rename(columns={"call": "call_gex", "put": "put_gex"}).reset_index()
    for column in ("call_gex", "put_gex"):
        if column not in pivot:
            pivot[column] = 0.0
    pivot["net_gex"] = pivot["call_gex"] + pivot["put_gex"]
    return pivot.sort_values("expiration_date")


def summarize(
    symbol: str,
    snapshot_date: date,
    enriched: pd.DataFrame,
    curve: pd.DataFrame,
    assumption: str,
    min_dte: int,
    max_dte: int,
) -> PositioningSummary:
    spot = float(enriched["underlyingPrice"].dropna().median())
    profile = strike_profile(enriched)
    call_rows = enriched[enriched["side"] == "call"]
    put_rows = enriched[enriched["side"] == "put"]
    call_oi = int(call_rows["openInterest"].sum())
    put_oi = int(put_rows["openInterest"].sum())
    call_wall = float(
        profile.loc[profile["call_gex"].abs().idxmax(), "strike"]
    )
    put_wall = float(
        profile.loc[profile["put_gex"].abs().idxmax(), "strike"]
    )
    flip = find_gamma_flip(curve, spot)
    return PositioningSummary(
        symbol=symbol.upper(),
        snapshot_date=snapshot_date.isoformat(),
        spot=spot,
        net_gex=float(enriched["gex"].sum()),
        gross_gex=float(enriched["gex"].abs().sum()),
        gamma_flip=flip,
        flip_distance_pct=((flip / spot) - 1.0) if flip is not None else None,
        call_wall=call_wall,
        put_wall=put_wall,
        call_open_interest=call_oi,
        put_open_interest=put_oi,
        put_call_oi_ratio=(put_oi / call_oi) if call_oi else None,
        net_delta_exposure=float(enriched["delta_exposure"].sum()),
        contract_count=len(enriched),
        min_dte=min_dte,
        max_dte=max_dte,
        assumption=assumption,
    )


def snapshot_record(
    summary: PositioningSummary,
    profile: pd.DataFrame,
) -> dict[str, Any]:
    record = summary.to_dict()
    compact = profile[
        ["strike", "call_gex", "put_gex", "net_gex", "call_open_interest", "put_open_interest"]
    ].copy()
    record["strike_profile"] = compact.round(4).to_dict(orient="records")
    return record
