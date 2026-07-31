from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from math import erf, exp, log, pi, sqrt
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


def black_scholes_delta(
    spot: float | np.ndarray,
    strike: np.ndarray,
    time_years: np.ndarray,
    volatility: np.ndarray,
    sides: pd.Series | np.ndarray,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    spot_array = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_years = np.maximum(np.asarray(time_years, dtype=float), 1.0 / 3650.0)
    volatility = np.asarray(volatility, dtype=float)
    root_time = np.sqrt(time_years)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (
            np.log(spot_array / strike)
            + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_years
        ) / (volatility * root_time)
    normal_cdf = np.vectorize(lambda value: 0.5 * (1.0 + erf(value / sqrt(2.0))))
    call_delta = np.exp(-dividend_yield * time_years) * normal_cdf(d1)
    side_values = np.asarray(sides, dtype=str)
    return np.where(
        np.char.lower(side_values) == "call",
        call_delta,
        call_delta - np.exp(-dividend_yield * time_years),
    )


def black_scholes_price(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    side: str,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    if spot <= 0 or strike <= 0 or time_years <= 0 or volatility <= 0:
        return float("nan")
    root_time = sqrt(time_years)
    d1 = (
        log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    normal_cdf = lambda value: 0.5 * (1.0 + erf(value / sqrt(2.0)))
    discounted_spot = spot * exp(-dividend_yield * time_years)
    discounted_strike = strike * exp(-risk_free_rate * time_years)
    if side.lower() == "call":
        return discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
    return discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)


def implied_volatility_from_price(
    option_price: float,
    spot: float,
    strike: float,
    time_years: float,
    side: str,
    risk_free_rate: float,
    dividend_yield: float,
    max_iterations: int = 60,
) -> float:
    values = (option_price, spot, strike, time_years)
    if not all(np.isfinite(value) for value in values):
        return float("nan")
    if option_price <= 0 or spot <= 0 or strike <= 0 or time_years <= 0:
        return float("nan")

    lower_volatility, upper_volatility = 1e-4, 10.0
    lower_price = black_scholes_price(
        spot,
        strike,
        time_years,
        lower_volatility,
        side,
        risk_free_rate,
        dividend_yield,
    )
    upper_price = black_scholes_price(
        spot,
        strike,
        time_years,
        upper_volatility,
        side,
        risk_free_rate,
        dividend_yield,
    )
    tolerance = max(1e-6, option_price * 1e-7)
    if option_price < lower_price - tolerance or option_price > upper_price + tolerance:
        return float("nan")

    for _ in range(max_iterations):
        midpoint = (lower_volatility + upper_volatility) / 2.0
        model_price = black_scholes_price(
            spot,
            strike,
            time_years,
            midpoint,
            side,
            risk_free_rate,
            dividend_yield,
        )
        if abs(model_price - option_price) <= tolerance:
            return midpoint
        if model_price < option_price:
            lower_volatility = midpoint
        else:
            upper_volatility = midpoint
    return (lower_volatility + upper_volatility) / 2.0


def _quote_candidates(row: pd.Series) -> list[float]:
    candidates: list[float] = []
    def numeric_value(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    mid = numeric_value(row.get("mid"))
    bid = numeric_value(row.get("bid"))
    ask = numeric_value(row.get("ask"))
    last = numeric_value(row.get("last"))
    if np.isfinite(mid) and mid > 0:
        candidates.append(float(mid))
    if np.isfinite(bid) and np.isfinite(ask) and 0 <= bid <= ask and ask > 0:
        candidates.append(float((bid + ask) / 2.0))
    if np.isfinite(last) and last > 0:
        candidates.append(float(last))
    return list(dict.fromkeys(candidates))


def derive_implied_volatility(
    frame: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> np.ndarray:
    derived = np.full(len(frame), np.nan, dtype=float)
    for output_index, (_, row) in enumerate(frame.iterrows()):
        for option_price in _quote_candidates(row):
            volatility = implied_volatility_from_price(
                option_price=option_price,
                spot=spot,
                strike=float(row["strike"]),
                time_years=max(float(row["dte"]) / 365.0, 1.0 / 3650.0),
                side=str(row["side"]),
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            if np.isfinite(volatility) and volatility > 0:
                derived[output_index] = volatility
                break
    return derived


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
    vendor_iv = pd.to_numeric(frame["iv"], errors="coerce").to_numpy(float)
    valid_vendor_iv = np.isfinite(vendor_iv) & (vendor_iv > 0)
    derived_iv = np.full(len(frame), np.nan, dtype=float)
    missing_iv = ~valid_vendor_iv
    if missing_iv.any():
        derived_iv[missing_iv] = derive_implied_volatility(
            frame.loc[missing_iv],
            spot,
            risk_free_rate,
            dividend_yield,
        )
    valid_derived_iv = np.isfinite(derived_iv) & (derived_iv > 0)
    volatility = np.where(valid_vendor_iv, vendor_iv, derived_iv)
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
        & (model_gamma > 0)
    )
    valid_vendor_gamma = np.isfinite(vendor_gamma) & (vendor_gamma > 0)

    # MarketData's reported gamma is intentionally low precision and frequently
    # arrives as zero. Recalculate it from the supplied IV whenever possible so
    # those rounded zeros do not erase otherwise meaningful aggregate exposure.
    gamma = np.where(
        valid_model_gamma,
        model_gamma,
        np.where(valid_vendor_gamma, vendor_gamma, 0.0),
    )
    model_delta = black_scholes_delta(
        spot,
        frame["strike"].to_numpy(float),
        frame["dte"].to_numpy(float) / 365.0,
        volatility,
        frame["side"].to_numpy(str),
        risk_free_rate,
        dividend_yield,
    )
    vendor_delta = pd.to_numeric(frame["delta"], errors="coerce").to_numpy(float)
    valid_model_delta = (
        np.isfinite(volatility)
        & (volatility > 0)
        & np.isfinite(model_delta)
    )
    delta = np.where(
        valid_model_delta,
        model_delta,
        np.where(np.isfinite(vendor_delta), vendor_delta, 0.0),
    )
    frame["dealer_sign"] = signs
    frame["iv_used"] = volatility
    frame["iv_source"] = np.select(
        [valid_vendor_iv, valid_derived_iv],
        ["vendor", "derived from option price"],
        default="unavailable",
    )
    frame["gamma_source"] = np.where(
        valid_model_gamma,
        "modelled from IV",
        np.where(valid_vendor_gamma, "vendor fallback", "unavailable"),
    )
    frame["gamma_used"] = gamma
    frame["delta_used"] = delta
    frame["delta_source"] = np.where(
        valid_model_delta,
        "modelled from IV",
        np.where(np.isfinite(vendor_delta), "vendor fallback", "unavailable"),
    )
    frame["gex"] = (
        gamma
        * frame["openInterest"].to_numpy(float)
        * CONTRACT_MULTIPLIER
        * spot**2
        * 0.01
        * signs
    )
    frame["delta_exposure"] = (
        delta
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
    iv_column = "iv_used" if "iv_used" in chain.columns else "iv"
    iv = pd.to_numeric(chain[iv_column], errors="coerce").to_numpy(float)
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

    magnitude = float(np.max(np.abs(values)))
    tolerance = max(1e-9, magnitude * 1e-12)
    nonzero_indices = np.flatnonzero(np.abs(values) > tolerance)
    if len(nonzero_indices) < 2:
        return None

    candidates: list[float] = []
    for left_index, right_index in zip(nonzero_indices[:-1], nonzero_indices[1:]):
        y1, y2 = values[left_index], values[right_index]
        if np.signbit(y1) == np.signbit(y2):
            continue
        x1, x2 = prices[left_index], prices[right_index]
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
    gross_gex = float(enriched["gex"].abs().sum())
    if not np.isfinite(gross_gex) or gross_gex <= 1e-6:
        raise ValueError(
            "No usable gamma exposure could be calculated from this chain. "
            "The provider returned no valid IV/Greeks and the option quotes could not be used to derive them."
        )
    if profile["call_gex"].abs().max() <= 1e-6 or profile["put_gex"].abs().max() <= 1e-6:
        raise ValueError("The chain does not contain usable gamma exposure for both calls and puts.")
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
        gross_gex=gross_gex,
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
