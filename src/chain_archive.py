from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import requests

from src.analytics import (
    black_scholes_delta,
    black_scholes_gamma,
    derive_implied_volatility,
)
from src.storage import SnapshotStore, SnapshotStoreError


CHAIN_ARCHIVE_BUCKET = "options-chain-archive"
CALCULATION_VERSION = "surface_v3_gex"
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_DIVIDEND_YIELD = 0.0


@dataclass(frozen=True)
class ArchiveResult:
    path: str
    byte_count: int
    contract_count: int


def prepare_chain_for_archive(
    symbol: str,
    snapshot_date: date,
    chain: pd.DataFrame,
) -> pd.DataFrame:
    """Return the paid-for bounded chain with reproducible local IV/Greeks."""
    if chain is None or chain.empty:
        raise ValueError("Cannot archive an empty option chain.")

    frame = chain.copy()
    frame["symbol"] = symbol.upper()
    frame["snapshot_date"] = pd.Timestamp(snapshot_date)

    for column in (
        "strike",
        "dte",
        "underlyingPrice",
        "bid",
        "ask",
        "mid",
        "last",
        "openInterest",
        "volume",
        "iv",
        "delta",
        "gamma",
        "theta",
        "vega",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["side"] = frame["side"].astype(str).str.lower()
    if "expiration" in frame.columns:
        frame["expiration"] = pd.to_datetime(frame["expiration"], errors="coerce")

    iv_used = np.full(len(frame), np.nan, dtype=float)
    delta_used = np.full(len(frame), np.nan, dtype=float)
    gamma_used = np.full(len(frame), np.nan, dtype=float)
    iv_source = np.full(len(frame), "unavailable", dtype=object)
    delta_source = np.full(len(frame), "unavailable", dtype=object)
    gamma_source = np.full(len(frame), "unavailable", dtype=object)

    # Work expiry-by-expiry so spot and model inputs are internally consistent.
    if "expiration" in frame.columns:
        groups = frame.groupby("expiration", dropna=False).groups.values()
    else:
        groups = [frame.index]

    position_by_index = {index: position for position, index in enumerate(frame.index)}
    for indices in groups:
        expiry = frame.loc[list(indices)].copy()
        spot_values = pd.to_numeric(expiry.get("underlyingPrice"), errors="coerce")
        spot = float(spot_values.median()) if spot_values.notna().any() else float("nan")
        if not np.isfinite(spot) or spot <= 0:
            continue

        vendor_iv = pd.to_numeric(expiry.get("iv"), errors="coerce").to_numpy(float)
        valid_vendor_iv = np.isfinite(vendor_iv) & (vendor_iv > 0)
        local_iv = vendor_iv.copy()
        missing_iv = ~valid_vendor_iv
        if missing_iv.any():
            derived = derive_implied_volatility(
                expiry.loc[missing_iv],
                spot,
                DEFAULT_RISK_FREE_RATE,
                DEFAULT_DIVIDEND_YIELD,
            )
            local_iv[missing_iv] = derived

        valid_iv = np.isfinite(local_iv) & (local_iv > 0)
        strikes = pd.to_numeric(expiry["strike"], errors="coerce").to_numpy(float)
        times = pd.to_numeric(expiry["dte"], errors="coerce").to_numpy(float) / 365.0
        sides = expiry["side"].astype(str).str.lower().to_numpy()

        model_delta = black_scholes_delta(
            spot,
            strikes,
            times,
            local_iv,
            sides,
            DEFAULT_RISK_FREE_RATE,
            DEFAULT_DIVIDEND_YIELD,
        )
        vendor_delta = pd.to_numeric(expiry.get("delta"), errors="coerce").to_numpy(float)
        valid_vendor_delta = (
            np.isfinite(vendor_delta)
            & (
                ((sides == "call") & (vendor_delta > 0) & (vendor_delta <= 1))
                | ((sides == "put") & (vendor_delta < 0) & (vendor_delta >= -1))
            )
        )
        valid_model_delta = valid_iv & np.isfinite(model_delta)
        local_delta = np.where(
            valid_vendor_delta,
            vendor_delta,
            np.where(valid_model_delta, model_delta, np.nan),
        )

        # Historical vendor gamma can be null or rounded to zero. Prefer a
        # reproducible Black-Scholes gamma whenever IV is usable; retain vendor
        # gamma as a fallback and preserve its original column separately.
        model_gamma = black_scholes_gamma(
            spot,
            strikes,
            times,
            local_iv,
            DEFAULT_RISK_FREE_RATE,
            DEFAULT_DIVIDEND_YIELD,
        )
        vendor_gamma = pd.to_numeric(expiry.get("gamma"), errors="coerce").to_numpy(float)
        valid_model_gamma = valid_iv & np.isfinite(model_gamma) & (model_gamma > 0)
        valid_vendor_gamma = np.isfinite(vendor_gamma) & (vendor_gamma > 0)
        local_gamma = np.where(
            valid_model_gamma,
            model_gamma,
            np.where(valid_vendor_gamma, vendor_gamma, np.nan),
        )

        for local_position, index in enumerate(expiry.index):
            output_position = position_by_index[index]
            iv_used[output_position] = local_iv[local_position]
            if valid_vendor_iv[local_position]:
                iv_source[output_position] = "vendor"
            elif valid_iv[local_position]:
                iv_source[output_position] = "derived_from_option_price"

            delta_used[output_position] = local_delta[local_position]
            if valid_vendor_delta[local_position]:
                delta_source[output_position] = "vendor"
            elif valid_model_delta[local_position]:
                delta_source[output_position] = "black_scholes_from_iv"

            gamma_used[output_position] = local_gamma[local_position]
            if valid_model_gamma[local_position]:
                gamma_source[output_position] = "black_scholes_from_iv"
            elif valid_vendor_gamma[local_position]:
                gamma_source[output_position] = "vendor"

    frame["iv_used"] = iv_used
    frame["iv_source"] = iv_source
    frame["delta_used"] = delta_used
    frame["delta_source"] = delta_source
    frame["gamma_used"] = gamma_used
    frame["gamma_source"] = gamma_source
    frame["calculation_version"] = CALCULATION_VERSION

    preferred = [
        "symbol",
        "snapshot_date",
        "optionSymbol",
        "expiration",
        "dte",
        "side",
        "strike",
        "underlyingPrice",
        "bid",
        "ask",
        "mid",
        "last",
        "openInterest",
        "volume",
        "iv",
        "iv_used",
        "iv_source",
        "delta",
        "delta_used",
        "delta_source",
        "gamma",
        "gamma_used",
        "gamma_source",
        "theta",
        "vega",
        "calculation_version",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns].reset_index(drop=True)


def archive_chain(
    store: SnapshotStore,
    symbol: str,
    snapshot_date: date,
    chain: pd.DataFrame,
) -> ArchiveResult:
    if not store.enabled:
        raise SnapshotStoreError("Supabase is not configured.")

    archived = prepare_chain_for_archive(symbol, snapshot_date, chain)
    buffer = BytesIO()
    archived.to_parquet(buffer, index=False, compression="zstd")
    payload = buffer.getvalue()
    path = (
        f"{CALCULATION_VERSION}/{snapshot_date:%Y/%m/%d}/"
        f"{symbol.upper()}.parquet"
    )

    headers = {
        "apikey": store.key,
        "Content-Type": "application/vnd.apache.parquet",
        "x-upsert": "true",
    }
    # Legacy service-role JWTs require Bearer auth. New sb_secret_* keys are
    # intentionally sent as API keys only, matching SnapshotStore behavior.
    if store.key and not store.key.startswith(("sb_secret_", "sb_publishable_")):
        headers["Authorization"] = f"Bearer {store.key}"

    response = requests.post(
        f"{store.url}/storage/v1/object/{CHAIN_ARCHIVE_BUCKET}/{path}",
        headers=headers,
        data=payload,
        timeout=max(store.timeout, 60),
    )
    if response.status_code not in {200, 201}:
        raise SnapshotStoreError(
            f"Supabase chain archive upload failed ({response.status_code}): "
            f"{response.text[:300]}"
        )

    return ArchiveResult(
        path=f"{CHAIN_ARCHIVE_BUCKET}/{path}",
        byte_count=len(payload),
        contract_count=len(archived),
    )
