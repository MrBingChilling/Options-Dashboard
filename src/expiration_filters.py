from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


FILTER_21_60 = "21–60 DTE"
FILTER_61_120 = "61–120 DTE"
FILTER_121_240 = "121–240 DTE"
FILTER_241_365 = "241–365 DTE"
FILTER_OVER_ONE_YEAR = "Over one year"
FILTER_MONTHLY = "Monthly expirations only"
FILTER_ALL = "All expirations"
FILTER_CUSTOM = "Custom DTE range"

EXPIRATION_FILTERS = (
    FILTER_21_60,
    FILTER_61_120,
    FILTER_121_240,
    FILTER_241_365,
    FILTER_OVER_ONE_YEAR,
    FILTER_MONTHLY,
    FILTER_ALL,
)

EXPIRATION_CHOICES = (*EXPIRATION_FILTERS, FILTER_CUSTOM)

# The slider covers the practical listed-equity/ETF LEAPS horizon. The
# dedicated "Over one year" and "All expirations" presets remain open-ended.
CUSTOM_DTE_MAX = 1095

PRESET_DTE_RANGES = {
    FILTER_21_60: (21, 60),
    FILTER_61_120: (61, 120),
    FILTER_121_240: (121, 240),
    FILTER_241_365: (241, 365),
}


@dataclass(frozen=True)
class ExpirationSelection:
    label: str
    min_dte: int
    max_dte: int
    monthly_only: bool = False
    complete_chain: bool = False

    def request_params(self, snapshot_date: date) -> dict[str, str | int]:
        params: dict[str, str | int] = {}
        if self.complete_chain or self.monthly_only:
            params["expiration"] = "all"
        if self.monthly_only:
            params["monthly"] = "true"
        if not self.complete_chain and not self.monthly_only:
            params["from"] = (snapshot_date + timedelta(days=self.min_dte)).isoformat()
            if self.max_dte < 3650:
                params["to"] = (snapshot_date + timedelta(days=self.max_dte)).isoformat()
        return params


def resolve_expiration_filter(label: str) -> ExpirationSelection:
    selections = {
        FILTER_21_60: ExpirationSelection(FILTER_21_60, 21, 60),
        FILTER_61_120: ExpirationSelection(FILTER_61_120, 61, 120),
        FILTER_121_240: ExpirationSelection(FILTER_121_240, 121, 240),
        FILTER_241_365: ExpirationSelection(FILTER_241_365, 241, 365),
        FILTER_OVER_ONE_YEAR: ExpirationSelection(FILTER_OVER_ONE_YEAR, 366, 3650),
        FILTER_MONTHLY: ExpirationSelection(
            FILTER_MONTHLY,
            0,
            3650,
            monthly_only=True,
        ),
        FILTER_ALL: ExpirationSelection(FILTER_ALL, 0, 3650, complete_chain=True),
    }
    try:
        return selections[label]
    except KeyError as exc:
        raise ValueError(f"Unknown expiration filter: {label}") from exc


def custom_expiration_selection(min_dte: int, max_dte: int) -> ExpirationSelection:
    if min_dte < 0 or max_dte < min_dte:
        raise ValueError("Maximum DTE must be greater than or equal to minimum DTE.")
    label = f"Custom {min_dte} DTE" if min_dte == max_dte else f"Custom {min_dte}–{max_dte} DTE"
    return ExpirationSelection(label, min_dte, max_dte)
