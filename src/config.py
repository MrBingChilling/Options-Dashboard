from __future__ import annotations

import os
from typing import Any


def get_setting(name: str, default: Any = None) -> Any:
    """Read an environment variable first, then a Streamlit secret."""
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except (FileNotFoundError, RuntimeError, KeyError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
