from __future__ import annotations

from copy import deepcopy
from importlib.metadata import distribution
from pathlib import Path

import streamlit.components.v1 as components


COMPONENT_NAME = "lightweight_charts_v5_component"
_dist = distribution("streamlit-lightweight-charts-v5")
_build_dir = Path(_dist.locate_file("lightweight_charts_v5/frontend/build"))
_component_func = components.declare_component(COMPONENT_NAME, path=str(_build_dir))


def _with_explicit_point_markers(charts):
    patched = deepcopy(charts)
    for pane in patched or []:
        for series in pane.get("series", []):
            if series.get("type") != "Line":
                continue
            data = series.get("data") or []
            if not data:
                continue
            options = series.setdefault("options", {})
            color = options.get("color", "#5B8FF9")
            series["markers"] = [
                {
                    "time": point["time"],
                    "position": "inBar",
                    "color": color,
                    "shape": "circle",
                    "text": "",
                    "size": 0.6,
                }
                for point in data
                if point.get("time") is not None and point.get("value") is not None
            ]
            options["pointMarkersVisible"] = False
            if len(data) == 1:
                options["color"] = "rgba(0,0,0,0)"
                options["lineVisible"] = False
                options["lastValueVisible"] = False
                options["title"] = ""
                options["crosshairMarkerVisible"] = False
    return patched


def lightweight_charts_v5_component(
    name,
    data=None,
    charts=None,
    height: int = 400,
    take_screenshot: bool = False,
    zoom_level: int = 200,
    fonts=None,
    configure_time_scale: bool = False,
    key=None,
):
    default_value = None if take_screenshot else 0
    if charts is not None:
        return _component_func(
            name=name,
            charts=_with_explicit_point_markers(charts),
            height=height,
            take_screenshot=take_screenshot,
            zoom_level=zoom_level,
            fonts=fonts or [],
            key=key,
            configureTimeScale=configure_time_scale,
            default=default_value,
        )
    return _component_func(
        name=name,
        data=data,
        height=height,
        take_screenshot=take_screenshot,
        zoom_level=zoom_level,
        key=key,
        configureTimeScale=configure_time_scale,
        default=default_value,
    )
