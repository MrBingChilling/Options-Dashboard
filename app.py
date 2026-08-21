import sys
from pathlib import Path

# Ensure repo-local modules take precedence over same-named installed packages.
# This is important for the customized lightweight_charts_v5 wrapper used by
# the Historical IV & Skew chart.
ROOT = Path(__file__).resolve().parent
root_text = str(ROOT)
if root_text in sys.path:
    sys.path.remove(root_text)
sys.path.insert(0, root_text)

import streamlit as st

# Production refresh marker: confirmed earnings calendar v2 (2026-08-21).

# Vega/Altair tooltips can remain pinned after a tap on touch devices and cover
# much of the chart. Keep desktop hover tooltips, but suppress them on coarse
# touch screens where ticker/value labels and the details table are available.
st.markdown(
    """
    <style>
    @media (hover: none) and (pointer: coarse) {
      #vg-tooltip-element,
      .vg-tooltip,
      [id^="vg-tooltip"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use explicit navigation rather than Streamlit's legacy automatic pages list.
# IV & Skew is the first/default route; Gamma Profile follows. Do not pass
# text glyphs as icons here: st.Page validates icons as emoji/Material icons,
# and the old glyphs caused the entrypoint to fail before navigation loaded.
pages = [
    st.Page("pages/2_IV_and_Skew.py", title="IV & Skew", default=True),
    st.Page("pages/2_Gamma_Profile.py", title="Gamma Profile"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
