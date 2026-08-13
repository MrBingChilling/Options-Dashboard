import streamlit as st

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

pages = [
    st.Page("pages/1_IV_and_Skew.py", title="IV & Skew", icon="↕", default=True),
    st.Page("pages/2_Gamma_Profile.py", title="Gamma Profile", icon="◈"),
]

st.navigation(pages, position="sidebar").run()
