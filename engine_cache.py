# SPDX-License-Identifier: MIT
"""Shared, process-wide VGG19 engine cache for all Streamlit pages."""

import streamlit as st

from style_transfer import StyleTransferEngine


@st.cache_resource(show_spinner=False)
def get_engine() -> StyleTransferEngine:
    return StyleTransferEngine()
