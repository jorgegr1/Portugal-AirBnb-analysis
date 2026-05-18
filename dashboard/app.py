"""Streamlit entry point. Multi-page app; pages live under dashboard/pages/."""
from __future__ import annotations

import streamlit as st

from egd_airbnb.config import PROCESSED_DIR

st.set_page_config(page_title="EGD-AirBnB", page_icon="🏠", layout="wide")

st.title("EGD — AirBnB (Porto · Lisbon)")
st.markdown(
    """
This dashboard explores the analytical and ML outputs of the EGD-AirBnB Spark pipeline.

**Pages** (sidebar):
1. *City Overview* — headline metrics per city.
2. *Price Explorer* — interactive price filtering.
3. *Map View* — geographic distribution of listings.
4. *Query Results* — Q1–Q10 results.
5. *Price Predictor* — live ML inference.

Data is read from `data/processed/` (cleaned listings/calendar/reviews) and
`reports/results/` (query outputs + ML metrics).
"""
)
st.caption(f"Data root: `{PROCESSED_DIR}`")
