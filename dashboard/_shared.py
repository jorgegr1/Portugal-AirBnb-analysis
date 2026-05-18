"""Shared helpers for dashboard pages — caching, paths, lazy Spark."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from egd_airbnb.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR


@st.cache_data(show_spinner="Loading processed listings…")
def load_listings() -> pd.DataFrame:
    path = PROCESSED_DIR / "listings"
    if not path.exists():
        st.error(f"No processed listings at {path}. Run `make clean` first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_query_result(query_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{query_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def get_dashboard_spark():
    """Lazy Spark for the model-inference page only."""
    from egd_airbnb.spark_session import get_spark
    return get_spark("dashboard", master="local[2]")


def latest_price_model_dir() -> Path | None:
    candidates = sorted((MODELS_DIR).glob("price_*"))
    return candidates[-1] if candidates else None
