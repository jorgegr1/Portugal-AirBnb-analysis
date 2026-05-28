"""Shared helpers for dashboard pages — caching, paths, lazy Spark."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from egd_airbnb.config import BENCH_DIR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR


@st.cache_data(show_spinner="Loading processed listings…")
def load_listings() -> pd.DataFrame:
    path = PROCESSED_DIR / "listings"
    if not path.exists():
        st.error(f"No processed listings at {path}. Run `make ingest && make clean` first.")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner="Loading neighbourhoods…")
def load_neighbourhoods() -> pd.DataFrame:
    path = PROCESSED_DIR / "neighbourhoods"
    if not path.exists():
        return pd.DataFrame(columns=["city", "neighbourhood_group", "neighbourhood"])
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_query_result(query_id: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{query_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_benchmark_summary() -> pd.DataFrame:
    path = BENCH_DIR / "summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def get_dashboard_spark():
    from egd_airbnb.spark_session import get_spark

    return get_spark("dashboard", master="local[2]")


def latest_price_model_dir() -> Path | None:
    candidates = [p for p in MODELS_DIR.glob("price_*") if not p.suffix == ".json"]
    return sorted(candidates)[-1] if candidates else None


def latest_occupancy_model_dir() -> Path | None:
    candidates = [p for p in MODELS_DIR.glob("occupancy_*") if not p.suffix == ".json"]
    return sorted(candidates)[-1] if candidates else None


def load_inference_stats(model_dir: Path) -> dict:
    """Load the neighbourhood target-encoding stats saved at training time."""
    stats_file = model_dir.parent / f"{model_dir.name}_inference_stats.json"
    if not stats_file.exists():
        return {"global_mean_log_price": 4.6, "global_mean_count": 50.0, "neighbourhood_stats": {}}
    return json.loads(stats_file.read_text())


def engineer_row(
    *,
    city: str,
    room_type: str,
    neighbourhood_group: str,
    neighbourhood: str,
    property_type: str,
    host_is_superhost: str,
    instant_bookable: str,
    minimum_nights: float,
    number_of_reviews: float,
    number_of_reviews_ltm: float,
    reviews_per_month: float,
    calculated_host_listings_count: float,
    availability_365: float,
    latitude: float,
    longitude: float,
    accommodates: float,
    bedrooms: float,
    beds: float,
    bathrooms: float,
    review_scores_rating: float,
    review_scores_cleanliness: float,
    review_scores_location: float,
    review_scores_value: float,
    amenity_count: float,
    estimated_occupancy_l365d: float,
    inference_stats: dict,
) -> dict:
    """Compute all engineered features for a single listing row (pure Python, no Spark)."""
    # Bucket minimum nights exactly as engineer_features() does.
    if minimum_nights == 1:
        min_nights_cat = "one_night"
    elif minimum_nights <= 3:
        min_nights_cat = "weekend"
    elif minimum_nights <= 6:
        min_nights_cat = "week_short"
    else:
        min_nights_cat = "long_stay"

    occupancy_rate = 1.0 - availability_365 / 365.0
    log_reviews = math.log1p(number_of_reviews)
    log_host_listings = math.log1p(calculated_host_listings_count)
    rar = number_of_reviews_ltm / (number_of_reviews + 1.0)

    nb_stats = inference_stats.get("neighbourhood_stats", {}).get(neighbourhood, {})
    nb_mean = nb_stats.get(
        "neighbourhood_mean_log_price", inference_stats.get("global_mean_log_price", 4.6)
    )
    nb_count = nb_stats.get(
        "neighbourhood_listing_count", inference_stats.get("global_mean_count", 50.0)
    )

    return {
        # categorical
        "city": city,
        "room_type": room_type,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood,
        "min_nights_cat": min_nights_cat,
        "property_type": property_type,
        "host_is_superhost": host_is_superhost,
        "instant_bookable": instant_bookable,
        # raw numeric
        "minimum_nights": float(minimum_nights),
        "number_of_reviews": float(number_of_reviews),
        "number_of_reviews_ltm": float(number_of_reviews_ltm),
        "reviews_per_month": float(reviews_per_month),
        "calculated_host_listings_count": float(calculated_host_listings_count),
        "availability_365": float(availability_365),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "accommodates": float(accommodates),
        "bedrooms": float(bedrooms),
        "beds": float(beds),
        "bathrooms": float(bathrooms),
        "review_scores_rating": float(review_scores_rating),
        "review_scores_cleanliness": float(review_scores_cleanliness),
        "review_scores_location": float(review_scores_location),
        "review_scores_value": float(review_scores_value),
        "amenity_count": float(amenity_count),
        "estimated_occupancy_l365d": float(estimated_occupancy_l365d),
        # engineered
        "occupancy_rate": occupancy_rate,
        "log_reviews": log_reviews,
        "log_host_listings": log_host_listings,
        "recent_activity_ratio": rar,
        "neighbourhood_listing_count": float(nb_count),
        "neighbourhood_mean_log_price": float(nb_mean),
    }
