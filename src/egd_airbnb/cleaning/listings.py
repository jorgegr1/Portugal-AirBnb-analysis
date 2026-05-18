"""Clean the unified listings DataFrame.

Implements the rules established by the EDA (notebooks/01_eda.ipynb):
- Price cast from "$1,234.00" → DoubleType.
- Booleans "t"/"f" → BooleanType.
- Drop fully-null columns (e.g. calendar_updated).
- Replace `neighbourhood` with `neighbourhood_cleansed` (45% null on the former).
- Fill superhost / reviews_per_month sensible defaults.
- Derive host_type and is_high_occupancy.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..utils.io import read_parquet, write_parquet

_NUMERIC_COLS = [
    "accommodates", "bedrooms", "beds", "minimum_nights", "maximum_nights",
    "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d",
    "reviews_per_month", "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
    "estimated_occupancy_l365d", "estimated_revenue_l365d",
    "latitude", "longitude",
]

_BOOL_COLS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified",
              "instant_bookable", "has_availability"]

_DATE_COLS = ["host_since", "first_review", "last_review", "last_scraped"]

_HIGH_OCCUPANCY_THRESHOLD = 0.70  # is_high_occupancy ↔ availability_365/365 < 0.30


def _parse_price(col: str) -> "Column":  # type: ignore[name-defined]
    return F.regexp_replace(F.col(col), r"[\$,]", "").cast(T.DoubleType())


def _parse_bool(col: str) -> "Column":  # type: ignore[name-defined]
    return F.when(F.col(col) == "t", True).when(F.col(col) == "f", False).otherwise(None)


def clean_listings(df: DataFrame) -> DataFrame:
    # Drop fully-null columns (EDA confirmed calendar_updated is 100% null).
    drop_cols = [c for c in ("calendar_updated",) if c in df.columns]
    df = df.drop(*drop_cols)

    out = df
    if "price" in out.columns:
        out = out.withColumn("price", _parse_price("price"))
    for c in _NUMERIC_COLS:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast(T.DoubleType()))
    for c in _BOOL_COLS:
        if c in out.columns:
            out = out.withColumn(c, _parse_bool(c))
    for c in _DATE_COLS:
        if c in out.columns:
            out = out.withColumn(c, F.to_date(c))

    # Fills based on EDA missing-value profile. Defensive about column presence so a
    # future snapshot missing one of these does not fail the whole pipeline.
    fill_map: dict = {}
    if "host_is_superhost"  in out.columns: fill_map["host_is_superhost"]  = False
    if "reviews_per_month"  in out.columns: fill_map["reviews_per_month"]  = 0.0
    if fill_map:
        out = out.fillna(fill_map)
    if "id" in out.columns:
        out = out.dropDuplicates(["id"])

    # Prefer the cleansed neighbourhood label (45% null on the raw one).
    if "neighbourhood_cleansed" in out.columns:
        out = out.withColumn("neighbourhood", F.col("neighbourhood_cleansed"))

    # Derived columns.
    if "calculated_host_listings_count" in out.columns:
        out = out.withColumn(
            "host_type",
            F.when(F.col("calculated_host_listings_count") > 1, "multi").otherwise("single"),
        )
    if "availability_365" in out.columns:
        out = out.withColumn(
            "is_high_occupancy",
            (F.lit(1.0) - F.col("availability_365") / F.lit(365.0)) > F.lit(_HIGH_OCCUPANCY_THRESHOLD),
        )

    return out


def run(spark, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "listings")
    cleaned = clean_listings(df)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "listings", partition_by="city")
    return cleaned
