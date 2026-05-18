"""Clean the unified listings DataFrame (summary format, 18 columns).

New Inside Airbnb summary format columns:
    id, name, host_id, host_name, neighbourhood_group, neighbourhood,
    latitude, longitude, room_type, price, minimum_nights, number_of_reviews,
    last_review, reviews_per_month, calculated_host_listings_count,
    availability_365, number_of_reviews_ltm, license

Key changes from the old detailed format:
- price is a plain integer (no "$" prefix) — just cast to DoubleType.
- neighbourhood / neighbourhood_group replace neighbourhood_cleansed.
- No property_type, bedrooms, beds, bathrooms, host_is_superhost, review_scores_*.
- No estimated_revenue_l365d → we derive it as price × booked_nights.
"""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..utils.io import read_parquet, write_parquet

_NUMERIC_COLS = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "number_of_reviews_ltm",
]

_HIGH_OCCUPANCY_THRESHOLD = 0.70   # 1 - availability_365/365 > 0.70


def clean_listings(df: DataFrame) -> DataFrame:
    out = df

    # Numeric casts (price is already a number — just ensure DoubleType).
    for c in _NUMERIC_COLS:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Date fields.
    if "last_review" in out.columns:
        out = out.withColumn("last_review", F.to_date("last_review"))

    # Fills.
    fill_map: dict = {}
    if "reviews_per_month" in out.columns:
        fill_map["reviews_per_month"] = 0.0
    if fill_map:
        out = out.fillna(fill_map)

    # Drop rows with no price (11-13% in the summary format).
    if "price" in out.columns:
        out = out.filter(F.col("price").isNotNull() & (F.col("price") > 0))

    # Dedup.
    if "id" in out.columns:
        out = out.dropDuplicates(["id"])

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
        # Estimated booked nights and revenue proxy.
        out = out.withColumn("booked_nights_365", F.lit(365.0) - F.col("availability_365"))
        if "price" in out.columns:
            out = out.withColumn(
                "estimated_revenue_365",
                F.col("price") * F.col("booked_nights_365"),
            )

    return out


def run(spark, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "listings")
    cleaned = clean_listings(df)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "listings", partition_by="city")
    return cleaned
