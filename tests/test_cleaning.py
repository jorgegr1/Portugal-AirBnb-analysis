"""Sanity checks for cleaning rules — run on tiny synthetic inputs (no fixtures on disk)."""
from __future__ import annotations

from pyspark.sql import types as T

from egd_airbnb.cleaning.calendar import clean_calendar
from egd_airbnb.cleaning.listings import clean_listings
from egd_airbnb.cleaning.reviews import clean_reviews


def test_listings_price_parses_and_booleans(spark):
    rows = [
        {"id": "1", "city": "porto", "price": "$1,250.00", "host_is_superhost": "t",
         "calculated_host_listings_count": "3", "availability_365": "30",
         "neighbourhood": None, "neighbourhood_cleansed": "Centro",
         "reviews_per_month": None},
        {"id": "2", "city": "lisbon", "price": "$99.50", "host_is_superhost": "f",
         "calculated_host_listings_count": "1", "availability_365": "365",
         "neighbourhood": "Old", "neighbourhood_cleansed": "Alfama",
         "reviews_per_month": "0.5"},
    ]
    df = spark.createDataFrame(rows)
    out = clean_listings(df).orderBy("id").collect()

    assert out[0]["price"] == 1250.0
    assert out[1]["price"] == 99.5
    assert out[0]["host_is_superhost"] is True
    assert out[1]["host_is_superhost"] is False
    assert out[0]["host_type"] == "multi"
    assert out[1]["host_type"] == "single"
    # availability_365=30 → occupancy ≈ 0.918 → True; availability_365=365 → 0.0 → False
    assert out[0]["is_high_occupancy"] is True
    assert out[1]["is_high_occupancy"] is False
    # reviews_per_month null fill
    assert out[0]["reviews_per_month"] == 0.0
    # neighbourhood replaced with cleansed value
    assert out[0]["neighbourhood"] == "Centro"


def test_listings_high_occupancy_threshold(spark):
    df = spark.createDataFrame([
        {"id": "x", "city": "porto", "availability_365": "50",
         "price": "$100.00", "calculated_host_listings_count": "1"},
        {"id": "y", "city": "porto", "availability_365": "300",
         "price": "$100.00", "calculated_host_listings_count": "1"},
    ])
    out = {r["id"]: r["is_high_occupancy"] for r in clean_listings(df).collect()}
    assert out["x"] is True   # 1 - 50/365 = 0.86 > 0.70
    assert out["y"] is False  # 1 - 300/365 = 0.18


def test_calendar_drops_null_price_column(spark):
    # Explicit schema — Spark can't infer the type of an all-null column.
    schema = T.StructType([
        T.StructField("city",           T.StringType()),
        T.StructField("listing_id",     T.StringType()),
        T.StructField("date",           T.StringType()),
        T.StructField("available",      T.StringType()),
        T.StructField("price",          T.StringType()),
        T.StructField("adjusted_price", T.StringType()),
    ])
    df = spark.createDataFrame(
        [("porto", "1", "2025-07-01", "t", None, "$120.00")],
        schema=schema,
    )
    out = clean_calendar(df)
    assert "price" not in out.columns
    row = out.first()
    assert row["adjusted_price"] == 120.0
    assert row["available"] is True
    assert row["year"]  == 2025
    assert row["month"] == 7


def test_reviews_dedup(spark):
    df = spark.createDataFrame([
        {"listing_id": "1", "reviewer_id": "10", "date": "2024-01-01", "city": "porto", "comments": "a"},
        {"listing_id": "1", "reviewer_id": "10", "date": "2024-01-01", "city": "porto", "comments": "b"},
    ])
    assert clean_reviews(df).count() == 1
