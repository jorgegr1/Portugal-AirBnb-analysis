"""Sanity checks for cleaning rules — run on tiny synthetic inputs (no fixtures on disk)."""
from __future__ import annotations

from pyspark.sql import types as T

from egd_airbnb.cleaning.calendar import clean_calendar
from egd_airbnb.cleaning.listings import clean_listings
from egd_airbnb.cleaning.neighbourhoods import clean_neighbourhoods
from egd_airbnb.cleaning.reviews import clean_reviews


def test_listings_price_cast_and_derived_cols(spark):
    """Summary-format listings: price is a plain integer, no $ stripping needed."""
    rows = [
        {"id": "1", "city": "porto", "price": "75",
         "calculated_host_listings_count": "3", "availability_365": "30",
         "reviews_per_month": None, "minimum_nights": "2",
         "neighbourhood_group": "PORTO", "neighbourhood": "Bonfim",
         "latitude": "41.14", "longitude": "-8.61",
         "room_type": "Entire home/apt", "number_of_reviews": "10",
         "number_of_reviews_ltm": "3"},
        {"id": "2", "city": "lisbon", "price": "120",
         "calculated_host_listings_count": "1", "availability_365": "300",
         "reviews_per_month": "1.5", "minimum_nights": "1",
         "neighbourhood_group": "LISBOA", "neighbourhood": "Alfama",
         "latitude": "38.71", "longitude": "-9.14",
         "room_type": "Private room", "number_of_reviews": "50",
         "number_of_reviews_ltm": "12"},
    ]
    df = spark.createDataFrame(rows)
    out = clean_listings(df).orderBy("id").collect()

    assert out[0]["price"] == 75.0
    assert out[1]["price"] == 120.0
    assert out[0]["host_type"] == "multi"
    assert out[1]["host_type"] == "single"
    # availability_365=30 → occupancy ≈ 0.918 → True
    assert out[0]["is_high_occupancy"] is True
    # availability_365=300 → occupancy ≈ 0.178 → False
    assert out[1]["is_high_occupancy"] is False
    # reviews_per_month null fill
    assert out[0]["reviews_per_month"] == 0.0
    # estimated_revenue_365 = price × (365 - availability_365)
    assert out[0]["estimated_revenue_365"] == 75.0 * (365 - 30)
    assert out[1]["estimated_revenue_365"] == 120.0 * (365 - 300)


def test_listings_drops_null_prices(spark):
    rows = [
        {"id": "x", "city": "porto", "price": None,  "availability_365": "100",
         "calculated_host_listings_count": "1"},
        {"id": "y", "city": "porto", "price": "0",   "availability_365": "100",
         "calculated_host_listings_count": "1"},
        {"id": "z", "city": "porto", "price": "50",  "availability_365": "100",
         "calculated_host_listings_count": "1"},
    ]
    df = spark.createDataFrame(rows)
    out = clean_listings(df)
    assert out.count() == 1
    assert out.first()["id"] == "z"


def test_listings_high_occupancy_threshold(spark):
    df = spark.createDataFrame([
        {"id": "x", "city": "porto", "availability_365": "50",
         "price": "100", "calculated_host_listings_count": "1"},
        {"id": "y", "city": "porto", "availability_365": "300",
         "price": "100", "calculated_host_listings_count": "1"},
    ])
    out = {r["id"]: r["is_high_occupancy"] for r in clean_listings(df).collect()}
    assert out["x"] is True   # 1 - 50/365 = 0.86 > 0.70
    assert out["y"] is False  # 1 - 300/365 = 0.18


def test_calendar_drops_null_price_column(spark):
    schema = T.StructType([
        T.StructField("city",           T.StringType()),
        T.StructField("listing_id",     T.StringType()),
        T.StructField("date",           T.StringType()),
        T.StructField("available",      T.StringType()),
        T.StructField("price",          T.StringType()),
        T.StructField("adjusted_price", T.StringType()),
    ])
    df = spark.createDataFrame(
        [("porto", "1", "2025-07-01", "t", None, None)],
        schema=schema,
    )
    out = clean_calendar(df)
    assert "price" not in out.columns
    row = out.first()
    assert row["available"] is True
    assert row["year"] == 2025
    assert row["month"] == 7


def test_reviews_dedup(spark):
    # Summary format only has listing_id + date (no reviewer_id).
    df = spark.createDataFrame([
        {"listing_id": "1", "date": "2024-01-01", "city": "porto"},
        {"listing_id": "1", "date": "2024-01-01", "city": "porto"},
        {"listing_id": "1", "date": "2024-02-01", "city": "porto"},
    ])
    out = clean_reviews(df)
    assert out.count() == 2
    assert out.filter(out.month == 1).count() == 1


def test_neighbourhoods_dedup(spark):
    df = spark.createDataFrame([
        {"city": "porto", "neighbourhood_group": "PORTO", "neighbourhood": "Bonfim"},
        {"city": "porto", "neighbourhood_group": "PORTO", "neighbourhood": "Bonfim"},
        {"city": "porto", "neighbourhood_group": "PORTO", "neighbourhood": "Campanhã"},
    ])
    out = clean_neighbourhoods(df)
    assert out.count() == 2
