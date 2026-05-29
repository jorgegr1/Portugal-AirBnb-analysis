"""Q2 — Availability seasonality: % of listings available per month per city.

Calendar prices are null in the current snapshot, so we use the `available` flag
as the demand signal. Low availability → high demand. We also enrich with the
listing price to compute an estimated monthly revenue proxy.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    calendar = (
        read_parquet(spark, PROCESSED_DIR / "calendar")
        .select("listing_id", "city", "date", "available", "year", "month")
        .filter(F.col("date").isNotNull())
    )
    listings = read_parquet(spark, PROCESSED_DIR / "listings").select(
        F.col("id").alias("listing_id"),
        "price",
        "room_type",
        F.col("city").alias("listing_city"),
    )

    joined = calendar.join(listings, on="listing_id", how="inner")

    return (
        joined.groupBy("city", "year", "month")
        .agg(
            F.avg(F.col("available").cast("int")).alias("availability_rate"),
            (F.lit(1.0) - F.avg(F.col("available").cast("int"))).alias("occupancy_rate"),
            F.count("*").alias("n_calendar_days"),
            # Revenue proxy: listing price × booked days
            F.sum(F.when(~F.col("available"), F.col("price")).otherwise(F.lit(0.0))).alias(
                "estimated_revenue_proxy"
            ),
        )
        .orderBy("city", "year", "month")
    )
