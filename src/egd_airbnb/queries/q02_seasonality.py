"""Q2 — Median adjusted_price per month per city, joining calendar × listings."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    calendar = (
        read_parquet(spark, PROCESSED_DIR / "calendar")
        .filter(F.col("adjusted_price").isNotNull())
        .filter(F.col("adjusted_price").between(10, 2000))
    )
    listings = read_parquet(spark, PROCESSED_DIR / "listings").select(
        F.col("id").alias("listing_id"), F.col("city").alias("listing_city"), "room_type",
    )

    joined = calendar.join(listings, on="listing_id", how="inner")

    return (
        joined
        .groupBy("city", "year", "month")
        .agg(
            F.expr("percentile_approx(adjusted_price, 0.5)").alias("median_price"),
            F.avg("adjusted_price").alias("mean_price"),
            F.count("*").alias("n_observations"),
        )
        .orderBy("city", "year", "month")
    )
