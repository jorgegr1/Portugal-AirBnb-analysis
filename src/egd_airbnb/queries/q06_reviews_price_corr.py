"""Q6 — Pearson correlation between reviews_per_month and price per neighbourhood."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = (
        read_parquet(spark, PROCESSED_DIR / "listings")
        .filter(F.col("price").isNotNull() & F.col("reviews_per_month").isNotNull())
        .filter(F.col("price").between(10, 1000))
    )

    return (
        listings
        .groupBy("city", "neighbourhood_cleansed")
        .agg(
            F.corr("reviews_per_month", "price").alias("corr_rpm_price"),
            F.count("id").alias("n_listings"),
        )
        .filter(F.col("n_listings") >= 50)
        .orderBy(F.col("corr_rpm_price").desc_nulls_last())
    )
