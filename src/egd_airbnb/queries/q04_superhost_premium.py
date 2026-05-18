"""Q4 — Superhost vs regular host: price, rating, reviews_per_month."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    return (
        listings
        .filter(F.col("price").isNotNull())
        .groupBy("city", "host_is_superhost")
        .agg(
            F.expr("percentile_approx(price, 0.5)").alias("median_price"),
            F.avg("price").alias("mean_price"),
            F.avg("review_scores_rating").alias("mean_rating"),
            F.avg("reviews_per_month").alias("mean_reviews_per_month"),
            F.count("id").alias("n_listings"),
        )
        .orderBy("city", F.col("host_is_superhost").desc())
    )
