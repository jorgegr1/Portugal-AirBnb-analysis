"""Q4 — Multi-listing vs single-listing host comparison.

host_is_superhost is not in the summary format.
We compare multi-listing hosts (professional operators) vs single-listing hosts
on price, occupancy, and review activity.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    return (
        listings.groupBy("city", "host_type")
        .agg(
            F.expr("percentile_approx(price, 0.5)").alias("median_price"),
            F.avg("price").alias("mean_price"),
            F.avg("reviews_per_month").alias("mean_reviews_per_month"),
            (F.lit(1.0) - F.avg("availability_365") / F.lit(365.0)).alias("avg_occupancy_rate"),
            F.count("id").alias("n_listings"),
            F.countDistinct("host_id").alias("n_hosts"),
        )
        .orderBy("city", "host_type")
    )
