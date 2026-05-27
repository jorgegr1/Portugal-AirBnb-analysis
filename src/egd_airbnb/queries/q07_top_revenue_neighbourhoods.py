"""Q7 — Top 10 neighbourhoods by estimated annual revenue per city.

estimated_revenue_365 = price × (365 − availability_365) is derived during cleaning.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    agg = (
        listings
        .filter(F.col("estimated_revenue_365").isNotNull())
        .groupBy("city", "neighbourhood_group_cleansed", "neighbourhood_cleansed")
        .agg(
            F.sum("estimated_revenue_365").alias("total_estimated_revenue"),
            F.avg("estimated_revenue_365").alias("avg_estimated_revenue"),
            F.expr("percentile_approx(price, 0.5)").alias("median_price"),
            F.count("id").alias("n_listings"),
        )
    )
    w = Window.partitionBy("city").orderBy(F.col("total_estimated_revenue").desc())
    return (
        agg.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= 10)
        .orderBy("city", "rank")
    )
