"""Q7 — Top 10 neighbourhoods by total estimated_revenue_l365d per city."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    agg = (
        listings
        .filter(F.col("estimated_revenue_l365d").isNotNull())
        .groupBy("city", "neighbourhood_cleansed")
        .agg(
            F.sum("estimated_revenue_l365d").alias("total_estimated_revenue"),
            F.avg("estimated_revenue_l365d").alias("avg_estimated_revenue"),
            F.count("id").alias("n_listings"),
        )
    )
    w = Window.partitionBy("city").orderBy(F.col("total_estimated_revenue").desc())
    return (
        agg.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= 10)
        .orderBy("city", "rank")
    )
