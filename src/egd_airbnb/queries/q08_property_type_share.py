"""Q8 — Room-type market share per city (entire home / private room / hotel / shared)."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    counts = listings.groupBy("city", "room_type").agg(F.count("id").alias("n_listings"))
    totals = counts.groupBy("city").agg(F.sum("n_listings").alias("total"))
    return (
        counts.join(totals, "city")
        .withColumn("share", F.col("n_listings") / F.col("total"))
        .drop("total")
        .orderBy("city", F.col("share").desc())
    )
