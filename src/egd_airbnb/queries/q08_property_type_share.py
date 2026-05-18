"""Q8 — Property-type market share per city (entire home / private / hotel / shared)."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    counts = (
        listings
        .groupBy("city", "room_type")
        .agg(F.count("id").alias("n_listings"))
    )
    totals = counts.groupBy("city").agg(F.sum("n_listings").alias("total"))
    return (
        counts.join(totals, "city")
        .withColumn("share", F.col("n_listings") / F.col("total"))
        .orderBy("city", F.col("share").desc())
        .drop("total")
    )
