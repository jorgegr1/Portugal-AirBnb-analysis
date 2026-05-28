"""Q3 — Average availability (lower = higher demand) by room_type × city."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    return (
        listings.filter(F.col("availability_365").isNotNull())
        .groupBy("city", "room_type")
        .agg(
            F.avg("availability_365").alias("avg_availability_365"),
            (F.lit(1.0) - F.avg("availability_365") / F.lit(365.0)).alias("avg_occupancy_rate"),
            F.count("id").alias("n_listings"),
        )
        .orderBy("city", "avg_occupancy_rate")
    )
