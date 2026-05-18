"""Q10 — Share of listings priced above {€500, €1000, €2000} per city × room_type."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet

_THRESHOLDS = (500, 1000, 2000)


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings").filter(F.col("price").isNotNull())

    agg = listings.groupBy("city", "room_type").agg(
        F.count("id").alias("n_listings"),
        *(F.sum((F.col("price") > t).cast("int")).alias(f"n_above_{t}") for t in _THRESHOLDS),
        F.max("price").alias("max_price"),
    )
    for t in _THRESHOLDS:
        agg = agg.withColumn(f"pct_above_{t}", F.col(f"n_above_{t}") / F.col("n_listings"))
    return agg.orderBy("city", "room_type")
