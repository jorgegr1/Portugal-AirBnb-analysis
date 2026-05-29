"""Q10 — Share of listings priced above {€100, €250, €500} per city × room_type.

Thresholds adjusted down from old detailed format because the summary dataset
prices are plain integers (already clean) and the distribution is different.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet

_THRESHOLDS = (100, 250, 500)


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")

    agg = listings.groupBy("city", "room_type").agg(
        F.count("id").alias("n_listings"),
        F.expr("percentile_approx(price, 0.5)").alias("median_price"),
        F.max("price").alias("max_price"),
        *(F.sum((F.col("price") > t).cast("int")).alias(f"n_above_{t}") for t in _THRESHOLDS),
    )
    for t in _THRESHOLDS:
        agg = agg.withColumn(f"pct_above_{t}", F.col(f"n_above_{t}") / F.col("n_listings"))
    return agg.orderBy("city", "room_type")
