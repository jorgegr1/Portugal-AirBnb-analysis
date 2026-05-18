"""Q9 — Monthly review count + YoY growth per city (2018+ to focus on stable period)."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    reviews = read_parquet(spark, PROCESSED_DIR / "reviews").filter(F.col("year") >= 2018)

    monthly = (
        reviews
        .groupBy("city", "year", "month")
        .agg(F.count("*").alias("n_reviews"))
    )

    # YoY: same (city, month), previous year.
    w = Window.partitionBy("city", "month").orderBy("year")
    return (
        monthly
        .withColumn("prev_year_reviews", F.lag("n_reviews").over(w))
        .withColumn(
            "yoy_growth",
            F.when(
                F.col("prev_year_reviews").isNotNull() & (F.col("prev_year_reviews") > 0),
                (F.col("n_reviews") - F.col("prev_year_reviews")) / F.col("prev_year_reviews"),
            ),
        )
        .orderBy("city", "year", "month")
    )
