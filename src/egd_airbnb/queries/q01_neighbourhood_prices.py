"""Q1 — Top 10 most expensive and 10 cheapest neighbourhoods per city (by median price)."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")

    agg = (
        listings
        .groupBy("city", "neighbourhood_group", "neighbourhood")
        .agg(
            F.expr("percentile_approx(price, 0.5)").alias("median_price"),
            F.avg("price").alias("mean_price"),
            F.count("id").alias("n_listings"),
        )
        .filter(F.col("n_listings") >= 10)
    )

    w_top = Window.partitionBy("city").orderBy(F.col("median_price").desc())
    w_bottom = Window.partitionBy("city").orderBy(F.col("median_price").asc())

    top = (
        agg.withColumn("rank", F.row_number().over(w_top))
        .filter(F.col("rank") <= 10)
        .withColumn("bucket", F.lit("top"))
    )
    bottom = (
        agg.withColumn("rank", F.row_number().over(w_bottom))
        .filter(F.col("rank") <= 10)
        .withColumn("bucket", F.lit("bottom"))
    )

    return top.unionByName(bottom).orderBy("city", "bucket", "rank")
