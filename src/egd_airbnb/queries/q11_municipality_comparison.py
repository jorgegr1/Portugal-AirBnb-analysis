"""Q11 — Municipality-level comparison using neighbourhood_group.

The neighbourhoods.csv provides the full municipality → neighbourhood hierarchy.
This query aggregates listings by neighbourhood_group (municipality) to compare
Porto Metro vs Lisbon Metro sub-regions: listing density, price, and occupancy.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")

    # Municipality stats from listings (neighbourhood_group_cleansed in processed listings).
    municipality = (
        listings
        .groupBy("city", "neighbourhood_group_cleansed")
        .agg(
            F.count("id").alias("n_listings"),
            F.countDistinct("host_id").alias("n_hosts"),
            F.expr("percentile_approx(price, 0.5)").alias("median_price"),
            F.avg("price").alias("mean_price"),
            (F.lit(1.0) - F.avg("availability_365") / F.lit(365.0)).alias("avg_occupancy_rate"),
            F.avg("reviews_per_month").alias("avg_reviews_per_month"),
            F.avg("minimum_nights").alias("avg_minimum_nights"),
        )
    )

    # Join full neighbourhood count from the reference table.
    nb = (
        read_parquet(spark, PROCESSED_DIR / "neighbourhoods")
        .groupBy("city", F.col("neighbourhood_group").alias("neighbourhood_group_cleansed"))
        .agg(F.countDistinct("neighbourhood").alias("n_neighbourhoods_in_municipality"))
    )

    return (
        municipality
        .join(nb, ["city", "neighbourhood_group_cleansed"], "left")
        .withColumn("listings_per_neighbourhood", F.col("n_listings") / F.col("n_neighbourhoods_in_municipality"))
        .orderBy("city", F.col("n_listings").desc())
    )
