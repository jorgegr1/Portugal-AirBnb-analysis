"""Q5 — Pareto: cumulative share of listings held by the top % of hosts per city."""
from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..config import PROCESSED_DIR
from ..utils.io import read_parquet


def run(spark: SparkSession) -> DataFrame:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")

    per_host = (
        listings
        .groupBy("city", "host_id")
        .agg(F.count("id").alias("n_listings"))
    )

    w = Window.partitionBy("city").orderBy(F.col("n_listings").desc())
    totals = per_host.groupBy("city").agg(F.sum("n_listings").alias("total_listings"))

    ranked = (
        per_host
        .withColumn("rank", F.row_number().over(w))
        .withColumn("cum_listings", F.sum("n_listings").over(w.rowsBetween(Window.unboundedPreceding, 0)))
        .join(totals, "city")
        .withColumn("host_pct",     F.col("rank") / F.count("host_id").over(Window.partitionBy("city")))
        .withColumn("listings_pct", F.col("cum_listings") / F.col("total_listings"))
    )

    # Sample points at host-percentile deciles for plotting.
    deciles = [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]
    rows = []
    for d in deciles:
        rows.append(
            ranked
            .filter(F.col("host_pct") <= d)
            .groupBy("city")
            .agg(F.max("listings_pct").alias("listings_share"))
            .withColumn("host_share", F.lit(d))
        )
    out = rows[0]
    for r in rows[1:]:
        out = out.unionByName(r)
    return out.orderBy("city", "host_share")
