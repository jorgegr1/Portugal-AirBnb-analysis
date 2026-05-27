"""Standalone benchmark script for Dataproc.

Usage (submitted via gcloud dataproc jobs submit pyspark):
    python benchmark_dataproc.py --workload query_seasonality --num-workers 2 --runs 3

Reads data from GCS (DATA_ROOT env var), prints timing results to stdout.
Results are collected manually into reports/benchmarks/.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from pyspark.sql import SparkSession


def get_spark(app_name: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


# ---------------------------------------------------------------------------
# Workload 1: query_seasonality (calendar x listings join — ~12M rows)
# ---------------------------------------------------------------------------
def workload_query_seasonality(spark: SparkSession, data_root: str) -> None:
    """Q2 — calendar x listings join, aggregation by city/year/month."""
    from pyspark.sql import functions as F

    calendar = (
        spark.read.parquet(f"{data_root}/processed/calendar")
        .select("listing_id", "city", "date", "available", "year", "month")
        .filter(F.col("date").isNotNull())
    )
    listings = spark.read.parquet(f"{data_root}/processed/listings").select(
        F.col("id").alias("listing_id"), "price", "room_type",
        F.col("city").alias("listing_city"),
    )
    joined = calendar.join(listings, on="listing_id", how="inner")
    result = (
        joined
        .groupBy("city", "year", "month")
        .agg(
            F.avg((F.col("available") == True).cast("int")).alias("availability_rate"),
            F.count("*").alias("n_calendar_days"),
        )
        .orderBy("city", "year", "month")
    )
    n = result.count()
    print(f"[workload] query_seasonality produced {n} rows")


# ---------------------------------------------------------------------------
# Workload 2: train_rf (ML pipeline on listings — ~35k rows)
# ---------------------------------------------------------------------------
def workload_train_rf(spark: SparkSession, data_root: str) -> None:
    """Train a Random Forest price regression model."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as F

    listings = spark.read.parquet(f"{data_root}/processed/listings")

    numeric_cols = [
        "accommodates", "bedrooms", "beds", "minimum_nights",
        "number_of_reviews", "reviews_per_month", "review_scores_rating",
        "latitude", "longitude",
    ]
    df = listings.select(*numeric_cols, "room_type", "city", "price")
    df = df.withColumn("price", F.col("price").cast("double"))
    for c in numeric_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    df = df.dropna(subset=["price"]).filter(
        (F.col("price") >= 10) & (F.col("price") <= 1000)
    ).fillna(0)

    df = df.withColumn("log_price", F.log1p("price"))
    n_train = df.count()
    print(f"[workload] train_rf: {n_train} training rows")

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in ["room_type", "city"]
    ]
    feature_cols = numeric_cols + [f"{c}_idx" for c in ["room_type", "city"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    rf = RandomForestRegressor(
        featuresCol="features", labelCol="log_price",
        numTrees=50, maxDepth=8, seed=42,
    )
    pipeline = Pipeline(stages=indexers + [assembler, rf])

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train_df)
    preds = model.transform(test_df)

    evaluator = RegressionEvaluator(labelCol="log_price", predictionCol="prediction")
    rmse = evaluator.setMetricName("rmse").evaluate(preds)
    r2 = evaluator.setMetricName("r2").evaluate(preds)
    print(f"[workload] train_rf: RMSE={rmse:.4f}, R2={r2:.4f}")


# ---------------------------------------------------------------------------
# Workload 3: review_demand (reviews x listings join — millions of rows)
# ---------------------------------------------------------------------------
def workload_review_demand(spark: SparkSession, data_root: str) -> None:
    """Reviews x listings join: monthly review volume by price range and city."""
    from pyspark.sql import functions as F
    from pyspark.sql import Window

    reviews = spark.read.parquet(f"{data_root}/processed/reviews")
    n_reviews = reviews.count()
    print(f"[workload] review_demand: {n_reviews} reviews loaded")

    listings = spark.read.parquet(f"{data_root}/processed/listings").select(
        F.col("id").alias("listing_id"),
        "price", "room_type", "neighbourhood_cleansed",
        F.col("city").alias("listing_city"),
    )

    # Join reviews with listings to get price and room_type per review
    joined = reviews.join(listings, on="listing_id", how="inner")

    # Create price buckets
    joined = joined.withColumn(
        "price_range",
        F.when(F.col("price") < 30, "budget_<30")
         .when(F.col("price") < 75, "mid_30-75")
         .when(F.col("price") < 150, "upper_75-150")
         .otherwise("luxury_150+")
    )

    # Aggregate: monthly review count by city, price range, room type
    result = (
        joined
        .groupBy("city", "year", "month", "price_range", "room_type")
        .agg(
            F.count("*").alias("n_reviews"),
            F.countDistinct("listing_id").alias("n_unique_listings"),
            F.avg("price").alias("avg_price"),
        )
        .orderBy("city", "year", "month")
    )
    n = result.count()
    print(f"[workload] review_demand produced {n} rows")


# ---------------------------------------------------------------------------
# Workload 4: calendar_revenue (heavy calendar aggregation — ~12M rows)
# ---------------------------------------------------------------------------
def workload_calendar_revenue(spark: SparkSession, data_root: str) -> None:
    """Calendar x listings: estimated revenue per neighbourhood per month."""
    from pyspark.sql import functions as F
    from pyspark.sql import Window

    calendar = spark.read.parquet(f"{data_root}/processed/calendar")
    n_cal = calendar.count()
    print(f"[workload] calendar_revenue: {n_cal} calendar rows loaded")

    listings = spark.read.parquet(f"{data_root}/processed/listings").select(
        F.col("id").alias("listing_id"),
        "price", "room_type", "neighbourhood_cleansed",
        F.col("city").alias("listing_city"),
    )

    # Join calendar with listings to get price per day
    joined = calendar.join(listings, on="listing_id", how="inner")

    # Estimate revenue: price x booked days (available == false)
    joined = joined.withColumn(
        "daily_revenue",
        F.when(F.col("available") == False, F.col("price")).otherwise(F.lit(0.0))
    )

    # Aggregate by city, neighbourhood, year, month
    monthly = (
        joined
        .groupBy("city", "neighbourhood_cleansed", "year", "month")
        .agg(
            F.sum("daily_revenue").alias("total_revenue"),
            F.avg("daily_revenue").alias("avg_daily_revenue"),
            F.count("*").alias("n_days"),
            F.sum((F.col("available") == False).cast("int")).alias("booked_days"),
            F.sum((F.col("available") == True).cast("int")).alias("available_days"),
            F.countDistinct("listing_id").alias("n_listings"),
        )
    )

    # Add occupancy rate
    monthly = monthly.withColumn(
        "occupancy_rate",
        F.col("booked_days") / (F.col("booked_days") + F.col("available_days"))
    )

    # Rank neighbourhoods by revenue within each city+month
    w = Window.partitionBy("city", "year", "month").orderBy(F.col("total_revenue").desc())
    ranked = monthly.withColumn("revenue_rank", F.row_number().over(w))

    n = ranked.count()
    print(f"[workload] calendar_revenue produced {n} rows")


# ---------------------------------------------------------------------------
# Workload 5: full_pipeline (all 3 tables joined — heaviest workload)
# ---------------------------------------------------------------------------
def workload_full_pipeline(spark: SparkSession, data_root: str) -> None:
    """Three-way join: calendar x listings x reviews aggregate."""
    from pyspark.sql import functions as F

    calendar = spark.read.parquet(f"{data_root}/processed/calendar")
    listings = spark.read.parquet(f"{data_root}/processed/listings")
    reviews = spark.read.parquet(f"{data_root}/processed/reviews")

    print(f"[workload] full_pipeline: calendar={calendar.count()}, "
          f"listings={listings.count()}, reviews={reviews.count()}")

    # Step 1: monthly occupancy per listing from calendar
    occupancy = (
        calendar
        .groupBy("listing_id", "city", "year", "month")
        .agg(
            F.sum((F.col("available") == False).cast("int")).alias("booked_days"),
            F.count("*").alias("total_days"),
        )
        .withColumn("occupancy_rate", F.col("booked_days") / F.col("total_days"))
    )

    # Step 2: monthly review count per listing from reviews
    review_counts = (
        reviews
        .groupBy("listing_id", F.year("date").alias("year"), F.month("date").alias("month"))
        .agg(F.count("*").alias("n_reviews_month"))
    )

    # Step 3: join occupancy + review_counts + listing attributes
    listings_slim = listings.select(
        F.col("id").alias("listing_id"),
        "price", "room_type", "neighbourhood_cleansed",
        F.col("city").alias("listing_city"),
    )

    combined = (
        occupancy
        .join(listings_slim, on="listing_id", how="inner")
        .join(review_counts, on=["listing_id", "year", "month"], how="left")
    )
    combined = combined.fillna({"n_reviews_month": 0})

    # Step 4: aggregate by city, room_type, month
    result = (
        combined
        .groupBy("city", "room_type", "year", "month")
        .agg(
            F.avg("occupancy_rate").alias("avg_occupancy"),
            F.avg("price").alias("avg_price"),
            F.sum("n_reviews_month").alias("total_reviews"),
            F.countDistinct("listing_id").alias("n_listings"),
            F.avg("booked_days").alias("avg_booked_days"),
        )
        .orderBy("city", "room_type", "year", "month")
    )
    n = result.count()
    print(f"[workload] full_pipeline produced {n} rows")


WORKLOADS = {
    "query_seasonality": workload_query_seasonality,
    "train_rf": workload_train_rf,
    "review_demand": workload_review_demand,
    "calendar_revenue": workload_calendar_revenue,
    "full_pipeline": workload_full_pipeline,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workload", required=True, choices=list(WORKLOADS))
    p.add_argument("--num-workers", type=int, required=True, help="Label for CSV output")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "gs://egd-airbnb-bucket/data"))
    args = p.parse_args()

    # On YARN (Dataproc) we must reuse a single SparkSession for all runs.
    spark = get_spark(f"bench-{args.workload}-w{args.num_workers}")
    results = []
    for i in range(args.runs):
        # Clear cached data between runs so each run starts fresh.
        spark.catalog.clearCache()
        t0 = time.perf_counter()
        WORKLOADS[args.workload](spark, args.data_root)
        elapsed = time.perf_counter() - t0
        results.append({"run": i, "wall_secs": round(elapsed, 4)})
        print(f"[bench] workers={args.num_workers} run={i} {args.workload} -> {elapsed:.2f}s")
    spark.stop()

    # Print summary as JSON for easy parsing
    summary = {
        "workload": args.workload,
        "platform": "dataproc",
        "num_workers": args.num_workers,
        "runs": results,
    }
    print(f"\n===BENCHMARK_RESULT===\n{json.dumps(summary, indent=2)}\n===END===")


if __name__ == "__main__":
    main()
