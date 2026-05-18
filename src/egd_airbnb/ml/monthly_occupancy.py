"""Month-aware occupancy classifier.

Dataset: calendar (listing_id, month, year, available) × listings (features)
aggregated to one row per (listing_id, year, month).

Target: monthly_occupancy_rate > 0.50 → is_high_demand_month
  (lower threshold than the annual model because some months are systematically
   low — 70% monthly would be very high; 50% is a meaningful cutoff)

Extra feature vs the annual model: `month` (1-12) and its cyclic encoding
  sin_month = sin(2π × month/12)
  cos_month = cos(2π × month/12)
These two continuous features encode the cyclical nature of seasonality without
treating month as an arbitrary ordinal.
"""
from __future__ import annotations

import json
import math

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from ..utils.io import read_parquet
from .evaluation import classification_metrics
from .features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_feature_stages,
    compute_train_stats,
    engineer_features,
    select_and_fill,
)

# Availability columns leak target — exclude them.
LEAK = [
    "availability_30", "availability_60", "availability_90", "availability_365",
    "occupancy_rate", "booked_nights_365", "estimated_revenue_365",
]

# Extra temporal features added to the numeric set.
TEMPORAL_NUMERIC = ["month", "sin_month", "cos_month"]


def _build_monthly_dataset(spark: SparkSession) -> DataFrame:
    """Join calendar × listings and aggregate to (listing, year, month)."""
    calendar = read_parquet(spark, PROCESSED_DIR / "calendar").filter(
        F.col("date").isNotNull()
    )
    listings = read_parquet(spark, PROCESSED_DIR / "listings")

    monthly = (
        calendar
        .groupBy("listing_id", "city", "year", "month")
        .agg(
            # fraction of days that were UNavailable (= booked)
            F.avg((F.col("available") == False).cast("double")).alias("monthly_occupancy_rate"),
            F.count("*").alias("n_days"),
        )
        .filter(F.col("n_days") >= 20)          # only months with enough observations
    )

    joined = monthly.join(
        listings.drop("city"),   # city already in monthly via calendar
        monthly.listing_id == listings.id,
        how="inner",
    )

    # Cyclic month encoding.
    joined = (
        joined
        .withColumn("month",     F.col("month").cast("double"))
        .withColumn("sin_month", F.sin(F.col("month") * F.lit(2 * math.pi / 12)))
        .withColumn("cos_month", F.cos(F.col("month") * F.lit(2 * math.pi / 12)))
        .withColumn("is_high_demand_month",
                    (F.col("monthly_occupancy_rate") > 0.50).cast("double"))
    )
    return joined


def train(spark: SparkSession, sample: float | None = None) -> dict:
    df = _build_monthly_dataset(spark)
    if sample:
        df = df.sample(False, sample, seed=42)

    train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)
    train_raw.cache()

    train_stats = compute_train_stats(train_raw)

    # Feature engineering — temporal cols (month, sin_month, cos_month) are already
    # in train_raw and passed through via extra_numeric in select_and_fill.
    extra_num = [c for c in TEMPORAL_NUMERIC if c in train_raw.columns]
    train_fe = engineer_features(train_raw, train_stats)
    test_fe  = engineer_features(test_raw,  train_stats)

    train_df = (select_and_fill(train_fe, leak_cols=LEAK, extra_numeric=extra_num)
                .withColumn("label", F.col("is_high_demand_month")))
    test_df  = (select_and_fill(test_fe,  leak_cols=LEAK, extra_numeric=extra_num)
                .withColumn("label", F.col("is_high_demand_month")))

    avail_num = [c for c in NUMERIC_FEATURES + TEMPORAL_NUMERIC if c in train_df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]

    base_rate = float(train_df.agg(F.avg("label")).first()[0])
    weight_col = None
    if base_rate < 0.35 or base_rate > 0.65:
        train_df = train_df.withColumn(
            "weight",
            F.when(F.col("label") == 1.0, 1.0 / base_rate)
             .otherwise(1.0 / (1.0 - base_rate)),
        )
        test_df = test_df.withColumn("weight", F.lit(1.0))
        weight_col = "weight"

    clf_kwargs: dict = dict(featuresCol="features", labelCol="label",
                            numTrees=150, maxDepth=12, seed=42)
    if weight_col:
        clf_kwargs["weightCol"] = weight_col
    clf = RandomForestClassifier(**clf_kwargs)
    stages = build_feature_stages(categorical=avail_cat, numeric=avail_num) + [clf]
    model: PipelineModel = Pipeline(stages=stages).fit(train_df)
    pred = model.transform(test_df)

    metrics = classification_metrics(pred)
    out = {
        "model": "monthly_occupancy_rf",
        "base_rate": base_rate,
        "n_train": train_raw.count(),
        "n_test":  test_raw.count(),
        "metrics": metrics,
    }

    model_path = MODELS_DIR / "monthly_occupancy_rf"
    model.write().overwrite().save(str(model_path))

    # Save inference stats.
    nb_pd = train_stats["nb_stats"].toPandas()
    inference_stats = {
        "global_mean_log_price": train_stats["global_mean_log_price"],
        "global_mean_count":     train_stats["global_mean_count"],
        "neighbourhood_stats":   nb_pd.set_index("neighbourhood").to_dict(orient="index"),
    }
    (MODELS_DIR / "monthly_occupancy_rf_inference_stats.json").write_text(
        json.dumps(inference_stats, indent=2)
    )

    out_dir = RESULTS_DIR / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "monthly_occupancy_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out
