"""High-occupancy classifier: predict `is_high_occupancy` (1 - availability_365/365 > 0.70).

Leaky features excluded:
  - availability_* (target is derived from availability_365)
  - occupancy_rate (= 1 - availability_365/365, same info as target)
  - booked_nights_365 (= 365 - availability_365)
  - estimated_revenue_365 (uses booked_nights_365)
"""
from __future__ import annotations

import json

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from ..utils.io import read_parquet
from .evaluation import classification_metrics
from .features import (
    build_feature_stages,
    compute_train_stats,
    engineer_features,
    select_and_fill,
)

# Columns that would leak the target (derived from availability_365).
LEAK = [
    "availability_30", "availability_60", "availability_90", "availability_365",
    "occupancy_rate",        # engineered from availability_365
    "booked_nights_365",     # engineered from availability_365
    "estimated_revenue_365", # uses booked_nights_365
]

ALGOS = {
    "logreg": lambda wc: LogisticRegression(
        featuresCol="features", labelCol="label", maxIter=100,
        weightCol=wc, family="binomial",
    ),
    "rf": lambda wc: RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=150, maxDepth=12, seed=42, weightCol=wc,
    ),
}


def train(spark: SparkSession, algo: str = "rf", sample: float | None = None) -> dict:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    df = listings.filter(F.col("is_high_occupancy").isNotNull())
    if sample:
        df = df.sample(False, sample, seed=42)

    train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)
    train_raw.cache()

    # Feature engineering (uses train_stats for target encoding).
    train_stats = compute_train_stats(train_raw)
    train_df = select_and_fill(engineer_features(train_raw, train_stats), leak_cols=LEAK)
    test_df  = select_and_fill(engineer_features(test_raw,  train_stats), leak_cols=LEAK)

    train_df = train_df.withColumn("label", F.col("is_high_occupancy").cast("double"))
    test_df  = test_df.withColumn("label",  F.col("is_high_occupancy").cast("double"))

    # Class-imbalance weighting.
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

    from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    avail_num = [c for c in NUMERIC_FEATURES if c in train_df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES if c in train_df.columns]
    estimator = ALGOS[algo](weight_col if weight_col else None)
    # Ensure weightCol is not passed as None (causes NPE in PySpark JVM).
    if weight_col is None and hasattr(estimator, "weightCol") and estimator.isDefined(estimator.weightCol):
        estimator.clear(estimator.weightCol)
    stages = build_feature_stages(categorical=avail_cat, numeric=avail_num) + [estimator]
    model: PipelineModel = Pipeline(stages=stages).fit(train_df)
    pred = model.transform(test_df)

    metrics = classification_metrics(pred)
    out = {
        "algo": algo,
        "base_rate": base_rate,
        "n_train": train_raw.count(),
        "n_test":  test_raw.count(),
        "metrics": metrics,
    }

    model_path = MODELS_DIR / f"occupancy_{algo}"
    model.write().overwrite().save(str(model_path))
    metrics_path = RESULTS_DIR / "ml"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / f"occupancy_{algo}_metrics.json").write_text(json.dumps(out, indent=2))

    # Save inference stats (same neighbourhood target-encoding used at training time).
    nb_pd = train_stats["nb_stats"].toPandas()
    inference_stats = {
        "global_mean_log_price": train_stats["global_mean_log_price"],
        "global_mean_count":     train_stats["global_mean_count"],
        "neighbourhood_stats":   nb_pd.set_index("neighbourhood").to_dict(orient="index"),
    }
    (MODELS_DIR / f"occupancy_{algo}_inference_stats.json").write_text(
        json.dumps(inference_stats, indent=2)
    )

    print(json.dumps(out, indent=2))
    return out


def train_all(spark: SparkSession, sample: float | None = None) -> list[dict]:
    return [train(spark, a, sample=sample) for a in ALGOS]
