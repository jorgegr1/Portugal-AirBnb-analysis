"""High-occupancy classifier: predict `is_high_occupancy` (availability_365/365 < 0.30)."""
from __future__ import annotations

import json

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from ..utils.io import read_parquet
from .evaluation import classification_metrics
from .features import build_feature_stages, select_features

# Drop availability_* features to avoid leakage (target is derived from availability_365).
LEAK = ["availability_30", "availability_60", "availability_90", "availability_365"]

ALGOS = {
    "logreg": lambda weight_col: LogisticRegression(
        featuresCol="features", labelCol="label", maxIter=50,
        weightCol=weight_col, family="binomial",
    ),
    "rf": lambda weight_col: RandomForestClassifier(
        featuresCol="features", labelCol="label", numTrees=100, maxDepth=10, seed=42,
        weightCol=weight_col,
    ),
}


def _prepare(df: DataFrame) -> DataFrame:
    df = df.filter(F.col("is_high_occupancy").isNotNull())
    feats = select_features(df, leak_cols=LEAK)
    return feats.withColumn("label", F.col("is_high_occupancy").cast("double"))


def train(spark: SparkSession, algo: str = "rf", sample: float | None = None) -> dict:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    df = _prepare(listings)
    if sample:
        df = df.sample(False, sample, seed=42)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    base_rate = float(train_df.agg(F.avg("label")).first()[0])
    weight_col = None
    if base_rate < 0.35 or base_rate > 0.65:
        train_df = train_df.withColumn(
            "weight",
            F.when(F.col("label") == 1.0, F.lit(1.0 / base_rate))
             .otherwise(F.lit(1.0 / (1.0 - base_rate))),
        )
        weight_col = "weight"
        # need same column in test to avoid evaluator complaints (unused there)
        test_df = test_df.withColumn("weight", F.lit(1.0))

    stages = build_feature_stages() + [ALGOS[algo](weight_col)]
    model: PipelineModel = Pipeline(stages=stages).fit(train_df)
    pred = model.transform(test_df)

    metrics = classification_metrics(pred)
    out = {"algo": algo, "n_train": train_df.count(), "n_test": test_df.count(), "metrics": metrics}

    model_path = MODELS_DIR / f"occupancy_{algo}"
    model.write().overwrite().save(str(model_path))
    metrics_path = RESULTS_DIR / "ml"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / f"occupancy_{algo}_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def train_all(spark: SparkSession, sample: float | None = None) -> list[dict]:
    return [train(spark, a, sample=sample) for a in ALGOS]
