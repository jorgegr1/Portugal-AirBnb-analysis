"""Price regression: predict log1p(price), report RMSE/MAE/R² back on €."""
from __future__ import annotations

import json
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from ..utils.io import read_parquet
from .evaluation import regression_metrics, regression_metrics_eur
from .features import build_feature_stages, select_features

ALGOS = {
    "lr":  lambda: LinearRegression(featuresCol="features", labelCol="label"),
    "rf":  lambda: RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=100, maxDepth=10, seed=42),
    "gbt": lambda: GBTRegressor(featuresCol="features", labelCol="label", maxIter=100, maxDepth=6, seed=42),
}


def _prepare(df: DataFrame) -> DataFrame:
    """Filter implausible prices, add log-target."""
    feats = select_features(df.filter(F.col("price").between(10, 1000)))
    return feats.withColumn("label", F.log1p(F.col("price")))


def train(spark: SparkSession, algo: str = "rf", sample: float | None = None) -> dict:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    df = _prepare(listings)
    if sample:
        df = df.sample(False, sample, seed=42)

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    stages = build_feature_stages() + [ALGOS[algo]()]
    pipeline = Pipeline(stages=stages)

    model: PipelineModel = pipeline.fit(train_df)
    pred = model.transform(test_df)
    pred = pred.withColumn("prediction_price", F.expm1("prediction"))

    metrics_log = regression_metrics(pred)
    metrics_eur = regression_metrics_eur(pred)

    out_metrics = {
        "algo": algo,
        "n_train": train_df.count(),
        "n_test":  test_df.count(),
        "metrics_log": metrics_log,
        "metrics_eur": metrics_eur,
    }

    model_path = MODELS_DIR / f"price_{algo}"
    model.write().overwrite().save(str(model_path))
    metrics_path = RESULTS_DIR / "ml"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / f"price_{algo}_metrics.json").write_text(json.dumps(out_metrics, indent=2))
    print(json.dumps(out_metrics, indent=2))
    return out_metrics


def train_all(spark: SparkSession, sample: float | None = None) -> list[dict]:
    return [train(spark, a, sample=sample) for a in ALGOS]
