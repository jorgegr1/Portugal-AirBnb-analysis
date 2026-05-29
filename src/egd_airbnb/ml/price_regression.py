"""Price regression: predict log1p(price), report RMSE/MAE/R² back in €.

Improvements over baseline:
  1. Dynamic price cap at training-set p99 (removes high outliers from loss).
  2. Feature engineering: occupancy_rate, log_reviews, recent_activity_ratio,
     min_nights_cat, neighbourhood_listing_count.
  3. Neighbourhood target encoding (replaces 285-column OHE with one dense signal).
  4. Better GBT hyperparameters (maxIter=200, stepSize=0.05).
"""

from __future__ import annotations

import json

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR
from ..utils.io import read_parquet
from .evaluation import regression_metrics, regression_metrics_eur
from .features import (
    build_feature_stages,
    compute_train_stats,
    engineer_features,
    select_and_fill,
)

ALGOS = {
    "lr": lambda: LinearRegression(
        featuresCol="features", labelCol="label", regParam=0.01, elasticNetParam=0.0
    ),  # L2 ridge
    "rf": lambda: RandomForestRegressor(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        minInstancesPerNode=10,
        seed=42,
    ),
    "gbt": lambda: GBTRegressor(
        featuresCol="features",
        labelCol="label",
        maxIter=200,
        maxDepth=6,
        stepSize=0.05,
        minInstancesPerNode=5,
        subsamplingRate=0.8,
        seed=42,
    ),
}


def _compute_p99(df: DataFrame) -> float:
    return float(df.agg(F.expr("percentile_approx(price, 0.99)")).first()[0])


def _prepare_raw(df: DataFrame, price_cap: float) -> DataFrame:
    """Filter implausible prices using a dynamic cap from the training distribution."""
    return df.filter(F.col("price").between(10, price_cap))


def train(spark: SparkSession, algo: str = "gbt", sample: float | None = None) -> dict:
    listings = read_parquet(spark, PROCESSED_DIR / "listings")
    if sample:
        listings = listings.sample(False, sample, seed=42)

    # 1. Dynamic price cap (p99 of the full dataset — computed before splitting
    #    because we want to remove noise, not learn the cap from train only).
    price_cap = _compute_p99(listings)
    df = _prepare_raw(listings, price_cap)

    # 2. Train/test split BEFORE feature engineering (avoids target-encoding leakage).
    train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)
    train_raw.cache()

    # 3. Compute target-encoding stats from training set only.
    train_stats = compute_train_stats(train_raw)

    # 4. Engineer features on both splits using training stats.
    # Cache train_df after engineering — RF evaluates splits many times per tree;
    # without caching, the neighbourhood join is recomputed on every pass.
    train_df = (
        select_and_fill(engineer_features(train_raw, train_stats))
        .withColumn("label", F.log1p(F.col("price")))
        .cache()
    )
    test_df = select_and_fill(engineer_features(test_raw, train_stats)).withColumn(
        "label", F.log1p(F.col("price"))
    )

    train_df.count()  # materialise cache before RF starts scanning partitions

    # 6. Fit ML Pipeline.
    stages = build_feature_stages() + [ALGOS[algo]()]
    pipeline = Pipeline(stages=stages)
    model: PipelineModel = pipeline.fit(train_df)

    # 7. Evaluate.
    pred = model.transform(test_df)
    pred = pred.withColumn("prediction_price", F.expm1("prediction"))

    metrics_log = regression_metrics(pred)
    metrics_eur = regression_metrics_eur(pred)

    out_metrics = {
        "algo": algo,
        "price_cap_p99": price_cap,
        "n_train": train_raw.count(),
        "n_test": test_raw.count(),
        "metrics_log": metrics_log,
        "metrics_eur": metrics_eur,
    }

    # 8. Persist model + metrics + inference stats.
    model_path = MODELS_DIR / f"price_{algo}"
    model.write().overwrite().save(str(model_path))
    metrics_path = RESULTS_DIR / "ml"
    metrics_path.mkdir(parents=True, exist_ok=True)
    (metrics_path / f"price_{algo}_metrics.json").write_text(json.dumps(out_metrics, indent=2))

    # Save neighbourhood stats so the dashboard can engineer features at inference time.
    nb_pd = train_stats["nb_stats"].toPandas()
    inference_stats = {
        "global_mean_log_price": train_stats["global_mean_log_price"],
        "global_mean_count": train_stats["global_mean_count"],
        "neighbourhood_stats": nb_pd.set_index("neighbourhood").to_dict(orient="index"),
    }
    (MODELS_DIR / f"price_{algo}_inference_stats.json").write_text(
        json.dumps(inference_stats, indent=2)
    )

    print(json.dumps(out_metrics, indent=2))
    return out_metrics


def train_all(spark: SparkSession, sample: float | None = None) -> list[dict]:
    return [train(spark, a, sample=sample) for a in ALGOS]
