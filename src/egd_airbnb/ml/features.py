"""Shared feature engineering for ML pipelines.

Builds a list of PipelineStages that callers can extend with an estimator.
"""
from __future__ import annotations

from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

NUMERIC_FEATURES = [
    "accommodates", "bedrooms", "beds", "minimum_nights",
    "number_of_reviews", "number_of_reviews_ltm", "reviews_per_month",
    "review_scores_rating", "calculated_host_listings_count",
    "latitude", "longitude",
]

CATEGORICAL_FEATURES = [
    "city", "room_type", "property_type",
    "neighbourhood_cleansed", "host_is_superhost",
]


def select_features(df: DataFrame, *, leak_cols: list[str] | None = None) -> DataFrame:
    """Project to the columns we care about + sensible imputations."""
    leak = set(leak_cols or [])
    kept = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in leak and c in df.columns]
    out = df.select(*kept, *(c for c in df.columns if c not in kept and c in {"id", "price", "is_high_occupancy"}))

    # Median-ish imputation: 0 for review counts, conservative defaults elsewhere.
    fills = {
        "bedrooms": 1, "beds": 1, "minimum_nights": 1,
        "number_of_reviews": 0, "number_of_reviews_ltm": 0, "reviews_per_month": 0,
        "review_scores_rating": 4.5, "calculated_host_listings_count": 1,
    }
    for c, v in fills.items():
        if c in out.columns:
            out = out.fillna({c: v})

    # Cast boolean superhost to string so StringIndexer behaves consistently.
    if "host_is_superhost" in out.columns:
        out = out.withColumn("host_is_superhost", F.col("host_is_superhost").cast("string"))
    return out


def build_feature_stages(categorical: list[str] | None = None, numeric: list[str] | None = None) -> list:
    cat = categorical or CATEGORICAL_FEATURES
    num = numeric or NUMERIC_FEATURES

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat],
        outputCols=[f"{c}_oh"  for c in cat],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=num + [f"{c}_oh" for c in cat],
        outputCol="features_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False)
    return [*indexers, encoder, assembler, scaler]


def fit_features_only(df: DataFrame, categorical: list[str] | None = None, numeric: list[str] | None = None) -> PipelineModel:
    """Convenience: fit the feature transformer alone (useful for benchmarking)."""
    return Pipeline(stages=build_feature_stages(categorical, numeric)).fit(df)
