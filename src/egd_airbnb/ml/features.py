"""Shared feature engineering for ML pipelines (summary format, 18-column schema)."""
from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Available numeric features in the summary-format listings.
NUMERIC_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "latitude",
    "longitude",
]

# Available categorical features.
CATEGORICAL_FEATURES = [
    "city",
    "room_type",
    "neighbourhood_group",
    "neighbourhood",
]


def select_features(df: DataFrame, *, leak_cols: list[str] | None = None) -> DataFrame:
    """Project to usable columns and apply sensible imputations."""
    leak = set(leak_cols or [])
    num = [c for c in NUMERIC_FEATURES if c not in leak and c in df.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c not in leak and c in df.columns]

    target_cols = {c for c in ("id", "price", "is_high_occupancy") if c in df.columns}
    out = df.select(*num, *cat, *target_cols)

    fills = {
        "minimum_nights": 1,
        "number_of_reviews": 0,
        "number_of_reviews_ltm": 0,
        "reviews_per_month": 0.0,
        "calculated_host_listings_count": 1,
    }
    for c, v in fills.items():
        if c in out.columns:
            out = out.fillna({c: v})

    return out


def build_feature_stages(
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> list:
    cat = categorical or CATEGORICAL_FEATURES
    num = numeric or NUMERIC_FEATURES

    # Only include cats/nums that exist — guarded by caller via select_features.
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat],
        outputCols=[f"{c}_oh" for c in cat],
        handleInvalid="keep",
    )
    assembler = VectorAssembler(
        inputCols=num + [f"{c}_oh" for c in cat],
        outputCol="features_raw",
        handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False)
    return [*indexers, encoder, assembler, scaler]


def fit_features_only(
    df: DataFrame,
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> PipelineModel:
    return Pipeline(stages=build_feature_stages(categorical, numeric)).fit(df)
