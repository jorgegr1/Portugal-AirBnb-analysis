"""Feature engineering for ML pipelines (summary format).

Two-phase approach:
  1. engineer_features() — adds derived columns from raw listing fields.
     Requires `train_stats` for target encoding (computed on training set only).
  2. build_feature_stages() — returns the MLlib Pipeline stages
     (Indexer → OHE → Assembler → Scaler) that operate on the engineered frame.
"""
from __future__ import annotations

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    Bucketizer,
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Numeric features — includes engineered ones added by engineer_features().
NUMERIC_FEATURES = [
    # raw — summary format
    "minimum_nights",
    "number_of_reviews",
    "number_of_reviews_ltm",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "latitude",
    "longitude",
    # raw — from detailed CSV (present when detailed join was run)
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "amenity_count",
    "estimated_occupancy_l365d",
    # engineered
    "occupancy_rate",
    "log_reviews",
    "log_host_listings",
    "recent_activity_ratio",
    "neighbourhood_listing_count",
    "neighbourhood_mean_log_price",   # target encoding — no leakage if from train only
]

# Categorical features — neighbourhood replaced by target encoding above.
CATEGORICAL_FEATURES = [
    "city",
    "room_type",
    "neighbourhood_group",
    "min_nights_cat",
    # from detailed CSV
    "property_type",
    "host_is_superhost",
    "instant_bookable",
]


# ── Derived feature construction ────────────────────────────────────────────

def compute_train_stats(train_df: DataFrame) -> dict:
    """Compute per-neighbourhood stats from the TRAINING set only (no leakage).

    Returns a dict of DataFrames to join onto any split.
    """
    # Target encoding: neighbourhood → mean log1p(price) on train.
    nb_stats = (
        train_df
        .groupBy("neighbourhood")
        .agg(
            F.avg(F.log1p("price")).alias("neighbourhood_mean_log_price"),
            F.count("id").alias("neighbourhood_listing_count"),
        )
    )
    # Global fallback for unseen neighbourhoods (use overall mean).
    global_mean = float(
        train_df.agg(F.avg(F.log1p("price"))).first()[0]
    )
    global_count = float(train_df.count()) / float(train_df.select("neighbourhood").distinct().count())
    return {
        "nb_stats": nb_stats,
        "global_mean_log_price": global_mean,
        "global_mean_count": global_count,
    }


def engineer_features(df: DataFrame, train_stats: dict) -> DataFrame:
    """Add all engineered columns to df. Call with train_stats from training set."""
    nb_stats    = train_stats["nb_stats"]
    global_mean = train_stats["global_mean_log_price"]
    global_cnt  = train_stats["global_mean_count"]

    out = df

    # 1. Occupancy rate (demand proxy from availability).
    out = out.withColumn("occupancy_rate",
        F.lit(1.0) - F.col("availability_365") / F.lit(365.0))

    # 2. Log-scale review count (right-skewed distribution).
    out = out.withColumn("log_reviews",     F.log1p(F.col("number_of_reviews")))
    out = out.withColumn("log_host_listings", F.log1p(F.col("calculated_host_listings_count")))

    # 3. Recent activity ratio: % of reviews from last year.
    out = out.withColumn("recent_activity_ratio",
        F.col("number_of_reviews_ltm") / (F.col("number_of_reviews") + F.lit(1.0)))

    # 4. Minimum-nights bucket (hosts set this intentionally as a signal of guest type).
    out = out.withColumn(
        "min_nights_cat",
        F.when(F.col("minimum_nights") == 1, "one_night")
         .when(F.col("minimum_nights").between(2, 3), "weekend")
         .when(F.col("minimum_nights").between(4, 6), "week_short")
         .otherwise("long_stay"),
    )

    # 5. Neighbourhood target encoding (join from training stats).
    out = (
        out
        .join(nb_stats, on="neighbourhood", how="left")
        .withColumn(
            "neighbourhood_mean_log_price",
            F.coalesce(F.col("neighbourhood_mean_log_price"), F.lit(global_mean)),
        )
        .withColumn(
            "neighbourhood_listing_count",
            F.coalesce(F.col("neighbourhood_listing_count"), F.lit(global_cnt)),
        )
    )

    return out


def select_and_fill(
    df: DataFrame,
    *,
    leak_cols: list[str] | None = None,
    extra_numeric: list[str] | None = None,
) -> DataFrame:
    """Project to model columns and fill sensible defaults.

    extra_numeric: additional numeric columns to keep beyond NUMERIC_FEATURES
    (e.g. temporal features for the monthly occupancy model).
    """
    leak = set(leak_cols or [])
    num  = [c for c in NUMERIC_FEATURES + (extra_numeric or [])
            if c not in leak and c in df.columns]
    cat  = [c for c in CATEGORICAL_FEATURES if c not in leak and c in df.columns]
    keep = {c for c in ("id", "price", "is_high_occupancy", "is_high_demand_month",
                        "monthly_occupancy_rate", "year", "month") if c in df.columns}

    out = df.select(*num, *cat, *keep)

    # StringIndexer requires string or numeric — cast booleans to string.
    _bool_cats = {"host_is_superhost", "instant_bookable"}
    for c in _bool_cats:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast("string"))

    # StringIndexer throws NPE on null strings even with handleInvalid="keep".
    for c in cat:
        if c in out.columns:
            out = out.withColumn(c, F.coalesce(F.col(c), F.lit("unknown")))

    fills = {
        "minimum_nights": 1, "number_of_reviews": 0, "number_of_reviews_ltm": 0,
        "reviews_per_month": 0.0, "calculated_host_listings_count": 1,
        "recent_activity_ratio": 0.0, "occupancy_rate": 0.5,
        # detailed cols — median-ish defaults
        "accommodates": 2, "bedrooms": 1, "beds": 1, "bathrooms": 1.0,
        "review_scores_rating": 4.7, "review_scores_cleanliness": 4.7,
        "review_scores_location": 4.7, "review_scores_value": 4.5,
        "amenity_count": 20, "estimated_occupancy_l365d": 0.0,
    }
    for c, v in fills.items():
        if c in out.columns:
            out = out.fillna({c: v})
    return out


# ── MLlib pipeline stages ───────────────────────────────────────────────────

def build_feature_stages(
    categorical: list[str] | None = None,
    numeric: list[str] | None = None,
) -> list:
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


def fit_features_only(df: DataFrame) -> PipelineModel:
    return Pipeline(stages=build_feature_stages()).fit(df)
