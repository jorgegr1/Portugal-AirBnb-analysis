"""Clean the unified listings DataFrame (summary format, 18 columns) and
optionally enrich it by left-joining the detailed listings CSV.

The detailed listings CSV (listings-detailed.csv) provides the high-value
structural features missing from the summary format:
  accommodates, bedrooms, beds, bathrooms, property_type,
  host_is_superhost, review_scores_*, amenities (→ amenity_count),
  estimated_revenue_l365d, estimated_occupancy_l365d.

Price in the detailed CSV uses "$X.XX" format and replaces the summary integer.
Join is performed on `id`; cities without a detailed file are kept as-is
(left join — no rows are lost).
"""
from __future__ import annotations

import json

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..config import CITIES, INTERIM_DIR, PROCESSED_DIR, RAW_DIR
from ..utils.io import read_parquet, write_parquet

# ── Column lists ─────────────────────────────────────────────────────────────

_SUMMARY_NUMERIC = [
    "latitude", "longitude", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "number_of_reviews_ltm",
]

# Columns to pull from the detailed CSV (beyond what summary provides).
_DETAIL_COLS = [
    "id",
    "price",                    # "$X.XX" format — replaces summary price
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "property_type",
    "host_is_superhost",        # "t"/"f"
    "amenities",                # JSON array string → amenity_count
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "estimated_revenue_l365d",
    "estimated_occupancy_l365d",
    "instant_bookable",
]

_HIGH_OCCUPANCY_THRESHOLD = 0.70


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_dollar_price(col: str):
    return F.regexp_replace(F.col(col), r"[\$,]", "").cast(T.DoubleType())


def _parse_bool(col: str):
    return F.when(F.col(col) == "t", True).when(F.col(col) == "f", False).otherwise(None)


def _count_amenities(col: str):
    # Amenities is a JSON array string. Count commas+1 as a fast proxy.
    return (F.size(F.from_json(F.col(col), T.ArrayType(T.StringType())))).cast(T.IntegerType())


# ── Detailed enrichment ──────────────────────────────────────────────────────

def _read_detailed(spark: SparkSession, city_key: str) -> DataFrame | None:
    path = RAW_DIR / CITIES[city_key] / "listings-detailed.csv"
    if not path.exists():
        return None
    available = [c for c in _DETAIL_COLS
                 if c in spark.read.option("header", True).csv(str(path)).columns]
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("multiLine", True)
        .option("escape", '"').option("quote", '"')
        .csv(str(path))
        .select(*available)
        .withColumn("city", F.lit(city_key))
    )
    if "price" in df.columns:
        df = df.withColumn("price_detailed", _parse_dollar_price("price")).drop("price")
    if "host_is_superhost" in df.columns:
        df = df.withColumn("host_is_superhost", _parse_bool("host_is_superhost"))
    if "instant_bookable" in df.columns:
        df = df.withColumn("instant_bookable", _parse_bool("instant_bookable"))
    if "amenities" in df.columns:
        df = df.withColumn("amenity_count", _count_amenities("amenities")).drop("amenities")
    for c in ["accommodates","bedrooms","beds","bathrooms",
              "review_scores_rating","review_scores_cleanliness",
              "review_scores_location","review_scores_value",
              "estimated_revenue_l365d","estimated_occupancy_l365d"]:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(T.DoubleType()))
    return df


def _build_detailed_enrichment(spark: SparkSession) -> DataFrame | None:
    """Union detailed CSVs for all available cities."""
    frames = [_read_detailed(spark, c) for c in CITIES if _read_detailed(spark, c) is not None]
    if not frames:
        return None
    # Re-read to avoid double I/O from the check above
    frames = []
    for city_key in CITIES:
        df = _read_detailed(spark, city_key)
        if df is not None:
            frames.append(df)
    if not frames:
        return None
    common = sorted(set.intersection(*(set(df.columns) for df in frames)))
    unified = frames[0].select(common)
    for df in frames[1:]:
        unified = unified.unionByName(df.select(common))
    return unified


# ── Main cleaning function ───────────────────────────────────────────────────

def clean_listings(df: DataFrame, detailed: DataFrame | None = None) -> DataFrame:
    out = df

    # Cast summary numerics.
    for c in _SUMMARY_NUMERIC:
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Summary price is a plain integer in the new format.
    if "price" in out.columns:
        out = out.withColumn("price", F.col("price").cast(T.DoubleType()))

    if "last_review" in out.columns:
        out = out.withColumn("last_review", F.to_date("last_review"))

    # Fills.
    fill_map: dict = {}
    if "reviews_per_month" in out.columns:
        fill_map["reviews_per_month"] = 0.0
    if fill_map:
        out = out.fillna(fill_map)

    # Enrich with detailed CSV (left join on id).
    if detailed is not None:
        out = out.join(
            detailed.drop("city"),   # city already present from summary
            on="id",
            how="left",
        )
        # Prefer the detailed price (cleaner, $-formatted) where available.
        if "price_detailed" in out.columns:
            out = out.withColumn(
                "price",
                F.coalesce(F.col("price_detailed"), F.col("price")),
            ).drop("price_detailed")

    # Drop rows with no price.
    out = out.filter(F.col("price").isNotNull() & (F.col("price") > 0))
    out = out.dropDuplicates(["id"])

    # Derived columns.
    if "calculated_host_listings_count" in out.columns:
        out = out.withColumn(
            "host_type",
            F.when(F.col("calculated_host_listings_count") > 1, "multi").otherwise("single"),
        )
    if "availability_365" in out.columns:
        out = out.withColumn(
            "is_high_occupancy",
            (F.lit(1.0) - F.col("availability_365") / F.lit(365.0)) > F.lit(_HIGH_OCCUPANCY_THRESHOLD),
        )
        out = out.withColumn("booked_nights_365", F.lit(365.0) - F.col("availability_365"))
        if "price" in out.columns:
            out = out.withColumn(
                "estimated_revenue_365",
                F.col("price") * F.col("booked_nights_365"),
            )

    return out


def run(spark: SparkSession, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "listings")
    detailed = _build_detailed_enrichment(spark)
    if detailed is not None:
        n = detailed.count()
        print(f"[clean] detailed enrichment: {n:,} rows joined")
    cleaned = clean_listings(df, detailed)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "listings", partition_by="city")
    return cleaned
