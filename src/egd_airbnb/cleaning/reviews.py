"""Clean the unified reviews DataFrame."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..utils.io import read_parquet, write_parquet


def clean_reviews(df: DataFrame) -> DataFrame:
    out = (
        df
        .withColumn("date", F.to_date("date"))
        .withColumn("year",  F.year("date"))
        .withColumn("month", F.month("date"))
        .dropDuplicates(["listing_id", "reviewer_id", "date"])
    )
    return out


def run(spark, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "reviews")
    cleaned = clean_reviews(df)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "reviews", partition_by="city")
    return cleaned
