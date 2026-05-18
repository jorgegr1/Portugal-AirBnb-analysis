"""Clean the unified calendar DataFrame."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..utils.io import read_parquet, write_parquet


def _parse_price(col: str):
    return F.regexp_replace(F.col(col), r"[\$,]", "").cast(T.DoubleType())


def _parse_bool(col: str):
    return F.when(F.col(col) == "t", True).when(F.col(col) == "f", False).otherwise(None)


def clean_calendar(df: DataFrame) -> DataFrame:
    # EDA confirmed `price` is 100% null — drop it; keep adjusted_price (with $ cast).
    if "price" in df.columns:
        df = df.drop("price")

    out = df.withColumn("date", F.to_date("date"))

    if "available" in out.columns:
        out = out.withColumn("available", _parse_bool("available"))
    if "adjusted_price" in out.columns:
        out = out.withColumn("adjusted_price", _parse_price("adjusted_price"))
    for c in ("minimum_nights", "maximum_nights"):
        if c in out.columns:
            out = out.withColumn(c, F.col(c).cast(T.IntegerType()))

    out = (
        out
        .withColumn("year",  F.year("date"))
        .withColumn("month", F.month("date"))
        .withColumn("dow",   F.dayofweek("date"))
    )
    return out


def run(spark, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "calendar")
    cleaned = clean_calendar(df)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "calendar", partition_by="city")
    return cleaned
