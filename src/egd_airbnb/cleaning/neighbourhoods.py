"""Clean the neighbourhoods reference table (neighbourhood_group → neighbourhood hierarchy)."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..utils.io import read_parquet, write_parquet


def clean_neighbourhoods(df: DataFrame) -> DataFrame:
    return (
        df.filter(F.col("neighbourhood_group").isNotNull() & F.col("neighbourhood").isNotNull())
        .dropDuplicates(["city", "neighbourhood_group", "neighbourhood"])
        .orderBy("city", "neighbourhood_group", "neighbourhood")
    )


def run(spark, write: bool = True) -> DataFrame:
    df = read_parquet(spark, INTERIM_DIR / "neighbourhoods")
    cleaned = clean_neighbourhoods(df)
    if write:
        write_parquet(cleaned, PROCESSED_DIR / "neighbourhoods", partition_by="city")
    return cleaned
