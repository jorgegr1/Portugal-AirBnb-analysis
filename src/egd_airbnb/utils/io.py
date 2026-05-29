"""Thin Parquet I/O wrappers — keep job code free of repetitive options."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


def read_parquet(spark: SparkSession, path: str | Path) -> DataFrame:
    return spark.read.parquet(str(path))


def write_parquet(
    df: DataFrame,
    path: str | Path,
    *,
    partition_by: str | list[str] | None = None,
    mode: str = "overwrite",
) -> None:
    writer = df.write.mode(mode)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.parquet(str(path))
