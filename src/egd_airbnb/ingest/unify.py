"""Read raw CSVs under data/raw/<City>/ and write unified Parquet per dataset.

Output layout (partitioned by city):
    data/interim/listings/city=porto/...
    data/interim/calendar/city=porto/...
    data/interim/reviews/city=porto/...

Reference tables (not partitioned):
    data/interim/neighbourhoods/city=porto/...
"""
from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import CITIES, DATASETS, INTERIM_DIR, REFERENCE_DATASETS, RAW_DIR
from ..utils.io import write_parquet


def _read_city_csv(spark: SparkSession, city_key: str, dataset: str) -> DataFrame:
    folder = CITIES[city_key]
    path_csv = RAW_DIR / folder / f"{dataset}.csv"
    path_gz = RAW_DIR / folder / f"{dataset}.csv.gz"
    if path_csv.exists():
        path = path_csv
    elif path_gz.exists():
        path = path_gz
    else:
        raise FileNotFoundError(f"Missing raw file: {path_csv} or {path_gz}")
    return (
        spark.read
        .option("header", True)
        .option("multiLine", True)
        .option("escape", '"')
        .option("quote", '"')
        .option("inferSchema", False)      # types handled in cleaning/
        .csv(str(path))
        .withColumn("city", F.lit(city_key))
    )


def unify_dataset(spark: SparkSession, dataset: str, cities: list[str]) -> Path:
    dfs = [_read_city_csv(spark, c, dataset) for c in cities]
    common = sorted(set.intersection(*(set(df.columns) for df in dfs)))
    unified = dfs[0].select(common)
    for df in dfs[1:]:
        unified = unified.unionByName(df.select(common))
    out = INTERIM_DIR / dataset
    write_parquet(unified, out, partition_by="city")
    return out


def unify_neighbourhoods(spark: SparkSession, cities: list[str]) -> Path:
    """neighbourhoods.csv is a reference table: neighbourhood_group → neighbourhood."""
    dfs = [_read_city_csv(spark, c, "neighbourhoods") for c in cities]
    common = sorted(set.intersection(*(set(df.columns) for df in dfs)))
    unified = dfs[0].select(common)
    for df in dfs[1:]:
        unified = unified.unionByName(df.select(common))
    out = INTERIM_DIR / "neighbourhoods"
    write_parquet(unified, out, partition_by="city")
    return out


def unify_all(spark: SparkSession, cities: list[str] | None = None) -> None:
    cities = cities or list(CITIES)
    for ds in DATASETS:
        out = unify_dataset(spark, ds, cities)
        print(f"[unify] {ds} → {out}")
    out = unify_neighbourhoods(spark, cities)
    print(f"[unify] neighbourhoods → {out}")
