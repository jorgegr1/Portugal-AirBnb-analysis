"""Single entry point for building SparkSessions across jobs and notebooks."""
from __future__ import annotations

from pyspark.sql import SparkSession

from .config import DEFAULT_MASTER, SHUFFLE_PARTITIONS, SPARK_DRIVER_MEMORY


def get_spark(app_name: str, master: str | None = None, *, extra_conf: dict | None = None) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master or DEFAULT_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    for k, v in (extra_conf or {}).items():
        builder = builder.config(k, str(v))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
