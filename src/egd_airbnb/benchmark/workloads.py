"""Definitions of the workloads we time."""

from __future__ import annotations

from collections.abc import Callable

from pyspark.sql import SparkSession

from ..ml.price_regression import train as train_price_rf
from ..queries.q02_seasonality import run as q02_run


def workload_query_seasonality(spark: SparkSession) -> None:
    # Force materialisation with .count() — cheap action that triggers the full plan.
    q02_run(spark).count()


def workload_train_rf(spark: SparkSession) -> None:
    # Force full training; small sample only if dev environment is constrained.
    train_price_rf(spark, algo="rf")


WORKLOADS: dict[str, Callable[[SparkSession], None]] = {
    "query_seasonality": workload_query_seasonality,
    "train_rf": workload_train_rf,
}
