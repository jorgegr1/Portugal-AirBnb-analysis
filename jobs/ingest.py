"""Read data/raw/ CSVs → data/interim/ unified Parquet (partitioned by city)."""

from __future__ import annotations

import argparse

from egd_airbnb.config import CITIES, ensure_dirs
from egd_airbnb.ingest.unify import unify_all
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default=",".join(CITIES), help="Comma-separated city keys")
    args = p.parse_args()

    ensure_dirs()
    spark = get_spark("ingest")
    try:
        unify_all(spark, args.cities.split(","))
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
