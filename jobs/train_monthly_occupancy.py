"""Train the month-aware occupancy model."""
from __future__ import annotations

import argparse

from egd_airbnb.config import ensure_dirs
from egd_airbnb.ml.monthly_occupancy import train
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=float, default=None)
    args = p.parse_args()

    ensure_dirs()
    spark = get_spark("train_monthly_occupancy")
    try:
        train(spark, sample=args.sample)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
