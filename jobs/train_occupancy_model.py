"""Train one or all occupancy classifiers."""

from __future__ import annotations

import argparse

from egd_airbnb.config import ensure_dirs
from egd_airbnb.ml.occupancy_classifier import train, train_all
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all")
    p.add_argument("--sample", type=float, default=None)
    args = p.parse_args()

    ensure_dirs()
    spark = get_spark("train_occupancy")
    try:
        if args.algo == "all":
            train_all(spark, sample=args.sample)
        else:
            train(spark, args.algo, sample=args.sample)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
