"""Train one or all price-regression algorithms."""

from __future__ import annotations

import argparse

from egd_airbnb.config import ensure_dirs
from egd_airbnb.ml.price_regression import ALGOS, train, train_all
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="all", help=f"one of {list(ALGOS)} or 'all'")
    p.add_argument(
        "--sample", type=float, default=None, help="Optional sampling fraction (0,1] for quick runs"
    )
    args = p.parse_args()

    ensure_dirs()
    spark = get_spark("train_price")
    try:
        if args.algo == "all":
            train_all(spark, sample=args.sample)
        else:
            train(spark, args.algo, sample=args.sample)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
