"""Apply cleaning rules to interim Parquet and emit processed Parquet."""
from __future__ import annotations

import argparse

from egd_airbnb.cleaning import calendar as clean_calendar
from egd_airbnb.cleaning import listings as clean_listings
from egd_airbnb.cleaning import reviews  as clean_reviews
from egd_airbnb.config import CITIES, ensure_dirs
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cities", default=",".join(CITIES))
    p.add_argument("--datasets", default="listings,calendar,reviews")
    args = p.parse_args()

    ensure_dirs()
    spark = get_spark("clean")
    try:
        wanted = set(args.datasets.split(","))
        if "listings" in wanted:
            clean_listings.run(spark)
            print("[clean] listings done")
        if "calendar" in wanted:
            clean_calendar.run(spark)
            print("[clean] calendar done")
        if "reviews" in wanted:
            clean_reviews.run(spark)
            print("[clean] reviews done")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
