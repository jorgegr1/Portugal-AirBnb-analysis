"""Run one or all analytical queries and persist results to reports/results/qNN.parquet."""
from __future__ import annotations

import argparse

from egd_airbnb.config import RESULTS_DIR, ensure_dirs
from egd_airbnb.queries.registry import DESCRIPTIONS, QUERIES
from egd_airbnb.spark_session import get_spark


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="all", help="qNN or 'all'")
    p.add_argument("--csv", action="store_true", help="Also export a single-file CSV alongside the Parquet")
    args = p.parse_args()

    ensure_dirs()
    targets = sorted(QUERIES) if args.query in ("all", "*") else [args.query]
    spark = get_spark("queries")
    try:
        for q in targets:
            print(f"[query] {q} — {DESCRIPTIONS[q]}")
            df = QUERIES[q](spark)
            out = RESULTS_DIR / f"{q}.parquet"
            df.write.mode("overwrite").parquet(str(out))
            if args.csv:
                df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(out.with_suffix(".csv")))
            print(f"[query] wrote {out}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
