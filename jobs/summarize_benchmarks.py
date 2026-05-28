"""Build reports/benchmarks/summary.csv from benchmark log files."""

from __future__ import annotations

import argparse
from pathlib import Path

from egd_airbnb.benchmark.summary import write_benchmark_summary
from egd_airbnb.config import BENCH_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = write_benchmark_summary(args.log_dir, args.out)
    print(f"[bench-summary] wrote {out}")


if __name__ == "__main__":
    main()
