"""Run benchmark workloads and emit CSV + speedup plots."""
from __future__ import annotations

import argparse

from egd_airbnb.benchmark.plot import plot_workload
from egd_airbnb.benchmark.runner import run_local, run_remote
from egd_airbnb.benchmark.workloads import WORKLOADS
from egd_airbnb.config import ensure_dirs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workload",    required=True, choices=list(WORKLOADS))
    p.add_argument("--platform",    default="local", choices=["local", "dataproc"])
    p.add_argument("--runs",        type=int, default=3)
    p.add_argument("--parallelism", type=int, help="Dataproc only: number of workers/cores label")
    p.add_argument("--plot",        action="store_true", help="Plot after running")
    args = p.parse_args()

    ensure_dirs()
    if args.platform == "local":
        run_local(args.workload, runs=args.runs)
    else:
        if args.parallelism is None:
            raise SystemExit("--parallelism required when --platform dataproc")
        run_remote(args.workload, parallelism=args.parallelism, runs=args.runs)

    if args.plot:
        for f in plot_workload(args.workload):
            print(f"[plot] {f}")


if __name__ == "__main__":
    main()
