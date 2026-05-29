"""Time a workload across parallelism levels and persist results to CSV."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from ..config import BENCH_DIR
from ..spark_session import get_spark
from .workloads import WORKLOADS

# Local thread-level parallelism we sweep.
LOCAL_PARALLELISM = (1, 2, 4, 8)


def _time_once(workload: str, master: str, parallelism: int) -> float:
    spark = get_spark(
        f"bench-{workload}-{parallelism}",
        master=master,
        extra_conf={
            "spark.default.parallelism": parallelism,
            "spark.sql.shuffle.partitions": max(parallelism * 8, 16),
        },
    )
    t0 = time.perf_counter()
    try:
        WORKLOADS[workload](spark)
    finally:
        elapsed = time.perf_counter() - t0
        spark.stop()
    return elapsed


def run_local(workload: str, runs: int = 3, levels: tuple[int, ...] = LOCAL_PARALLELISM) -> Path:
    out = BENCH_DIR / f"{workload}_local.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "platform", "parallelism", "wall_secs"])
        for p in levels:
            for i in range(runs):
                secs = _time_once(workload, master=f"local[{p}]", parallelism=p)
                w.writerow([i, "local", p, f"{secs:.4f}"])
                print(f"[bench] local[{p}] run={i} {workload} → {secs:.2f}s")
    return out


def run_remote(workload: str, parallelism: int, runs: int = 3, master: str = "yarn") -> Path:
    """Dataproc path — master is set by spark-submit; runner just times runs and writes CSV."""
    out = BENCH_DIR / f"{workload}_dataproc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run_id", "platform", "parallelism", "wall_secs"])
        for i in range(runs):
            secs = _time_once(workload, master=master, parallelism=parallelism)
            w.writerow([i, "dataproc", parallelism, f"{secs:.4f}"])
            print(f"[bench] dataproc workers={parallelism} run={i} {workload} → {secs:.2f}s")
    return out
