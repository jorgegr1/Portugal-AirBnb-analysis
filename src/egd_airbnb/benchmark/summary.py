"""Parse Dataproc benchmark logs and write a compact CSV summary."""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

from ..config import BENCH_DIR

_LOG_NAME_RE = re.compile(r"(?P<workload>.+)_(?P<workers>\d+)w\.txt$")
_WALL_SECS_RE = re.compile(r'"wall_secs":\s*(?P<secs>[0-9.]+)')


def _read_wall_times(path: Path) -> list[float]:
    return [float(m.group("secs")) for m in _WALL_SECS_RE.finditer(path.read_text())]


def collect_benchmark_rows(log_dir: Path = BENCH_DIR) -> list[dict[str, str | int | float]]:
    """Collect median wall times and scaling metrics from benchmark text logs."""
    rows: list[dict[str, str | int | float]] = []
    for path in sorted(log_dir.glob("*_*.txt")):
        match = _LOG_NAME_RE.match(path.name)
        if not match:
            continue
        times = _read_wall_times(path)
        if not times:
            continue
        rows.append(
            {
                "workload": match.group("workload"),
                "platform": "dataproc",
                "workers": int(match.group("workers")),
                "runs": len(times),
                "median_wall_secs": round(statistics.median(times), 4),
                "min_wall_secs": round(min(times), 4),
                "max_wall_secs": round(max(times), 4),
            }
        )

    baselines = {row["workload"]: row["median_wall_secs"] for row in rows if row["workers"] == 1}
    for row in rows:
        baseline = float(baselines.get(row["workload"], 0.0))
        workers = int(row["workers"])
        median = float(row["median_wall_secs"])
        if baseline and median:
            speedup = baseline / median
            row["speedup_vs_1w"] = round(speedup, 4)
            row["efficiency_vs_1w"] = round(speedup / workers, 4)
        else:
            row["speedup_vs_1w"] = ""
            row["efficiency_vs_1w"] = ""

    return sorted(rows, key=lambda r: (str(r["workload"]), int(r["workers"])))


def write_benchmark_summary(
    log_dir: Path = BENCH_DIR,
    out_path: Path | None = None,
) -> Path:
    """Write reports/benchmarks/summary.csv from Dataproc benchmark logs."""
    rows = collect_benchmark_rows(log_dir)
    out = out_path or log_dir / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "workload",
        "platform",
        "workers",
        "runs",
        "median_wall_secs",
        "min_wall_secs",
        "max_wall_secs",
        "speedup_vs_1w",
        "efficiency_vs_1w",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out
