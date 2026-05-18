"""Speedup / efficiency plots for the benchmark CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BENCH_DIR, FIGURES_DIR


def _summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return (
        df.groupby(["platform", "parallelism"])["wall_secs"]
          .agg(["median", "min", "max", "count"])
          .reset_index()
          .sort_values(["platform", "parallelism"])
    )


def plot_workload(workload: str) -> list[Path]:
    import matplotlib.pyplot as plt  # noqa: I001

    out_files: list[Path] = []
    frames = []
    for path in BENCH_DIR.glob(f"{workload}_*.csv"):
        s = _summary(path)
        frames.append(s)
    if not frames:
        print(f"[plot] no benchmark CSVs for {workload}")
        return out_files
    summary = pd.concat(frames, ignore_index=True)

    # Wall-time figure.
    fig, ax = plt.subplots(figsize=(6, 4))
    for platform, sub in summary.groupby("platform"):
        ax.plot(sub["parallelism"], sub["median"], marker="o", label=platform)
    ax.set_xlabel("Parallelism (cores or workers)")
    ax.set_ylabel("Wall time (seconds, median over runs)")
    ax.set_title(f"{workload}: wall-time vs parallelism")
    ax.legend()
    ax.grid(True, alpha=0.3)
    f1 = FIGURES_DIR / f"bench_{workload}_walltime.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    out_files.append(f1)

    # Speedup / efficiency.
    fig, ax = plt.subplots(figsize=(6, 4))
    for platform, sub in summary.groupby("platform"):
        baseline = sub.loc[sub["parallelism"] == sub["parallelism"].min(), "median"].iloc[0]
        sub = sub.assign(speedup=baseline / sub["median"], efficiency=lambda d: d["speedup"] / d["parallelism"])
        ax.plot(sub["parallelism"], sub["speedup"], marker="o", label=f"{platform} speedup")
        ax.plot(sub["parallelism"], sub["efficiency"], marker="x", linestyle="--", label=f"{platform} efficiency")
    ax.set_xlabel("Parallelism")
    ax.set_ylabel("Speedup / Efficiency")
    ax.set_title(f"{workload}: speedup & efficiency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    f2 = FIGURES_DIR / f"bench_{workload}_speedup.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    out_files.append(f2)
    return out_files
