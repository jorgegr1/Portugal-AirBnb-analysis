"""Central paths and constants. Import these instead of hardcoding."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports"
RESULTS_DIR   = REPORTS_DIR / "results"
FIGURES_DIR   = REPORTS_DIR / "figures"
BENCH_DIR     = REPORTS_DIR / "benchmarks"
MODELS_DIR    = PROJECT_ROOT / "models"

# data/raw/ uses capitalised city folder names as supplied.
CITIES: dict[str, str] = {
    "porto":  "Porto",
    "lisbon": "Lisbon",
    "madrid": "Madrid",
}

DATASETS = ("listings", "calendar", "reviews")
REFERENCE_DATASETS = ("neighbourhoods",)

SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
DEFAULT_MASTER      = os.environ.get("SPARK_MASTER", "local[*]")
SHUFFLE_PARTITIONS  = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "200"))


def ensure_dirs() -> None:
    for p in (INTERIM_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, BENCH_DIR, MODELS_DIR):
        p.mkdir(parents=True, exist_ok=True)

# Convenience path helpers
def processed(dataset: str) -> "Path":  # type: ignore[name-defined]
    return PROCESSED_DIR / dataset
