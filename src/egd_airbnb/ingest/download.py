"""Optional helper to re-download Inside Airbnb snapshots.

Not used in the default flow — raw CSVs are already present under data/raw/.
Kept for reproducibility / future snapshots.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests
from tqdm import tqdm

from ..config import CITIES, DATASETS, RAW_DIR

DEFAULT_BASE = os.environ.get("INSIDE_AIRBNB_BASE_URL", "https://data.insideairbnb.com")


def _url(base: str, city: str, snapshot: str, dataset: str) -> str:
    # NOTE: Inside Airbnb URL format may change; verify on insideairbnb.com/get-the-data
    return f"{base}/portugal/{city}/{snapshot}/data/{dataset}.csv.gz"


def download_city(city_key: str, snapshot: str, base: str = DEFAULT_BASE) -> None:
    folder = CITIES[city_key]
    out_dir = RAW_DIR / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        out = out_dir / f"{ds}.csv.gz"
        url = _url(base, folder.lower(), snapshot, ds)
        print(f"GET {url} → {out}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with out.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_all(snapshot: str, cities: list[str] | None = None) -> None:
    for c in (cities or list(CITIES)):
        download_city(c, snapshot)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", required=True, help="Snapshot date, e.g. 2025-06-25")
    p.add_argument("--cities", default=",".join(CITIES))
    args = p.parse_args()
    download_all(args.snapshot, args.cities.split(","))
