"""Extract and save feature importances from all trained tree models."""
from __future__ import annotations

import json

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel, VectorAssembler

from egd_airbnb.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, ensure_dirs
from egd_airbnb.ml.features import compute_train_stats, engineer_features, select_and_fill
from egd_airbnb.spark_session import get_spark

LEAK_OCC = [
    "availability_30", "availability_60", "availability_90", "availability_365",
    "occupancy_rate", "booked_nights_365", "estimated_revenue_365",
]


def _load_train_stats(spark, stats_path):
    raw = json.loads(stats_path.read_text())
    nb_df = spark.createDataFrame([
        {"neighbourhood": k,
         "neighbourhood_mean_log_price": v["neighbourhood_mean_log_price"],
         "neighbourhood_listing_count": v["neighbourhood_listing_count"]}
        for k, v in raw["neighbourhood_stats"].items()
    ])
    return {"nb_stats": nb_df,
            "global_mean_log_price": raw["global_mean_log_price"],
            "global_mean_count": raw["global_mean_count"]}


def _extract_importance(model: PipelineModel, sample_df) -> pd.Series:
    indexers  = [s for s in model.stages if isinstance(s, StringIndexerModel)]
    assembler = next(s for s in model.stages if isinstance(s, VectorAssembler))
    estimator = model.stages[-1]

    num_cols = [c for c in assembler.getInputCols() if not c.endswith("_oh")]
    feature_names = list(num_cols)
    for si in indexers:
        cat = si.getInputCol()
        for lbl in si.labels:
            feature_names.append(f"{cat}={lbl}")

    importances = estimator.featureImportances.toArray()
    rows = []
    for i, imp in enumerate(importances):
        name = feature_names[i] if i < len(feature_names) else f"__ohe_unknown_{i}"
        base = name.split("=")[0]
        rows.append({"feature": base, "importance": imp})

    return (pd.DataFrame(rows)
              .groupby("feature")["importance"].sum()
              .sort_values(ascending=False))


def main() -> None:
    ensure_dirs()
    spark = get_spark("feature_importance")
    listings = spark.read.parquet(str(PROCESSED_DIR / "listings"))

    stats_file = MODELS_DIR / "price_gbt_inference_stats.json"
    if not stats_file.exists():
        raise SystemExit("Run `make ml-price` first to generate inference stats.")
    train_stats = _load_train_stats(spark, stats_file)

    out_dir = RESULTS_DIR / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        ("price_gbt",    [],        "price_importance.csv"),
        ("occupancy_rf", LEAK_OCC,  "occupancy_importance.csv"),
    ]

    for model_name, leak, out_name in targets:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            print(f"[importance] {model_name} not found — skipping")
            continue

        model   = PipelineModel.load(str(model_path))
        sample  = select_and_fill(
            engineer_features(listings.limit(1000), train_stats),
            leak_cols=leak,
        )
        imp = _extract_importance(model, sample)
        imp = imp[~imp.index.str.startswith("__ohe")]   # drop OHE unknown slots

        out_path = out_dir / out_name
        imp.reset_index().rename(columns={"importance": "importance"}).to_csv(out_path, index=False)
        print(f"\n[importance] {model_name} → {out_path}")
        print(imp.head(20).to_string())

    spark.stop()


if __name__ == "__main__":
    main()
