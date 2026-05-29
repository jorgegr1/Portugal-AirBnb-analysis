"""Evaluation helpers — regression (RMSE/MAE/R²) and classification (AUC/F1)."""

from __future__ import annotations

from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def regression_metrics(
    pred: DataFrame, *, label_col: str = "label", prediction_col: str = "prediction"
) -> dict:
    out = {}
    for m in ("rmse", "mae", "r2"):
        out[m] = float(
            RegressionEvaluator(
                labelCol=label_col, predictionCol=prediction_col, metricName=m
            ).evaluate(pred)
        )
    return out


def regression_metrics_eur(
    pred: DataFrame, *, label_col: str = "price", prediction_col: str = "prediction_price"
) -> dict:
    """Same metrics but on the back-transformed € price (after expm1)."""
    return regression_metrics(pred, label_col=label_col, prediction_col=prediction_col)


def classification_metrics(
    pred: DataFrame, *, label_col: str = "label", prediction_col: str = "prediction"
) -> dict:
    binary = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction")
    multi = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
    auc = float(binary.evaluate(pred, {binary.metricName: "areaUnderROC"}))
    f1 = float(multi.evaluate(pred, {multi.metricName: "f1"}))
    acc = float(multi.evaluate(pred, {multi.metricName: "accuracy"}))
    base_rate = float(pred.agg(F.avg(F.col(label_col).cast("double"))).first()[0])
    return {"auc": auc, "f1": f1, "accuracy": acc, "base_rate": base_rate}
