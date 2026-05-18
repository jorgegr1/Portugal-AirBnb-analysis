"""Registry: query name → callable returning a DataFrame.

Used by jobs/run_queries.py and by the Streamlit dashboard's Query Results page.
"""
from __future__ import annotations

from typing import Callable

from pyspark.sql import DataFrame, SparkSession

from . import (
    q01_neighbourhood_prices,
    q02_seasonality,
    q03_room_type_occupancy,
    q04_superhost_premium,
    q05_host_concentration,
    q06_reviews_price_corr,
    q07_top_revenue_neighbourhoods,
    q08_property_type_share,
    q09_review_demand_trend,
    q10_price_outliers,
)

QueryFn = Callable[[SparkSession], DataFrame]

QUERIES: dict[str, QueryFn] = {
    "q01": q01_neighbourhood_prices.run,
    "q02": q02_seasonality.run,
    "q03": q03_room_type_occupancy.run,
    "q04": q04_superhost_premium.run,
    "q05": q05_host_concentration.run,
    "q06": q06_reviews_price_corr.run,
    "q07": q07_top_revenue_neighbourhoods.run,
    "q08": q08_property_type_share.run,
    "q09": q09_review_demand_trend.run,
    "q10": q10_price_outliers.run,
}

DESCRIPTIONS: dict[str, str] = {
    "q01": "Top/bottom 10 neighbourhoods by median price per city",
    "q02": "Price seasonality from calendar (median monthly price per city)",
    "q03": "Availability/occupancy by room type and city",
    "q04": "Superhost vs regular host premium",
    "q05": "Host concentration (Pareto curve)",
    "q06": "Correlation: reviews_per_month × price per neighbourhood",
    "q07": "Top 10 neighbourhoods by estimated revenue",
    "q08": "Property-type market share per city",
    "q09": "Monthly review trend + YoY growth per city",
    "q10": "Price outlier prevalence per city × room_type",
}
