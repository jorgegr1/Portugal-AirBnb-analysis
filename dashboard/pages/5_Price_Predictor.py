"""Price predictor — uses saved GBT model.

Inputs trimmed to features that actually matter (per feature importance):
  REMOVED (0% importance): host_is_superhost, instant_bookable, log_reviews.
  REMOVED (redundant given GBT): separate log_* inputs.

Seasonal context from Q2 results shows how demand varies by month in the city.
"""

import math

import streamlit as st
from _shared import (
    engineer_row,
    get_dashboard_spark,
    latest_price_model_dir,
    load_inference_stats,
    load_listings,
    load_query_result,
)

st.title("Price predictor")

model_dir = latest_price_model_dir()
if model_dir is None:
    st.warning("No trained price model found. Run `make ml-price` first.")
    st.stop()

inference_stats = load_inference_stats(model_dir)
listings = load_listings()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location")
    city = st.selectbox("City", sorted(listings["city"].unique()), key="p_city")
    municipality = st.selectbox(
        "Municipality",
        sorted(listings[listings["city"] == city]["neighbourhood_group"].dropna().unique()),
        key="p_mun",
    )
    neighbourhood = st.selectbox(
        "Neighbourhood",
        sorted(
            listings[listings["neighbourhood_group"] == municipality]["neighbourhood"]
            .dropna()
            .unique()
        ),
        key="p_nb",
    )

with col2:
    st.subheader("Property")
    room_type = st.selectbox(
        "Room type", sorted(listings["room_type"].dropna().unique()), key="p_rt"
    )
    property_type = st.selectbox(
        "Property type",
        sorted(listings["property_type"].dropna().unique())
        if "property_type" in listings
        else ["Entire rental unit"],
        key="p_pt",
    )
    accommodates = st.slider("Accommodates", 1, 16, 2, key="p_acc")
    bedrooms = st.slider("Bedrooms", 0, 10, 1, key="p_bed")
    bathrooms = st.slider("Bathrooms", 0.5, 6.0, 1.0, 0.5, key="p_bath")
    amenity_count = st.slider("Amenities (count)", 0, 80, 25, key="p_am")

with col3:
    st.subheader("Activity & host")
    host_listings = st.slider(
        "Host total listings",
        1,
        50,
        1,
        key="p_hl",
        help="Top feature: professional multi-listing hosts price differently",
    )
    availability = st.slider("Availability (days/year)", 0, 365, 120, key="p_av")
    minimum_nights = st.slider("Minimum nights", 1, 30, 2, key="p_mn")
    rpm = st.slider("Reviews/month", 0.0, 20.0, 1.0, 0.1, key="p_rpm")
    n_reviews = st.slider("Total reviews", 0, 500, 30, key="p_nr")
    rating = st.slider("Rating", 0.0, 5.0, 4.7, 0.05, key="p_rat")

if st.button("Predict nightly price", type="primary"):
    lat = float(listings[listings["neighbourhood_group"] == municipality]["latitude"].mean())
    lon = float(listings[listings["neighbourhood_group"] == municipality]["longitude"].mean())

    row_dict = engineer_row(
        city=city,
        room_type=room_type,
        neighbourhood_group=municipality,
        neighbourhood=neighbourhood,
        property_type=property_type,
        host_is_superhost="false",
        instant_bookable="false",  # zero importance — use neutral
        minimum_nights=float(minimum_nights),
        number_of_reviews=float(n_reviews),
        number_of_reviews_ltm=float(n_reviews * 0.3),
        reviews_per_month=float(rpm),
        calculated_host_listings_count=float(host_listings),
        availability_365=float(availability),
        latitude=lat,
        longitude=lon,
        accommodates=float(accommodates),
        bedrooms=float(bedrooms),
        beds=float(bedrooms),
        bathrooms=float(bathrooms),
        review_scores_rating=float(rating),
        review_scores_cleanliness=float(rating),
        review_scores_location=float(rating),
        review_scores_value=float(rating),
        amenity_count=float(amenity_count),
        estimated_occupancy_l365d=float(365 - availability),
        inference_stats=inference_stats,
    )

    spark = get_dashboard_spark()
    from pyspark.ml import PipelineModel

    model = PipelineModel.load(str(model_dir))
    pred = model.transform(spark.createDataFrame([row_dict])).select("prediction").first()[0]
    eur = math.expm1(pred)

    st.success(f"### Predicted nightly price: **€{eur:.0f}**")

    # Market context: comparable listings
    similar = listings[
        (listings["city"] == city)
        & (listings["neighbourhood_group"] == municipality)
        & (listings["room_type"] == room_type)
    ]["price"]
    if not similar.empty:
        st.caption(
            f"Comparable listings in **{municipality}** ({room_type}): "
            f"median €{similar.median():.0f} · p25 €{similar.quantile(0.25):.0f} · p75 €{similar.quantile(0.75):.0f}"
        )

st.divider()

# Seasonal demand context from Q2 seasonality query.
st.subheader("Monthly demand context")
st.caption(
    "Occupancy rate by month shows when your listing will face highest competition / demand."
)
q2 = load_query_result("q02")
if not q2.empty:
    city_q2 = q2[q2["city"] == city].copy()
    if not city_q2.empty:
        city_q2["month_label"] = (
            city_q2["month"]
            .astype(int)
            .map(
                {
                    1: "Jan",
                    2: "Feb",
                    3: "Mar",
                    4: "Apr",
                    5: "May",
                    6: "Jun",
                    7: "Jul",
                    8: "Aug",
                    9: "Sep",
                    10: "Oct",
                    11: "Nov",
                    12: "Dec",
                }
            )
        )
        pivot = city_q2.sort_values("month").set_index("month_label")
        st.bar_chart(pivot["occupancy_rate"], use_container_width=True)
        peak = city_q2.loc[city_q2["occupancy_rate"].idxmax()]
        trough = city_q2.loc[city_q2["occupancy_rate"].idxmin()]
        st.caption(
            f"Peak demand: month {int(peak['month'])} (occupancy {peak['occupancy_rate'] * 100:.1f}%) · "
            f"Lowest: month {int(trough['month'])} ({trough['occupancy_rate'] * 100:.1f}%)"
        )
else:
    st.info("Run `make queries QUERY=q02` to see seasonal demand context.")
