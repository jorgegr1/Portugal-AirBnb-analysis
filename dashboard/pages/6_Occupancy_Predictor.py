"""Occupancy predictors — annual (listing-level) and monthly (month-aware).

Annual model: predicts long-run occupancy class (high/low) from listing features.
Monthly model: trained on calendar × listings data with cyclic month encoding —
  predicts whether a specific listing will be in high demand in a given month.

Inputs trimmed per feature importance:
  REMOVED (0%): host_is_superhost, instant_bookable.
"""
import math

import streamlit as st

from _shared import (
    engineer_row,
    get_dashboard_spark,
    latest_occupancy_model_dir,
    load_inference_stats,
    load_listings,
    load_query_result,
)
from egd_airbnb.config import MODELS_DIR

st.title("Occupancy predictor")

# Check available models
annual_dir  = latest_occupancy_model_dir()
monthly_dir = next(iter(sorted(MODELS_DIR.glob("monthly_occupancy_*"), reverse=True)
                        if MODELS_DIR.exists() else []), None)

if not annual_dir and not monthly_dir:
    st.warning("No occupancy models found. Run `make ml-occupancy && make ml-monthly`.")
    st.stop()

listings = load_listings()

# ── Shared inputs ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Location")
    city          = st.selectbox("City", sorted(listings["city"].unique()), key="o_city")
    municipality  = st.selectbox("Municipality",
                        sorted(listings[listings["city"] == city]["neighbourhood_group"].dropna().unique()), key="o_mun")
    neighbourhood = st.selectbox("Neighbourhood",
                        sorted(listings[listings["neighbourhood_group"] == municipality]["neighbourhood"].dropna().unique()), key="o_nb")

with col2:
    st.subheader("Property")
    room_type     = st.selectbox("Room type", sorted(listings["room_type"].dropna().unique()), key="o_rt")
    property_type = st.selectbox("Property type",
                        sorted(listings["property_type"].dropna().unique()) if "property_type" in listings else ["Entire rental unit"], key="o_pt")
    accommodates  = st.slider("Accommodates", 1, 16, 2, key="o_acc")
    bedrooms      = st.slider("Bedrooms",     0, 10, 1, key="o_bed")
    bathrooms     = st.slider("Bathrooms",    0.5, 6.0, 1.0, 0.5, key="o_bath")
    amenity_count = st.slider("Amenities (count)", 0, 80, 25, key="o_am",
                               help="Top 4 feature: more amenities → higher demand")

with col3:
    st.subheader("Activity & host")
    host_listings  = st.slider("Host total listings", 1, 50, 1, key="o_hl",
                                help="Professional hosts tend to have better-optimised listings")
    minimum_nights = st.slider("Minimum nights", 1, 30, 2, key="o_mn")
    rpm            = st.slider("Reviews/month", 0.0, 20.0, 1.0, 0.1, key="o_rpm")
    n_reviews      = st.slider("Total reviews", 0, 500, 30, key="o_nr")
    rating         = st.slider("Rating", 0.0, 5.0, 4.7, 0.05, key="o_rat")
    target_month   = st.selectbox("Target month (for monthly model)",
                        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                        index=6, key="o_month",
                        help="Only used by the monthly model — annual model ignores this")

month_num = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"].index(target_month) + 1

if st.button("Predict occupancy", type="primary"):
    lat = float(listings[listings["neighbourhood_group"] == municipality]["latitude"].mean())
    lon = float(listings[listings["neighbourhood_group"] == municipality]["longitude"].mean())

    tab_annual, tab_monthly = st.tabs(["Annual (long-run)", f"Monthly ({target_month})"])

    # ── Annual prediction ──────────────────────────────────────────────────
    with tab_annual:
        if annual_dir:
            inference_stats = load_inference_stats(annual_dir)
            row_dict = engineer_row(
                city=city, room_type=room_type,
                neighbourhood_group=municipality, neighbourhood=neighbourhood,
                property_type=property_type,
                host_is_superhost="false", instant_bookable="false",
                minimum_nights=float(minimum_nights),
                number_of_reviews=float(n_reviews),
                number_of_reviews_ltm=float(n_reviews * 0.3),
                reviews_per_month=float(rpm),
                calculated_host_listings_count=float(host_listings),
                availability_365=180.0,   # neutral — feature excluded from this model
                latitude=lat, longitude=lon,
                accommodates=float(accommodates), bedrooms=float(bedrooms),
                beds=float(bedrooms), bathrooms=float(bathrooms),
                review_scores_rating=float(rating),
                review_scores_cleanliness=float(rating),
                review_scores_location=float(rating),
                review_scores_value=float(rating),
                amenity_count=float(amenity_count),
                estimated_occupancy_l365d=0.0,
                inference_stats=inference_stats,
            )
            spark = get_dashboard_spark()
            from pyspark.ml import PipelineModel
            model  = PipelineModel.load(str(annual_dir))
            result = model.transform(spark.createDataFrame([row_dict])).select("prediction","probability").first()
            pred, prob = int(result["prediction"]), result["probability"]
            high_prob = float(prob[1])

            if pred == 1:
                st.success(f"**High-demand listing** ({high_prob*100:.1f}% confidence)")
                st.write("Long-run occupancy predicted **>70%** — fewer than ~110 available days/year.")
            else:
                st.info(f"**Low-to-medium demand** ({float(prob[0])*100:.1f}% confidence)")
            st.progress(high_prob, text=f"High-occupancy probability: {high_prob*100:.1f}%")
        else:
            st.warning("Annual model not found. Run `make ml-occupancy`.")

    # ── Monthly prediction ─────────────────────────────────────────────────
    with tab_monthly:
        if monthly_dir and not str(monthly_dir).endswith(".json"):
            infer_m = load_inference_stats(monthly_dir)
            sin_m   = math.sin(2 * math.pi * month_num / 12)
            cos_m   = math.cos(2 * math.pi * month_num / 12)

            row_m = engineer_row(
                city=city, room_type=room_type,
                neighbourhood_group=municipality, neighbourhood=neighbourhood,
                property_type=property_type,
                host_is_superhost="false", instant_bookable="false",
                minimum_nights=float(minimum_nights),
                number_of_reviews=float(n_reviews),
                number_of_reviews_ltm=float(n_reviews * 0.3),
                reviews_per_month=float(rpm),
                calculated_host_listings_count=float(host_listings),
                availability_365=180.0,
                latitude=lat, longitude=lon,
                accommodates=float(accommodates), bedrooms=float(bedrooms),
                beds=float(bedrooms), bathrooms=float(bathrooms),
                review_scores_rating=float(rating),
                review_scores_cleanliness=float(rating),
                review_scores_location=float(rating),
                review_scores_value=float(rating),
                amenity_count=float(amenity_count),
                estimated_occupancy_l365d=0.0,
                inference_stats=infer_m,
            )
            # Add temporal features.
            row_m["month"]     = float(month_num)
            row_m["sin_month"] = sin_m
            row_m["cos_month"] = cos_m

            spark = get_dashboard_spark()
            from pyspark.ml import PipelineModel
            model_m = PipelineModel.load(str(monthly_dir))
            result_m = model_m.transform(spark.createDataFrame([row_m])).select("prediction","probability").first()
            pred_m = int(result_m["prediction"])
            prob_m = float(result_m["probability"][1])

            if pred_m == 1:
                st.success(f"**High demand in {target_month}** ({prob_m*100:.1f}% confidence)")
                st.write(f"This listing type tends to be **>50% booked** during {target_month}.")
            else:
                st.info(f"**Lower demand in {target_month}** ({(1-prob_m)*100:.1f}% confidence)")
            st.progress(prob_m, text=f"High-demand probability for {target_month}: {prob_m*100:.1f}%")
        else:
            st.warning("Monthly model not found. Run `make ml-monthly`.")

st.divider()

# Seasonal overview from Q2
st.subheader("Seasonal occupancy pattern")
q2 = load_query_result("q02")
if not q2.empty:
    city_q2 = q2[q2["city"] == city].copy()
    if not city_q2.empty:
        city_q2["month_label"] = city_q2["month"].astype(int).map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        )
        st.bar_chart(
            city_q2.sort_values("month").set_index("month_label")["occupancy_rate"],
            use_container_width=True,
        )
else:
    st.info("Run `make queries QUERY=q02` to see seasonal pattern.")
