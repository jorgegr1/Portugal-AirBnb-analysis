import math

import streamlit as st

from _shared import get_dashboard_spark, latest_price_model_dir, load_listings

st.title("Price predictor")

model_dir = latest_price_model_dir()
if model_dir is None:
    st.warning("No trained price model found. Run `make ml-price` first.")
    st.stop()

st.caption(f"Loading model from `{model_dir}` (Spark local[2])")

listings = load_listings()
cities = sorted(listings["city"].unique())
room_types = sorted(listings["room_type"].dropna().unique())
neighbourhoods = sorted(listings["neighbourhood_cleansed"].dropna().unique())
property_types = sorted(listings["property_type"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    city = st.selectbox("City", cities)
    room_type = st.selectbox("Room type", room_types)
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
    property_type = st.selectbox("Property type", property_types)
    superhost = st.checkbox("Superhost")
with col2:
    accommodates = st.slider("Accommodates", 1, 16, 2)
    bedrooms = st.slider("Bedrooms", 0, 10, 1)
    beds = st.slider("Beds", 0, 16, 1)
    rating = st.slider("Review score", 0.0, 5.0, 4.7, 0.1)
    rpm = st.slider("Reviews/month", 0.0, 30.0, 1.5, 0.1)

if st.button("Predict price"):
    spark = get_dashboard_spark()
    from pyspark.ml import PipelineModel
    model = PipelineModel.load(str(model_dir))

    row = spark.createDataFrame([{
        "city": city, "room_type": room_type, "neighbourhood_cleansed": neighbourhood,
        "property_type": property_type, "host_is_superhost": str(bool(superhost)).lower(),
        "accommodates": float(accommodates), "bedrooms": float(bedrooms), "beds": float(beds),
        "minimum_nights": 1.0, "number_of_reviews": 0.0, "number_of_reviews_ltm": 0.0,
        "reviews_per_month": float(rpm), "review_scores_rating": float(rating),
        "calculated_host_listings_count": 1.0,
        "latitude": float(listings.query("city == @city")["latitude"].mean()),
        "longitude": float(listings.query("city == @city")["longitude"].mean()),
    }])
    pred = model.transform(row).select("prediction").first()[0]
    eur = math.expm1(pred)
    st.success(f"Predicted nightly price: **€{eur:.0f}**  (log-pred {pred:.3f})")
