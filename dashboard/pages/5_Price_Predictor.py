import math

import streamlit as st

from _shared import get_dashboard_spark, latest_price_model_dir, load_listings

st.title("Price predictor")

model_dir = latest_price_model_dir()
if model_dir is None:
    st.warning("No trained price model found. Run `make ml-price` first.")
    st.stop()

st.caption(f"Model: `{model_dir.name}` · Spark local[2]")

listings = load_listings()
cities          = sorted(listings["city"].unique())
room_types      = sorted(listings["room_type"].dropna().unique())
municipalities  = sorted(listings["neighbourhood_group"].dropna().unique())
neighbourhoods  = sorted(listings["neighbourhood"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    city             = st.selectbox("City", cities)
    municipality     = st.selectbox("Municipality", sorted(listings[listings["city"] == city]["neighbourhood_group"].dropna().unique()))
    neighbourhood    = st.selectbox("Neighbourhood", sorted(listings[listings["neighbourhood_group"] == municipality]["neighbourhood"].dropna().unique()))
    room_type        = st.selectbox("Room type", room_types)
with col2:
    minimum_nights   = st.slider("Minimum nights", 1, 30, 2)
    n_reviews        = st.slider("Number of reviews", 0, 500, 20)
    rpm              = st.slider("Reviews/month", 0.0, 20.0, 1.0, 0.1)
    availability     = st.slider("Availability (days/year)", 0, 365, 120)
    host_listings    = st.slider("Host's total listings", 1, 50, 1)

if st.button("Predict price", type="primary"):
    spark = get_dashboard_spark()
    from pyspark.ml import PipelineModel
    model = PipelineModel.load(str(model_dir))

    lat = float(listings[listings["city"] == city]["latitude"].mean())
    lon = float(listings[listings["city"] == city]["longitude"].mean())

    row = spark.createDataFrame([{
        "city": city,
        "room_type": room_type,
        "neighbourhood_group": municipality,
        "neighbourhood": neighbourhood,
        "minimum_nights": float(minimum_nights),
        "number_of_reviews": float(n_reviews),
        "number_of_reviews_ltm": float(n_reviews),
        "reviews_per_month": float(rpm),
        "calculated_host_listings_count": float(host_listings),
        "availability_365": float(availability),
        "latitude": lat,
        "longitude": lon,
    }])
    pred = model.transform(row).select("prediction").first()[0]
    eur = math.expm1(pred)
    st.success(f"Predicted nightly price: **€{eur:.0f}**")
