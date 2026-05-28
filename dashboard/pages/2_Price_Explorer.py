import streamlit as st
from _shared import load_listings

st.title("Price explorer")

df = load_listings()

col1, col2 = st.columns(2)
with col1:
    cities = st.multiselect(
        "City", sorted(df["city"].unique()), default=sorted(df["city"].unique())
    )
    room_types = st.multiselect(
        "Room type",
        sorted(df["room_type"].dropna().unique()),
        default=sorted(df["room_type"].dropna().unique()),
    )
with col2:
    all_groups = sorted(df["neighbourhood_group"].dropna().unique())
    selected_groups = st.multiselect(
        "Municipality (neighbourhood_group)", all_groups, default=all_groups
    )
    price_cap = st.slider("Price cap (€)", 50, 1000, 300, 25)

sub = df[
    df["city"].isin(cities)
    & df["room_type"].isin(room_types)
    & df["neighbourhood_group"].isin(selected_groups)
    & df["price"].between(1, price_cap)
]

st.metric("Selected listings", f"{len(sub):,}")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Median price by room type")
    st.bar_chart(sub.groupby("room_type")["price"].median())
with col_b:
    st.subheader("Median price by municipality")
    st.bar_chart(sub.groupby("neighbourhood_group")["price"].median().sort_values(ascending=False))

st.subheader("Price distribution")
st.bar_chart(sub["price"].round(-1).value_counts().sort_index().head(40))
