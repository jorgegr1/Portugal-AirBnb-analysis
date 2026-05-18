import streamlit as st

from _shared import load_listings

st.title("Price explorer")

df = load_listings()
cities = st.multiselect("City", sorted(df["city"].unique()), default=sorted(df["city"].unique()))
room_types = st.multiselect("Room type", sorted(df["room_type"].dropna().unique()), default=sorted(df["room_type"].dropna().unique()))
price_cap = st.slider("Price cap (€)", 50, 1500, 500, 50)

sub = df[df["city"].isin(cities) & df["room_type"].isin(room_types)]
sub = sub[sub["price"].between(10, price_cap)]

st.metric("Selected listings", f"{len(sub):,}")
st.bar_chart(sub.groupby("room_type")["price"].median())
st.subheader("Price histogram")
st.bar_chart(sub["price"].round(-1).value_counts().sort_index())
