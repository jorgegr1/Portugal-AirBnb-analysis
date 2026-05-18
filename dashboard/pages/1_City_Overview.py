import streamlit as st

from _shared import load_listings

st.title("City overview")

df = load_listings()
city = st.selectbox("City", sorted(df["city"].unique()))
sub = df[df["city"] == city]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", f"{len(sub):,}")
c2.metric("Hosts", f"{sub['host_id'].nunique():,}")
c3.metric("Median price (€)", f"{sub['price'].median():.0f}")
c4.metric("Superhost share", f"{sub['host_is_superhost'].mean() * 100:.1f}%")

st.subheader("Top 10 neighbourhoods by listing count")
top = sub.groupby("neighbourhood_cleansed")["id"].count().sort_values(ascending=False).head(10)
st.bar_chart(top)

st.subheader("Room type distribution")
st.bar_chart(sub["room_type"].value_counts(normalize=True))
