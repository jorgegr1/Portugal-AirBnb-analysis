import streamlit as st

from _shared import load_listings, load_neighbourhoods

st.title("City overview")

listings = load_listings()
nb = load_neighbourhoods()

city = st.selectbox("City", sorted(listings["city"].unique()))
sub = listings[listings["city"] == city]

# Headline metrics.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings",      f"{len(sub):,}")
c2.metric("Hosts",         f"{sub['host_id'].nunique():,}")
c3.metric("Median price (€)", f"{sub['price'].median():.0f}" if sub["price"].notna().any() else "N/A")
c4.metric("Multi-listing host share", f"{(sub['host_type'] == 'multi').mean() * 100:.1f}%")

st.divider()

# Municipality breakdown.
st.subheader("By municipality (neighbourhood_group)")
mun = (
    sub.groupby("neighbourhood_group")
    .agg(n_listings=("id", "count"), median_price=("price", "median"))
    .sort_values("n_listings", ascending=False)
    .reset_index()
)
st.dataframe(mun, use_container_width=True)

st.subheader("Top 10 neighbourhoods by listing count")
top_nb = sub.groupby("neighbourhood")["id"].count().sort_values(ascending=False).head(10)
st.bar_chart(top_nb)

st.subheader("Room type distribution")
st.bar_chart(sub["room_type"].value_counts(normalize=True))

# Reference table: full neighbourhood hierarchy for this city.
if not nb.empty:
    with st.expander("Full neighbourhood hierarchy"):
        nb_city = nb[nb["city"] == city][["neighbourhood_group", "neighbourhood"]].sort_values(
            ["neighbourhood_group", "neighbourhood"]
        )
        st.dataframe(nb_city, use_container_width=True)
