import pydeck as pdk
import streamlit as st
from _shared import load_listings

st.title("Map view")

df = load_listings().dropna(subset=["latitude", "longitude", "price"]).query("price > 0")
city = st.selectbox("City", sorted(df["city"].unique()))

all_groups = sorted(df[df["city"] == city]["neighbourhood_group"].dropna().unique())
selected_groups = st.multiselect("Municipality filter", all_groups, default=all_groups)

sub = df[(df["city"] == city) & (df["neighbourhood_group"].isin(selected_groups))]
price_cap = st.slider("Price cap for colour scale (€)", 50, 1000, 300, 25)
sub = sub[sub["price"] <= price_cap * 2]  # keep outliers visible but cap colour


def _color(p: float) -> list[int]:
    q1, q2, q3 = sub["price"].quantile([0.25, 0.5, 0.75])
    if p <= q1:
        return [0, 200, 0, 160]
    if p <= q2:
        return [255, 200, 0, 160]
    if p <= q3:
        return [255, 120, 0, 160]
    return [220, 0, 0, 160]


sub = sub.assign(color=sub["price"].apply(_color))

layer = pdk.Layer(
    "ScatterplotLayer",
    sub,
    get_position="[longitude, latitude]",
    get_color="color",
    get_radius=40,
    pickable=True,
)
view = pdk.ViewState(latitude=sub["latitude"].mean(), longitude=sub["longitude"].mean(), zoom=12)
st.pydeck_chart(
    pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "€{price}/night\n{room_type}\n{neighbourhood}\n({neighbourhood_group})"},
    )
)

st.caption("Green = cheapest quartile · Yellow = 2nd · Orange = 3rd · Red = most expensive")
