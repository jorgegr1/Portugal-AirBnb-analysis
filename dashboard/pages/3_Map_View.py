import pydeck as pdk
import streamlit as st

from _shared import load_listings

st.title("Map view")

df = load_listings()
df = df.dropna(subset=["latitude", "longitude", "price"]).query("10 <= price <= 1000")
city = st.selectbox("City", sorted(df["city"].unique()))
sub = df[df["city"] == city]

quartile = sub["price"].quantile([0.25, 0.5, 0.75]).to_dict()


def _color(p: float) -> list[int]:
    if p <= quartile[0.25]: return [0, 200, 0, 140]
    if p <= quartile[0.5]:  return [255, 200, 0, 140]
    if p <= quartile[0.75]: return [255, 120, 0, 140]
    return [255, 0, 0, 140]


sub = sub.assign(color=sub["price"].apply(_color))

layer = pdk.Layer(
    "ScatterplotLayer",
    sub,
    get_position="[longitude, latitude]",
    get_color="color",
    get_radius=30,
    pickable=True,
)
view = pdk.ViewState(latitude=sub["latitude"].mean(), longitude=sub["longitude"].mean(), zoom=12)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                         tooltip={"text": "€{price}\n{room_type}\n{neighbourhood_cleansed}"}))
