import streamlit as st

from _shared import load_query_result
from egd_airbnb.queries.registry import DESCRIPTIONS

st.title("Query results")

q = st.selectbox("Query", list(DESCRIPTIONS), format_func=lambda k: f"{k} — {DESCRIPTIONS[k]}")
df = load_query_result(q)
if df.empty:
    st.warning(f"No result for {q}. Run `make queries QUERY={q}` or `make queries`.")
    st.stop()

st.dataframe(df, use_container_width=True)

st.subheader("Auto-chart")
num_cols = df.select_dtypes("number").columns.tolist()
cat_cols = df.select_dtypes("object").columns.tolist()
if num_cols and cat_cols:
    col = st.selectbox("Metric to chart", num_cols)
    grp = st.selectbox("Group by", cat_cols)
    chart_data = df.groupby(grp)[col].mean().sort_values(ascending=False).head(20)
    st.bar_chart(chart_data)
elif num_cols:
    st.bar_chart(df[num_cols[0]])
