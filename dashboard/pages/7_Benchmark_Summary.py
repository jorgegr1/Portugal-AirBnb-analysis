import streamlit as st
from _shared import load_benchmark_summary

st.title("Benchmark summary")

df = load_benchmark_summary()
if df.empty:
    st.warning("No benchmark summary found. Run `make bench-summary` first.")
    st.stop()

workloads = sorted(df["workload"].unique())
selected = st.multiselect("Workload", workloads, default=workloads)
sub = df[df["workload"].isin(selected)].sort_values(["workload", "workers"])

st.dataframe(
    sub[
        [
            "workload",
            "platform",
            "workers",
            "runs",
            "median_wall_secs",
            "speedup_vs_1w",
            "efficiency_vs_1w",
        ]
    ],
    use_container_width=True,
)

wide = sub.pivot(index="workers", columns="workload", values="median_wall_secs")
st.subheader("Median wall-clock seconds")
st.line_chart(wide)

speedup = sub.pivot(index="workers", columns="workload", values="speedup_vs_1w")
st.subheader("Speedup vs 1 worker")
st.line_chart(speedup)

best = (
    sub.sort_values(["workload", "median_wall_secs"])
    .groupby("workload", as_index=False)
    .first()[["workload", "workers", "median_wall_secs"]]
)
best["best_config"] = best["workers"].astype(str) + " workers"

st.subheader("Fastest observed configuration")
st.dataframe(
    best[["workload", "best_config", "median_wall_secs"]],
    use_container_width=True,
    hide_index=True,
)
