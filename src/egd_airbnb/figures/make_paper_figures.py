"""Generate publication figures for the final report.

Reads the Q2 (monthly seasonality) and Q5 (host concentration) query outputs
from ``reports/results/`` and writes single-column matplotlib figures to
``reports/figures/``.

Styling targets the IEEE conference template: serif fonts to match the body,
Okabe–Ito colour palette (colour-blind safe), no chartjunk, vector PDF +
high-DPI PNG fallback.

Run from the repository root::

    python -m egd_airbnb.figures.make_paper_figures
    # or
    python src/egd_airbnb/figures/make_paper_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "reports" / "results"
FIG_DIR = REPO_ROOT / "reports" / "figures"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Okabe–Ito palette (colour-blind safe). Mapped so adjacent-tour cities are
# visually distinguishable in the figure and consistent across both plots.
COLOURS = {
    "porto":     "#0072B2",  # blue
    "lisbon":    "#E69F00",  # orange
    "madrid":    "#009E73",  # bluish green
    "barcelona": "#D55E00",  # vermillion
}
MARKERS = {
    "porto":     "o",
    "lisbon":    "s",
    "madrid":    "^",
    "barcelona": "D",
}
LABELS = {
    "porto":     "Porto",
    "lisbon":    "Lisbon",
    "madrid":    "Madrid",
    "barcelona": "Barcelona",
}
# Order kept stable across both figures and the legend.
CITY_ORDER = ["porto", "lisbon", "madrid", "barcelona"]

# IEEE single-column figure: \columnwidth ≈ 3.5 in. Keep aspect ratio compact.
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.4

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # TrueType, editable in vector tools
    "ps.fonttype": 42,
})

MONTH_TICKS = list(range(1, 13))
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


# ---------------------------------------------------------------------------
# Figure 1 — Monthly occupancy seasonality per city
# ---------------------------------------------------------------------------

def make_seasonality_figure(out_stem: Path) -> dict:
    """Plot monthly occupancy rate per city, averaged across available years.

    Returns the per-city summary statistics so the caller (or the paper) can
    cross-check the JJA-vs-annual lift used in the body text.
    """
    df = pd.read_parquet(RESULTS_DIR / "q02.parquet")
    monthly = (
        df.groupby(["city", "month"], as_index=False)["occupancy_rate"]
        .mean()
        .pivot(index="month", columns="city", values="occupancy_rate")
        .reindex(MONTH_TICKS)
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    for city in CITY_ORDER:
        ax.plot(
            monthly.index,
            monthly[city],
            color=COLOURS[city],
            marker=MARKERS[city],
            label=LABELS[city],
        )

    ax.set_xticks(MONTH_TICKS)
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Mean occupancy rate")
    ax.set_ylim(0.10, 0.70)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.7)

    # JJA shading to make the summer window legible at a glance.
    ax.axvspan(5.5, 8.5, color="0.85", alpha=0.35, zorder=0, linewidth=0)

    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.3)

    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"))
    plt.close(fig)

    summary = {}
    for city in CITY_ORDER:
        col = monthly[city]
        jja = col.loc[[6, 7, 8]].mean()
        annual = col.mean()
        summary[city] = {
            "annual_mean": round(annual, 3),
            "jja_mean": round(jja, 3),
            "jja_vs_annual_pct": round((jja - annual) / annual * 100, 1),
            "peak_month": int(col.idxmax()),
            "peak_value": round(col.max(), 3),
        }
    return summary


# ---------------------------------------------------------------------------
# Figure 2 — Pareto curve of host concentration
# ---------------------------------------------------------------------------

def make_pareto_figure(out_stem: Path) -> dict:
    """Plot cumulative inventory share vs cumulative host share per city."""
    df = pd.read_parquet(RESULTS_DIR / "q05.parquet")
    df = df.sort_values(["city", "host_share"]).reset_index(drop=True)

    # The Q5 output is a few quantiles per city. We anchor the curve at the
    # origin to make the Pareto shape obvious, then connect through the
    # provided percentile points up to (1, 1).
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    # Reference line: equal distribution (no concentration).
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=0.7,
            label="Equal share")

    for city in CITY_ORDER:
        sub = df[df["city"] == city].sort_values("host_share")
        xs = [0.0, *sub["host_share"].tolist()]
        ys = [0.0, *sub["listings_share"].tolist()]
        ax.plot(
            xs, ys,
            color=COLOURS[city],
            marker=MARKERS[city],
            label=LABELS[city],
        )

    # Highlight the top-decile line (host_share == 0.10) as a visual anchor
    # for the claim in §III.
    ax.axvline(0.10, color="0.55", linestyle=":", linewidth=0.6)
    ax.text(0.105, 0.07, "top 10 %\nof hosts",
            fontsize=6, color="0.35", va="bottom")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Cumulative share of hosts (ranked by listing count)")
    ax.set_ylabel("Cumulative share of listings")
    ax.set_xticks([0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="both", linestyle=":", linewidth=0.4, alpha=0.7)
    ax.legend(loc="lower right", handlelength=1.6, borderaxespad=0.3)

    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"))
    plt.close(fig)

    top10 = (df[df["host_share"] == 0.10]
             .set_index("city")["listings_share"]
             .round(3).to_dict())
    return {"top_decile_listings_share": top10}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    seasonality = make_seasonality_figure(FIG_DIR / "seasonality_by_city")
    pareto = make_pareto_figure(FIG_DIR / "host_pareto")

    print("[seasonality] written:",
          (FIG_DIR / "seasonality_by_city.pdf").relative_to(REPO_ROOT))
    for city, stats in seasonality.items():
        print(f"  {city}: annual={stats['annual_mean']:.3f}, "
              f"JJA={stats['jja_mean']:.3f}, "
              f"lift={stats['jja_vs_annual_pct']:+.1f}%, "
              f"peak=M{stats['peak_month']:02d} ({stats['peak_value']:.3f})")
    print("[pareto] written:",
          (FIG_DIR / "host_pareto.pdf").relative_to(REPO_ROOT))
    for city, share in pareto["top_decile_listings_share"].items():
        print(f"  top-10%-of-hosts share in {city}: {share*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
