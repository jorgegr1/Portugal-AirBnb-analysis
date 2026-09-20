# EGD-AirBnB

End-to-end Big Data pipeline for the analytical exploration and predictive
modelling of [Inside Airbnb](https://insideairbnb.com/get-the-data/) short-term
rental data across four Iberian cities: **Porto, Lisbon, Madrid and
Barcelona**. Built on Apache Spark.

Course project for *Engenharia de Grandes Dados* (EGD), FEUP - MECD.

The pipeline ingests **68,521 listings**, a **25.8 M-row** daily availability
calendar and **4.90 M reviews** from raw CSV into partitioned Parquet, runs
eleven cross-city analytical queries, trains three MLlib models, and benchmarks
five Spark workloads on Google Dataproc across 1/2/4 worker nodes. Results feed
a Streamlit dashboard and the IEEE-style report
([`reports/EGD_Report.pdf`](reports/EGD_Report.pdf)).

---

## Pipeline

```
ingest  →  clean  →  queries  →  ml  →  benchmark
 CSV       Parquet    q01-q11    models    Dataproc
        (by city)              + metrics   1/2/4 nodes
```

| Stage | Module | Output |
|-------|--------|--------|
| **Ingest** | `ingest/unify.py` | `data/interim/` — unified Parquet, one `city` column per dataset |
| **Clean** | `cleaning/{listings,calendar,reviews,neighbourhoods}.py` | `data/processed/` — typed, deduplicated, partitioned by `city` |
| **Queries** | `queries/q01…q11_*.py` | `reports/results/qNN.parquet` |
| **ML** | `ml/{price_regression,occupancy_classifier,monthly_occupancy}.py` | `models/`, `reports/results/ml/*.json` |
| **Benchmark** | `benchmark/`, `jobs/benchmark_dataproc.py` | `reports/benchmarks/*.txt` |

### Analytical queries (11)

Each lives in `src/egd_airbnb/queries/qNN_*.py` as `def run(spark) -> DataFrame`,
registered in `registry.py`, runnable via `--query qNN` or `--all`.

1. Top/bottom neighbourhoods by median price
2. Price/occupancy seasonality per month per city
3. Availability (demand proxy) by room type × city
4. Superhost vs regular host premium
5. Host concentration (Pareto curve)
6. Reviews-per-month × price correlation per neighbourhood
7. Top neighbourhoods by estimated revenue
8. Property-type market share per city
9. Monthly review count + YoY growth
10. Price-outlier prevalence per city × room type
11. Municipality-level comparison via `neighbourhood_group`

### ML models

| Model | Task | Headline metric |
|-------|------|-----------------|
| Gradient-boosted regressor (`price_regression.py`) | nightly `log1p(price)` | R²_log = 0.72, RMSE €78 |
| Random forest (`occupancy_classifier.py`) | long-run high-occupancy listing | AUC = 0.73 |
| Month-aware random forest (`monthly_occupancy.py`) | high-demand month (cyclic month encoding) | AUC = 0.77 |

All three share the leakage-safe feature pipeline in `ml/features.py`
(`StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler`, plus
train-split-only target encoding of the high-cardinality neighbourhood field).

---

## Repository layout

```
EGD-AirBnb/
├── src/egd_airbnb/          installable package
│   ├── config.py            paths, city list, Spark configs
│   ├── spark_session.py     get_spark(app_name, master=None)
│   ├── ingest/              download + unify raw CSVs
│   ├── cleaning/            per-dataset cleaning modules
│   ├── queries/             q01–q11 + registry
│   ├── ml/                  features, price, occupancy, monthly + evaluation
│   ├── benchmark/           workloads, runner, plot
│   ├── figures/             make_paper_figures.py (report figures)
│   └── utils/               read_parquet / write_parquet helpers
├── jobs/                    thin entry points (python -m jobs.*)
├── dashboard/               Streamlit app + 6 pages
├── dataproc/                gcloud cluster create/submit/delete/init scripts
├── notebooks/               01_eda.ipynb, 02_eda_4cities.ipynb
├── reports/
│   ├── figures/             paper-ready PNG/PDF
│   ├── benchmarks/          Dataproc run logs
│   ├── results/             query Parquet + ml/*.json
│   └── EGD_Report.pdf       final IEEE-style report (deliverable)
├── docs/                    BENCHMARK_GUIDE.md + course briefs (PDF)
├── tests/                   pytest cleaning sanity checks
├── run_benchmarks.bat       full Dataproc benchmark suite (Windows)
├── Makefile                 make ingest|clean|queries|ml|bench-local|dash|test
├── pyproject.toml           deps + ruff/pytest config
└── requirements.txt         pip fallback
```

`data/`, `models/`, `*.parquet`, `*.csv` and LaTeX build artefacts are
gitignored; only `.gitkeep` placeholders are tracked.

---

## Setup

```bash
# prerequisites: Python 3.10+, Java 17, ~8 GB free RAM
export JAVA_HOME=$(/usr/libexec/java_home -v17)   # macOS

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,dashboard]"                 # or: make install
cp .env.example .env                              # set Spark / GCP vars
```

Place the raw Inside Airbnb CSVs under `data/raw/<City>/`
(`listings.csv`, `listings-detailed.csv`, `calendar.csv`, `reviews.csv`,
`neighbourhoods.csv`). Use **June 2025 snapshots** — later snapshots have null
listing prices (see `docs/BENCHMARK_GUIDE.md`).

### Run the pipeline

```bash
make ingest        # data/raw/ CSVs → data/interim/ Parquet
make clean         # data/interim/ → data/processed/ (partitioned by city)
make queries       # → reports/results/qNN.parquet   (QUERY=q03 for one)
make ml            # train price + occupancy + monthly models, write metrics
make dash          # streamlit run dashboard/app.py
make test          # pytest cleaning sanity checks
```

### Benchmark on Dataproc

The scalability study runs five workloads
(`query_seasonality`, `train_rf`, `review_demand`, `calendar_revenue`,
`full_pipeline`) across 1/2/4 worker `n2-standard-2` nodes. See
[`docs/BENCHMARK_GUIDE.md`](docs/BENCHMARK_GUIDE.md) for the full procedure.

```bash
# one cluster size at a time (create → submit → delete)
bash dataproc/create_cluster.sh egd-cluster 2
bash dataproc/submit_job.sh egd-cluster benchmark_dataproc.py --workload query_seasonality --runs 3
bash dataproc/delete_cluster.sh egd-cluster
```

`run_benchmarks.bat` automates the full create/submit/delete sweep across all
cluster sizes on Windows.

---

## Dashboard

`streamlit run dashboard/app.py` — reads exclusively from `reports/results/`,
so it stays decoupled from Spark. Six pages: City Overview, Price Explorer,
Map View, Query Results, Price Predictor, Occupancy Predictor (the predictor
pages load the saved MLlib models for live single-row inference).

---

## Technical stack

| Concern | Choice |
|---------|--------|
| Language / engine | Python 3.10+ · Apache Spark 3.5.1 (PySpark) |
| Storage | Apache Parquet (Snappy), partitioned by `city` |
| ML | Spark MLlib Pipeline API |
| Dashboard | Streamlit + PyDeck |
| Cloud (benchmark) | Google Dataproc (`2.2-debian12`) + GCS |
| Lint / format / test | Ruff · Pytest |
| Build | `pyproject.toml` (setuptools), editable install |
