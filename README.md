# EGD-AirBnB — Project Plan

> Project plan and technical specification for the FEUP/MECD Engenharia de Grandes Dados (EGD) course. Implementation should follow this document section-by-section.

---

## 1. Context

**Course:** Engenharia de Grandes Dados (EGD), FEUP — MECD.
**Goal:** Apply a full Big Data Engineering pipeline (ingest → clean → query → ML → benchmark) to Inside Airbnb data for **Porto + Lisbon** (extensible to more cities), using **Apache Spark**. Deliverables driven by the course brief (`EGD_Proj.pdf`):

- Spark-based processing platform (Local + Google Dataproc).
- Analytical SQL queries **and** an ML pipeline (we do both — required and ML for credit).
- Performance evaluation across **different number of computing nodes**.
- 6-page report (Intro, Related Work, Dataset Profiling, Queries/ML, Performance, Results).
- Presentation: **2026-05-22** · Delivery: **2026-05-29**.

**Status today (2026-05-18):**
- Inside Airbnb cleaned Parquet datasets already exist (used by the EDA): `listings_cleaned.parquet` (~34.6K rows · 86 cols), `calendar_cleaned.parquet` (~14.4M rows), `reviews_cleaned.parquet` (~2.69M rows). City split: Lisbon 62 % / Porto 38 %.
- A complete PySpark **EDA notebook** lives at `notebooks/01_eda.ipynb` — covers price distributions, room types, neighbourhood concentration, host concentration, calendar availability, review trends. **No ML, no formal ingestion script, no benchmarks yet.**

**Strategic choices (confirmed):**
- **Compute:** Develop everything locally on Spark; use the Google $300 free credit only for the *performance/benchmark* section, running the same jobs on Dataproc with 1 / 2 / 4 workers.
- **ML:** Price regression (primary) + high-occupancy classification (bonus).
- **Extras:** Lightweight **Streamlit dashboard**.
- **Plan style:** Task-oriented (not per-person). EDA already exists — migrate it. Ingestion is included briefly (not yet done).

---

## 2. Repository Structure

```
EGD-AirBnb/
├── README.md                       ← this document
├── pyproject.toml                  ← deps + tool config (ruff, pytest)
├── requirements.txt                ← pip-installable fallback
├── Makefile                        ← `make clean`, `make queries`, `make ml`, `make bench`, `make dash`
├── .env.example                    ← INSIDE_AIRBNB_BASE_URL, GCP_PROJECT, GCS_BUCKET
├── .gitignore                      ← data/, .venv/, __pycache__/, *.parquet
│
├── data/
│   ├── raw/                        ← raw CSVs per city (gitignored) — already present
│   │   ├── Porto/{listings,calendar,reviews}.csv
│   │   └── Lisbon/{listings,calendar,reviews}.csv
│   ├── interim/                    ← unified, typed Parquet (gitignored)
│   └── processed/                  ← final cleaned Parquet partitioned by city
│       ├── listings/city=porto/…
│       ├── calendar/city=porto/…
│       └── reviews/city=porto/…
│
├── notebooks/
│   ├── 01_eda.ipynb                ← existing EDA (already moved here)
│   ├── 02_queries_exploration.ipynb
│   ├── 03_ml_exploration.ipynb
│   └── 04_benchmark_analysis.ipynb
│
├── src/egd_airbnb/                 ← installable package
│   ├── __init__.py
│   ├── config.py                   ← paths, city list, spark configs
│   ├── spark_session.py            ← get_spark(app_name, master=None)
│   ├── ingest/
│   │   ├── download.py             ← fetch Inside Airbnb snapshots
│   │   └── unify.py                ← add `city` col, conform schemas across cities
│   ├── cleaning/
│   │   ├── listings.py             ← type casts, null handling, derived cols
│   │   ├── calendar.py
│   │   └── reviews.py
│   ├── queries/
│   │   ├── q01_neighbourhood_prices.py
│   │   ├── q02_seasonality.py
│   │   ├── q03_room_type_occupancy.py
│   │   ├── q04_superhost_premium.py
│   │   ├── q05_host_concentration.py
│   │   ├── q06_reviews_price_corr.py
│   │   ├── q07_top_revenue_neighbourhoods.py
│   │   ├── q08_property_type_share.py
│   │   ├── q09_review_demand_trend.py
│   │   ├── q10_price_outliers.py
│   │   └── registry.py             ← name → callable map for runner & dashboard
│   ├── ml/
│   │   ├── features.py             ← shared feature pipeline (Indexer/OHE/Assembler/Scaler)
│   │   ├── price_regression.py     ← LR / RF / GBT for log-price
│   │   ├── occupancy_classifier.py ← LogReg / RF for high-demand flag
│   │   └── evaluation.py           ← RMSE/MAE/R², AUC/F1 helpers + CV harness
│   ├── benchmark/
│   │   ├── workloads.py            ← chosen heavy jobs to time
│   │   ├── runner.py               ← run workload × parallelism matrix
│   │   └── plot.py                 ← speedup/efficiency charts
│   └── utils/
│       └── io.py                   ← read_parquet / write_parquet helpers
│
├── jobs/                           ← thin spark-submit entry points
│   ├── ingest.py
│   ├── clean.py                    ← runs listings + calendar + reviews
│   ├── run_queries.py              ← `--query qNN | --all`
│   ├── train_price_model.py
│   ├── train_occupancy_model.py
│   └── benchmark.py                ← `--workload {query_seasonality|train_rf} --cores N`
│
├── dashboard/
│   ├── app.py                      ← Streamlit entry
│   └── pages/
│       ├── 1_City_Overview.py
│       ├── 2_Price_Explorer.py
│       ├── 3_Map_View.py           ← folium / pydeck
│       ├── 4_Query_Results.py
│       └── 5_Price_Predictor.py    ← live ML inference
│
├── dataproc/
│   ├── create_cluster.sh           ← gcloud dataproc clusters create … (parameterised)
│   ├── submit_job.sh
│   ├── delete_cluster.sh
│   └── init.sh                     ← pip-install package on workers
│
├── reports/
│   ├── figures/                    ← all paper-ready PNGs (auto-generated)
│   ├── benchmarks/                 ← CSVs of run times
│   ├── results/                    ← query output Parquet/CSV
│   └── final_report/               ← LaTeX/Word draft + assets
│
└── tests/
    └── test_cleaning.py            ← schema + non-null sanity checks
```

---

## 3. Technical Stack

| Concern              | Choice                                                    | Notes |
|----------------------|-----------------------------------------------------------|-------|
| Language             | Python 3.10+                                              | matches PySpark 3.5 |
| Distributed engine   | Apache Spark 3.5                                          | local & Dataproc image `2.2-debian12` (Spark 3.5) |
| File format          | Apache Parquet (Snappy)                                   | partitioned by `city` |
| ML                   | Spark MLlib                                               | Pipeline API throughout |
| Notebooks            | Jupyter                                                   | only for exploration & report figures |
| Plotting             | Matplotlib + Seaborn                                      | reuse EDA style |
| Maps                 | Folium (notebook) + PyDeck (Streamlit)                    | |
| Dashboard            | Streamlit                                                 | reads `reports/results/*.parquet` |
| Cloud (benchmark)    | Google Dataproc + GCS                                     | only for §8 |
| Packaging            | `pyproject.toml` (setuptools) + editable install          | `pip install -e .` |
| Lint / format        | Ruff                                                      | |
| Tests                | Pytest                                                    | smoke tests only |
| Java runtime         | Temurin / OpenJDK 17                                      | required by Spark 3.5 |
| Build automation     | `make`                                                    | thin wrapper over `python -m jobs.*` |

---

## 4. Setup Instructions

### 4.1 Local

```bash
# prerequisites: Python 3.10+, Java 17, ~8 GB free RAM
brew install openjdk@17 python@3.11        # macOS; adjust per OS
export JAVA_HOME=$(/usr/libexec/java_home -v17)

git clone <repo> && cd EGD-AirBnb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                    # installs egd_airbnb + dev tools

cp .env.example .env                       # set Spark / GCP / GCS variables
make ingest                                # data/raw/ CSVs → data/interim/ Parquet
make clean                                 # data/interim/ → data/processed/ partitioned by city
make queries                               # → reports/results/*.parquet
make ml                                    # train + save models/, write metrics
make dash                                  # streamlit run dashboard/app.py
```

### 4.2 Google Dataproc (benchmark only)

```bash
# one-time
gcloud auth login
gcloud config set project $GCP_PROJECT
gsutil mb -l europe-west1 gs://$GCS_BUCKET
gsutil -m cp -r data/processed gs://$GCS_BUCKET/processed/

# package upload
python -m build && gsutil cp dist/egd_airbnb-*.whl gs://$GCS_BUCKET/dist/

# create a 1-worker cluster (rerun with 2 and 4 workers for the matrix)
bash dataproc/create_cluster.sh egd-1w 1
bash dataproc/submit_job.sh egd-1w benchmark.py --workload train_rf

# tear down to stop billing
bash dataproc/delete_cluster.sh egd-1w
```

Recommended machine type: `n2-standard-4` (4 vCPU, 16 GB) for both master and workers — keeps cost low and matches local laptop CPU class so the comparison is honest.

---

## 5. Pipeline Stages

### 5.1 Ingest

- **Source**: raw CSVs already present at `data/raw/Porto/{listings,calendar,reviews}.csv` and `data/raw/Lisbon/...` (~1.5 GB total).
- `src/egd_airbnb/ingest/unify.py` reads each city's CSVs into Spark, **adds `city` column**, conforms columns across cities, and writes one Parquet per dataset partitioned by `city` under `data/interim/`. This is the only place CSVs are touched.
- `src/egd_airbnb/ingest/download.py` is an optional helper to re-fetch Inside Airbnb snapshots (URL format may need updating per `insideairbnb.com/get-the-data`).
- Driver script: `jobs/ingest.py --cities porto,lisbon`.

### 5.2 Clean

Formalises (and slightly extends) what the EDA already does ad-hoc. One module per dataset under `src/egd_airbnb/cleaning/`:

- **`listings.py`**
  - Cast: `price` (`"$1,234.00"` → DoubleType), `*_rate` columns, booleans (`"t"/"f"` → BooleanType), dates.
  - Drop fully-null columns (e.g. `calendar_updated`).
  - Fill: `host_is_superhost` → `false`, `reviews_per_month` → `0.0`.
  - Replace `neighbourhood` with `neighbourhood_cleansed` (EDA showed 45 % null on the former).
  - Derive: `host_type ∈ {single, multi}` from `calculated_host_listings_count > 1`; `is_high_occupancy = (1 - availability_365/365) > 0.7` (target for classifier).
  - Dedup on `id`. Output: `data/processed/listings/city=…/`.
- **`calendar.py`**
  - Drop `calendar.price` (100 % null — EDA confirmed). Keep `adjusted_price`.
  - Cast `date`, parse `available` (`"t"/"f"` → bool).
  - Derive `year`, `month`, `dow`.
- **`reviews.py`**
  - Cast `date`. Derive `year`, `month`. Drop duplicates on `(listing_id, reviewer_id, date)`.

All write modes: `overwrite`, compression `snappy`, partitioned by `city`.

### 5.3 Migration of existing EDA

- **DONE**: `EDG_airbnb_eda.ipynb` is now at `notebooks/01_eda.ipynb` (moved via `git mv` to preserve history).
- **Refactor (pending)**: extract reusable helpers (price casting, missing-value summary, percentile-approx, monthly aggregations) into `src/egd_airbnb/cleaning/*` and `src/egd_airbnb/utils/io.py`, then have the notebook import them — keeps the notebook readable and the code testable.
- **Pin paths**: notebook should read from `data/processed/` (not Colab Drive). Add a top cell that calls `get_spark("eda", master="local[*]")`.

---

## 6. Analytical Queries

10 queries informed by the EDA's findings — each lives in `src/egd_airbnb/queries/qNN_*.py` as a function `def run(spark) -> DataFrame`, registered in `registry.py`, runnable via `python -m jobs.run_queries --query q03` or `--all`. Results land in `reports/results/qNN.parquet` and are picked up by the Streamlit dashboard.

| # | Question                                                                                  | Source(s)            | Key motivation (from EDA) |
|---|-------------------------------------------------------------------------------------------|----------------------|---------------------------|
| 1 | Top / bottom 10 neighbourhoods by median price per city                                  | listings             | Strong price spread between neighbourhoods |
| 2 | Price seasonality: median `adjusted_price` per month per city                            | calendar + listings  | Calendar covers Jun-2025 → Jul-2026 |
| 3 | Average availability (=demand proxy) by `room_type` × city                               | listings             | EDA showed bimodal availability |
| 4 | Superhost vs regular host: price, rating, reviews_per_month                              | listings             | `host_is_superhost` clean after fill |
| 5 | Host concentration (Pareto curve: % of listings held by top 10 % of hosts)               | listings             | 75-78 % multi-listing hosts |
| 6 | Pearson correlation: `reviews_per_month` × `price` per neighbourhood                     | listings             | Demand vs price elasticity |
| 7 | Top 10 neighbourhoods by `estimated_revenue_l365d`                                       | listings             | Field present, untouched in EDA |
| 8 | Property-type market-share per city (entire home vs private room vs hotel vs shared)     | listings             | EDA: 84/77 % entire-home |
| 9 | Monthly review count + YoY growth 2023-2025 per city                                     | reviews              | Lisbon 1.7× Porto |
| 10| Price outlier prevalence: % listings above {€500, €1 000, €2 000} per city × room_type   | listings             | 99-pct €1 450 vs €639 |

Implementation rules:
- Always include `city` as a grouping/output column for cross-city comparison.
- Cap visualisation prices at €500 to mirror EDA conventions (the *outlier* query is the one exception).
- Each query writes a small (≤ 10 K rows) result; the dashboard reads these directly.

---

## 7. ML Pipeline

Two related problems, sharing one feature module.

### 7.1 Shared features (`src/egd_airbnb/ml/features.py`)

Numeric: `accommodates`, `bedrooms`, `beds`, `bathrooms` (if present), `minimum_nights`, `number_of_reviews`, `number_of_reviews_ltm`, `reviews_per_month`, `review_scores_rating`, `calculated_host_listings_count`, `latitude`, `longitude`.
Categorical (one-hot): `city`, `room_type`, `property_type`, `neighbourhood_cleansed`, `host_is_superhost`.
Pipeline stages: `StringIndexer(handleInvalid='keep')` → `OneHotEncoder` → `VectorAssembler` → `StandardScaler(withMean=False)`.

### 7.2 Price regression (primary — `price_regression.py`)

- Target: `log1p(price)` (heavy right skew confirmed by EDA).
- Filter to `price between 10 and 1000` (drops ~1 % outliers).
- Split: 80/20 random, seed = 42, stratified-ish by city (use weighted sampling).
- Models:
  1. `LinearRegression` — interpretable baseline.
  2. `RandomForestRegressor(numTrees=100, maxDepth=10)`.
  3. `GBTRegressor(maxIter=100, maxDepth=6)`.
- Tuning: `CrossValidator` (3 folds) over a small grid on the winning model — `maxDepth ∈ {6,10,14}` and `numTrees ∈ {50,150}` for RF.
- Metrics: **RMSE, MAE, R²** (on hold-out, computed back in € by `expm1`). Per-city breakdown.
- Output: model saved under `models/price_<algo>/`, metrics JSON under `reports/results/ml/price_metrics.json`, feature-importance bar chart for the report.

### 7.3 Occupancy classification (bonus — `occupancy_classifier.py`)

- Target: `is_high_occupancy` derived in cleaning (`availability_365/365 < 0.30`).
- Same feature set minus `availability_*` (would leak).
- Models: `LogisticRegression` (baseline) + `RandomForestClassifier`.
- Metrics: **AUC-ROC, F1, precision/recall, confusion matrix**.
- Class-imbalance: report base rate, use `weightCol` if rate < 35 %.

### 7.4 Reproducibility

- Set `spark.sql.shuffle.partitions=200` for local, `=cores*3` on Dataproc.
- Fix seeds (`42`) on every random op.
- Persist input features as Parquet between feature-engineering and model fitting to avoid re-computation across trials.

---

## 8. Performance / Benchmarking

The single most-credit-bearing section of the report. Run **two workloads** across **two platforms × multiple parallelism levels**, each at least **3 times** (report median + p95).

### 8.1 Workloads (`benchmark/workloads.py`)

1. **Seasonality query** (Q2) — heavy because of the calendar × listings join over 14.4 M rows.
2. **RandomForest price training** — heaviest CPU job; numTrees=100, maxDepth=10, full feature set.

### 8.2 Configuration matrix

| Platform     | Parallelism levels                                          | Notes |
|--------------|-------------------------------------------------------------|-------|
| Local        | `local[1]`, `local[2]`, `local[4]`, `local[8]`              | thread-level, fast feedback |
| Dataproc     | 1 worker, 2 workers, 4 workers (each `n2-standard-4`)       | **the real "different number of nodes" datum** |

Reading the brief carefully, "different number of computing nodes" is best satisfied by Dataproc — local thread scaling is reported as a complementary curve.

### 8.3 Measurement

- `runner.py` wraps each workload in a context manager: triggers an action (`.count()` or `.write()` to a temp sink), records `time.perf_counter()` deltas, Spark stage metrics via `spark.sparkContext.statusTracker()`, and shuffle bytes from the Spark UI history server (Dataproc only).
- Output: `reports/benchmarks/{workload}_{platform}.csv` with columns `run_id, parallelism, wall_secs, shuffle_mb`.
- `plot.py` builds two figures per workload: (a) wall-time vs parallelism, (b) speedup + efficiency curves (`S(n) = T(1)/T(n)`, `E(n) = S(n)/n`).

### 8.4 Cost guardrails for Dataproc

- Keep cluster lifetime under 2 hours per session: create → submit → delete.
- Use ephemeral GCS staging; do **not** keep raw data in BigQuery for this project.
- Estimated cost: ≤ €15 for the whole benchmark suite at current pricing.

---

## 9. Streamlit Dashboard

`dashboard/app.py` — multi-page; reads exclusively from `reports/results/` so it stays decoupled from Spark.

| Page              | Content |
|-------------------|---------|
| City Overview     | KPIs (listings, hosts, median price, % superhost), filterable by city |
| Price Explorer    | Histograms, box-plot by room type, neighbourhood drill-down |
| Map View          | PyDeck scatter coloured by price quartile; toggle Porto/Lisbon |
| Query Results     | Tabbed view of Q1–Q10 (table + auto chart based on schema) |
| Price Predictor   | Form (room_type, accommodates, neighbourhood, …) → loads saved RF/GBT model → predicted nightly price + 80 % interval |

Run locally with `streamlit run dashboard/app.py`. Model loading uses Spark in `local[2]` mode — fast enough for single-row inference.

---

## 10. Deliverables checklist (report alignment)

| Report section            | Backed by                                                       |
|---------------------------|-----------------------------------------------------------------|
| Introduction              | This README §1; problem statement                               |
| Related Work              | Brief literature on Airbnb price modelling (Kalehbasti et al. 2019; Tang & Sangani 2015); Inside Airbnb methodology |
| Dataset Profiling         | Notebook `01_eda.ipynb` + figures exported to `reports/figures/`|
| Queries                   | `reports/results/q*.parquet` + commentary                       |
| ML Pipeline               | `reports/results/ml/*.json` + feature-importance plot           |
| Performance / Scalability | `reports/benchmarks/*.csv` + speedup/efficiency plots           |
| Results Analysis          | Cross-cutting discussion (per-city differences, model errors, scaling bottlenecks) |

---

## 11. Suggested Timeline (today = 2026-05-18; presentation 2026-05-22)

| Day        | Output                                                                                 |
|------------|----------------------------------------------------------------------------------------|
| Mon 05-18  | Repo scaffold; move EDA to `notebooks/01_eda.ipynb`; write `ingest` + `cleaning` modules; re-emit `data/processed/` |
| Tue 05-19  | Implement queries Q1–Q10; export result Parquets + first figures                       |
| Wed 05-20  | ML pipelines (price + occupancy); save models + metrics; draft Streamlit dashboard     |
| Thu 05-21  | Local benchmarks; Dataproc setup + cloud benchmarks; speedup plots; finish dashboard   |
| Fri 05-22  | **Presentation.** Slides built from notebook figures + dashboard demo                  |
| 05-23 → 27 | Write 6-page report; iterate figures; polish                                           |
| Thu 05-28  | Internal review + report freeze                                                        |
| Fri 05-29  | **Delivery.**                                                                          |

Buffer is thin — keep query and ML scope at the level above; do not add features.

---

## 12. Verification

End-to-end smoke test (≤ 5 min on a laptop):

```bash
make ingest CITIES=porto                       # one city only for the quick check
make clean
pytest                                         # cleaning rules verified on synthetic input
make queries QUERY=q01                         # writes reports/results/q01.parquet
python -m jobs.train_price_model --algo lr --sample 0.1   # quick LR training run
make dash                                      # dashboard loads, all pages render
```

Production / Dataproc verification:

```bash
bash dataproc/create_cluster.sh egd-1w 1
bash dataproc/submit_job.sh egd-1w benchmark.py --workload query_seasonality --runs 3
gsutil cp gs://$GCS_BUCKET/reports/benchmarks/*.csv reports/benchmarks/
bash dataproc/delete_cluster.sh egd-1w
```

Report-time spot-check: each figure in `reports/figures/` is regenerable from a single notebook cell or script; each number cited in the report is traceable to a file under `reports/{results,benchmarks}/`.

---

## 13. Critical files to create / modify

- **DONE** `EDG_airbnb_eda.ipynb` → `notebooks/01_eda.ipynb` (its data paths still need updating to `data/processed/`).
- **CREATE** repo scaffold per §2 (empty `__init__.py` files where applicable).
- **CREATE** `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`.
- **CREATE** `src/egd_airbnb/{config,spark_session}.py`, full `ingest/`, `cleaning/`, `queries/`, `ml/`, `benchmark/` modules.
- **CREATE** `jobs/*.py` thin spark-submit entry points.
- **CREATE** `dataproc/*.sh` cluster-management scripts.
- **CREATE** `dashboard/app.py` + pages.
- **CREATE** `tests/test_cleaning.py` — at minimum: schema, non-null assertions on `price`, `city`, `room_type`.

Out of scope (deliberately): Airflow/Prefect orchestration, Delta Lake, BigQuery, CI/CD, Docker images. Reach for these only if §11 buffer allows.
