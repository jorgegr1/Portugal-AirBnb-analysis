# Guia de Benchmarking — Dataproc

Documentacao completa do processo de benchmarking no Google Cloud Dataproc.
Cobre: configuracao do ambiente, pipeline de dados, execucao de benchmarks e interpretacao de resultados.

---

## 1. Arquitetura Geral

```
[Dados Raw (CSV.gz)]          [Google Cloud]
  data/raw/                     GCS: gs://egd-airbnb-bucket/
    Porto/                        data/processed/
    Lisbon/                         listings/city=porto/...
    Madrid/                         calendar/city=porto/...
    Barcelona/                      reviews/city=porto/...

[Pipeline Local (Docker)]     [Dataproc Cluster]
  ingest -> clean -> queries    Spark on YARN
  Parquet partitioned by city   Le dados do GCS
                                Executa workloads de benchmark
```

**Fluxo resumido:**
1. Descarregar dados raw do Inside Airbnb
2. Correr pipeline local (ingest + clean) via Docker
3. Upload dos dados processados para GCS
4. Criar cluster Dataproc e submeter benchmarks
5. Recolher resultados e apagar cluster

---

## 2. Pre-requisitos

### 2.1 Software local
- **Docker Desktop** (para correr PySpark localmente no Windows)
- **gcloud CLI** instalado e autenticado (`gcloud auth login`)
- **Git** para gestao de versoes

### 2.2 Google Cloud
- Projeto GCP criado (ex: `egd-airbnb-2026`)
- APIs ativadas:
  - Dataproc API
  - Cloud Storage API
  - Compute Engine API
- Bucket GCS criado: `gs://egd-airbnb-bucket`
- Quota minima: 12 vCPUs em `CPUS_ALL_REGIONS` (permite ate 4 workers n2-standard-2 + 1 master)

### 2.3 Variaveis de ambiente (.env)
```
GCP_PROJECT=egd-airbnb-2026
GCP_REGION=europe-west1
GCS_BUCKET=egd-airbnb-bucket
DATAPROC_IMAGE_VERSION=2.2-debian12
DATAPROC_MACHINE_TYPE=n2-standard-2
```

---

## 3. Dados

### 3.1 Fonte
Inside Airbnb (https://insideairbnb.com/get-the-data/)

### 3.2 Cidades e snapshots
| Cidade    | Snapshot   | Pasta raw        |
|-----------|------------|------------------|
| Porto     | 2025-06-15 | `data/raw/Porto/`     |
| Lisbon    | 2025-06-15 | `data/raw/Lisbon/`    |
| Madrid    | 2025-06-12 | `data/raw/Madrid/`    |
| Barcelona | 2025-06-12 | `data/raw/Barcelona/` |

**NOTA:** O snapshot de dezembro 2025 tem precos NULL nos listings — usar sempre snapshots de junho.

### 3.3 Datasets por cidade
| Dataset          | Ficheiro              | Tamanho aprox. (4 cidades) |
|------------------|-----------------------|----------------------------|
| listings         | `listings.csv.gz`     | ~42 MB comprimido          |
| calendar         | `calendar.csv.gz`     | ~59 MB comprimido          |
| reviews          | `reviews.csv.gz`      | ~605 MB comprimido         |
| neighbourhoods   | `neighbourhoods.csv`  | ~9 KB (nao comprimido)     |

**NOTA:** O `neighbourhoods.csv` esta em `/visualisations/` no URL (nao em `/data/`).

### 3.4 Download manual (PowerShell)
```powershell
# Exemplo para uma cidade (ajustar URL e pasta conforme necessario)
$base = "data\raw\Madrid"
mkdir $base -Force
$snap = "2025-06-12"
$url = "https://data.insideairbnb.com/spain/comunidad-de-madrid/madrid/$snap"

Invoke-WebRequest -Uri "$url/data/listings.csv.gz" -OutFile "$base\listings.csv.gz"
Invoke-WebRequest -Uri "$url/data/calendar.csv.gz" -OutFile "$base\calendar.csv.gz"
Invoke-WebRequest -Uri "$url/data/reviews.csv.gz" -OutFile "$base\reviews.csv.gz"
Invoke-WebRequest -Uri "$url/visualisations/neighbourhoods.csv" -OutFile "$base\neighbourhoods.csv"
```

### 3.5 URLs Inside Airbnb por cidade
| Cidade    | URL path                              |
|-----------|---------------------------------------|
| Porto     | `portugal/norte/porto`                |
| Lisbon    | `portugal/lisbon/lisbon`              |
| Madrid    | `spain/comunidad-de-madrid/madrid`    |
| Barcelona | `spain/catalonia/barcelona`           |

---

## 4. Pipeline Local (Docker)

### 4.1 Arrancar o container Docker
```bash
# Na pasta raiz do projeto
docker run -it --name spark-dev -p 8888:8888 -v "%cd%":/home/jovyan/work jupyter/pyspark-notebook bash
```

Se o container ja existir mas estiver parado:
```bash
docker start spark-dev
docker exec -it spark-dev bash
```

### 4.2 Instalar o package e correr o pipeline
```bash
cd /home/jovyan/work
pip install -e .

# 1. Ingest: raw CSV -> interim Parquet (unifica colunas, adiciona coluna city)
python -m jobs.ingest

# 2. Clean: interim -> processed (type casts, null handling, colunas derivadas)
python -m jobs.clean

# 3. Queries: corre as 11 queries analiticas, guarda em reports/results/
python -m jobs.run_queries
```

### 4.3 Problemas conhecidos e solucoes

| Problema | Causa | Solucao |
|----------|-------|---------|
| `ModuleNotFoundError: egd_airbnb` | Package nao instalado no container | `pip install -e .` |
| Listings com 0 rows apos clean | Price format `"$48.00"` nao parseado | Usar `_parse_dollar_price()` em `cleaning/listings.py` |
| `neighbourhood_group` not found nas queries | Coluna chama-se `neighbourhood_group_cleansed` nos processed listings | Actualizar queries para usar `_cleansed` |
| `neighbourhoods.csv` not found | URL esta em `/visualisations/` e nao `/data/` | Download manual com URL correto |
| Unicode `->` error no Windows | Caracter `→` incompativel com cp1252 | Substituir por `->` nos prints |

---

## 5. Upload para Google Cloud Storage

Depois de correr o pipeline local com sucesso:

```bash
# No terminal Windows (nao no Docker)

# Apagar dados antigos (se existirem)
gcloud storage rm -r gs://egd-airbnb-bucket/data/processed/

# Upload dos novos dados processados
gcloud storage cp -r data/processed gs://egd-airbnb-bucket/data/processed

# Verificar
gcloud storage ls gs://egd-airbnb-bucket/data/processed/
```

**Esperar:** 4 pastas — `listings/`, `calendar/`, `reviews/`, `neighbourhoods/`
Cada uma com subpastas `city=porto/`, `city=lisbon/`, `city=madrid/`, `city=barcelona/`.

---

## 6. Benchmarking no Dataproc

### 6.1 Workloads disponiveis

O ficheiro `jobs/benchmark_dataproc.py` contem 5 workloads:

| Workload | Tabelas | Volume aprox. | Descricao |
|----------|---------|---------------|-----------|
| `query_seasonality` | calendar + listings | ~20M rows | Join calendar x listings, agregacao por mes |
| `train_rf` | listings | ~60k rows | Pipeline ML com Random Forest (50 arvores) |
| `review_demand` | reviews + listings | milhoes | Join reviews x listings, agregacao por faixa de preco |
| `calendar_revenue` | calendar + listings | ~20M rows | Revenue estimado por bairro/mes + ranking Window |
| `full_pipeline` | calendar + listings + reviews | o maior | Join das 3 tabelas, ocupacao + reviews por listing |

### 6.2 Configuracoes de cluster testadas

| Config | Tipo | Workers | Master | vCPUs total | Disco |
|--------|------|---------|--------|-------------|-------|
| Single-node | `--single-node` | 0 | n2-standard-2 | 2 | 50 GB |
| 2 workers | standard | 2 x n2-standard-2 | n2-standard-2 | 6 | 50 GB cada |
| 4 workers | standard | 4 x n2-standard-2 | n2-standard-2 | 10 | 50 GB cada |

**NOTA:** 8 workers nao e possivel com a quota gratuita (precisa 18 vCPUs, limite e 12).

### 6.3 Comandos manuais

#### Criar cluster
```bash
# Single-node (1 no)
gcloud dataproc clusters create egd-cluster --region=europe-west1 --single-node --master-machine-type=n2-standard-2 --image-version=2.2-debian12 --master-boot-disk-size=50

# 2 workers
gcloud dataproc clusters create egd-cluster --region=europe-west1 --num-workers=2 --master-machine-type=n2-standard-2 --worker-machine-type=n2-standard-2 --image-version=2.2-debian12 --master-boot-disk-size=50 --worker-boot-disk-size=50

# 4 workers
gcloud dataproc clusters create egd-cluster --region=europe-west1 --num-workers=4 --master-machine-type=n2-standard-2 --worker-machine-type=n2-standard-2 --image-version=2.2-debian12 --master-boot-disk-size=50 --worker-boot-disk-size=50
```

#### Submeter um benchmark
```bash
gcloud dataproc jobs submit pyspark jobs/benchmark_dataproc.py --cluster=egd-cluster --region=europe-west1 -- --workload query_seasonality --num-workers 2 --runs 3
```

#### Apagar cluster (IMPORTANTE — evitar custos)
```bash
gcloud dataproc clusters delete egd-cluster --region=europe-west1 --quiet
```

#### Verificar clusters ativos
```bash
gcloud dataproc clusters list --region=europe-west1
```

### 6.4 Script automatizado

O ficheiro `run_benchmarks.bat` automatiza todo o processo:
- Cria cluster single-node -> corre 5 workloads -> apaga
- Cria cluster 2 workers -> corre 5 workloads -> apaga
- Cria cluster 4 workers -> corre 5 workloads -> apaga
- Tenta 8 workers (falha por quota, salta automaticamente)

```bash
# Na pasta raiz do projeto, terminal Windows
run_benchmarks.bat
```

**Duracao total:** ~45-60 minutos (inclui criar/apagar clusters)

**Output:** Um ficheiro `.txt` por workload/config em `reports/benchmarks/`:
```
reports/benchmarks/
  query_seasonality_1w.txt
  query_seasonality_2w.txt
  query_seasonality_4w.txt
  train_rf_1w.txt
  ...
```

Cada ficheiro contem o output completo do job, incluindo um bloco JSON com resultados:
```
===BENCHMARK_RESULT===
{
  "workload": "query_seasonality",
  "platform": "dataproc",
  "num_workers": 2,
  "runs": [
    {"run": 0, "wall_secs": 46.10},
    {"run": 1, "wall_secs": 22.48},
    {"run": 2, "wall_secs": 13.21}
  ]
}
===END===
```

### 6.5 Notas tecnicas sobre Dataproc

- **YARN:** Dataproc usa YARN como gestor de recursos. Nao e possivel criar multiplas SparkSessions — o script reutiliza uma unica sessao com `spark.catalog.clearCache()` entre runs.
- **Run 0 (cold start):** A primeira execucao inclui overhead de inicializacao (carregar JARs, alocar containers YARN). Para analise, usar apenas Run 1 e Run 2.
- **Disco:** O default e 500 GB por no, que excede a quota de `DISKS_TOTAL_GB` (2048 GB). Usar `--master-boot-disk-size=50 --worker-boot-disk-size=50`.
- **Regiao:** Usar `europe-west1` (Belgica) para baixa latencia com GCS.

---

## 7. Resultados do Benchmark (4 cidades — Maio 2026)

Media de Run 1 e Run 2 (excluindo cold start):

| Workload | 1 node | 2 workers | 4 workers | Speedup 1->2 | Speedup 1->4 |
|----------|--------|-----------|-----------|--------------|--------------|
| query_seasonality | 21.6s | 17.8s | 13.4s | 1.21x | 1.61x |
| train_rf | 54.0s | 39.2s | 38.9s | 1.38x | 1.39x |
| review_demand | 34.6s | 17.1s | 13.9s | 2.02x | 2.49x |
| calendar_revenue | 23.7s | 14.4s | 15.3s | 1.64x | 1.55x |
| full_pipeline | 35.7s | 21.4s | 19.2s | 1.67x | 1.86x |

### 7.1 Interpretacao

- **review_demand** tem o melhor speedup (2.49x com 4 workers) — e o workload com mais dados (milhoes de reviews), onde o paralelismo compensa o overhead.
- **1->2 workers** mostra melhorias consistentes em todos os workloads (1.2x a 2.0x).
- **2->4 workers** mostra ganhos adicionais nos workloads pesados (review_demand, full_pipeline) mas estagnacao/regressao nos mais leves.
- **train_rf** praticamente nao beneficia de ir de 2 para 4 workers — o ML pipeline tem componente sequencial significativa (fitting do modelo).
- **calendar_revenue** piora ligeiramente de 2->4 — o overhead de coordenacao entre nos supera o ganho com este volume de dados.
- Estes resultados sao consistentes com a **Lei de Amdahl**: o speedup e limitado pela fracao sequencial do workload.

---

## 8. Estrutura do Codigo Relevante

```
src/egd_airbnb/
  config.py              # CITIES dict, paths, Spark configs
  ingest/
    download.py          # URL_PATHS por cidade, download de snapshots
    unify.py             # Unifica CSVs de todas as cidades em Parquet
  cleaning/
    listings.py          # _parse_dollar_price() para format "$48.00"
    calendar.py          # Drop price (null), parse adjusted_price
    reviews.py           # Cast date, dedup
    neighbourhoods.py    # Tabela de referencia neighbourhood_group -> neighbourhood
  queries/
    q01 a q11            # Queries analiticas
    registry.py          # Mapa nome -> funcao

jobs/
  benchmark_dataproc.py  # Script standalone com 5 workloads para Dataproc

run_benchmarks.bat       # Automatiza criacao/destruicao de clusters e execucao
```

### 8.1 Como adicionar uma nova cidade

1. Adicionar ao `CITIES` dict em `config.py`
2. Adicionar URL path ao `URL_PATHS` dict em `ingest/download.py`
3. Criar pasta `data/raw/<NomeCidade>/` com os 4 ficheiros
4. Re-correr pipeline: `ingest -> clean -> queries`
5. Re-upload para GCS

### 8.2 Como adicionar um novo workload de benchmark

1. Criar funcao `workload_<nome>(spark, data_root)` em `benchmark_dataproc.py`
2. Adicionar ao dict `WORKLOADS`
3. Actualizar `run_benchmarks.bat` com as linhas de execucao para cada config

---

## 9. Troubleshooting

| Erro | Causa | Solucao |
|------|-------|---------|
| `INSUFFICIENT CPUS_ALL_REGIONS quota` | Excedeu limite de vCPUs | Reduzir workers ou usar maquinas mais pequenas |
| `DISKS_TOTAL_GB quota exceeded` | Disco default muito grande | Adicionar `--master-boot-disk-size=50 --worker-boot-disk-size=50` |
| `subnetwork not ready` | Compute Engine API nao ativada | `gcloud services enable compute.googleapis.com` |
| `gsutil 401 Unauthorized` | Autenticacao expirada | Usar `gcloud storage` em vez de `gsutil`, ou `gcloud auth login` |
| `No such container: spark-dev` | Container Docker foi removido | Criar novo com `docker run ...` |
| Job Dataproc falha silenciosamente | Erro no codigo Python | Ver output completo no ficheiro .txt em `reports/benchmarks/` |
| Benchmark script para apos criar cluster | Falta `call` antes de `gcloud` no .bat | Ja corrigido no script atual |

---

## 10. Custos Estimados (Google Cloud)

| Recurso | Custo por hora aprox. |
|---------|-----------------------|
| n2-standard-2 (por no) | ~$0.10/h |
| Cluster 1 no (single-node) | ~$0.10/h |
| Cluster 2 workers + master | ~$0.30/h |
| Cluster 4 workers + master | ~$0.50/h |
| GCS storage (< 1 GB) | negligivel |

**Suite completa de benchmarks (1+2+4 workers):** ~$1-2 por execucao

**Regra de ouro:** SEMPRE apagar o cluster apos uso. Verificar com:
```bash
gcloud dataproc clusters list --region=europe-west1
```
