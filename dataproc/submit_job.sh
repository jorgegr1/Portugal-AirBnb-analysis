#!/usr/bin/env bash
# Usage: bash submit_job.sh <cluster-name> <job-script> [args...]
# Example: bash submit_job.sh egd-1w benchmark.py --workload query_seasonality --runs 3

set -euo pipefail
[ -f .env ] && set -a && . .env && set +a

CLUSTER="${1:?cluster name required}"
SCRIPT="${2:?job script (e.g. benchmark.py) required}"
shift 2

# Stage the package wheel to the cluster.
WHEEL=$(ls dist/egd_airbnb-*.whl | head -n1)

gcloud dataproc jobs submit pyspark \
  "jobs/${SCRIPT}" \
  --cluster="$CLUSTER" \
  --region="${GCP_REGION:-europe-west1}" \
  --py-files="$WHEEL" \
  --properties="spark.sql.parquet.compression.codec=snappy" \
  -- "$@"
