#!/usr/bin/env bash
# Usage: bash delete_cluster.sh <cluster-name>

set -euo pipefail
[ -f .env ] && set -a && . .env && set +a

NAME="${1:?cluster name required}"
gcloud dataproc clusters delete "$NAME" \
  --project="$GCP_PROJECT" \
  --region="${GCP_REGION:-europe-west1}" \
  --quiet
