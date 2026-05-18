#!/usr/bin/env bash
# Usage: bash create_cluster.sh <cluster-name> <num-workers>
# Reads GCP_PROJECT, GCP_REGION, DATAPROC_IMAGE_VERSION, DATAPROC_MACHINE_TYPE from .env

set -euo pipefail
[ -f .env ] && set -a && . .env && set +a

NAME="${1:?cluster name required}"
WORKERS="${2:?num workers required}"
REGION="${GCP_REGION:-europe-west1}"
IMAGE="${DATAPROC_IMAGE_VERSION:-2.2-debian12}"
MTYPE="${DATAPROC_MACHINE_TYPE:-n2-standard-4}"

gcloud dataproc clusters create "$NAME" \
  --project="$GCP_PROJECT" \
  --region="$REGION" \
  --image-version="$IMAGE" \
  --master-machine-type="$MTYPE" \
  --worker-machine-type="$MTYPE" \
  --num-workers="$WORKERS" \
  --master-boot-disk-size=100 \
  --worker-boot-disk-size=100 \
  --initialization-actions="gs://$GCS_BUCKET/dataproc/init.sh" \
  --metadata="PIP_PACKAGES=egd_airbnb" \
  --max-idle=30m
