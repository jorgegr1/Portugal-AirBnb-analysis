#!/usr/bin/env bash
# Dataproc initialization action: install the project's pip dependencies on every node.
# Upload this file to gs://$GCS_BUCKET/dataproc/init.sh before creating the cluster.

set -euo pipefail

PIP_PACKAGES="${PIP_PACKAGES:-pandas numpy}"

/opt/conda/default/bin/pip install --upgrade pip
/opt/conda/default/bin/pip install $PIP_PACKAGES
