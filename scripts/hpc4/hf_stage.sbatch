#!/usr/bin/env bash
#SBATCH --job-name=prorm-hf-stage
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

set -euo pipefail
: "${PRORM_REPO_ROOT:?}"
: "${PRORM_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_HF_CACHE:?}"

mkdir -p "${PRORM_HF_CACHE}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
apptainer exec --cleanenv \
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE}" \
  --pwd "${PRORM_REPO_ROOT}" \
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" \
  --env "HF_HOME=${PRORM_HF_CACHE}" \
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub" \
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets" \
  "${PRORM_IMAGE}" \
  python scripts/hpc4/stage_hf_assets.py "${PRORM_CONFIG}" "${PRORM_HF_CACHE}"
