#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

container=(prorm_apptainer_exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}" --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  "${PRORM_IMAGE}")
mapfile -t seeds < <("${container[@]}" python - "${PRORM_CONFIG}" <<'PY'
import sys
from smart_reward.config import load_config
for seed in load_config(sys.argv[1])["run"]["seeds"]:
    print(seed)
PY
)
inputs=()
for seed in "${seeds[@]}"; do
  inputs+=("${PRORM_RUN_ROOT}/seed-${seed}/fisher_crossfit.json")
done
"${container[@]}" python -m smart_reward.cli select-fisher \
  "${PRORM_CONFIG}" "${PRORM_RUN_ROOT}/fisher_selection.json" \
  --crossfit-results "${inputs[@]}"
