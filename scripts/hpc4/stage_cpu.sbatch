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
: "${SLURM_ARRAY_TASK_ID:?}"

container=(
  apptainer exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  "${PRORM_IMAGE}"
)
mapfile -t seeds < <(
  "${container[@]}" python - "${PRORM_CONFIG}" <<'PY'
import sys
from smart_reward.config import load_config
for seed in load_config(sys.argv[1])["run"]["seeds"]:
    print(seed)
PY
)
index="${SLURM_ARRAY_TASK_ID}"
(( index >= 0 && index < ${#seeds[@]} )) || { echo "invalid seed index" >&2; exit 2; }
seed="${seeds[index]}"
"${container[@]}" python -m smart_reward.cli run-stage \
  "${PRORM_CONFIG}" "${PRORM_RUN_ROOT}/seed-${seed}" \
  --stage rollout-aggregate --seed "${seed}" --device cpu
