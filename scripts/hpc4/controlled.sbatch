#!/usr/bin/env bash
#SBATCH --job-name=prorm-main
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_CACHE:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_RUN_ROOT:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

container=(
  apptainer exec --cleanenv --nv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub"
  --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "HF_DATASETS_OFFLINE=1"
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
(( index >= 0 && index < ${#seeds[@]} )) || { echo "invalid array index" >&2; exit 2; }
seed="${seeds[index]}"
output="${PRORM_RUN_ROOT}/seed-${seed}"
[[ ! -e "${output}" ]] || { echo "refusing existing output: ${output}" >&2; exit 2; }

"${container[@]}" python -m smart_reward.cli run-seed \
  "${PRORM_CONFIG}" "${output}" --seed "${seed}" --device cuda
