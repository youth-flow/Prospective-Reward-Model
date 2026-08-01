#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"
container=(
  prorm_apptainer_exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
  --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}"
  --env "MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}"
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
seed_root="${PRORM_RUN_ROOT}/seed-${seed}"
mkdir -p "${seed_root}"
"${container[@]}" python -m smart_reward.cli evaluate-ngd \
  "${PRORM_CONFIG}" \
  "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
  "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/reward_result.json" \
  "${seed_root}/evaluation.json" \
  --seed "${seed}" --device cpu
