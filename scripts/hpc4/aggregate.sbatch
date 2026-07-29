#!/usr/bin/env bash
#SBATCH --job-name=prorm-aggregate
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2

set -euo pipefail
: "${PRORM_REPO_ROOT:?}"
: "${PRORM_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"

container=(apptainer exec --cleanenv
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
reward_results=()
rollout_results=()
for seed in "${seeds[@]}"; do
  reward_results+=("${PRORM_RUN_ROOT}/seed-${seed}/reward_result.json")
  rollout_results+=("${PRORM_RUN_ROOT}/seed-${seed}/policy_utility/metrics.json")
done
"${container[@]}" python -m smart_reward.cli aggregate \
  "${PRORM_CONFIG}" "${PRORM_RUN_ROOT}/aggregate.json" \
  --reward-results "${reward_results[@]}" \
  --rollout-results "${rollout_results[@]}"
