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

container=(apptainer exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}" --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
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
