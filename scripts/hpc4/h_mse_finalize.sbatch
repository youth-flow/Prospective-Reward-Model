#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

set -euo pipefail
umask 027
: "${PRORM_REPO_ROOT:?}" "${PRORM_H_CONFIG:?}" "${PRORM_IMAGE:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}" "${PRORM_SOURCE_M6_ROOT:?}" "${PRORM_RUN_ROOT:?}"
: "${PRORM_ARCHIVE_ROOT:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
verify_bound_runtime "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT:?}" \
  "${PRORM_GIT_COMMIT:?}" "${PRORM_REPO_ROOT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"
binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_SOURCE_M6_ROOT}:${PRORM_SOURCE_M6_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
container=(prorm_apptainer_exec --cleanenv --bind "${binds}" --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256:?}" --env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256:?}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" "${PRORM_IMAGE}")

mapfile -t seeds < <(
  "${container[@]}" python - "${PRORM_H_CONFIG}" <<'PY'
import sys
from smart_reward.h_ablation import load_h_ablation_config
for value in load_h_ablation_config(sys.argv[1])["experiment"]["seeds"]:
    print(value)
PY
)
reward_results=()
policy_results=()
for seed in "${seeds[@]}"; do
  target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
  "${container[@]}" python -m smart_reward.cli assemble-h-mse-policy-seed \
    "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact" \
    "${target_seed}/reward_result.json" "${target_seed}/adapters" \
    "${PRORM_SOURCE_M6_ROOT}/seed-${seed}" "${target_seed}/policy_rollouts" \
    "${target_seed}/evaluation.json" --seed "${seed}"
  reward_results+=("${target_seed}/reward_result.json")
  policy_results+=("${target_seed}/evaluation.json")
done
"${container[@]}" python -m smart_reward.cli aggregate-h-mse-rewards \
  "${PRORM_H_CONFIG}" "${PRORM_RUN_ROOT}/reward-aggregate.json" --results "${reward_results[@]}"
"${container[@]}" python -m smart_reward.cli aggregate-h-mse-policy \
  "${PRORM_H_CONFIG}" "${PRORM_RUN_ROOT}/aggregate.json" --results "${policy_results[@]}"
"${container[@]}" python -m smart_reward.cli audit-h-mse-run \
  "${PRORM_H_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}" "${PRORM_SOURCE_M6_ROOT}" \
  "${PRORM_RUN_ROOT}" "${PRORM_RUN_ROOT}/integrity-audit.json"
"${PRORM_REPO_ROOT}/scripts/hpc4/archive_audited_run.sh" \
  "${PRORM_RUN_ROOT}" "${PRORM_ARCHIVE_ROOT}" "${PRORM_RUN_ROOT}/integrity-audit.json" \
  prorm-h-mse-integrity-audit/v1
