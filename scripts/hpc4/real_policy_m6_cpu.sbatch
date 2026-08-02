#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_EXTENSION_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}"
: "${PRORM_BASE_REAL_RUN_ROOT:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_REAL_STAGE:?}"

if [[ -n "${PRORM_IMAGE_SOURCE_COMMIT:-}" ]]; then
  source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
  verify_bound_runtime \
    "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT}" \
    "${PRORM_GIT_COMMIT}" "${PRORM_REPO_ROOT}"
else
  source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
  verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
fi
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"
container=(
  prorm_apptainer_exec --cleanenv
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_BASE_REAL_RUN_ROOT}:${PRORM_BASE_REAL_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
  --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}"
  --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
)
if [[ -n "${PRORM_IMAGE_SOURCE_COMMIT:-}" ]]; then
  container+=(--env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}")
fi
container+=("${PRORM_IMAGE}")

mapfile -t seeds < <(
  "${container[@]}" python - "${PRORM_EXTENSION_CONFIG}" <<'PY'
import sys
from smart_reward.real_policy_extension import load_real_policy_extension_config
for seed in load_real_policy_extension_config(sys.argv[1])["experiment"]["seeds"]:
    print(seed)
PY
)

case "${PRORM_REAL_STAGE}" in
  seed-aggregate)
    : "${SLURM_ARRAY_TASK_ID:?}"
    index="${SLURM_ARRAY_TASK_ID}"
    (( index >= 0 && index < ${#seeds[@]} )) || { echo "invalid seed index" >&2; exit 2; }
    seed="${seeds[index]}"
    source_seed="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}"
    base_seed="${PRORM_BASE_REAL_RUN_ROOT}/seed-${seed}"
    target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
    "${container[@]}" python -m smart_reward.cli assemble-extended-real-policy-seed \
      "${PRORM_EXTENSION_CONFIG}" \
      "${source_seed}/artifact" "${source_seed}/reward_result.json" \
      "${base_seed}/adapters" "${base_seed}/policy_rollouts" \
      "${target_seed}/policy_rollouts" "${target_seed}/evaluation.json" \
      --seed "${seed}"
    ;;
  aggregate)
    results=()
    for seed in "${seeds[@]}"; do
      results+=("${PRORM_RUN_ROOT}/seed-${seed}/evaluation.json")
    done
    "${container[@]}" python -m smart_reward.cli aggregate-extended-real-policy \
      "${PRORM_EXTENSION_CONFIG}" "${PRORM_RUN_ROOT}/aggregate.json" \
      --results "${results[@]}"
    ;;
  audit)
    "${container[@]}" python -m smart_reward.cli audit-extended-real-policy \
      "${PRORM_EXTENSION_CONFIG}" "${PRORM_SOURCE_RUN_ROOT}" \
      "${PRORM_BASE_REAL_RUN_ROOT}" "${PRORM_RUN_ROOT}" \
      "${PRORM_RUN_ROOT}/integrity-audit.json"
    ;;
  *)
    echo "unknown m=6 real-policy CPU stage: ${PRORM_REAL_STAGE}" >&2
    exit 2
    ;;
esac
