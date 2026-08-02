#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_EXTENSION_CONFIG:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_HF_CACHE:?}"
: "${PRORM_HF_INVENTORY_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_SOURCE_RUN_ROOT:?}"
: "${PRORM_BASE_REAL_RUN_ROOT:?}"
: "${PRORM_RUN_ROOT:?}"
: "${PRORM_REAL_WORKERS:?}"
: "${PRORM_REAL_GPUS_PER_JOB:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

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

binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_BASE_REAL_RUN_ROOT}:${PRORM_BASE_REAL_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
container_prefix=(
  prorm_apptainer_exec --cleanenv --nv
  --bind "${binds}"
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
)
if [[ -n "${PRORM_IMAGE_SOURCE_COMMIT:-}" ]]; then
  container_prefix+=(--env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}")
fi

run_container() {
  "${container_prefix[@]}" "${PRORM_IMAGE}" "$@"
}

run_on_gpu() {
  local device="$1"
  shift
  "${container_prefix[@]}" --env "CUDA_VISIBLE_DEVICES=${device}" "${PRORM_IMAGE}" "$@"
}

mapfile -t seeds < <(
  run_container python - "${PRORM_EXTENSION_CONFIG}" <<'PY'
import sys
from smart_reward.real_policy_extension import load_real_policy_extension_config
for seed in load_real_policy_extension_config(sys.argv[1])["experiment"]["seeds"]:
    print(seed)
PY
)
mapfile -t policies < <(run_container python -m smart_reward.cli real-policy-names)
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
(( ${#visible_gpus[@]} == PRORM_REAL_GPUS_PER_JOB )) || {
  echo "allocated GPU visibility does not match PRORM_REAL_GPUS_PER_JOB" >&2
  exit 2
}
worker_offset="${PRORM_REAL_WORKER_OFFSET:-0}"
[[ "${worker_offset}" =~ ^[0-9]+$ ]] || {
  echo "PRORM_REAL_WORKER_OFFSET must be a non-negative integer" >&2
  exit 2
}
total=$(( ${#seeds[@]} * ${#policies[@]} ))

run_slot() {
  local worker="$1"
  local device="$2"
  local task seed_index policy_index seed policy source_seed base_seed target_seed
  for ((task = worker; task < total; task += PRORM_REAL_WORKERS)); do
    seed_index=$((task / ${#policies[@]}))
    policy_index=$((task % ${#policies[@]}))
    seed="${seeds[seed_index]}"
    policy="${policies[policy_index]}"
    source_seed="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}"
    base_seed="${PRORM_BASE_REAL_RUN_ROOT}/seed-${seed}"
    target_seed="${PRORM_RUN_ROOT}/seed-${seed}"
    mkdir -p "${target_seed}/policy_rollouts"
    run_on_gpu "${device}" python -m smart_reward.cli extend-real-policy-rollout-to-six \
      "${PRORM_EXTENSION_CONFIG}" \
      "${source_seed}/artifact" "${source_seed}/reward_result.json" \
      "${base_seed}/adapters" "${base_seed}/policy_rollouts/${policy}" \
      "${target_seed}/policy_rollouts/${policy}" \
      --policy-name "${policy}" --seed "${seed}" --device cuda
  done
}

pids=()
for ((local_worker = 0; local_worker < PRORM_REAL_GPUS_PER_JOB; local_worker++)); do
  worker=$((
    worker_offset + SLURM_ARRAY_TASK_ID * PRORM_REAL_GPUS_PER_JOB + local_worker
  ))
  (( worker < PRORM_REAL_WORKERS )) || continue
  run_slot "${worker}" "${visible_gpus[local_worker]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
