#!/usr/bin/env bash
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
: "${PRORM_STAGE:?}"
: "${SLURM_ARRAY_TASK_ID:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_image_revision.sh"
verify_image_revision "${PRORM_IMAGE}" "${PRORM_GIT_COMMIT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

container_prefix=(
  prorm_apptainer_exec --cleanenv --nv
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
)

run_container() {
  "${container_prefix[@]}" "${PRORM_IMAGE}" "$@"
}

run_container_on_gpu() {
  local device="$1"
  shift
  "${container_prefix[@]}" --env "CUDA_VISIBLE_DEVICES=${device}" "${PRORM_IMAGE}" "$@"
}

mapfile -t seeds < <(
  run_container python - "${PRORM_CONFIG}" <<'PY'
import sys
from smart_reward.config import load_config
for seed in load_config(sys.argv[1])["run"]["seeds"]:
    print(seed)
PY
)

task="${SLURM_ARRAY_TASK_ID}"
if [[ "${PRORM_STAGE}" = "kl-calibration-worker" ]]; then
  : "${PRORM_CALIBRATION_WORKERS:?}"
  : "${PRORM_CALIBRATION_GPUS_PER_JOB:?}"
  mapfile -t policies < <(
    run_container python -m smart_reward.cli policy-names "${PRORM_CONFIG}" | tail -n +2
  )
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
  (( ${#visible_gpus[@]} == PRORM_CALIBRATION_GPUS_PER_JOB )) || {
    echo "allocated GPU visibility does not match PRORM_CALIBRATION_GPUS_PER_JOB" >&2
    exit 2
  }
  total_tasks=$(( ${#seeds[@]} * ${#policies[@]} ))

  run_calibration_slot() {
    local global_worker="$1"
    local device="$2"
    local calibration_task seed_index policy_index seed
    for ((calibration_task = global_worker; calibration_task < total_tasks; calibration_task += PRORM_CALIBRATION_WORKERS)); do
      seed_index=$((calibration_task / ${#policies[@]}))
      policy_index=$((calibration_task % ${#policies[@]}))
      seed="${seeds[seed_index]}"
      run_container_on_gpu "${device}" python -m smart_reward.cli run-stage \
        "${PRORM_CONFIG}" "${PRORM_RUN_ROOT}/seed-${seed}" \
        --stage kl-calibration-policy --seed "${seed}" --device cuda \
        --policy-name "${policies[policy_index]}"
    done
  }

  pids=()
  for ((local_worker = 0; local_worker < PRORM_CALIBRATION_GPUS_PER_JOB; local_worker++)); do
    global_worker=$((task * PRORM_CALIBRATION_GPUS_PER_JOB + local_worker))
    (( global_worker < PRORM_CALIBRATION_WORKERS )) || continue
    run_calibration_slot "${global_worker}" "${visible_gpus[local_worker]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  exit "${failed}"
elif [[ "${PRORM_STAGE}" = "rollout-worker" ]]; then
  : "${PRORM_ROLLOUT_WORKERS:?}"
  : "${PRORM_ROLLOUT_GPUS_PER_JOB:?}"
  mapfile -t policies < <(run_container python -m smart_reward.cli policy-names "${PRORM_CONFIG}")
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES:?}"
  (( ${#visible_gpus[@]} == PRORM_ROLLOUT_GPUS_PER_JOB )) || {
    echo "allocated GPU visibility does not match PRORM_ROLLOUT_GPUS_PER_JOB" >&2
    exit 2
  }
  total_tasks=$(( ${#seeds[@]} * ${#policies[@]} ))

  run_rollout_slot() {
    local global_worker="$1"
    local device="$2"
    local rollout_task seed_index policy_index seed
    for ((rollout_task = global_worker; rollout_task < total_tasks; rollout_task += PRORM_ROLLOUT_WORKERS)); do
      seed_index=$((rollout_task / ${#policies[@]}))
      policy_index=$((rollout_task % ${#policies[@]}))
      seed="${seeds[seed_index]}"
      run_container_on_gpu "${device}" python -m smart_reward.cli run-stage \
        "${PRORM_CONFIG}" "${PRORM_RUN_ROOT}/seed-${seed}" \
        --stage rollout-policy --seed "${seed}" --device cuda \
        --policy-name "${policies[policy_index]}"
    done
  }

  pids=()
  for ((local_worker = 0; local_worker < PRORM_ROLLOUT_GPUS_PER_JOB; local_worker++)); do
    global_worker=$((task * PRORM_ROLLOUT_GPUS_PER_JOB + local_worker))
    (( global_worker < PRORM_ROLLOUT_WORKERS )) || continue
    run_rollout_slot "${global_worker}" "${visible_gpus[local_worker]}" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  exit "${failed}"
else
  (( task >= 0 && task < ${#seeds[@]} )) || { echo "invalid seed index" >&2; exit 2; }
  seed="${seeds[task]}"
  seed_root="${PRORM_RUN_ROOT}/seed-${seed}"
  extra_args=()
  if [[ "${PRORM_STAGE}" = "materialize" && -n "${PRORM_SOURCE_RUN_ROOT:-}" ]]; then
    extra_args+=(--reuse-splits-from "${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact")
  fi
  if [[ "${PRORM_STAGE}" = "reward" && -n "${PRORM_SOURCE_RUN_ROOT:-}" ]]; then
    mkdir -p "${seed_root}/reward_provenance"
    source_reward="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/reward_result.json"
    target_reward="${seed_root}/reward_provenance/mle-source.json"
    if [[ -f "${target_reward}" ]]; then
      cmp -s "${source_reward}" "${target_reward}" || {
        echo "existing MLE source differs from the requested formal source" >&2
        exit 2
      }
    else
      cp --preserve=mode,timestamps "${source_reward}" "${target_reward}"
    fi
  fi
  run_container python -m smart_reward.cli run-stage \
    "${PRORM_CONFIG}" "${seed_root}" \
    --stage "${PRORM_STAGE}" --seed "${seed}" --device cuda \
    "${extra_args[@]}"
fi
