#!/usr/bin/env bash
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}" "${PRORM_EXTENSION_CONFIG:?}" "${PRORM_IMAGE:?}"
: "${PRORM_HF_CACHE:?}" "${PRORM_SOURCE_RUN_ROOT:?}" "${PRORM_RUN_ROOT:?}"
: "${PRORM_DIRECT_STAGE:?}" "${PRORM_GPU_WORKERS_PER_JOB:?}" "${SLURM_ARRAY_TASK_ID:?}"

source "${PRORM_REPO_ROOT}/scripts/hpc4/verify_bound_runtime.sh"
verify_bound_runtime "${PRORM_IMAGE}" "${PRORM_IMAGE_SOURCE_COMMIT:?}" \
  "${PRORM_GIT_COMMIT:?}" "${PRORM_REPO_ROOT}"
source "${PRORM_REPO_ROOT}/scripts/hpc4/runtime.sh"

binds="${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT},${PRORM_HF_CACHE}:${PRORM_HF_CACHE},${PRORM_SOURCE_RUN_ROOT}:${PRORM_SOURCE_RUN_ROOT},${PRORM_RUN_ROOT}:${PRORM_RUN_ROOT}"
container_prefix=(
  prorm_apptainer_exec --cleanenv --nv --bind "${binds}" --pwd "${PRORM_REPO_ROOT}"
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" --env "HF_HOME=${PRORM_HF_CACHE}"
  --env "HF_HUB_CACHE=${PRORM_HF_CACHE}/hub" --env "HF_DATASETS_CACHE=${PRORM_HF_CACHE}/datasets"
  --env "HF_HUB_OFFLINE=1" --env "TRANSFORMERS_OFFLINE=1" --env "HF_DATASETS_OFFLINE=1"
  --env "PRORM_GIT_COMMIT=${PRORM_GIT_COMMIT}" --env "PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256:?}"
  --env "PRORM_IMAGE_SOURCE_COMMIT=${PRORM_IMAGE_SOURCE_COMMIT}"
  --env "PRORM_HF_INVENTORY_SHA256=${PRORM_HF_INVENTORY_SHA256:?}"
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
)
run_gpu() {
  local device="$1"
  shift
  "${container_prefix[@]}" --env "CUDA_VISIBLE_DEVICES=${device}" "${PRORM_IMAGE}" "$@"
}

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:?}"
(( ${#visible_devices[@]} == PRORM_GPU_WORKERS_PER_JOB )) || {
  echo "GPU visibility differs from PRORM_GPU_WORKERS_PER_JOB" >&2
  exit 2
}
mapfile -t conditions < <(
  run_gpu "${visible_devices[0]}" python - \
    "${PRORM_EXTENSION_CONFIG}" "${PRORM_DIRECT_STAGE}" <<'PY'
import itertools
import sys
from smart_reward.direct_preference import load_direct_preference_config

config = load_direct_preference_config(sys.argv[1])
if sys.argv[2] == "reference":
    for seed in config["experiment"]["seeds"]:
        print(seed)
elif sys.argv[2] == "train":
    for seed, beta, method in itertools.product(
        config["experiment"]["seeds"],
        config["experiment"]["betas"],
        config["experiment"]["methods"],
    ):
        print(seed, beta, method)
else:
    raise SystemExit("stage must be reference or train")
PY
)

default_worker_offset=$((SLURM_ARRAY_TASK_ID * PRORM_GPU_WORKERS_PER_JOB))
default_total_workers=$((SLURM_ARRAY_TASK_COUNT * PRORM_GPU_WORKERS_PER_JOB))
worker_offset="${PRORM_WORKER_OFFSET:-${default_worker_offset}}"
total_workers="${PRORM_TOTAL_WORKERS:-${default_total_workers}}"
[[ "${worker_offset}" =~ ^[0-9]+$ && "${total_workers}" =~ ^[1-9][0-9]*$ ]] || {
  echo "worker offset and total must be nonnegative integers" >&2
  exit 2
}
(( worker_offset + PRORM_GPU_WORKERS_PER_JOB <= total_workers )) || {
  echo "GPU worker range exceeds the declared global worker count" >&2
  exit 2
}
run_slot() {
  local worker="$1" device="$2" index seed beta method seed_root artifact
  for ((index=worker; index<${#conditions[@]}; index+=total_workers)); do
    read -r seed beta method <<< "${conditions[index]}"
    seed_root="${PRORM_RUN_ROOT}/seed-${seed}"
    artifact="${PRORM_SOURCE_RUN_ROOT}/seed-${seed}/artifact"
    mkdir -p "${seed_root}"
    if [[ "${PRORM_DIRECT_STAGE}" = "reference" ]]; then
      run_gpu "${device}" python -m smart_reward.cli cache-direct-reference \
        "${PRORM_EXTENSION_CONFIG}" "${artifact}" "${seed_root}/reference" \
        --seed "${seed}" --device cuda
    else
      run_gpu "${device}" python -m smart_reward.cli train-direct-preference \
        "${PRORM_EXTENSION_CONFIG}" "${artifact}" "${seed_root}/reference" \
        "${seed_root}/fits" --seed "${seed}" --beta "${beta}" \
        --method "${method}" --device cuda
    fi
  done
}

pids=()
for ((local_worker=0; local_worker<PRORM_GPU_WORKERS_PER_JOB; local_worker++)); do
  global_worker=$((worker_offset + local_worker))
  (( global_worker < ${#conditions[@]} )) || continue
  run_slot "${global_worker}" "${visible_devices[local_worker]}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
exit "${failed}"
