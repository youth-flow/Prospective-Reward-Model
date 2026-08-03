#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 9 ]] || {
  echo "usage: $0 CONFIG IMAGE IMAGE_SOURCE_COMMIT HF_CACHE SOURCE_RUN RUN_ROOT STAGE JOBS GPUS_PER_JOB" >&2
  exit 2
}
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
image_source_commit="$3"
hf_cache="$(realpath -e "$4")"
source_run="$(realpath -e "$5")"
run_root="$(realpath -m "$6")"
stage="$7"
jobs="$8"
gpus_per_job="$9"
[[ "${stage}" =~ ^(reference|train)$ ]] || { echo "stage must be reference or train" >&2; exit 2; }
[[ "${jobs}" =~ ^[1-2]$ ]] || { echo "jobs must be 1 or 2" >&2; exit 2; }
[[ "${gpus_per_job}" =~ ^[1-4]$ ]] || { echo "gpus per job must be in [1,4]" >&2; exit 2; }
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || { echo "submission requires a clean worktree" >&2; exit 2; }
if [[ "${stage}" = "reference" ]]; then
  [[ ! -e "${run_root}" ]] || { echo "refusing an existing run root" >&2; exit 2; }
  mkdir -p "${run_root}/logs"
else
  run_root="$(realpath -e "${run_root}")"
fi

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
source_sha="$(PYTHONPATH="${repo_root}/src" python3 -c 'import sys; from smart_reward.config import config_hash,load_config; print(config_hash(load_config(sys.argv[1])))' "${repo_root}/configs/fisher_trpo_main.yaml")"
inventory="${hf_cache}/inventories/${source_sha}.json"
[[ -f "${inventory}" ]] || { echo "missing HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_EXTENSION_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SOURCE_COMMIT=${image_source_commit},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_RUN_ROOT=${run_root},PRORM_DIRECT_STAGE=${stage},PRORM_GPU_WORKERS_PER_JOB=${gpus_per_job}"
walltime="01:00:00"
[[ "${stage}" = "train" ]] && walltime="12:00:00"
job="$(sbatch --parsable --job-name="prorm-converged-${stage}" --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time="${walltime}" --array="0-$((jobs - 1))" \
  --gpus-per-node="${gpus_per_job}" --output="${run_root}/logs/${stage}-%A_%a.out" \
  --export="${common}" "${repo_root}/scripts/hpc4/direct_converged_gpu.sbatch")"
printf 'stage=%s\njob_id=%s\nrun_root=%s\n' "${stage}" "${job}" "${run_root}"
