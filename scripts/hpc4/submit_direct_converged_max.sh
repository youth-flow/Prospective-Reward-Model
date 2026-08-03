#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 7 ]] || {
  echo "usage: $0 CONFIG IMAGE IMAGE_SOURCE_COMMIT HF_CACHE SOURCE_RUN RUN_ROOT WALLTIME" >&2
  exit 2
}
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
image_source_commit="$3"
hf_cache="$(realpath -e "$4")"
source_run="$(realpath -e "$5")"
run_root="$(realpath -e "$6")"
walltime="$7"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "submission requires a clean worktree" >&2
  exit 2
}

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
source_sha="$(PYTHONPATH="${repo_root}/src" python3 -c 'import sys; from smart_reward.config import config_hash,load_config; print(config_hash(load_config(sys.argv[1])))' "${repo_root}/configs/fisher_trpo_main.yaml")"
inventory="${hf_cache}/inventories/${source_sha}.json"
[[ -f "${inventory}" ]] || { echo "missing HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_EXTENSION_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SOURCE_COMMIT=${image_source_commit},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_RUN_ROOT=${run_root},PRORM_DIRECT_STAGE=train,PRORM_TOTAL_WORKERS=6"

# Two jobs are the HPC4 l20_qos maximum. A 4+2 split fills six GPUs without
# requiring two nodes to each expose three contiguous free devices.
job_a="$(sbatch --parsable --job-name=prorm-converged-train-a --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time="${walltime}" --array=0 --gpus-per-node=4 \
  --output="${run_root}/logs/train-a-%A_%a.out" \
  --export="${common},PRORM_GPU_WORKERS_PER_JOB=4,PRORM_WORKER_OFFSET=0" \
  "${repo_root}/scripts/hpc4/direct_converged_gpu.sbatch")"
job_b="$(sbatch --parsable --job-name=prorm-converged-train-b --partition=gpu-l20 \
  --exclude=gpu18,gpu19 --time="${walltime}" --array=0 --gpus-per-node=2 \
  --output="${run_root}/logs/train-b-%A_%a.out" \
  --export="${common},PRORM_GPU_WORKERS_PER_JOB=2,PRORM_WORKER_OFFSET=4" \
  "${repo_root}/scripts/hpc4/direct_converged_gpu.sbatch")"
printf 'train_job_a=%s\ntrain_job_b=%s\nrun_root=%s\n' "${job_a}" "${job_b}" "${run_root}"
