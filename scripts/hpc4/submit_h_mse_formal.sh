#!/usr/bin/env bash
set -euo pipefail
umask 027
[[ $# -eq 9 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE SOURCE_RUN SOURCE_M6 RUN_ROOT ARCHIVE_ROOT IMAGE_SOURCE_COMMIT SMOKE_JSON" >&2
  exit 2
}
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"; image="$(realpath -e "$2")"; hf_cache="$(realpath -e "$3")"
source_run="$(realpath -e "$4")"; source_m6="$(realpath -e "$5")"
run_root="$(realpath -m "$6")"; archive_root="$(realpath -m "$7")"
image_source_commit="$8"; smoke_json="$(realpath -e "$9")"
python3 - "${smoke_json}" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
assert x.get("schema") == "prorm-h-mse-smoke/v1" and x.get("status") == "passed"
assert x.get("new_reward_count") == 4 and x.get("new_policy_count") == 4
PY
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || { echo "submission requires clean worktree" >&2; exit 2; }
[[ ! -e "${run_root}" ]] || { echo "refusing to reuse run root" >&2; exit 2; }
[[ ! -e "${archive_root}" ]] || { echo "refusing to overwrite archive" >&2; exit 2; }
mkdir -p "${run_root}/logs"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"; image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
source_sha="0ff8cb872c5bdecc33bd2e5ded7d9c3adcbc43d7b6c355b40f8d34a1ae95ce92"
inventory="${hf_cache}/inventories/${source_sha}.json"; [[ -f "${inventory}" ]] || { echo "missing HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_H_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SOURCE_COMMIT=${image_source_commit},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_SOURCE_M6_ROOT=${source_m6},PRORM_RUN_ROOT=${run_root},PRORM_ARCHIVE_ROOT=${archive_root}"
prepare="$(sbatch --parsable --job-name=prorm-h-mse-prep --partition=gpu-l20 --time=03:00:00 \
  --gpus-per-node=3 --output="${run_root}/logs/prepare-%j.out" \
  --export="${common},PRORM_H_STAGE=prepare" "${repo_root}/scripts/hpc4/h_mse_gpu.sbatch")"
rollout_a="$(sbatch --parsable --job-name=prorm-h-mse-roll-a --partition=gpu-l20 --time=08:00:00 \
  --gpus-per-node=4 --dependency="afterok:${prepare}" --output="${run_root}/logs/rollout-a-%j.out" \
  --export="${common},PRORM_H_STAGE=rollout,PRORM_H_WORKERS=8,PRORM_H_GPUS_PER_JOB=4,SLURM_ARRAY_TASK_ID=0" \
  "${repo_root}/scripts/hpc4/h_mse_gpu.sbatch")"
rollout_b="$(sbatch --parsable --job-name=prorm-h-mse-roll-b --partition=gpu-l20 --time=08:00:00 \
  --gpus-per-node=4 --dependency="afterok:${prepare}" --output="${run_root}/logs/rollout-b-%j.out" \
  --export="${common},PRORM_H_STAGE=rollout,PRORM_H_WORKERS=8,PRORM_H_GPUS_PER_JOB=4,SLURM_ARRAY_TASK_ID=1" \
  "${repo_root}/scripts/hpc4/h_mse_gpu.sbatch")"
finalize="$(sbatch --parsable --job-name=prorm-h-mse-final --partition=amd --time=02:00:00 \
  --dependency="afterok:${rollout_a}:${rollout_b}" --output="${run_root}/logs/finalize-%j.out" \
  --export="${common}" "${repo_root}/scripts/hpc4/h_mse_finalize.sbatch")"
printf 'run_root=%s\narchive_root=%s\nprepare_job=%s\nrollout_jobs=%s,%s\nfinalize_job=%s\n' \
  "${run_root}" "${archive_root}" "${prepare}" "${rollout_a}" "${rollout_b}" "${finalize}"
