#!/usr/bin/env bash
set -euo pipefail
umask 027
[[ $# -eq 7 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE SOURCE_RUN SOURCE_M6 SMOKE_ROOT IMAGE_SOURCE_COMMIT" >&2
  exit 2
}
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"; image="$(realpath -e "$2")"; hf_cache="$(realpath -e "$3")"
source_run="$(realpath -e "$4")"; source_m6="$(realpath -e "$5")"
smoke_root="$(realpath -m "$6")"; image_source_commit="$7"
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || { echo "submission requires clean worktree" >&2; exit 2; }
if [[ -e "${smoke_root}" && -e "${smoke_root}/smoke.json" ]]; then
  echo "refusing to overwrite a completed smoke root" >&2
  exit 2
fi
mkdir -p "${smoke_root}/logs"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"; image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
source_sha="0ff8cb872c5bdecc33bd2e5ded7d9c3adcbc43d7b6c355b40f8d34a1ae95ce92"
inventory="${hf_cache}/inventories/${source_sha}.json"; [[ -f "${inventory}" ]] || { echo "missing HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_H_CONFIG=${config},PRORM_IMAGE=${image},PRORM_IMAGE_SOURCE_COMMIT=${image_source_commit},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_SOURCE_M6_ROOT=${source_m6},PRORM_RUN_ROOT=${smoke_root},PRORM_H_STAGE=smoke"
job="$(sbatch --parsable --job-name=prorm-h-mse-smoke --partition=gpu-l20 --time=02:00:00 \
  --gpus-per-node=1 --output="${smoke_root}/logs/smoke-%j.out" --export="${common}" \
  "${repo_root}/scripts/hpc4/h_mse_gpu.sbatch")"
printf 'smoke_root=%s\nsmoke_job=%s\n' "${smoke_root}" "${job}"
