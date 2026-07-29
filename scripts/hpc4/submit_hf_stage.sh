#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || {
  echo "usage: $0 CONFIG IMAGE HF_CACHE CPU_PARTITION WALLTIME" >&2
  exit 2
}
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -m "$3")"
partition="$4"
walltime="$5"

[[ "${partition}" = "amd" || "${partition}" = "intel" ]] || {
  echo "HF staging requires an HPC4 CPU partition: amd or intel" >&2
  exit 2
}
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "HF staging requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
mkdir -p "${hf_cache}/logs"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
sbatch --parsable --partition="${partition}" --time="${walltime}" \
  --output="${hf_cache}/logs/hf-stage-%j.out" \
  --export=ALL,PRORM_REPO_ROOT="${repo_root}",PRORM_CONFIG="${config}",PRORM_IMAGE="${image}",PRORM_IMAGE_SHA256="${image_sha}",PRORM_GIT_COMMIT="${git_commit}",PRORM_HF_CACHE="${hf_cache}" \
  "${repo_root}/scripts/hpc4/hf_stage.sbatch"
