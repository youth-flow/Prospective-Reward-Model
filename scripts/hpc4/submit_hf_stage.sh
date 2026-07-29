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
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
mkdir -p "${hf_cache}"
sbatch --parsable --partition="${partition}" --time="${walltime}" \
  --export=ALL,PRORM_REPO_ROOT="${repo_root}",PRORM_CONFIG="${config}",PRORM_IMAGE="${image}",PRORM_HF_CACHE="${hf_cache}" \
  "${repo_root}/scripts/hpc4/hf_stage.sbatch"
