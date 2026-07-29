#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 CONFIG IMAGE RUN_ROOT PARTITION WALLTIME" >&2; exit 2; }
repo_root="$(git rev-parse --show-toplevel)"
config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
run_root="$(realpath -e "$3")"
partition="$4"
walltime="$5"
[[ "${partition}" = "amd" || "${partition}" = "intel" ]] || {
  echo "aggregation requires an HPC4 CPU partition: amd or intel" >&2
  exit 2
}
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "aggregation requires a clean worktree" >&2
  exit 2
}
case "${config}" in "${repo_root}/configs/"*.yaml) ;; *) echo "config must be in configs/" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;; esac
for seed_dir in "${run_root}"/seed-*; do
  [[ -f "${seed_dir}/reward_result.json" && -f "${seed_dir}/policy_utility/metrics.json" ]] || {
    echo "incomplete seed directory: ${seed_dir}" >&2
    exit 2
  }
done
sbatch --parsable --partition="${partition}" --time="${walltime}" \
  --output="${run_root}/logs/prorm-aggregate-%j.out" \
  --export=ALL,PRORM_REPO_ROOT="${repo_root}",PRORM_CONFIG="${config}",PRORM_IMAGE="${image}",PRORM_RUN_ROOT="${run_root}" \
  "${repo_root}/scripts/hpc4/aggregate.sbatch"
