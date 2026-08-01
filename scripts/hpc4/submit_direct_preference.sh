#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 8 ]] || {
  echo "usage: $0 EXTENSION_CONFIG IMAGE HF_CACHE SOURCE_RUN BASELINE_RUN RUN_ROOT STAGE WORKERS" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
extension_config="$(realpath -e "$1")"
image="$(realpath -e "$2")"
hf_cache="$(realpath -e "$3")"
source_run="$(realpath -e "$4")"
baseline_run="$(realpath -e "$5")"
run_root="$(realpath -m "$6")"
stage="$7"
workers="$8"
[[ "${stage}" =~ ^(reference|train|evaluate|aggregate|audit)$ ]] || {
  echo "stage must be reference, train, evaluate, aggregate, or audit" >&2
  exit 2
}
[[ "${workers}" =~ ^[1-2]$ ]] || { echo "workers must be 1 or 2" >&2; exit 2; }
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "direct-preference submission requires a clean worktree" >&2
  exit 2
}
case "${extension_config}" in
  "${repo_root}/configs/"*.yaml) ;;
  *) echo "extension config must be under configs/" >&2; exit 2 ;;
esac
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${hf_cache}" in /project/sigroup/*) ;; *) echo "HF cache must be under /project/sigroup" >&2; exit 2 ;; esac
case "${source_run}" in /project/sigroup/*) ;; *) echo "source run must be archived under /project/sigroup" >&2; exit 2 ;; esac
case "${baseline_run}" in /project/sigroup/*) ;; *) echo "baseline run must be archived under /project/sigroup" >&2; exit 2 ;; esac
case "${run_root}" in "/scratch/${USER}/"*) ;; *) echo "run root must be under user scratch" >&2; exit 2 ;; esac
if [[ "${stage}" = "reference" ]]; then
  [[ ! -e "${run_root}" ]] || { echo "refusing an existing new run root" >&2; exit 2; }
  mkdir -p "${run_root}/logs"
else
  run_root="$(realpath -e "${run_root}")"
fi

git_commit="$(git -C "${repo_root}" rev-parse HEAD)"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
mapfile -t identity < <(PYTHONPATH="${repo_root}/src" python3 - "${extension_config}" <<'PY'
import sys
from smart_reward.direct_preference import load_direct_preference_config, resolve_source_config
config = load_direct_preference_config(sys.argv[1])
_, source = resolve_source_config(sys.argv[1], config)
print(config["source_config_sha256"])
print(len(config["experiment"]["seeds"]))
PY
)
inventory="${hf_cache}/inventories/${identity[0]}.json"
[[ -f "${inventory}" ]] || { echo "missing source-config HF inventory" >&2; exit 2; }
inventory_sha="$(sha256sum "${inventory}" | cut -d' ' -f1)"
common="ALL,PRORM_REPO_ROOT=${repo_root},PRORM_EXTENSION_CONFIG=${extension_config},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${image_sha},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY_SHA256=${inventory_sha},PRORM_GIT_COMMIT=${git_commit},PRORM_SOURCE_RUN_ROOT=${source_run},PRORM_BASELINE_RUN_ROOT=${baseline_run},PRORM_RUN_ROOT=${run_root}"

case "${stage}" in
  reference|train)
    job="$(sbatch --parsable --job-name="prorm-direct-${stage}" --partition=gpu-l20 \
      --exclude=gpu18,gpu19 --time=24:00:00 --array="0-$((workers - 1))" \
      --output="${run_root}/logs/${stage}-%A_%a.out" \
      --export="${common},PRORM_DIRECT_STAGE=${stage},PRORM_DIRECT_WORKERS=${workers}" \
      "${repo_root}/scripts/hpc4/direct_gpu.sbatch")"
    ;;
  evaluate)
    job="$(sbatch --parsable --job-name=prorm-direct-eval --partition=amd \
      --time=04:00:00 --array="0-$((identity[1] - 1))" \
      --output="${run_root}/logs/evaluate-%A_%a.out" --export="${common}" \
      "${repo_root}/scripts/hpc4/direct_evaluate.sbatch")"
    ;;
  aggregate)
    job="$(sbatch --parsable --job-name=prorm-direct-aggregate --partition=amd \
      --time=00:20:00 --output="${run_root}/logs/aggregate-%j.out" \
      --export="${common}" "${repo_root}/scripts/hpc4/direct_aggregate.sbatch")"
    ;;
  audit)
    job="$(sbatch --parsable --job-name=prorm-direct-audit --partition=amd \
      --time=00:20:00 --output="${run_root}/logs/audit-%j.out" \
      --export="${common}" "${repo_root}/scripts/hpc4/direct_audit.sbatch")"
    ;;
esac
printf 'stage=%s\njob_id=%s\nrun_root=%s\n' "${stage}" "${job}" "${run_root}"
