#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 4 ]] || {
  echo "usage: $0 IMAGE REPORT_ROOT GPU_PARTITION WALLTIME" >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel)"
image="$(realpath -e "$1")"
report_root="$(realpath -m "$2")"
partition="$3"
walltime="$4"

[[ "${partition}" = "gpu-l20" ]] || {
  echo "formal smoke is frozen to the HPC4 gpu-l20 partition" >&2
  exit 2
}
case "${image}" in /project/sigroup/*) ;; *) echo "image must be under /project/sigroup" >&2; exit 2 ;; esac
case "${report_root}" in /project/sigroup/*) ;; *) echo "report root must be under /project/sigroup" >&2; exit 2 ;; esac
[[ -z "$(git -C "${repo_root}" status --porcelain)" ]] || {
  echo "GPU smoke requires a clean worktree" >&2
  exit 2
}
mkdir -p "${report_root}"
image_sha="$(sha256sum "${image}" | cut -d' ' -f1)"
git_commit="$(git -C "${repo_root}" rev-parse HEAD)"

sbatch --parsable --partition="${partition}" --time="${walltime}" \
  --output="${report_root}/prorm-gpu-smoke-%j.out" \
  --export=ALL,PRORM_REPO_ROOT="${repo_root}",PRORM_IMAGE="${image}",PRORM_IMAGE_SHA256="${image_sha}",PRORM_GIT_COMMIT="${git_commit}",PRORM_REPORT_ROOT="${report_root}" \
  "${repo_root}/scripts/hpc4/gpu_smoke.sbatch"
