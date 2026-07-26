#!/usr/bin/env bash
# Capture one fixed-three terminal and publish the five-endpoint aggregate in SIF.

#SBATCH --job-name=prorm-gatee-finalize
#SBATCH --account=sigroup
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'GateE finalizer fatal: %s\n' "$*" >&2
  exit 1
}

for name in \
  PRORM_PROJECT_ROOT PRORM_REPO_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_BUDGETED_ARRAY_JOB_ID PRORM_BUDGETED_DESIGN_SHA256 PRORM_GIT_COMMIT; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-l20" \
  && "${SLURM_RESTART_COUNT:-0}" == "0" ]] \
  || die "finalizer requires one unrequeued gpu-l20 allocation"
for command_name in apptainer git realpath sha256sum awk mkdir; do
  command -v "${command_name}" >/dev/null 2>&1 || die "${command_name} is unavailable"
done
host_python="$(realpath -e -- "${HOST_PYTHON}")" \
  || die "fixed host Python is unavailable"
[[ "${host_python}" == "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python must be a canonical executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
repo_root="$(realpath -e -- "${PRORM_REPO_ROOT}")"
image="$(realpath -e -- "${PRORM_IMAGE}")"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected production project root"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production repository root"
[[ "${PRORM_BUDGETED_ARRAY_JOB_ID}" =~ ^[1-9][0-9]*$ ]] \
  || die "array job ID is invalid"
[[ "${PRORM_BUDGETED_DESIGN_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "design SHA-256 is invalid"
[[ "${PRORM_GIT_COMMIT}" =~ ^[0-9a-f]{40,64}$ ]] || die "Git commit is invalid"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "image SHA-256 is invalid"
[[ "$(sha256sum -- "${image}" | awk '{print $1}')" == "${PRORM_IMAGE_SHA256}" ]] \
  || die "container image SHA-256 mismatch"
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" == "${PRORM_GIT_COMMIT}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "finalizer repository must be the clean producer commit"

campaign="${project_root}/runs/phase2-budgeted-end-to-end/${PRORM_BUDGETED_DESIGN_SHA256}"
artifact_campaign="${project_root}/artifacts/phase2-budgeted-end-to-end/${PRORM_BUDGETED_DESIGN_SHA256}"
aggregate_root="$(realpath -e -- "${project_root}/aggregates")"
[[ -d "${campaign}" && ! -L "${campaign}" ]] || die "campaign root is unavailable"
[[ -d "${artifact_campaign}" && ! -L "${artifact_campaign}" ]] \
  || die "artifact campaign root is unavailable"
terminal_parent="${campaign}/terminal-evidence"
if [[ -e "${terminal_parent}" || -L "${terminal_parent}" ]]; then
  [[ -d "${terminal_parent}" && ! -L "${terminal_parent}" ]] \
    || die "terminal-evidence parent is unsafe"
else
  mkdir -- "${terminal_parent}"
fi
terminal_parent="$(realpath -e -- "${terminal_parent}")"
terminal="${terminal_parent}/array-${PRORM_BUDGETED_ARRAY_JOB_ID}.json"
raw_terminal="${terminal_parent}/array-${PRORM_BUDGETED_ARRAY_JOB_ID}.all.sacct.psv"
[[ ! -e "${terminal}" && ! -L "${terminal}" \
  && ! -e "${raw_terminal}" && ! -L "${raw_terminal}" \
  && ! -e "${terminal_parent}/array-${PRORM_BUDGETED_ARRAY_JOB_ID}.sacct.psv" ]] \
  || die "terminal evidence already exists"

"${host_python}" -I -S \
  "${repo_root}/scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py" \
  capture-raw "${PRORM_BUDGETED_ARRAY_JOB_ID}" "${raw_terminal}"
"${host_python}" -I -S \
  "${repo_root}/scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py" \
  capture "${PRORM_BUDGETED_ARRAY_JOB_ID}" "${terminal}"
terminal_sha256="$(sha256sum -- "${terminal}" | awk '{print $1}')"
[[ "${terminal_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "terminal SHA-256 is invalid"

seeds=(20261001 20261002 20261003)
runs=()
for task in 0 1 2; do
  run="${campaign}/seed-${seeds[task]}/job-${PRORM_BUDGETED_ARRAY_JOB_ID}_${task}"
  [[ -d "${run}" && ! -L "${run}" && -f "${run}/SUCCESS" ]] \
    || die "seed ${seeds[task]} lacks its terminal successful run"
  runs+=("${run}")
done
output="${aggregate_root}/phase2-budgeted-end-to-end-${PRORM_BUDGETED_DESIGN_SHA256}.json"
[[ ! -e "${output}" && ! -L "${output}" \
  && ! -e "${output}.evidence.json" && ! -L "${output}.evidence.json" ]] \
  || die "fixed budgeted aggregate output already exists"

apptainer exec \
  --cleanenv \
  --nv \
  --no-mount home,cwd,bind-paths \
  --pwd "${repo_root}" \
  --bind "${repo_root}:${repo_root}:ro" \
  --bind "${campaign}:${campaign}:ro" \
  --bind "${artifact_campaign}:${artifact_campaign}:ro" \
  --bind "${aggregate_root}:${aggregate_root}:rw" \
  --env "PYTHONPATH=${repo_root}/src" \
  --env "PYTHONNOUSERSITE=1" \
  --env "PYTHONDONTWRITEBYTECODE=1" \
  "${image}" \
  python "${repo_root}/scripts/hpc4/aggregate_phase2_budgeted_end_to_end.py" \
    "${output}" \
    "${runs[@]}" \
    --project-root "${project_root}" \
    --repo-root "${repo_root}" \
    --terminal-evidence "${terminal}" \
    --terminal-evidence-sha256 "${terminal_sha256}" \
    --array-job-id "${PRORM_BUDGETED_ARRAY_JOB_ID}"

[[ -f "${output}" && ! -L "${output}" \
  && -f "${output}.evidence.json" && ! -L "${output}.evidence.json" ]] \
  || die "five-endpoint aggregate publication did not complete"
printf 'GATEE_AGGREGATE=%s\n' "${output}"
printf 'GATEE_AGGREGATE_SHA256=%s\n' "$(sha256sum -- "${output}" | awk '{print $1}')"
