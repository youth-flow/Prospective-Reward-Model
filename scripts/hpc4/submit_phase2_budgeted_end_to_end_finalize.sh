#!/usr/bin/env bash
# Submit the sole fixed-three terminal-capture and five-endpoint aggregator.

set -euo pipefail
umask 077

die() {
  printf 'GateE finalizer submit fatal: %s\n' "$*" >&2
  exit 1
}

[[ $# -eq 3 ]] \
  || die "usage: $0 <array-job-id> <design-sha256> <git-commit>"
array_job_id="$1"
design_sha256="$2"
git_commit="$3"
[[ "${array_job_id}" =~ ^[1-9][0-9]*$ ]] || die "array job ID is invalid"
[[ "${design_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "design SHA-256 is invalid"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "Git commit is invalid"
[[ -z "${SLURM_JOB_ID:-}" ]] || die "finalizer submit must run outside Slurm"
for name in PRORM_PROJECT_ROOT PRORM_REPO_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in env mkdir realpath rmdir sbatch; do
  command -v "${command_name}" >/dev/null 2>&1 || die "${command_name} is unavailable"
done
while IFS='=' read -r name _; do
  case "${name}" in
    APPTAINER*|SINGULARITY*|SBATCH_*|PRORM_BUDGETED_*|\
    PRORM_RECOVERY_AUTHORIZATION*)
      die "unset ambient finalizer control variable: ${name}"
      ;;
  esac
done < <(env)

repo_root="$(realpath -e -- "${PRORM_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_IMAGE}")"
[[ "${repo_root}" == "${PRORM_REPO_ROOT}" \
  && "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" \
  && -d "${repo_root}" && ! -L "${repo_root}" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "${PRORM_PROJECT_ROOT}" \
  && "${project_root}" == "/project/sigroup/smart-reward-model" \
  && -d "${project_root}" && ! -L "${project_root}" ]] \
  || die "unexpected production project root"
[[ "${image}" == "${PRORM_IMAGE}" && "${image}" == "${project_root}/"* \
  && -f "${image}" && ! -L "${image}" ]] \
  || die "container image is not a canonical project file"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die "image SHA-256 is invalid"
driver="${repo_root}/scripts/hpc4/phase2_budgeted_end_to_end_finalize.sbatch"
[[ -f "${driver}" && ! -L "${driver}" ]] || die "fixed finalizer driver is unavailable"
campaign="${project_root}/runs/phase2-budgeted-end-to-end/${design_sha256}"
[[ -d "${campaign}" && ! -L "${campaign}" ]] || die "budgeted campaign root is unavailable"
terminal="${campaign}/terminal-evidence/array-${array_job_id}.json"
output="${project_root}/aggregates/phase2-budgeted-end-to-end-${design_sha256}.json"
[[ ! -e "${terminal}" && ! -L "${terminal}" \
  && ! -e "${output}" && ! -L "${output}" ]] \
  || die "GateE finalizer evidence already exists"
submission_lock="${campaign}/finalizer-submit.lock"
mkdir -- "${submission_lock}" 2>/dev/null \
  || die "a GateE finalizer has already been submitted"
cleanup_lock=1
trap 'if [[ "${cleanup_lock}" == 1 ]]; then rmdir -- "${submission_lock}" 2>/dev/null || true; fi' EXIT

export_spec="PATH=/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_REPO_ROOT=${repo_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_BUDGETED_ARRAY_JOB_ID=${array_job_id},PRORM_BUDGETED_DESIGN_SHA256=${design_sha256},PRORM_GIT_COMMIT=${git_commit}"
job_id="$(
  sbatch \
    --parsable \
    --account=sigroup \
    --partition=gpu-l20 \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --no-requeue \
    --kill-on-invalid-dep=yes \
    --dependency="afterany:${array_job_id}" \
    --job-name=prorm-gatee-finalize \
    --chdir="${repo_root}" \
    --output="${campaign}/finalize-%j.out" \
    --error="${campaign}/finalize-%j.err" \
    --export="${export_spec}" \
    "${driver}"
)" || die "finalizer sbatch submission failed"
job_id="${job_id%%;*}"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf '%s\n' "${job_id}" >"${submission_lock}/job-id"
cleanup_lock=0
trap - EXIT
printf '%s\n' "${job_id}"
