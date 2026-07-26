#!/usr/bin/env bash
# Launch one dependency-bound R3 scheduler terminalizer on an HPC4 L20 node.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 terminalize launcher fatal: %s\n' "$*" >&2
  exit 1
}

usage() {
  die "usage: $0 gatep <gatep-attempt-root> | $0 primary <attempt-root> <task-id> <gatep-operational-bundle>"
}

[[ -z "${SLURM_JOB_ID:-}" ]] \
  || die "login-side terminalize launcher must run outside Slurm"
[[ -n "${USER:-}" && "${USER}" != *","* \
  && "${USER}" != *$'\n'* && "${USER}" != *$'\r'* ]] \
  || die "scheduler user is unavailable or unsafe"
for name in \
  PRORM_R3_REPO_ROOT PRORM_R3_PROJECT_ROOT PRORM_R3_IMAGE \
  PRORM_R3_IMAGE_SHA256; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in env git install realpath sha256sum srun; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done
while IFS='=' read -r name _; do
  case "${name}" in
    APPTAINER*|SINGULARITY*|SBATCH_*|PRORM_R3_TERMINALIZE_*)
      die "unset ambient terminalize control variable ${name}"
      ;;
  esac
done < <(env)

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_IMAGE}")"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production Git repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected production project root"
[[ "${image}" == "${project_root}/"* && -f "${image}" && ! -L "${image}" ]] \
  || die "container image is not a canonical project file"
[[ "${PRORM_R3_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "container SHA-256 is invalid"
printf '%s  %s\n' "${PRORM_R3_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"

host_python="$(realpath -e -- "${HOST_PYTHON}")"
[[ "${host_python}" == "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python is not a canonical non-symlink executable"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "Git commit is invalid"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout must be clean"

route_helper="${repo_root}/scripts/hpc4/phase2_r3_terminalize_stdlib.py"
driver="${repo_root}/scripts/hpc4/phase2_r3_terminalize.sbatch"
[[ -f "${route_helper}" && ! -L "${route_helper}" \
  && -f "${driver}" && ! -L "${driver}" ]] \
  || die "R3 terminalize execution surface is incomplete"

[[ $# -ge 1 ]] || usage
mode="$1"
case "${mode}" in
  gatep)
    [[ $# -eq 2 ]] || usage
    route_json="$(
      "${host_python}" -I -S "${route_helper}" plan-gatep \
        --project-root "${project_root}" \
        --attempt-root "$2"
    )" || die "Gate-P terminal route inspection failed"
    ;;
  primary)
    [[ $# -eq 4 ]] || usage
    [[ "$3" =~ ^[0-2]$ ]] || die "Gate-R task ID must be 0, 1, or 2"
    route_json="$(
      "${host_python}" -I -S "${route_helper}" plan-primary \
        --project-root "${project_root}" \
        --attempt-root "$2" \
        --task-id "$3" \
        --operational-bundle "$4"
    )" || die "Gate-R terminal route inspection failed"
    ;;
  *)
    usage
    ;;
esac

mapfile -t route_fields < <(
  printf '%s\n' "${route_json}" \
    | "${host_python}" -I -S -c '
import json,sys
value=json.load(sys.stdin)
for key in (
    "mode","job_selector","job_id_raw","route_status","finalizer_command",
    "attempt_root","operational_bundle","allocation_intent","runtime_receipt",
    "runtime_closure","raw_sacct","evidence_directory","task_id","segment_index",
):
    item=value.get(key)
    print("" if item is None else item)
'
)
[[ "${#route_fields[@]}" -eq 14 ]] || die "terminal route field count is invalid"
route_mode="${route_fields[0]}"
job_selector="${route_fields[1]}"
dependency_job_id="${route_fields[2]:-${job_selector}}"
route_status="${route_fields[3]}"
finalizer_command="${route_fields[4]}"
attempt_root="${route_fields[5]}"
operational_bundle="${route_fields[6]}"
allocation_intent="${route_fields[7]}"
runtime_receipt="${route_fields[8]}"
runtime_closure="${route_fields[9]}"
raw_sacct="${route_fields[10]}"
evidence_directory="${route_fields[11]}"
task_id="${route_fields[12]}"
segment_index="${route_fields[13]}"
[[ "${route_mode}" == "${mode}" ]] || die "terminal route mode changed"
[[ "${job_selector}" =~ ^[1-9][0-9]*(_[0-9]+)?$ \
  && "${dependency_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "terminal route returned an invalid Slurm identity"
for value in \
  "${attempt_root}" "${operational_bundle}" "${allocation_intent}" \
  "${runtime_receipt}" "${runtime_closure}" "${raw_sacct}" \
  "${evidence_directory}"; do
  [[ "${value}" != *","* && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "terminal route contains an unsafe export value"
done

install -d -m 0750 \
  "${attempt_root}/terminal-raw" \
  "${attempt_root}/terminal-evidence"
[[ "$(realpath -e -- "$(dirname -- "${raw_sacct}")")" == \
  "${attempt_root}/terminal-raw" ]] \
  || die "raw sacct parent is not canonical"
[[ "$(realpath -e -- "$(dirname -- "${evidence_directory}")")" == \
  "${attempt_root}/terminal-evidence" ]] \
  || die "terminal evidence parent is not canonical"

sha256_file() {
  local path="$1" digest
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "terminalize input is missing or unsafe: ${path}"
  digest="$(sha256sum -- "${path}" | cut -d' ' -f1)"
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
    || die "failed to hash terminalize input: ${path}"
  printf '%s\n' "${digest}"
}

operational_bundle_sha256="$(sha256_file "${operational_bundle}")"
allocation_intent_sha256=""
runtime_receipt_sha256=""
runtime_closure_sha256=""
if [[ "${mode}" == "gatep" ]]; then
  allocation_intent_sha256="$(sha256_file "${allocation_intent}")"
  runtime_receipt_sha256="$(sha256_file "${runtime_receipt}")"
else
  runtime_closure_sha256="$(sha256_file "${runtime_closure}")"
fi

export_spec="PATH=/usr/bin:/bin"
for item in \
  "PRORM_R3_TERMINALIZE_MODE=${mode}" \
  "PRORM_R3_TERMINALIZE_REPO_ROOT=${repo_root}" \
  "PRORM_R3_TERMINALIZE_PROJECT_ROOT=${project_root}" \
  "PRORM_R3_TERMINALIZE_IMAGE=${image}" \
  "PRORM_R3_TERMINALIZE_IMAGE_SHA256=${PRORM_R3_IMAGE_SHA256}" \
  "PRORM_R3_TERMINALIZE_GIT_COMMIT=${commit}" \
  "PRORM_R3_TERMINALIZE_USER=${USER}" \
  "PRORM_R3_TERMINALIZE_JOB_SELECTOR=${job_selector}" \
  "PRORM_R3_TERMINALIZE_DEPENDENCY_JOB_ID=${dependency_job_id}" \
  "PRORM_R3_TERMINALIZE_ROUTE_STATUS=${route_status}" \
  "PRORM_R3_TERMINALIZE_FINALIZER_COMMAND=${finalizer_command}" \
  "PRORM_R3_TERMINALIZE_ATTEMPT_ROOT=${attempt_root}" \
  "PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE=${operational_bundle}" \
  "PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256=${operational_bundle_sha256}" \
  "PRORM_R3_TERMINALIZE_ALLOCATION_INTENT=${allocation_intent}" \
  "PRORM_R3_TERMINALIZE_ALLOCATION_INTENT_SHA256=${allocation_intent_sha256}" \
  "PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT=${runtime_receipt}" \
  "PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT_SHA256=${runtime_receipt_sha256}" \
  "PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE=${runtime_closure}" \
  "PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256=${runtime_closure_sha256}" \
  "PRORM_R3_TERMINALIZE_RAW_SACCT=${raw_sacct}" \
  "PRORM_R3_TERMINALIZE_EVIDENCE_DIRECTORY=${evidence_directory}" \
  "PRORM_R3_TERMINALIZE_TASK_ID=${task_id}" \
  "PRORM_R3_TERMINALIZE_SEGMENT_INDEX=${segment_index}"; do
  [[ "${item}" != *","* && "${item}" != *$'\n'* && "${item}" != *$'\r'* ]] \
    || die "terminalize export value is unsafe"
  export_spec+=",${item}"
done

printf 'R3_TERMINALIZE_MODE=%s\n' "${mode}"
printf 'R3_TERMINALIZE_JOB_SELECTOR=%s\n' "${job_selector}"
printf 'R3_TERMINALIZE_AFTERANY_DEPENDENCY=afterany:%s\n' \
  "${dependency_job_id}"
printf 'R3_TERMINALIZE_RAW_SACCT=%s\n' "${raw_sacct}"
exec srun \
  --dependency="afterany:${dependency_job_id}" \
  --account=sigroup \
  --partition=gpu-l20 \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --job-name="prorm-r3-${mode}-terminalize" \
  --chdir="${repo_root}" \
  --export="${export_spec}" \
  /usr/bin/bash "${driver}"
