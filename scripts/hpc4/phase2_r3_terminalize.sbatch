#!/usr/bin/env bash
# Compute-side driver for one dependency-bound R3 terminalization.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 terminalize compute fatal: %s\n' "$*" >&2
  exit 1
}

required_environment=(
  PRORM_R3_TERMINALIZE_MODE
  PRORM_R3_TERMINALIZE_REPO_ROOT
  PRORM_R3_TERMINALIZE_PROJECT_ROOT
  PRORM_R3_TERMINALIZE_IMAGE
  PRORM_R3_TERMINALIZE_IMAGE_SHA256
  PRORM_R3_TERMINALIZE_GIT_COMMIT
  PRORM_R3_TERMINALIZE_USER
  PRORM_R3_TERMINALIZE_JOB_SELECTOR
  PRORM_R3_TERMINALIZE_DEPENDENCY_JOB_ID
  PRORM_R3_TERMINALIZE_ROUTE_STATUS
  PRORM_R3_TERMINALIZE_FINALIZER_COMMAND
  PRORM_R3_TERMINALIZE_ATTEMPT_ROOT
  PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE
  PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256
  PRORM_R3_TERMINALIZE_RAW_SACCT
  PRORM_R3_TERMINALIZE_EVIDENCE_DIRECTORY
)
for name in "${required_environment[@]}"; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_ARRAY_JOB_ID:-}" ]] \
  || die "terminalize driver requires one non-array Slurm allocation"
[[ "${SLURM_CLUSTER_NAME:-hpc4}" == "hpc4" \
  && "${SLURM_JOB_ACCOUNT:-}" == "sigroup" \
  && "${SLURM_JOB_PARTITION:-}" == "gpu-l20" ]] \
  || die "terminalize allocation identity is invalid"
for command_name in apptainer git realpath sacct sha256sum squeue; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "${command_name} is unavailable"
done

host_python="$(realpath -e -- "${HOST_PYTHON}")"
[[ "${host_python}" == "${HOST_PYTHON}" && -f "${host_python}" \
  && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python is not canonical"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

repo_root="$(realpath -e -- "${PRORM_R3_TERMINALIZE_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_TERMINALIZE_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_TERMINALIZE_IMAGE}")"
attempt_root="$(realpath -e -- "${PRORM_R3_TERMINALIZE_ATTEMPT_ROOT}")"
operational_bundle="$(
  realpath -e -- "${PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE}"
)"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" \
  && "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "terminalize production roots are invalid"
[[ "${image}" == "${project_root}/"* && -f "${image}" && ! -L "${image}" ]] \
  || die "terminalize image is not a canonical project file"
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" == \
  "${PRORM_R3_TERMINALIZE_GIT_COMMIT}" ]] \
  || die "terminalize Git commit changed"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "terminalize checkout is not clean"

verify_sha256() {
  local path="$1" expected="$2"
  [[ "${expected}" =~ ^[0-9a-f]{64}$ ]] || die "invalid expected SHA-256"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "terminalize input SHA-256 mismatch: ${path}"
}
verify_sha256 "${image}" "${PRORM_R3_TERMINALIZE_IMAGE_SHA256}"
verify_sha256 \
  "${operational_bundle}" \
  "${PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256}"

route_helper="${repo_root}/scripts/hpc4/phase2_r3_terminalize_stdlib.py"
capture_cli="${repo_root}/scripts/hpc4/capture_phase2_r3_terminal.py"
reuse_verifier="${repo_root}/scripts/hpc4/verify_phase2_r3_terminalized.py"
for path in "${route_helper}" "${capture_cli}" "${reuse_verifier}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "terminalize source is missing or unsafe: ${path}"
done

mode="${PRORM_R3_TERMINALIZE_MODE}"
case "${mode}" in
  gatep)
    allocation_intent="$(
      realpath -e -- "${PRORM_R3_TERMINALIZE_ALLOCATION_INTENT}"
    )"
    runtime_receipt="$(
      realpath -e -- "${PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT}"
    )"
    verify_sha256 \
      "${allocation_intent}" \
      "${PRORM_R3_TERMINALIZE_ALLOCATION_INTENT_SHA256}"
    verify_sha256 \
      "${runtime_receipt}" \
      "${PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT_SHA256}"
    route_json="$(
      "${host_python}" -I -S "${route_helper}" plan-gatep \
        --project-root "${project_root}" \
        --attempt-root "${attempt_root}"
    )" || die "compute-side Gate-P route inspection failed"
    ;;
  primary)
    runtime_closure="$(
      realpath -e -- "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE}"
    )"
    verify_sha256 \
      "${runtime_closure}" \
      "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256}"
    route_json="$(
      "${host_python}" -I -S "${route_helper}" plan-primary \
        --project-root "${project_root}" \
        --attempt-root "${attempt_root}" \
        --task-id "${PRORM_R3_TERMINALIZE_TASK_ID}" \
        --operational-bundle "${operational_bundle}"
    )" || die "compute-side Gate-R route inspection failed"
    ;;
  *)
    die "terminalize mode must be gatep or primary"
    ;;
esac

"${host_python}" -I -S - "${route_json}" <<'PY' \
  || die "terminal route changed between login and compute"
import json
import os
import sys

route = json.loads(sys.argv[1])
expected = {
    "mode": os.environ["PRORM_R3_TERMINALIZE_MODE"],
    "job_selector": os.environ["PRORM_R3_TERMINALIZE_JOB_SELECTOR"],
    "route_status": os.environ["PRORM_R3_TERMINALIZE_ROUTE_STATUS"],
    "finalizer_command": os.environ["PRORM_R3_TERMINALIZE_FINALIZER_COMMAND"],
    "attempt_root": os.environ["PRORM_R3_TERMINALIZE_ATTEMPT_ROOT"],
    "operational_bundle": os.environ["PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE"],
    "raw_sacct": os.environ["PRORM_R3_TERMINALIZE_RAW_SACCT"],
    "evidence_directory": os.environ["PRORM_R3_TERMINALIZE_EVIDENCE_DIRECTORY"],
}
for key, value in expected.items():
    if route.get(key) != value:
        raise SystemExit(f"route field changed: {key}")
PY

raw_sacct="${PRORM_R3_TERMINALIZE_RAW_SACCT}"
evidence_directory="${PRORM_R3_TERMINALIZE_EVIDENCE_DIRECTORY}"
raw_capture="$(
  "${host_python}" -I -S "${route_helper}" capture \
    --job-selector "${PRORM_R3_TERMINALIZE_JOB_SELECTOR}" \
    --output "${raw_sacct}" \
    --user "${PRORM_R3_TERMINALIZE_USER}" \
    --attempts 30 \
    --interval-seconds 2
)" || die "raw sacct capture failed; no terminal capability was issued"
raw_sha256="$(
  printf '%s\n' "${raw_capture}" \
    | "${host_python}" -I -S -c \
      'import json,sys; print(json.load(sys.stdin)["raw_sha256"])'
)"
[[ "${raw_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "raw sacct capture returned an invalid SHA-256"
verify_sha256 "${raw_sacct}" "${raw_sha256}"

raw_sacct_canonical="$(realpath -e -- "${raw_sacct}")"
[[ "${raw_sacct_canonical}" == "${raw_sacct}" \
  && -f "${raw_sacct}" && ! -L "${raw_sacct}" ]] \
  || die "raw sacct artifact is not a canonical regular file"
terminal_evidence_parent="$(
  realpath -e -- "$(dirname -- "${evidence_directory}")"
)"
[[ "${terminal_evidence_parent}" == "${attempt_root}/terminal-evidence" \
  && -d "${terminal_evidence_parent}" && ! -L "${terminal_evidence_parent}" ]] \
  || die "terminal evidence parent is not canonical"

container=(
  apptainer exec
  --cleanenv
  --nv
  --no-mount home,cwd,bind-paths
  --pwd "${repo_root}"
  --bind "${repo_root}:${repo_root}:ro"
)
terminal_readonly_paths=("${operational_bundle}" "${raw_sacct}")
if [[ "${mode}" == "gatep" ]]; then
  terminal_readonly_paths+=("${allocation_intent}" "${runtime_receipt}")
else
  terminal_readonly_paths+=("${runtime_closure}")
fi
for path in "${terminal_readonly_paths[@]}"; do
  [[ "${path}" == "${project_root}/"* \
    && "${path}" != *","* && "${path}" != *":"* \
    && -f "${path}" && ! -L "${path}" ]] \
    || die "terminalize read-only bind path is unsafe: ${path}"
  container+=(--bind "${path}:${path}:ro")
done
container+=(
  --bind "${terminal_evidence_parent}:${terminal_evidence_parent}:rw"
  --env "PYTHONPATH=${repo_root}/src"
  --env "PYTHONNOUSERSITE=1"
  --env "PYTHONDONTWRITEBYTECODE=1"
  "${image}"
)

reused=false
if [[ -e "${evidence_directory}" || -L "${evidence_directory}" ]]; then
  reused=true
else
  if [[ "${mode}" == "gatep" ]]; then
    "${container[@]}" python3 "${capture_cli}" profile-finalize \
      --operational-bundle "${operational_bundle}" \
      --operational-bundle-file-sha256 \
        "${PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256}" \
      --allocation-intent "${allocation_intent}" \
      --allocation-intent-file-sha256 \
        "${PRORM_R3_TERMINALIZE_ALLOCATION_INTENT_SHA256}" \
      --runtime-receipt "${runtime_receipt}" \
      --runtime-receipt-file-sha256 \
        "${PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT_SHA256}" \
      --raw-sacct "${raw_sacct}" \
      --raw-sacct-sha256 "${raw_sha256}" \
      --evidence-directory "${evidence_directory}"
  else
    "${container[@]}" python3 "${capture_cli}" \
      "${PRORM_R3_TERMINALIZE_FINALIZER_COMMAND}" \
      --operational-bundle "${operational_bundle}" \
      --operational-bundle-file-sha256 \
        "${PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256}" \
      --runtime-closure "${runtime_closure}" \
      --runtime-closure-file-sha256 \
        "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256}" \
      --raw-sacct "${raw_sacct}" \
      --raw-sacct-sha256 "${raw_sha256}" \
      --evidence-directory "${evidence_directory}"
  fi
fi

[[ -d "${evidence_directory}" && ! -L "${evidence_directory}" ]] \
  || die "terminal finalizer did not publish its evidence directory"
manifest_sha256="$(
  sha256sum -- "${evidence_directory}/terminal-manifest.json" | cut -d' ' -f1
)"
bundle_raw_sha256="$(
  sha256sum -- "${evidence_directory}/raw-sacct.psv" | cut -d' ' -f1
)"
[[ "${manifest_sha256}" =~ ^[0-9a-f]{64}$ \
  && "${bundle_raw_sha256}" == "${raw_sha256}" ]] \
  || die "published terminal bundle has invalid hashes"

verify_arguments=(
  --operational-bundle "${operational_bundle}"
  --operational-bundle-file-sha256 \
    "${PRORM_R3_TERMINALIZE_OPERATIONAL_BUNDLE_SHA256}"
  --evidence-directory "${evidence_directory}"
)
if [[ "${mode}" == "gatep" ]]; then
  verify_mode=gatep
  verify_arguments+=(
    --allocation-intent "${allocation_intent}"
    --allocation-intent-file-sha256 \
      "${PRORM_R3_TERMINALIZE_ALLOCATION_INTENT_SHA256}"
    --runtime-receipt "${runtime_receipt}"
    --runtime-receipt-file-sha256 \
      "${PRORM_R3_TERMINALIZE_RUNTIME_RECEIPT_SHA256}"
  )
elif [[ "${PRORM_R3_TERMINALIZE_FINALIZER_COMMAND}" == \
  "primary-continuable-finalize" ]]; then
  verify_mode=primary-continuable
  verify_arguments+=(
    --runtime-closure "${runtime_closure}"
    --runtime-closure-file-sha256 \
      "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256}"
  )
else
  verify_mode=primary-completed
  verify_arguments+=(
    --runtime-closure "${runtime_closure}"
    --runtime-closure-file-sha256 \
      "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256}"
  )
fi
verified="$(
  "${container[@]}" python3 "${reuse_verifier}" \
    "${verify_mode}" "${verify_arguments[@]}"
)" || die "SIF terminal evidence revalidation failed"
mapfile -t verified_fields < <(
  printf '%s\n' "${verified}" \
    | "${host_python}" -I -S -c '
import json,sys
value=json.load(sys.stdin)
for key in ("status","manifest_file_sha256","raw_sacct_sha256","terminal_sha256"):
    print(value[key])
'
)
[[ "${#verified_fields[@]}" -eq 4 \
  && "${verified_fields[1]}" == "${manifest_sha256}" \
  && "${verified_fields[2]}" == "${raw_sha256}" \
  && "${verified_fields[3]}" =~ ^[0-9a-f]{64}$ ]] \
  || die "SIF terminal revalidation output is inconsistent"

printf 'R3_TERMINAL_MODE=%s\n' "${mode}"
printf 'R3_TERMINAL_JOB_SELECTOR=%s\n' \
  "${PRORM_R3_TERMINALIZE_JOB_SELECTOR}"
printf 'R3_TERMINAL_STATUS=%s\n' "${verified_fields[0]}"
printf 'R3_TERMINAL_EVIDENCE_DIRECTORY=%s\n' "${evidence_directory}"
printf 'R3_TERMINAL_MANIFEST_FILE_SHA256=%s\n' "${manifest_sha256}"
printf 'R3_TERMINAL_RAW_SACCT_SHA256=%s\n' "${raw_sha256}"
printf 'R3_TERMINAL_SHA256=%s\n' "${verified_fields[3]}"
printf 'R3_TERMINAL_REUSED=%s\n' "${reused}"
if [[ "${mode}" == "gatep" ]]; then
  printf 'R3_PROFILE_TERMINAL_EVIDENCE_DIRECTORY=%s\n' \
    "${evidence_directory}"
  printf 'R3_PROFILE_TERMINAL_MANIFEST_FILE_SHA256=%s\n' \
    "${manifest_sha256}"
  printf 'R3_PROFILE_TERMINAL_RAW_SACCT_SHA256=%s\n' "${raw_sha256}"
else
  task_id="${PRORM_R3_TERMINALIZE_TASK_ID}"
  printf 'R3_TASK_%s_RUNTIME_CLOSURE=%s\n' \
    "${task_id}" "${runtime_closure}"
  printf 'R3_TASK_%s_RUNTIME_CLOSURE_FILE_SHA256=%s\n' \
    "${task_id}" "${PRORM_R3_TERMINALIZE_RUNTIME_CLOSURE_SHA256}"
  printf 'R3_TASK_%s_TERMINAL_EVIDENCE_DIRECTORY=%s\n' \
    "${task_id}" "${evidence_directory}"
  printf 'R3_TASK_%s_TERMINAL_MANIFEST_FILE_SHA256=%s\n' \
    "${task_id}" "${manifest_sha256}"
  printf 'R3_TASK_%s_TERMINAL_RAW_SACCT_SHA256=%s\n' \
    "${task_id}" "${raw_sha256}"
fi
