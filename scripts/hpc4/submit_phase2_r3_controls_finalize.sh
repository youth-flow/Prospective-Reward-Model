#!/usr/bin/env bash
# Submit the fixed SIF finalizer for the complete external 3x3 Gate-C matrix.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 Gate-C formal-finalizer submit fatal: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 10 ]] || die \
  "usage: $0 PROFILE PROFILE_SHA PLAN PLAN_SHA ARRAY0 ARRAY1 ARRAY2 GATE_R GATE_R_SHA OUTPUT_LOG_DIR"
profile_requested="$1"
profile_file_sha256="$2"
plan_requested="$3"
plan_file_sha256="$4"
array0="$5"
array1="$6"
array2="$7"
gate_r_requested="$8"
gate_r_file_sha256="$9"
output_log_requested="${10}"
for digest in "${profile_file_sha256}" "${plan_file_sha256}" "${gate_r_file_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid input SHA-256"
done
for job_id in "${array0}" "${array1}" "${array2}"; do
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "invalid formal array job ID"
done

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT:-/home/yyangjo/Smart-Reward-Model}")"
project_root="$(
  realpath -e -- \
    "${PRORM_R3_PROJECT_ROOT:-/project/sigroup/smart-reward-model}"
)"
image="$(realpath -e -- "${PRORM_R3_IMAGE:?missing PRORM_R3_IMAGE}")"
profile="$(realpath -e -- "${profile_requested}")"
plan="$(realpath -e -- "${plan_requested}")"
gate_r="$(realpath -e -- "${gate_r_requested}")"
output_log_dir="$(realpath -e -- "${output_log_requested}")"
controls_config="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_r3_controls.yaml"
)"
inspector="$(
  realpath -e -- \
    "${repo_root}/scripts/hpc4/inspect_phase2_r3_controls_plan_stdlib.py"
)"
sbatch_script="$(
  realpath -e -- "${repo_root}/scripts/hpc4/phase2_r3_controls_finalize.sbatch"
)"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "${profile}" == "${project_root}/"* && "${plan}" == "${project_root}/"* ]] \
  || die "profile and plan must be durable project artifacts"
[[ "${gate_r}" == \
  "${project_root}/runs/phase2-recovery-r3/recovery-success-authorization.json" ]] \
  || die "Gate-R authorization is not at its fixed path"
[[ "${output_log_dir}" == \
  "${project_root}/runs/phase2-recovery-r3-controls/"* ]] \
  || die "finalizer logs must remain below the Gate-C project namespace"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "container must be a regular project file"

host_python="${HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ -f "${host_python}" && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python is not a regular non-symlink executable"
[[ "${host_python_target}" == "${host_python}" ]] \
  || die "fixed host Python path is not canonical"
printf '%s  %s\n' "${HOST_PYTHON_SHA256}" "${host_python}" \
  | sha256sum --check --status || die "fixed host Python SHA-256 mismatch"
[[ "$("${host_python}" --version 2>&1)" == "Python 3.12.2" ]] \
  || die "fixed host Python version mismatch"

commit="$(git -C "${repo_root}" rev-parse HEAD)"
[[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git commit"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "production checkout must be clean"
image_sha256="$(sha256sum -- "${image}" | cut -d' ' -f1)"
printf '%s  %s\n' "${gate_r_file_sha256}" "${gate_r}" \
  | sha256sum --check --status || die "Gate-R file SHA-256 mismatch"

inspect_json="$(
  "${host_python}" -I -S "${inspector}" \
    --controls-config "${controls_config}" \
    --profile "${profile}" \
    --profile-file-sha256 "${profile_file_sha256}" \
    --plan "${plan}" \
    --plan-file-sha256 "${plan_file_sha256}"
)"
mapfile -t locked < <(
  printf '%s\n' "${inspect_json}" | "${host_python}" -I -S -c '
import json,sys
v=json.load(sys.stdin)
for item in (v["plan_sha256"],v["git_commit"],v["container_sha256"]):
    print(item)
'
)
[[ "${#locked[@]}" -eq 3 ]] || die "plan inspection returned invalid output"
plan_sha256="${locked[0]}"
[[ "${locked[1]}" == "${commit}" ]] || die "plan Git commit differs"
[[ "${locked[2]}" == "${image_sha256}" ]] || die "plan container differs"
submission_parent="$(
  realpath -e -- \
    "${project_root}/runs/phase2-recovery-r3-controls/submissions/${plan_sha256}"
)"
[[ -d "${submission_parent}" && ! -L "${submission_parent}" ]] \
  || die "formal submission namespace is unavailable"
[[ ! -e "${project_root}/runs/phase2-recovery-r3-controls/gate-c-aggregate.json" ]] \
  || die "fixed Gate-C aggregate already exists"
[[ ! -e \
  "${project_root}/runs/phase2-recovery-r3-controls/gate-c-success-authorization.json" ]] \
  || die "fixed final authorization already exists"

submitted_count="$(squeue -r -h -u "${USER}" -o '%i' | sed '/^[[:space:]]*$/d' | wc -l)"
(( submitted_count <= 3 )) \
  || die "formal finalizer would exceed HPC4 MaxSubmitJobsPU=4"

export_spec="PATH=/usr/bin:/bin"
for item in \
  "PRORM_R3_GATEC_FINALIZE_REPO_ROOT=${repo_root}" \
  "PRORM_R3_GATEC_FINALIZE_PROJECT_ROOT=${project_root}" \
  "PRORM_R3_GATEC_FINALIZE_IMAGE=${image}" \
  "PRORM_R3_GATEC_FINALIZE_IMAGE_SHA256=${image_sha256}" \
  "PRORM_R3_GATEC_FINALIZE_GIT_COMMIT=${commit}" \
  "PRORM_R3_GATEC_FINALIZE_CONTROLS_CONFIG=${controls_config}" \
  "PRORM_R3_GATEC_FINALIZE_PROFILE=${profile}" \
  "PRORM_R3_GATEC_FINALIZE_PROFILE_FILE_SHA256=${profile_file_sha256}" \
  "PRORM_R3_GATEC_FINALIZE_PLAN=${plan}" \
  "PRORM_R3_GATEC_FINALIZE_PLAN_FILE_SHA256=${plan_file_sha256}" \
  "PRORM_R3_GATEC_FINALIZE_PLAN_SHA256=${plan_sha256}" \
  "PRORM_R3_GATEC_FINALIZE_ARRAY_JOB_IDS=${array0}:${array1}:${array2}" \
  "PRORM_R3_GATEC_FINALIZE_GATE_R_AUTHORIZATION=${gate_r}" \
  "PRORM_R3_GATEC_FINALIZE_GATE_R_FILE_SHA256=${gate_r_file_sha256}"; do
  [[ "${item}" != *","* ]] || die "export value contains a comma"
  export_spec+=",${item}"
done

job_id="$(
  sbatch --parsable \
    --dependency="afterany:${array0}:${array1}:${array2}" \
    --account=sigroup \
    --partition=gpu-l20 \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=0-01:00:00 \
    --no-requeue \
    --job-name=prorm-r3-gatec-finalize \
    --output="${output_log_dir}/gatec-finalize-%j.out" \
    --error="${output_log_dir}/gatec-finalize-%j.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf 'R3_GATEC_FINALIZER_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATEC_AGGREGATE=%s\n' \
  "${project_root}/runs/phase2-recovery-r3-controls/gate-c-aggregate.json"
printf 'R3_FINAL_AUTHORIZATION=%s\n' \
  "${project_root}/runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
