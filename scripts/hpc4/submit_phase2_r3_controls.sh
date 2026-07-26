#!/usr/bin/env bash
# Submit exactly one Gate-C family array. Resources come only from the profile.

set -euo pipefail
umask 077

readonly HOST_PYTHON="/opt/shared/spack/local/linux-rocky9-x86_64_v4/gcc-11.4.1/miniconda3-24.3.0-quc3pyudmzikgo2r4qsyqpwnrvzpin63/bin/python3.12"
readonly HOST_PYTHON_SHA256="9c91f9aa231c61c6bf2eabb9b93ebc5a8269a4126a36125e9548d8853e32da9c"

die() {
  printf 'R3 Gate-C submit fatal: %s\n' "$*" >&2
  exit 1
}

[[ "$#" -eq 5 ]] || die \
  "usage: $0 FAMILY_INDEX PROFILE PROFILE_FILE_SHA256 PLAN PLAN_FILE_SHA256"
family_index="$1"
profile_requested="$2"
profile_file_sha256="$3"
plan_requested="$4"
plan_file_sha256="$5"
[[ "${family_index}" =~ ^[0-2]$ ]] || die "FAMILY_INDEX must be 0, 1, or 2"
[[ "${profile_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid profile file SHA-256"
[[ "${plan_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid plan file SHA-256"

repo_root="$(realpath -e -- "${PRORM_R3_REPO_ROOT:-/home/yyangjo/Smart-Reward-Model}")"
project_root="$(
  realpath -e -- \
    "${PRORM_R3_PROJECT_ROOT:-/project/sigroup/smart-reward-model}"
)"
image="$(realpath -e -- "${PRORM_R3_IMAGE:?missing PRORM_R3_IMAGE}")"
profile="$(realpath -e -- "${profile_requested}")"
plan="$(realpath -e -- "${plan_requested}")"
controls_config="$(
  realpath -e -- "${repo_root}/configs/phase2_recovery_r3_controls.yaml"
)"
plan_inspector="$(
  realpath -e -- \
    "${repo_root}/scripts/hpc4/inspect_phase2_r3_controls_plan_stdlib.py"
)"
science_runner="${repo_root}/scripts/hpc4/run_phase2_r3_control_family.py"
sbatch_script="${repo_root}/scripts/hpc4/phase2_r3_controls.sbatch"

[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected production repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected persistent project root"
[[ -f "${plan_inspector}" && ! -L "${plan_inspector}" && -f "${sbatch_script}" ]] \
  || die "Gate-C committed execution surface is incomplete"
# The scientific producer is deliberately a separate fixed API. Until it is
# implemented and committed, submission fails here without consuming a GPU.
[[ -f "${science_runner}" ]] \
  || die "missing committed Gate-C family science runner; no job was submitted"
[[ -f "${image}" && ! -L "${image}" && "${image}" == "${project_root}/"* ]] \
  || die "container must be a regular file under the persistent project root"
[[ "${profile}" == "${project_root}/"* && "${plan}" == "${project_root}/"* ]] \
  || die "profile and plan must be durable project artifacts"

host_python="${HOST_PYTHON}"
host_python_target="$(realpath -e -- "${host_python}")"
[[ -f "${host_python}" && -x "${host_python}" && ! -L "${host_python}" ]] \
  || die "fixed host Python launcher is not a regular non-symlink executable"
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
[[ "${image_sha256}" =~ ^[0-9a-f]{64}$ ]] || die "invalid container SHA-256"

input_parent="${project_root}/runs/phase2-recovery-r3/inputs"
mkdir -p -- "${input_parent}"
input_parent="$(realpath -e -- "${input_parent}")"
[[ "${input_parent}" == "${project_root}/runs/phase2-recovery-r3/inputs" ]] \
  || die "Gate-C input parent is not canonical"
input_root="${input_parent}/${commit}"
mkdir -p -- "${input_root}"
input_root="$(realpath -e -- "${input_root}")"
[[ "${input_root}" == "${input_parent}/${commit}" && ! -L "${input_root}" ]] \
  || die "Gate-C clean-commit input namespace is invalid"

copy_no_overwrite_exact() {
  local source="$1"
  local destination="$2"
  if [[ -e "${destination}" || -L "${destination}" ]]; then
    [[ -f "${destination}" && ! -L "${destination}" ]] \
      || die "retained Gate-C input is not a regular file"
  else
    cp --no-clobber -- "${source}" "${destination}"
  fi
  cmp --silent -- "${source}" "${destination}" \
    || die "retained Gate-C input differs from the clean commit"
}
copy_no_overwrite_exact \
  "${repo_root}/configs/phase2_recovery_r3_science.yaml" \
  "${input_root}/phase2_recovery_r3_science.yaml"
copy_no_overwrite_exact \
  "${repo_root}/configs/common_beta_pilot_base.yaml" \
  "${input_root}/common_beta_pilot_base.yaml"
copy_no_overwrite_exact \
  "${repo_root}/configs/phase2_recovery_parent_failures.json" \
  "${input_root}/phase2_recovery_parent_failures.json"
parent_registry_file_sha256="$(
  sha256sum -- "${input_root}/phase2_recovery_parent_failures.json" \
    | cut -d' ' -f1
)"
[[ "${parent_registry_file_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid retained parent-registry SHA-256"

inspect_json="$(
  "${host_python}" -I -S "${plan_inspector}" \
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
r=v["resources"]
a=v["arrays"]
for item in (
    v["plan_sha256"],
    v["git_commit"], v["container_sha256"],
    v["controls_config_file_sha256"], v["controls_config_semantic_sha256"],
    r["cluster"], r["account"], r["partition"], r["gpu_name"],
    r["cpus_per_task"], r["memory_bytes"], r["array_concurrency"],
    r["requested_walltime_seconds"], r["signal_lead_seconds"],
    r["checkpoint_cadence_updates"], r["max_scheduler_segments"],
    a[int(sys.argv[1])]["family"], a[int(sys.argv[1])]["array_task_range"],
):
    print(item)
' "${family_index}"
)
[[ "${#locked[@]}" -eq 18 ]] || die "plan inspection returned an invalid schema"
plan_sha256="${locked[0]}"
plan_commit="${locked[1]}"
plan_image_sha256="${locked[2]}"
plan_config_file_sha256="${locked[3]}"
plan_config_semantic_sha256="${locked[4]}"
cluster="${locked[5]}"
account="${locked[6]}"
partition="${locked[7]}"
gpu_name="${locked[8]}"
cpus_per_task="${locked[9]}"
memory_bytes="${locked[10]}"
array_concurrency="${locked[11]}"
walltime_seconds="${locked[12]}"
signal_lead_seconds="${locked[13]}"
checkpoint_cadence_updates="${locked[14]}"
max_scheduler_segments="${locked[15]}"
family="${locked[16]}"
array_spec="${locked[17]}"

[[ "${cluster}" == "hpc4" && "${account}" == "sigroup" ]] \
  || die "plan does not target the locked HPC4 account"
[[ "${commit}" == "${plan_commit}" ]] || die "Git commit differs from the plan"
[[ "${image_sha256}" == "${plan_image_sha256}" ]] \
  || die "container SHA-256 mismatch"
[[ "$(sha256sum -- "${controls_config}" | cut -d' ' -f1)" == \
  "${plan_config_file_sha256}" ]] || die "controls config file SHA-256 mismatch"
[[ "${plan_config_semantic_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "invalid controls config semantic SHA-256"
[[ "${array_concurrency}" == "1" && "${array_spec}" == "0-2%1" ]] \
  || die "Gate-C must use one rolling family array at concurrency one"
[[ "${max_scheduler_segments}" == "1" ]] \
  || die "multi-segment Gate-C awaits a committed state-continuation runner"
(( walltime_seconds > signal_lead_seconds && walltime_seconds <= 172800 )) \
  || die "profile walltime/signal exceeds the two-day Gate-C envelope"

# ``squeue -r`` expands arrays. Three new tasks may be submitted only while at
# most one existing task remains, preserving MaxSubmitJobsPU=4. Concurrency one
# consumes at most one additional running slot, preserving MaxJobsPU=2.
submitted_count="$(squeue -r -h -u "${USER}" -o '%i' | sed '/^[[:space:]]*$/d' | wc -l)"
running_count="$(
  squeue -r -h -u "${USER}" -t RUNNING,COMPLETING -o '%i' \
    | sed '/^[[:space:]]*$/d' | wc -l
)"
(( submitted_count <= 1 )) \
  || die "rolling admission would exceed HPC4 MaxSubmitJobsPU=4"
(( running_count <= 1 )) \
  || die "rolling admission would exceed HPC4 MaxJobsPU=2"

memory_mib="$((memory_bytes / 1024 / 1024))"
(( memory_mib * 1024 * 1024 == memory_bytes )) \
  || die "profile memory bytes are not an exact MiB request"
days="$((walltime_seconds / 86400))"
hours="$(((walltime_seconds % 86400) / 3600))"
minutes="$(((walltime_seconds % 3600) / 60))"
seconds="$((walltime_seconds % 60))"
printf -v slurm_walltime '%d-%02d:%02d:%02d' \
  "${days}" "${hours}" "${minutes}" "${seconds}"

submission_parent="${project_root}/runs/phase2-recovery-r3-controls/submissions"
mkdir -p -- "${submission_parent}"
submission_root="${submission_parent}/${plan_sha256}/family-${family}"
[[ ! -e "${submission_root}" && ! -L "${submission_root}" ]] \
  || die "family submission namespace already exists; refusing replacement"
mkdir -p -- "${submission_root}/logs"
submission_root="$(realpath -e -- "${submission_root}")"

export_spec="PATH=/usr/bin:/bin"
for item in \
  "PRORM_R3_GATEC_REPO_ROOT=${repo_root}" \
  "PRORM_R3_GATEC_PROJECT_ROOT=${project_root}" \
  "PRORM_R3_GATEC_IMAGE=${image}" \
  "PRORM_R3_GATEC_IMAGE_SHA256=${image_sha256}" \
  "PRORM_R3_GATEC_GIT_COMMIT=${commit}" \
  "PRORM_R3_GATEC_CONTROLS_CONFIG=${controls_config}" \
  "PRORM_R3_GATEC_PROFILE=${profile}" \
  "PRORM_R3_GATEC_PROFILE_FILE_SHA256=${profile_file_sha256}" \
  "PRORM_R3_GATEC_PLAN=${plan}" \
  "PRORM_R3_GATEC_PLAN_FILE_SHA256=${plan_file_sha256}" \
  "PRORM_R3_GATEC_PLAN_SHA256=${plan_sha256}" \
  "PRORM_R3_GATEC_FAMILY_INDEX=${family_index}" \
  "PRORM_R3_GATEC_FAMILY=${family}" \
  "PRORM_R3_GATEC_PARENT_REGISTRY_FILE_SHA256=${parent_registry_file_sha256}" \
  "PRORM_R3_GATEC_SUBMISSION_ROOT=${submission_root}" \
  "PRORM_R3_GATEC_CHECKPOINT_CADENCE_UPDATES=${checkpoint_cadence_updates}"; do
  [[ "${item}" != *","* ]] || die "export value contains a comma"
  export_spec+=",${item}"
done

job_id="$(
  sbatch --parsable \
    --account="${account}" \
    --partition="${partition}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-node=1 \
    --cpus-per-task="${cpus_per_task}" \
    --mem="${memory_mib}M" \
    --time="${slurm_walltime}" \
    --signal="B:USR1@${signal_lead_seconds}" \
    --no-requeue \
    --array="${array_spec}" \
    --job-name="prorm-r3-gatec-${family}" \
    --output="${submission_root}/logs/%A_%a.out" \
    --error="${submission_root}/logs/%A_%a.err" \
    --export="${export_spec}" \
    "${sbatch_script}"
)"
[[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "sbatch returned an invalid job ID"
printf 'R3_GATEC_ARRAY_JOB_ID=%s\n' "${job_id}"
printf 'R3_GATEC_FAMILY=%s\n' "${family}"
printf 'R3_GATEC_SUBMISSION_ROOT=%s\n' "${submission_root}"
