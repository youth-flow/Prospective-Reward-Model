#!/usr/bin/env bash
# Terminalize three profile tasks and freeze the formal Gate-C profile/plan.

set -euo pipefail
umask 077

readonly SIF_PYTHON="/opt/conda/bin/python"
readonly SACCT_FORMAT="JobID%64,JobIDRaw%32,State%64,ExitCode%32,DerivedExitCode%32,Cluster%64,Account%64,Partition%64,QOS%64,NNodes%16,NCPUS%16,ReqTRES%512,AllocTRES%512,ElapsedRaw%32"

die() {
  printf 'R3 Gate-C profile finalizer fatal: %s\n' "$*" >&2
  exit 1
}

required=(
  PRORM_R3_GATEC_PROFILE_FINALIZE_REPO_ROOT
  PRORM_R3_GATEC_PROFILE_FINALIZE_PROJECT_ROOT
  PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE
  PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE_SHA256
  PRORM_R3_GATEC_PROFILE_FINALIZE_GIT_COMMIT
  PRORM_R3_GATEC_PROFILE_FINALIZE_CONTROLS_CONFIG
  PRORM_R3_GATEC_PROFILE_FINALIZE_ARRAY_JOB_ID
  PRORM_R3_GATEC_PROFILE_FINALIZE_SUBMISSION_ROOT
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done
[[ "${SLURM_JOB_ACCOUNT:-}" == "sigroup" ]] || die "unexpected Slurm account"
[[ "${SLURM_JOB_PARTITION:-}" == "gpu-l20" ]] \
  || die "unexpected Slurm partition"
[[ "${SLURM_NNODES:-}" == "1" && "${SLURM_NTASKS:-}" == "1" ]] \
  || die "finalizer requires one node and one task"
[[ "${SLURM_CPUS_PER_TASK:-}" == "8" ]] || die "unexpected CPU allocation"

repo_root="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_FINALIZE_REPO_ROOT}")"
project_root="$(
  realpath -e -- "${PRORM_R3_GATEC_PROFILE_FINALIZE_PROJECT_ROOT}"
)"
image="$(realpath -e -- "${PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE}")"
controls_config="$(
  realpath -e -- "${PRORM_R3_GATEC_PROFILE_FINALIZE_CONTROLS_CONFIG}"
)"
submission_root="$(
  realpath -e -- "${PRORM_R3_GATEC_PROFILE_FINALIZE_SUBMISSION_ROOT}"
)"
array_job_id="${PRORM_R3_GATEC_PROFILE_FINALIZE_ARRAY_JOB_ID}"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "${submission_root}" == \
  "${project_root}/runs/phase2-recovery-r3-controls/profile-attempts/"* ]] \
  || die "profile attempt is outside the durable namespace"
[[ "${array_job_id}" =~ ^[1-9][0-9]*$ ]] || die "invalid profile array job ID"
[[ "$(git -C "${repo_root}" rev-parse HEAD)" == \
  "${PRORM_R3_GATEC_PROFILE_FINALIZE_GIT_COMMIT}" ]] || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "Git checkout is dirty"
printf '%s  %s\n' \
  "${PRORM_R3_GATEC_PROFILE_FINALIZE_IMAGE_SHA256}" \
  "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"

evidence_cli="${repo_root}/scripts/hpc4/phase2_r3_controls_evidence.py"
[[ -f "${evidence_cli}" && ! -L "${evidence_cli}" ]] \
  || die "Gate-C evidence CLI is unavailable"
profile="${submission_root}/operational-profile.json"
plan="${submission_root}/execution-plan.json"
[[ ! -e "${profile}" && ! -L "${profile}" ]] \
  || die "operational profile already exists"
[[ ! -e "${plan}" && ! -L "${plan}" ]] || die "execution plan already exists"

run_in_sif() {
  apptainer exec \
    --cleanenv \
    --bind "${repo_root}:${repo_root}:ro" \
    --bind "${project_root}:${project_root}:rw" \
    --bind "${image}:${image}:ro" \
    --env "PYTHONPATH=${repo_root}/src" \
    --env "PYTHONNOUSERSITE=1" \
    --env "PYTHONDONTWRITEBYTECODE=1" \
    "${image}" \
    "${SIF_PYTHON}" "${evidence_cli}" "$@"
}

families=(
  exact_margin_prorm_plus
  exact_soft_label_bt
  low_dimensional_prorm_plus
)
evidence_root="${submission_root}/terminal-evidence"
if [[ -e "${evidence_root}" || -L "${evidence_root}" ]]; then
  [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
    || die "profile terminal-evidence root is unsafe"
else
  mkdir -- "${evidence_root}"
fi
evidence_root="$(realpath -e -- "${evidence_root}")"
raw_paths=()
raw_sha256s=()
raw_job_id_values=()

# Failure evidence comes first.  This loop has no dependency on any compute
# receipt, so every terminal array row is durably captured even when science
# setup or training failed.
for task_id in 0 1 2; do
  family="${families[task_id]}"
  task_evidence="${evidence_root}/task-${task_id}-${family}"
  if [[ -e "${task_evidence}" || -L "${task_evidence}" ]]; then
    [[ -d "${task_evidence}" && ! -L "${task_evidence}" ]] \
      || die "profile task evidence namespace is unsafe"
  else
    mkdir -- "${task_evidence}"
  fi
  task_evidence="$(realpath -e -- "${task_evidence}")"
  raw="${task_evidence}/profile-sacct.raw"
  if [[ -e "${raw}" || -L "${raw}" ]]; then
    [[ -f "${raw}" && ! -L "${raw}" ]] \
      || die "retained raw profile sacct is unsafe"
    raw_sha256="$(sha256sum -- "${raw}" | cut -d' ' -f1)"
  else
    temporary="$(mktemp --tmpdir="${task_evidence}" ".profile-sacct.XXXXXXXX")"
    capture_ok=0
    for _attempt in {1..20}; do
      if sacct \
        -X \
        -n \
        -P \
        -j "${array_job_id}_${task_id}" \
        "--format=${SACCT_FORMAT}" \
        > "${temporary}" && [[ -s "${temporary}" ]]; then
        capture_ok=1
        break
      fi
      sleep 15
    done
    [[ "${capture_ok}" == "1" ]] \
      || die "profile sacct row never became available"
    raw_sha256="$(sha256sum -- "${temporary}" | cut -d' ' -f1)"
    run_in_sif publish-captured-sacct \
      --source "${temporary}" \
      --source-sha256 "${raw_sha256}" \
      --output "${raw}"
    rm -f -- "${temporary}"
  fi
  mapfile -t raw_job_ids < <(cut -d'|' -f2 -- "${raw}")
  [[ "${#raw_job_ids[@]}" -eq 1 && "${raw_job_ids[0]}" =~ ^[1-9][0-9]*$ ]] \
    || die "profile sacct capture does not contain one exact raw job ID"
  raw_paths[task_id]="${raw}"
  raw_sha256s[task_id]="${raw_sha256}"
  raw_job_id_values[task_id]="${raw_job_ids[0]}"
done

# Success promotion starts only after the complete three-row raw terminal set
# exists.  A missing receipt now fails without erasing or preventing evidence.
measurement_arguments=()
for task_id in 0 1 2; do
  family="${families[task_id]}"
  task_root="$(
    realpath -e -- "${submission_root}/task-${task_id}-${family}"
  )"
  receipt="$(realpath -e -- "${task_root}/profile-compute-receipt.json")"
  [[ -f "${receipt}" && ! -L "${receipt}" ]] \
    || die "profile compute receipt is not a regular file"
  receipt_sha256="$(sha256sum -- "${receipt}" | cut -d' ' -f1)"
  measurement="${task_root}/profile-measurement.json"
  [[ ! -e "${measurement}" && ! -L "${measurement}" ]] \
    || die "profile measurement already exists"
  run_in_sif profile-measurement-finalize \
    --compute-receipt "${receipt}" \
    --compute-receipt-file-sha256 "${receipt_sha256}" \
    --raw-sacct "${raw_paths[task_id]}" \
    --raw-sacct-sha256 "${raw_sha256s[task_id]}" \
    --family "${family}" \
    --array-job-id "${array_job_id}" \
    --job-id-raw "${raw_job_id_values[task_id]}" \
    --output "${measurement}"
  measurement_sha256="$(sha256sum -- "${measurement}" | cut -d' ' -f1)"
  measurement_arguments+=(--measurement "${measurement}" "${measurement_sha256}")
done

run_in_sif profile-finalize \
  --controls-config "${controls_config}" \
  "${measurement_arguments[@]}" \
  --optimizer-schedule-sha256 \
    46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216 \
  --checkpoint-cadence-updates 200 \
  --walltime-safety-margin-fraction 0.25 \
  --fixed-walltime-margin-seconds 1800 \
  --memory-safety-margin-fraction 0.25 \
  --cluster hpc4 \
  --account sigroup \
  --partition gpu-l20 \
  --gpu-name "NVIDIA L20" \
  --cpus-per-task 8 \
  --memory-bytes 103079215104 \
  --array-concurrency 1 \
  --walltime-seconds 86400 \
  --signal-lead-seconds 1800 \
  --max-scheduler-segments 1 \
  --output "${profile}"
profile_file_sha256="$(sha256sum -- "${profile}" | cut -d' ' -f1)"

run_in_sif plan \
  --controls-config "${controls_config}" \
  --profile "${profile}" \
  --profile-file-sha256 "${profile_file_sha256}" \
  --output "${plan}"
plan_file_sha256="$(sha256sum -- "${plan}" | cut -d' ' -f1)"

printf 'R3_GATEC_PROFILE=%s\n' "${profile}"
printf 'R3_GATEC_PROFILE_FILE_SHA256=%s\n' "${profile_file_sha256}"
printf 'R3_GATEC_PLAN=%s\n' "${plan}"
printf 'R3_GATEC_PLAN_FILE_SHA256=%s\n' "${plan_file_sha256}"
