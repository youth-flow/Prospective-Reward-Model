#!/usr/bin/env bash
# Terminalize all nine formal Gate-C tasks, aggregate, and bridge Gate R+C.

set -euo pipefail
umask 077

readonly SIF_PYTHON="/opt/conda/bin/python"
readonly SACCT_FORMAT="JobID%64,JobIDRaw%32,State%64,ExitCode%32,DerivedExitCode%32,Cluster%64,Account%64,Partition%64,QOS%64,NNodes%16,NCPUS%16,ReqTRES%512,AllocTRES%512,ElapsedRaw%32"

die() {
  printf 'R3 Gate-C formal finalizer fatal: %s\n' "$*" >&2
  exit 1
}

required=(
  PRORM_R3_GATEC_FINALIZE_REPO_ROOT
  PRORM_R3_GATEC_FINALIZE_PROJECT_ROOT
  PRORM_R3_GATEC_FINALIZE_IMAGE
  PRORM_R3_GATEC_FINALIZE_IMAGE_SHA256
  PRORM_R3_GATEC_FINALIZE_GIT_COMMIT
  PRORM_R3_GATEC_FINALIZE_CONTROLS_CONFIG
  PRORM_R3_GATEC_FINALIZE_PROFILE
  PRORM_R3_GATEC_FINALIZE_PROFILE_FILE_SHA256
  PRORM_R3_GATEC_FINALIZE_PLAN
  PRORM_R3_GATEC_FINALIZE_PLAN_FILE_SHA256
  PRORM_R3_GATEC_FINALIZE_PLAN_SHA256
  PRORM_R3_GATEC_FINALIZE_ARRAY_JOB_IDS
  PRORM_R3_GATEC_FINALIZE_GATE_R_AUTHORIZATION
  PRORM_R3_GATEC_FINALIZE_GATE_R_FILE_SHA256
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

repo_root="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_REPO_ROOT}")"
project_root="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_PROJECT_ROOT}")"
image="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_IMAGE}")"
controls_config="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_CONTROLS_CONFIG}")"
profile="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_PROFILE}")"
plan="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_PLAN}")"
gate_r="$(realpath -e -- "${PRORM_R3_GATEC_FINALIZE_GATE_R_AUTHORIZATION}")"
plan_sha256="${PRORM_R3_GATEC_FINALIZE_PLAN_SHA256}"
[[ "${repo_root}" == "/home/yyangjo/Smart-Reward-Model" ]] \
  || die "unexpected repository root"
[[ "${project_root}" == "/project/sigroup/smart-reward-model" ]] \
  || die "unexpected project root"
[[ "${gate_r}" == \
  "${project_root}/runs/phase2-recovery-r3/recovery-success-authorization.json" ]] \
  || die "Gate-R authorization is not at its fixed path"
[[ "$(git -C "${repo_root}" rev-parse HEAD)" == \
  "${PRORM_R3_GATEC_FINALIZE_GIT_COMMIT}" ]] || die "Git commit drift"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=all)" ]] \
  || die "Git checkout is dirty"
printf '%s  %s\n' "${PRORM_R3_GATEC_FINALIZE_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "container SHA-256 mismatch"
printf '%s  %s\n' "${PRORM_R3_GATEC_FINALIZE_GATE_R_FILE_SHA256}" "${gate_r}" \
  | sha256sum --check --status || die "Gate-R file SHA-256 mismatch"

IFS=: read -r -a array_job_ids <<< "${PRORM_R3_GATEC_FINALIZE_ARRAY_JOB_IDS}"
[[ "${#array_job_ids[@]}" -eq 3 ]] || die "exactly three array job IDs are required"
for job_id in "${array_job_ids[@]}"; do
  [[ "${job_id}" =~ ^[1-9][0-9]*$ ]] || die "invalid formal array job ID"
done

evidence_cli="${repo_root}/scripts/hpc4/phase2_r3_controls_evidence.py"
[[ -f "${evidence_cli}" && ! -L "${evidence_cli}" ]] \
  || die "Gate-C evidence CLI is unavailable"
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
seeds=(20260801 20260802 20260803)
submission_parent="$(
  realpath -e -- \
    "${project_root}/runs/phase2-recovery-r3-controls/submissions/${plan_sha256}"
)"
evidence_root="${submission_parent}/terminal-evidence"
if [[ -e "${evidence_root}" || -L "${evidence_root}" ]]; then
  [[ -d "${evidence_root}" && ! -L "${evidence_root}" ]] \
    || die "formal terminal-evidence root is unsafe"
else
  mkdir -- "${evidence_root}"
fi
evidence_root="$(realpath -e -- "${evidence_root}")"
raw_paths=()
raw_sha256s=()
raw_job_id_values=()

# Capture every scheduler outcome before opening any success-only result or
# closure.  Thus a failed task still leaves the complete external 3x3 raw set.
for family_index in 0 1 2; do
  family="${families[family_index]}"
  for seed_index in 0 1 2; do
    seed="${seeds[seed_index]}"
    task_id="$((family_index * 3 + seed_index))"
    task_evidence="${evidence_root}/task-${task_id}-${family}-seed-${seed}"
    if [[ -e "${task_evidence}" || -L "${task_evidence}" ]]; then
      [[ -d "${task_evidence}" && ! -L "${task_evidence}" ]] \
        || die "formal task evidence namespace is unsafe"
    else
      mkdir -- "${task_evidence}"
    fi
    task_evidence="$(realpath -e -- "${task_evidence}")"
    raw="${task_evidence}/formal-sacct.raw"
    if [[ -e "${raw}" || -L "${raw}" ]]; then
      [[ -f "${raw}" && ! -L "${raw}" ]] \
        || die "retained raw formal sacct is unsafe"
      raw_sha256="$(sha256sum -- "${raw}" | cut -d' ' -f1)"
    else
      temporary="$(mktemp --tmpdir="${task_evidence}" ".formal-sacct.XXXXXXXX")"
      capture_ok=0
      for _attempt in {1..20}; do
        if sacct \
          -X \
          -n \
          -P \
          -j "${array_job_ids[family_index]}_${seed_index}" \
          "--format=${SACCT_FORMAT}" \
          > "${temporary}" && [[ -s "${temporary}" ]]; then
          capture_ok=1
          break
        fi
        sleep 15
      done
      [[ "${capture_ok}" == "1" ]] \
        || die "formal sacct row never became available"
      raw_sha256="$(sha256sum -- "${temporary}" | cut -d' ' -f1)"
      run_in_sif publish-captured-sacct \
        --source "${temporary}" \
        --source-sha256 "${raw_sha256}" \
        --output "${raw}"
      rm -f -- "${temporary}"
    fi
    mapfile -t raw_job_ids < <(cut -d'|' -f2 -- "${raw}")
    [[ "${#raw_job_ids[@]}" -eq 1 && "${raw_job_ids[0]}" =~ ^[1-9][0-9]*$ ]] \
      || die "formal sacct capture does not contain one exact raw job ID"
    raw_paths[task_id]="${raw}"
    raw_sha256s[task_id]="${raw_sha256}"
    raw_job_id_values[task_id]="${raw_job_ids[0]}"
  done
done

# Only a complete successful matrix can enter terminal promotion and
# aggregation.  Any missing result/closure fails here after all raw rows exist.
aggregate_arguments=()
for family_index in 0 1 2; do
  family="${families[family_index]}"
  family_root="$(
    realpath -e -- "${submission_parent}/family-${family}"
  )"
  for seed_index in 0 1 2; do
    seed="${seeds[seed_index]}"
    task_id="$((family_index * 3 + seed_index))"
    task_root="$(
      realpath -e -- "${family_root}/task-${seed_index}-seed-${seed}"
    )"
    result="$(realpath -e -- "${task_root}/family-result.json")"
    closure="$(realpath -e -- "${task_root}/task-closure.json")"
    [[ -f "${result}" && ! -L "${result}" ]] \
      || die "formal result ${task_id} is not a regular file"
    [[ -f "${closure}" && ! -L "${closure}" ]] \
      || die "formal closure ${task_id} is not a regular file"
    result_sha256="$(sha256sum -- "${result}" | cut -d' ' -f1)"
    closure_sha256="$(sha256sum -- "${closure}" | cut -d' ' -f1)"
    terminal="${task_root}/task-terminal.json"
    [[ ! -e "${terminal}" && ! -L "${terminal}" ]] \
      || die "formal terminal ${task_id} already exists"
    run_in_sif terminal \
      --controls-config "${controls_config}" \
      --profile "${profile}" \
      --profile-file-sha256 "${PRORM_R3_GATEC_FINALIZE_PROFILE_FILE_SHA256}" \
      --plan "${plan}" \
      --plan-file-sha256 "${PRORM_R3_GATEC_FINALIZE_PLAN_FILE_SHA256}" \
      --result "${result}" \
      --result-file-sha256 "${result_sha256}" \
      --closure "${closure}" \
      --closure-file-sha256 "${closure_sha256}" \
      --raw-sacct "${raw_paths[task_id]}" \
      --raw-sacct-sha256 "${raw_sha256s[task_id]}" \
      --array-job-id "${array_job_ids[family_index]}" \
      --job-id-raw "${raw_job_id_values[task_id]}" \
      --output "${terminal}"
    terminal_sha256="$(sha256sum -- "${terminal}" | cut -d' ' -f1)"
    aggregate_arguments+=(
      --entry
      "${result}" "${result_sha256}"
      "${closure}" "${closure_sha256}"
      "${terminal}" "${terminal_sha256}"
      "${raw_paths[task_id]}" "${raw_sha256s[task_id]}"
    )
  done
done

aggregate="${project_root}/runs/phase2-recovery-r3-controls/gate-c-aggregate.json"
authorization="$(
  printf '%s' \
    "${project_root}/runs/phase2-recovery-r3-controls/gate-c-success-authorization.json"
)"
[[ ! -e "${aggregate}" && ! -L "${aggregate}" ]] \
  || die "fixed Gate-C aggregate already exists"
[[ ! -e "${authorization}" && ! -L "${authorization}" ]] \
  || die "fixed R3 final authorization already exists"
run_in_sif aggregate \
  --controls-config "${controls_config}" \
  --profile "${profile}" \
  --profile-file-sha256 "${PRORM_R3_GATEC_FINALIZE_PROFILE_FILE_SHA256}" \
  --plan "${plan}" \
  --plan-file-sha256 "${PRORM_R3_GATEC_FINALIZE_PLAN_FILE_SHA256}" \
  "${aggregate_arguments[@]}" \
  --output "${aggregate}"
aggregate_sha256="$(sha256sum -- "${aggregate}" | cut -d' ' -f1)"

run_in_sif authorize \
  --aggregate "${aggregate}" \
  --aggregate-file-sha256 "${aggregate_sha256}" \
  --gate-r-authorization "${gate_r}" \
  --gate-r-authorization-file-sha256 \
    "${PRORM_R3_GATEC_FINALIZE_GATE_R_FILE_SHA256}" \
  --project-root "${project_root}" \
  --output "${authorization}"
authorization_sha256="$(sha256sum -- "${authorization}" | cut -d' ' -f1)"

printf 'R3_GATEC_AGGREGATE=%s\n' "${aggregate}"
printf 'R3_GATEC_AGGREGATE_FILE_SHA256=%s\n' "${aggregate_sha256}"
printf 'R3_FINAL_AUTHORIZATION=%s\n' "${authorization}"
printf 'R3_FINAL_AUTHORIZATION_FILE_SHA256=%s\n' "${authorization_sha256}"
