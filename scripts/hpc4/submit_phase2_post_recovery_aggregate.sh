#!/usr/bin/env bash
set -euo pipefail
die() { echo "error: $*" >&2; exit 2; }

if [[ $# -lt 10 ]]; then
  die "usage: $0 <authorization.json> <terminal.json> <array-job-id> <output.json> <cpu-partition> <walltime> <producer-commit> <run-0> <run-1> <run-2> [--overlay <yaml>] [--beta-source-aggregate <json>] [--horizon-parent-aggregate <json>]"
fi
authorization_input="$1"
terminal_input="$2"
array_job_id="$3"
output_input="$4"
partition="$5"
walltime="$6"
producer_commit="$7"
run_inputs=("$8" "$9" "${10}")
shift 10
overlay_input=""
beta_source_input=""
horizon_parent_input=""
while (( $# )); do
  case "$1" in
    --overlay)
      [[ $# -ge 2 && -z "${overlay_input}" ]] || die "--overlay requires one path"
      overlay_input="$2"
      shift 2
      ;;
    --beta-source-aggregate)
      [[ $# -ge 2 && -z "${beta_source_input}" ]] \
        || die "--beta-source-aggregate requires one path"
      beta_source_input="$2"
      shift 2
      ;;
    --horizon-parent-aggregate)
      [[ $# -ge 2 && -z "${horizon_parent_input}" ]] \
        || die "--horizon-parent-aggregate requires one path"
      horizon_parent_input="$2"
      shift 2
      ;;
    *) die "unknown post-recovery aggregate option: $1" ;;
  esac
done
[[ "${array_job_id}" =~ ^[1-9][0-9]*$ ]] || die "array job ID must be positive"
case "${partition}" in amd|intel) ;; *) die "aggregate partition must be amd or intel" ;; esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"
[[ "${producer_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "producer commit is invalid"
while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*|SBATCH_*)
      die "unset exported ${variable}; aggregate submission forbids ambient controls"
      ;;
  esac
done < <(compgen -e)
for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
for command_name in git python3 realpath sha256sum awk grep sbatch scontrol squeue sacct; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
aggregator_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "aggregate submission requires a clean committed worktree"
[[ "$(git -C "${repo_root}" rev-parse --verify "${producer_commit}^{commit}")" = \
  "${producer_commit}" ]] || die "producer commit is not an exact commit"
git -C "${repo_root}" merge-base --is-ancestor \
  "${producer_commit}" "${aggregator_commit}" \
  || die "producer commit must be an ancestor of aggregate commit"

if [[ -z "${overlay_input}" ]]; then
  overlay_relative="configs/common_beta_post_recovery_calibration.yaml"
  overlay="${repo_root}/${overlay_relative}"
else
  overlay="$(realpath -e -- "${overlay_input}")" \
    || die "post-recovery overlay cannot be resolved"
  [[ -f "${overlay}" && ! -L "${overlay}" ]] || die "post-recovery overlay is unsafe"
  case "${overlay}" in
    "${repo_root}"/configs/*.yaml) ;;
    *) die "post-recovery overlay must be a tracked configs/*.yaml file" ;;
  esac
  overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"
fi
base="$(
  PYTHONPATH="${repo_root}/src" python3 - "${overlay}" <<'PY'
import sys
from smart_reward.phase2_config import load_phase2_config_bundle
print(load_phase2_config_bundle(sys.argv[1]).base_config_path)
PY
)" || die "could not resolve aggregate overlay base config"
base="$(realpath -e -- "${base}")" || die "aggregate base config cannot be resolved"
case "${base}" in "${repo_root}"/configs/*.yaml) ;; *)
  die "aggregate base config must be a tracked configs/*.yaml file"
  ;;
esac
base_relative="$(realpath --relative-to="${repo_root}" "${base}")"
for relative in \
  "${overlay_relative}" "${base_relative}" \
  "src/smart_reward/phase2_post_recovery_aggregate.py" \
  "src/smart_reward/phase2_post_recovery_control.py" \
  "scripts/hpc4/submit_phase2_post_recovery_array_once.py" \
  "scripts/hpc4/submit_phase2_post_recovery_aggregate_attempt.py" \
  "scripts/hpc4/validate_phase2_post_recovery_submission.py" \
  "scripts/hpc4/validate_phase2_post_recovery_aggregate_submission.py" \
  "scripts/hpc4/capture_phase2_post_recovery_aggregate_terminal.py" \
  "scripts/hpc4/run_phase2_post_recovery_aggregate.py" \
  "scripts/hpc4/phase2_post_recovery_aggregate.sbatch"; do
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null \
    || die "required aggregate source is not tracked: ${relative}"
done
for path in "${overlay}" "${base}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || die "aggregate config is missing or unsafe"
done
overlay_sha256="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_sha256="$(sha256sum -- "${base}" | awk '{print $1}')"
for binding in \
  "${overlay_relative}:${overlay_sha256}" "${base_relative}:${base_sha256}"; do
  relative="${binding%%:*}"
  expected="${binding#*:}"
  for commit in "${producer_commit}" "${aggregator_commit}"; do
    observed="$(
      git -C "${repo_root}" cat-file blob "${commit}:${relative}" \
        | sha256sum | awk '{print $1}'
    )"
    [[ "${observed}" = "${expected}" ]] \
      || die "producer/aggregator config bytes differ: ${relative}"
  done
done

canonical_root() {
  local name="$1" raw="${!1}" resolved
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${raw}" && "${resolved}" != "/" \
    && "${resolved}" = "${raw}" ]] || die "${name} must be canonical"
  printf '%s\n' "${resolved}"
}
project_path() {
  local raw="$1" kind="$2" candidate resolved
  if [[ "${raw}" = /* ]]; then candidate="${raw}"; else candidate="${project_root}/${raw}"; fi
  resolved="$(realpath -e -- "${candidate}")" || die "project input cannot be resolved"
  case "${resolved}" in "${project_root}"/*) ;; *) die "project input escaped root" ;; esac
  case "${kind}" in
    file) [[ -f "${resolved}" && ! -L "${candidate}" ]] || die "project file is unsafe" ;;
    directory) [[ -d "${resolved}" && ! -L "${candidate}" ]] || die "project directory is unsafe" ;;
  esac
  printf '%s\n' "${resolved}"
}
reject_delimiters() {
  [[ "$2" != *","* && "$2" != *":"* && "$2" != *"="* \
    && "$2" != *$'\n'* && "$2" != *$'\r'* ]] || die "$1 contains an unsafe delimiter"
}
project_root="$(canonical_root PRORM_PROJECT_ROOT)"
scratch_root="$(canonical_root PRORM_SCRATCH_ROOT)"
case "${project_root}" in "${scratch_root}"|"${scratch_root}"/*) die "roots overlap" ;; esac
case "${scratch_root}" in "${project_root}"|"${project_root}"/*) die "roots overlap" ;; esac
image="$(project_path "${PRORM_IMAGE}" file)"
hf_cache="$(project_path "${PRORM_HF_CACHE}" directory)"
authorization="$(project_path "${authorization_input}" file)"
terminal="$(project_path "${terminal_input}" file)"
[[ "${authorization}" = \
  "${project_root}/runs/phase2-recovery-pilot/recovery-success-authorization.json" ]] \
  || die "authorization must be the locked recovery campaign receipt"
terminal_raw="$(
  realpath -e -- "$(dirname "${terminal}")/$(basename "${terminal}" .json).sacct.psv"
)" || die "raw calibration sacct evidence is missing"
[[ -f "${terminal_raw}" && ! -L "${terminal_raw}" ]] || die "raw sacct evidence is unsafe"
authorization_sha256="$(sha256sum -- "${authorization}" | awk '{print $1}')"
terminal_sha256="$(sha256sum -- "${terminal}" | awk '{print $1}')"
beta_source=""
beta_source_sha256=""
if [[ -n "${beta_source_input}" ]]; then
  beta_source="$(project_path "${beta_source_input}" file)"
  [[ "$(dirname "${beta_source}")" = "${project_root}/aggregates" ]] \
    || die "beta source must use the production aggregates directory"
  beta_source_sha256="$(sha256sum -- "${beta_source}" | awk '{print $1}')"
fi
horizon_parent=""
horizon_parent_sha256=""
if [[ -n "${horizon_parent_input}" ]]; then
  horizon_parent="$(project_path "${horizon_parent_input}" file)"
  [[ "$(dirname "${horizon_parent}")" = "${project_root}/aggregates" ]] \
    || die "horizon parent must use the production aggregates directory"
  horizon_parent_sha256="$(sha256sum -- "${horizon_parent}" | awk '{print $1}')"
fi

binding_json="$(
  PYTHONPATH="${repo_root}/src" python3 \
    "${repo_root}/scripts/hpc4/validate_phase2_recovery_authorization.py" \
    "${authorization}" "${overlay}" --expected-sha256 "${authorization_sha256}"
)" || die "authorization/config binding failed"
mapfile -t identities < <(
  PYTHONPATH="${repo_root}/src" python3 - \
    "${binding_json}" "${overlay}" \
    "${beta_source:-none}" "${horizon_parent:-none}" \
    "${overlay_relative}" <<'PY'
import json,re,sys
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    verify_post_recovery_aggregate_success_receipt,
)
value=json.loads(sys.argv[1])
for key in ("phase2_design_sha256","base_config_hash","optimizer_schedule_sha256"):
 item=value.get(key)
 if not isinstance(item,str) or re.fullmatch(r"[0-9a-f]{64}",item) is None:
  raise SystemExit(f"invalid identity {key}")
 print(item)
bundle=load_phase2_config_bundle(sys.argv[2])
config=bundle.config
phase=config["design"]["pilot_phase"]
for path in (sys.argv[3], sys.argv[4]):
 if path!="none":
  verify_post_recovery_aggregate_success_receipt(path)
beta=verify_beta_source_aggregate(
 config, None if sys.argv[3]=="none" else sys.argv[3]
)
horizon=verify_horizon_parent_aggregate(
 config, None if sys.argv[4]=="none" else sys.argv[4]
)
if phase=="calibration":
 index=config["evaluation"]["max_length"]["horizon_grid_index"]
 expected=(
  "configs/common_beta_post_recovery_calibration.yaml"
  if index==0 else f"configs/common_beta_post_recovery_calibration_horizon_{index}.yaml"
 )
 output=(
  "phase2-post-recovery-calibration-aggregate.json"
  if index==0 else f"phase2-post-recovery-calibration-horizon-{index}-aggregate.json"
 )
 if beta is not None or (index==0)!=(horizon is None):
  raise SystemExit("calibration predecessor bindings are invalid")
elif phase=="freeze":
 if beta is None or horizon is None:
  raise SystemExit("freeze predecessor bindings are incomplete")
 index=beta["beta_grid_index"]
 if not isinstance(index,int) or isinstance(index,bool) or index<0:
  raise SystemExit("freeze beta-grid index is invalid")
 expected=(
  "configs/common_beta_post_recovery_freeze.yaml"
  if index==0 else f"configs/common_beta_post_recovery_freeze_retry_{index}.yaml"
 )
 output=(
  "phase2-post-recovery-freeze-aggregate.json"
  if index==0 else f"phase2-post-recovery-freeze-retry-{index}-aggregate.json"
 )
else:
 raise SystemExit("aggregate overlay is not a pilot phase")
if sys.argv[5]!=expected:
 raise SystemExit("overlay filename differs from its semantic lineage")
print(phase)
print(output)
PY
)
[[ "${#identities[@]}" = 5 ]] || die "could not resolve aggregate identities"
design_sha256="${identities[0]}"
base_hash="${identities[1]}"
schedule_sha256="${identities[2]}"
pilot_phase="${identities[3]}"
semantic_output_name="${identities[4]}"
[[ "${terminal}" = \
  "${project_root}/runs/phase2-post-recovery-${pilot_phase}/terminal-${array_job_id}.json" ]] \
  || die "terminal evidence path differs from the semantic pilot phase"
PYTHONPATH="${repo_root}/src" python3 \
  "${repo_root}/scripts/hpc4/capture_phase2_post_recovery_terminal.py" verify \
  "${terminal}" --expected-sha256 "${terminal_sha256}" \
  --array-job-id "${array_job_id}" --pilot-phase "${pilot_phase}" >/dev/null \
  || die "post-recovery array lacks exact terminal sacct evidence"

inventory="$(realpath -e -- "${hf_cache}/inventories/${base_hash}.json")" \
  || die "HF inventory is missing"
[[ -f "${inventory}" && ! -L "${inventory}" ]] || die "HF inventory is unsafe"
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image SHA256 mismatch"

mkdir -p "${project_root}/aggregates"
output_candidate="${output_input}"
[[ "${output_candidate}" = /* ]] || output_candidate="${project_root}/${output_candidate}"
output_parent="$(realpath -e -- "$(dirname "${output_candidate}")")" \
  || die "aggregate output parent is missing"
output="${output_parent}/$(basename "${output_candidate}")"
case "${output}" in "${project_root}"/*) ;; *) die "aggregate output escaped root" ;; esac
[[ "${output}" = "${project_root}/aggregates/${semantic_output_name}" ]] \
  || die "post-recovery aggregate output must use its locked semantic path"
[[ ! -e "${output}" && ! -L "${output}" ]] || die "refusing to overwrite aggregate"
evidence_root="${output}.evidence"
overlay_evidence="${evidence_root}/${overlay_relative}"
base_evidence="${evidence_root}/${base_relative}"
if [[ -e "${evidence_root}" || -L "${evidence_root}" ]]; then
  [[ -d "${evidence_root}" && ! -L "${evidence_root}" \
    && -f "${overlay_evidence}" && ! -L "${overlay_evidence}" \
    && -f "${base_evidence}" && ! -L "${base_evidence}" ]] \
    || die "existing aggregate configuration evidence is unsafe"
  printf '%s  %s\n' "${overlay_sha256}" "${overlay_evidence}" \
    | sha256sum --check --status \
    || die "existing overlay evidence differs from the committed overlay"
  printf '%s  %s\n' "${base_sha256}" "${base_evidence}" \
    | sha256sum --check --status \
    || die "existing base evidence differs from the committed base"
fi

runs=()
file_bindings=()
for task in 0 1 2; do
  seed=$((20260801 + task))
  run="$(project_path "${run_inputs[$task]}" directory)"
  case "${run}" in
    "${project_root}/runs/phase2-post-recovery-${pilot_phase}/${design_sha256}/seed-${seed}/job-${array_job_id}_${task}") ;;
    *) die "run ${task} does not match the locked design/seed/array path" ;;
  esac
  marker="${run}/SUCCESS"
  grep -Fx 'status=SUCCESS' "${marker}" >/dev/null || die "run ${task} is not SUCCESS"
  allocation_job_id_raw="$(
    awk -F= '$1 == "allocation_job_id_raw" { print $2 }' "${marker}"
  )"
  [[ "${allocation_job_id_raw}" =~ ^[1-9][0-9]*$ ]] \
    || die "run ${task} SUCCESS marker lacks a valid allocation JobIDRaw"
  for expected in \
    "schema_version=prorm-phase2-post-recovery-pilot-run-status/v1" \
    "pilot_phase=${pilot_phase}" \
    "slurm_job_id=${allocation_job_id_raw}" \
    "slurm_array_task_job_id=${array_job_id}_${task}" \
    "array_job_id=${array_job_id}" "array_task_id=${task}" "seed=${seed}" \
    "phase2_design_sha256=${design_sha256}" "base_config_hash=${base_hash}" \
    "git_commit=${producer_commit}" \
    "recovery_authorization_sha256=${authorization_sha256}" \
    "optimizer_schedule_sha256=${schedule_sha256}" \
    "materialization_mode=fresh" "recovery_outputs_mounted=false"; do
    grep -Fx "${expected}" "${marker}" >/dev/null \
      || die "run ${task} SUCCESS marker lacks ${expected}"
  done
  for name in \
    SUCCESS phase2-pilot-diagnostics.json \
    phase2-pilot-diagnostics.diagnostics.jsonl run-manifest.json \
    phase2-output-verification.json post-recovery-output-verification.json \
    artifact/metadata.json; do
    path="${run}/${name}"
    [[ -f "${path}" && ! -L "${path}" ]] || die "run ${task} lacks ${name}"
    file_bindings+=("${path}:$(sha256sum -- "${path}" | awk '{print $1}')")
  done
  runs+=("${run}")
done

for value in \
  "${project_root}" "${scratch_root}" "${repo_root}" "${image}" "${hf_cache}" \
  "${inventory}" "${authorization}" "${terminal}" "${terminal_raw}" \
  "${beta_source}" "${horizon_parent}" \
  "${output}" "${evidence_root}" "${overlay_evidence}" "${base_evidence}" \
  "${runs[@]}"; do
  reject_delimiters path "${value}"
done
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${aggregator_commit}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "aggregate checkout changed during validation"
for binding in \
  "${authorization}:${authorization_sha256}" "${terminal}:${terminal_sha256}" \
  "${inventory}:${inventory_sha256}" "${image}:${PRORM_IMAGE_SHA256}" \
  "${beta_source:+${beta_source}:${beta_source_sha256}}" \
  "${horizon_parent:+${horizon_parent}:${horizon_parent_sha256}}" \
  "${file_bindings[@]}"; do
  [[ -z "${binding}" ]] && continue
  path="${binding%%:*}"
  expected="${binding#*:}"
  printf '%s  %s\n' "${expected}" "${path}" \
    | sha256sum --check --status || die "aggregate source changed: ${path}"
done

mkdir -p "${scratch_root}/phase2-post-recovery-aggregate-jobs" \
  "${project_root}/slurm-logs"
beta_source_present=0
horizon_parent_present=0
[[ -z "${beta_source}" ]] || beta_source_present=1
[[ -z "${horizon_parent}" ]] || horizon_parent_present=1
export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_REPO_ROOT=${repo_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_POST_RECOVERY_OVERLAY_REL=${overlay_relative},PRORM_PHASE2_BASE_REL=${base_relative},PRORM_POST_RECOVERY_OVERLAY_SHA256=${overlay_sha256},PRORM_PHASE2_BASE_SHA256=${base_sha256},PRORM_POST_RECOVERY_DESIGN_SHA256=${design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_hash},PRORM_RECOVERY_AUTHORIZATION=${authorization},PRORM_RECOVERY_AUTHORIZATION_SHA256=${authorization_sha256},PRORM_POST_RECOVERY_TERMINAL=${terminal},PRORM_POST_RECOVERY_TERMINAL_RAW=${terminal_raw},PRORM_POST_RECOVERY_TERMINAL_SHA256=${terminal_sha256},PRORM_POST_RECOVERY_ARRAY_JOB_ID=${array_job_id},PRORM_AGGREGATOR_GIT_COMMIT=${aggregator_commit},PRORM_PRODUCER_GIT_COMMIT=${producer_commit},PRORM_POST_RECOVERY_AGGREGATE_OUTPUT=${output},PRORM_POST_RECOVERY_EVIDENCE_ROOT=${evidence_root},PRORM_POST_RECOVERY_OVERLAY_EVIDENCE=${overlay_evidence},PRORM_POST_RECOVERY_BASE_EVIDENCE=${base_evidence},PRORM_POST_RECOVERY_PILOT_PHASE=${pilot_phase},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${beta_source_present},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${horizon_parent_present}"
if [[ -n "${beta_source}" ]]; then
  export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${beta_source},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${beta_source_sha256}"
fi
if [[ -n "${horizon_parent}" ]]; then
  export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${horizon_parent},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${horizon_parent_sha256}"
fi
for task in 0 1 2; do
  export_spec+=",PRORM_POST_RECOVERY_RUN_${task}=${runs[$task]}"
done
exec python3 \
  "${repo_root}/scripts/hpc4/submit_phase2_post_recovery_aggregate_attempt.py" \
  --project-root "${project_root}" \
  --repo-root "${repo_root}" \
  --pilot-phase "${pilot_phase}" \
  --design-sha256 "${design_sha256}" \
  --pilot-array-job-id "${array_job_id}" \
  --aggregator-git-commit "${aggregator_commit}" \
  --output "${output}" \
  --partition "${partition}" \
  --walltime "${walltime}" \
  --export-spec "${export_spec}" \
  --sbatch-script \
  "${repo_root}/scripts/hpc4/phase2_post_recovery_aggregate.sbatch"
