#!/usr/bin/env bash
set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -ne 5 ]]; then
  die "usage: $0 <overlay.yaml> <base.yaml> <accepted-freeze-aggregate.json> <gpu-partition> <walltime>"
fi

overlay_input="$1"
base_input="$2"
accepted_freeze_input="$3"
partition="$4"
walltime="$5"
case "${partition}" in
  gpu-l20) ;;
  *)
    die "Phase-2 confirmatory design is locked to HPC4 gpu-l20; refusing GPU partition: ${partition}"
    ;;
esac
[[ "${walltime}" =~ ^[1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

for variable in $(compgen -e); do
  case "${variable}" in
    APPTAINER*|SINGULARITY*)
      die "unset exported ${variable}; Phase-2 submission forbids ambient container controls"
      ;;
    SBATCH_*)
      die "unset exported ${variable}; Phase-2 submission forbids ambient sbatch option overrides"
      ;;
  esac
done

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "could not resolve a full Git HEAD"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "Phase-2 confirmatory submission requires a clean Git worktree"

overlay="$(realpath -e -- "${overlay_input}")" \
  || die "overlay does not exist or cannot be resolved: ${overlay_input}"
base_config="$(realpath -e -- "${base_input}")" \
  || die "base config does not exist or cannot be resolved: ${base_input}"
for path in "${overlay}" "${base_config}"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "configuration path is not a non-symlink regular file: ${path}"
  case "${path}" in
    "${repo_root}"/configs/*.yaml) ;;
    *) die "Phase-2 configs must be tracked configs/*.yaml files" ;;
  esac
done
overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"
base_relative="$(realpath --relative-to="${repo_root}" "${base_config}")"
[[ "${overlay_relative}" != "configs/common_beta_pilot.yaml" ]] \
  || die "confirmatory submission refuses the pilot overlay"
[[ "${base_relative}" != "configs/common_beta_pilot_base.yaml" ]] \
  || die "confirmatory submission refuses the pilot base config"
git -C "${repo_root}" ls-files --error-unmatch -- "${overlay_relative}" >/dev/null \
  || die "overlay is not tracked by Git"
git -C "${repo_root}" ls-files --error-unmatch -- "${base_relative}" >/dev/null \
  || die "base config is not tracked by Git"

identity_relative="configs/identities.json"
git -C "${repo_root}" ls-files --error-unmatch -- "${identity_relative}" >/dev/null \
  || die "configs/identities.json is not tracked by Git"
for command_name in \
  python3 sbatch scontrol squeue sacct id find flock realpath sha256sum awk git mktemp mv; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required Phase-2 submission command is unavailable: ${command_name}"
done
overlay_worktree_sha256="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_worktree_sha256="$(sha256sum -- "${base_config}" | awk '{print $1}')"
mapfile -t identity_info < <(
  python3 -I -S - \
    "${repo_root}" "${git_commit}" "${identity_relative}" \
    "${overlay_relative}" "${base_relative}" \
    "${overlay_worktree_sha256}" "${base_worktree_sha256}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys


(
    repo_root,
    commit,
    identity_relative,
    overlay_relative,
    base_relative,
    overlay_worktree_sha256,
    base_worktree_sha256,
) = sys.argv[1:]


def committed_blob(relative):
    result = subprocess.run(
        ["git", "-C", repo_root, "cat-file", "blob", f"{commit}:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"cannot read committed blob {relative}: {message}")
    return result.stdout


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


overlay_bytes = committed_blob(overlay_relative)
base_bytes = committed_blob(base_relative)
overlay_file_sha = hashlib.sha256(overlay_bytes).hexdigest()
base_file_sha = hashlib.sha256(base_bytes).hexdigest()
if overlay_file_sha != overlay_worktree_sha256:
    raise SystemExit("worktree overlay bytes do not match submitted Git commit")
if base_file_sha != base_worktree_sha256:
    raise SystemExit("worktree base config bytes do not match submitted Git commit")

identity_bytes = committed_blob(identity_relative)
identities = json.loads(
    identity_bytes.decode("utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {value}")
    ),
)
if identities.get("schema_version") != "prorm-config-identities/v1":
    raise SystemExit("unsupported config identity schema")
configs = identities.get("configs")
if not isinstance(configs, dict):
    raise SystemExit("config identity map is missing")


def entry(relative, file_sha):
    value = configs.get(relative)
    if not isinstance(value, dict) or set(value) != {
        "config_hash",
        "file_sha256",
        "seed_count",
    }:
        raise SystemExit(f"invalid config identity entry: {relative}")
    if value["file_sha256"] != file_sha:
        raise SystemExit(
            f"committed config bytes do not match committed identity: {relative}"
        )
    if not isinstance(value["config_hash"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["config_hash"]
    ):
        raise SystemExit(f"invalid semantic config identity: {relative}")
    if not isinstance(value["seed_count"], int) or value["seed_count"] <= 0:
        raise SystemExit(f"invalid seed count identity: {relative}")
    return value


overlay_entry = entry(overlay_relative, overlay_file_sha)
base_entry = entry(base_relative, base_file_sha)
if overlay_entry["seed_count"] != base_entry["seed_count"]:
    raise SystemExit("overlay and base seed-count identities differ")
if overlay_entry["seed_count"] != 30:
    raise SystemExit("the Phase-2 confirmatory identity must declare exactly 30 seeds")
if overlay_entry["config_hash"] == base_entry["config_hash"]:
    raise SystemExit("overlay design identity and base artifact identity must differ")

overlay_text = overlay_bytes.decode("utf-8")


def require_unique(pattern, description):
    values = re.findall(pattern, overlay_text, flags=re.MULTILINE)
    if len(values) != 1:
        raise SystemExit(f"overlay must contain exactly one {description}")
    return values[0]


schema = require_unique(
    r"^schema_version:[ \t]*(\S+)[ \t]*$",
    "schema_version",
)
if schema not in {
    "prorm-common-beta-config/v2",
    "prorm-common-beta-post-recovery-experiment/v1",
}:
    raise SystemExit("overlay is not a supported Phase-2 confirmatory schema")
source_path = require_unique(
    r"^  source_config:[ \t]*(\S+)[ \t]*$",
    "design.source_config binding",
)
source_hash = require_unique(
    r"^  source_config_hash:[ \t]*([0-9a-f]{64})[ \t]*$",
    "design.source_config_hash binding",
)
if source_path != base_relative:
    raise SystemExit("overlay source_config does not name the submitted base config")
if source_hash != base_entry["config_hash"]:
    raise SystemExit("overlay source_config_hash does not equal the base semantic identity")
if not re.search(r"(?m)^  stage:[ \t]*confirmatory[ \t]*$", overlay_text):
    raise SystemExit("overlay is not explicitly confirmatory")
if not re.search(r"(?m)^  formal_eligibility:[ \t]*true[ \t]*$", overlay_text):
    raise SystemExit("confirmatory overlay must be formally eligible")

run_match = re.search(
    r"(?m)^run:[ \t]*\n(?P<body>(?:(?:^[ \t].*)\n|^\n)*)",
    overlay_text,
)
if run_match is None:
    raise SystemExit("overlay is missing a top-level run section")
run_body = run_match.group("body")
formal_seeds = [
    int(value)
    for value in re.findall(r"(?<![0-9])202609[0-9]{2}(?![0-9])", run_body)
]
expected_seeds = list(range(20260901, 20260931))
if formal_seeds != expected_seeds:
    raise SystemExit(
        "confirmatory run.seeds must be exactly ordered 20260901 through 20260930"
    )
if not re.search(r"(?m)^  confirmatory:[ \t]*true[ \t]*$", run_body):
    raise SystemExit("confirmatory run flag is not true")
if not re.search(r"(?m)^  excluded_from_confirmatory_evidence:[ \t]*false[ \t]*$", run_body):
    raise SystemExit("formal run is incorrectly excluded from confirmatory evidence")

beta_source_sha = require_unique(
    r"^    beta_source_aggregate_sha256:[ \t]*([0-9a-f]{64})[ \t]*$",
    "objective.common_beta.beta_source_aggregate_sha256 binding",
)
horizon_parent_sha = require_unique(
    r"^    parent_pilot_aggregate_sha256:[ \t]*([0-9a-f]{64})[ \t]*$",
    "evaluation.max_length.parent_pilot_aggregate_sha256 binding",
)
if beta_source_sha != horizon_parent_sha:
    raise SystemExit("confirmatory beta and horizon must bind the same accepted freeze aggregate")
frozen_beta = require_unique(
    r"^    frozen_global_beta:[ \t]*([^# \t]+)[ \t]*$",
    "objective.common_beta.frozen_global_beta",
)
try:
    beta = float(frozen_beta)
except ValueError as error:
    raise SystemExit("confirmatory frozen_global_beta is not numeric") from error
if not (beta > 0.0 and beta < float("inf")):
    raise SystemExit("confirmatory frozen_global_beta must be finite and positive")

print(overlay_entry["seed_count"])
print(overlay_entry["config_hash"])
print(base_entry["config_hash"])
print(overlay_file_sha)
print(base_file_sha)
print(hashlib.sha256(identity_bytes).hexdigest())
print(beta_source_sha)
print(frozen_beta)
print(schema)
PY
)
[[ "${#identity_info[@]}" -eq 9 ]] \
  || die "failed to resolve committed confirmatory identities"
seed_count="${identity_info[0]}"
phase2_design_sha256="${identity_info[1]}"
base_config_hash="${identity_info[2]}"
overlay_file_sha256="${identity_info[3]}"
base_file_sha256="${identity_info[4]}"
identities_file_sha256="${identity_info[5]}"
bound_freeze_sha256="${identity_info[6]}"
frozen_global_beta="${identity_info[7]}"
phase2_schema_version="${identity_info[8]}"
if [[ "${phase2_schema_version}" = \
  "prorm-common-beta-post-recovery-experiment/v1" ]]; then
  [[ "${overlay_relative}" = \
    "configs/common_beta_post_recovery_confirmatory.yaml" \
    && "${base_relative}" = \
    "configs/common_beta_post_recovery_confirmatory_base.yaml" ]] \
    || die "post-recovery confirmatory configs must use their exact semantic paths"
fi
[[ "${seed_count}" = "30" ]] || die "invalid committed formal seed count"
for digest in \
  "${phase2_design_sha256}" "${base_config_hash}" "${overlay_file_sha256}" \
  "${base_file_sha256}" "${identities_file_sha256}" "${bound_freeze_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid committed SHA256 identity"
done

# Scheduler capacity is a campaign protocol, not a caller option.  The exact
# ordered 30-seed plan is precommitted below as seven four-task waves and one
# two-task wave.  Every wave uses global task IDs and a fixed `%2` throttle.
[[ -z "${PRORM_PHASE2_ARRAY_CONCURRENCY:-}" ]] \
  || die "formal fixed-wave concurrency is immutable; unset PRORM_PHASE2_ARRAY_CONCURRENCY"

normalize_root() {
  local name="$1" raw="${!1}" resolved=""
  [[ "${raw}" = /* ]] || die "${name} must be an absolute path"
  resolved="$(realpath -e -- "${raw}")" \
    || die "${name} does not exist or cannot be resolved: ${raw}"
  [[ -d "${resolved}" && "${resolved}" != "/" ]] \
    || die "${name} must be a non-root directory"
  printf '%s\n' "${resolved}"
}

project_root="$(normalize_root PRORM_PROJECT_ROOT)"
scratch_root="$(normalize_root PRORM_SCRATCH_ROOT)"
case "${project_root}" in
  "${scratch_root}"|"${scratch_root}"/*) die "project and scratch roots overlap" ;;
esac
case "${scratch_root}" in
  "${project_root}"|"${project_root}"/*) die "project and scratch roots overlap" ;;
esac
[[ -w "${project_root}" ]] || die "PRORM_PROJECT_ROOT is not writable"
[[ -w "${scratch_root}" ]] || die "PRORM_SCRATCH_ROOT is not writable"

resolve_project_path() {
  local raw="$1" kind="$2" candidate="" resolved=""
  if [[ "${raw}" = /* ]]; then candidate="${raw}"; else candidate="${project_root}/${raw}"; fi
  resolved="$(realpath -e -- "${candidate}")" \
    || die "project path does not exist or cannot be resolved: ${candidate}"
  case "${resolved}" in
    "${project_root}"/*) ;;
    *) die "project path escaped PRORM_PROJECT_ROOT: ${raw}" ;;
  esac
  case "${kind}" in
    file) [[ -f "${resolved}" && ! -L "${resolved}" ]] \
      || die "project path is not a non-symlink file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] \
      || die "project path is not a directory: ${resolved}" ;;
    *) die "invalid internal path kind" ;;
  esac
  printf '%s\n' "${resolved}"
}

image="$(resolve_project_path "${PRORM_IMAGE}" file)"
hf_cache="$(resolve_project_path "${PRORM_HF_CACHE}" directory)"
accepted_freeze="$(resolve_project_path "${accepted_freeze_input}" file)"
accepted_freeze_sha256="$(sha256sum -- "${accepted_freeze}" | awk '{print $1}')"
[[ "${accepted_freeze_sha256}" = "${bound_freeze_sha256}" ]] \
  || die "accepted freeze aggregate bytes do not match the confirmatory design binding"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status || die "image SHA256 mismatch"

inventory_expected="${hf_cache}/inventories/${base_config_hash}.json"
inventory="$(realpath -e -- "${inventory_expected}")" \
  || die "missing base-config HF inventory: ${inventory_expected}"
[[ -f "${inventory}" && ! -L "${inventory}" ]] \
  || die "base-config HF inventory is not a non-symlink regular file"
case "${inventory}" in
  "${hf_cache}"/inventories/*) ;;
  *) die "base-config inventory escaped HF cache" ;;
esac
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
[[ "${inventory_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash base-config HF inventory"

if [[ "${phase2_schema_version}" = "prorm-common-beta-config/v2" ]]; then
  python3 -I -S - "${accepted_freeze}" "${frozen_global_beta}" <<'PY'
import json
import math
import sys
from pathlib import Path


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


value = json.loads(
    Path(sys.argv[1]).read_text(encoding="utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda constant: (_ for _ in ()).throw(
        ValueError(f"non-finite JSON constant: {constant}")
    ),
)
selection = value.get("selection")
aggregation = value.get("aggregation_identity")
if (
    value.get("schema_version") != "common-beta-pilot-selection-aggregate/v2"
    or value.get("pilot_phase") != "freeze"
    or value.get("formal_eligibility") is not False
    or value.get("supports_formal_claim") is not False
    or not isinstance(aggregation, dict)
    or set(aggregation)
    != {
        "schema_version",
        "aggregator_git_commit",
        "producer_git_commit",
        "image_sha256",
        "hf_inventory_sha256",
        "validator_source_sha256",
    }
    or aggregation.get("schema_version")
    != "phase2-pilot-aggregation-identity/v1"
    or not isinstance(selection, dict)
    or selection.get("schema_version") != "pilot-freeze-selection/v1"
    or selection.get("selection_accepted") is not True
    or selection.get("accepted_for_confirmatory_identity") is not True
    or selection.get("all_seeds_and_arms_used_same_beta") is not True
    or selection.get("all_pre_oracle_safety_gates_passed") is not True
    or selection.get("all_length_gates_passed") is not True
    or selection.get("all_non_length_safety_gates_passed") is not True
    or selection.get("next_action") != "freeze_confirmatory_design_identity"
):
    raise SystemExit("accepted freeze aggregate did not pass every frozen safety gate")
beta = selection.get("frozen_global_beta")
if (
    isinstance(beta, bool)
    or not isinstance(beta, (int, float))
    or not math.isfinite(float(beta))
    or float(beta) != float(sys.argv[2])
):
    raise SystemExit("accepted freeze beta differs from confirmatory frozen_global_beta")
PY
else
  [[ "${phase2_schema_version}" = \
    "prorm-common-beta-post-recovery-experiment/v1" ]] \
    || die "unreachable confirmatory schema branch"
  authorization="${project_root}/runs/phase2-recovery-pilot/recovery-success-authorization.json"
  authorization="$(resolve_project_path "${authorization}" file)"
  authorization_sha256="$(sha256sum -- "${authorization}" | awk '{print $1}')"
  PYTHONPATH="${repo_root}/src" python3 - \
    "${overlay}" "${accepted_freeze}" "${frozen_global_beta}" \
    "${authorization}" "${authorization_sha256}" <<'PY'
import math
import sys
from smart_reward.phase2_config import load_phase2_config_bundle
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    verify_recovery_authorization_config_binding,
)

overlay, freeze, expected_beta, authorization, authorization_sha = sys.argv[1:]
bundle = load_phase2_config_bundle(overlay)
config = bundle.config
binding = verify_recovery_authorization_config_binding(
    authorization,
    overlay,
    expected_sha256=authorization_sha,
    expected_stage="confirmatory",
)
beta = verify_beta_source_aggregate(config, freeze)
horizon = verify_horizon_parent_aggregate(config, freeze)
if (
    beta is None
    or horizon is None
    or beta.get("source_pilot_phase") != "freeze"
    or horizon.get("source_pilot_phase") != "freeze"
    or binding.get("optimizer_schedule_sha256") != OPTIMIZER_SCHEDULE_SHA256
    or not math.isclose(
        float(beta["accepted_beta"]),
        float(expected_beta),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
):
    raise SystemExit("post-recovery accepted freeze failed recursive v3 verification")
PY
fi
attempt_index=1

validate_existing_formal_directory() {
  local label="$1" path="$2" resolved=""
  if [[ -e "${path}" || -L "${path}" ]]; then
    [[ -d "${path}" && ! -L "${path}" ]] \
      || die "${label} exists but is not a non-symlink directory: ${path}"
    resolved="$(realpath -e -- "${path}")" \
      || die "${label} cannot be resolved: ${path}"
    [[ "${resolved}" = "${path}" ]] \
      || die "${label} is not canonical: ${path}"
  fi
}

formal_runs_root="${project_root}/runs"
formal_phase2_root="${formal_runs_root}/phase2-confirmatory"
formal_design_root="${formal_phase2_root}/${phase2_design_sha256}"
validate_existing_formal_directory "project runs root" "${formal_runs_root}"
validate_existing_formal_directory "formal Phase-2 runs root" "${formal_phase2_root}"
validate_existing_formal_directory "formal design root" "${formal_design_root}"

ensure_submission_directory() {
  local label="$1" path="$2" resolved=""
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    if ! mkdir -- "${path}"; then
      [[ -e "${path}" || -L "${path}" ]] \
        || die "failed to create ${label}: ${path}"
    fi
  fi
  [[ -d "${path}" && ! -L "${path}" ]] \
    || die "${label} is not a non-symlink directory: ${path}"
  resolved="$(realpath -e -- "${path}")" \
    || die "${label} cannot be resolved: ${path}"
  [[ "${resolved}" = "${path}" ]] \
    || die "${label} is not canonical: ${path}"
  python3 -I -S - "${path}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
for directory in (path, path.parent):
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
}

ensure_submission_directory "project runs root" "${formal_runs_root}"
ensure_submission_directory "formal Phase-2 runs root" "${formal_phase2_root}"
ensure_submission_directory "formal design root" "${formal_design_root}"
campaign_registry="${formal_design_root}/campaign-registry"
registry_admissions="${campaign_registry}/admissions"
registry_submissions="${campaign_registry}/submissions"
registry_executions="${campaign_registry}/executions"
registry_recoveries="${campaign_registry}/recoveries"
registry_scheduler_terminals="${campaign_registry}/scheduler-terminals"
registry_staging="${campaign_registry}/.staging"
ensure_submission_directory "formal campaign registry" "${campaign_registry}"
ensure_submission_directory "formal wave admission registry" "${registry_admissions}"
ensure_submission_directory "formal submission registry" "${registry_submissions}"
ensure_submission_directory "formal execution registry" "${registry_executions}"
ensure_submission_directory "formal recovery registry" "${registry_recoveries}"
ensure_submission_directory \
  "formal scheduler terminal registry" "${registry_scheduler_terminals}"
ensure_submission_directory "non-authoritative registry staging" "${registry_staging}"
registry_lock="${campaign_registry}/registry.lock"
[[ ! -L "${registry_lock}" ]] || die "formal campaign registry lock is a symlink"

for value in \
  "${project_root}" "${scratch_root}" "${image}" "${hf_cache}" "${repo_root}" \
  "${overlay}" "${base_config}" "${inventory}" "${accepted_freeze}" \
  "${campaign_registry}"; do
  [[ "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "path contains an unsafe sbatch/export or Apptainer-bind delimiter"
done

[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" ]] \
  || die "Git HEAD changed while preparing the Phase-2 submission"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "Git worktree changed while preparing the Phase-2 submission"
printf '%s  %s\n' "${overlay_file_sha256}" "${overlay}" \
  | sha256sum --check --status || die "overlay changed during submission"
printf '%s  %s\n' "${base_file_sha256}" "${base_config}" \
  | sha256sum --check --status || die "base config changed during submission"
printf '%s  %s\n' "${accepted_freeze_sha256}" "${accepted_freeze}" \
  | sha256sum --check --status \
  || die "accepted freeze aggregate changed during submission"
slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${scratch_root}/phase2-confirmatory-jobs"
export_spec_base="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_REPO_ROOT=${repo_root},PRORM_PHASE2_OVERLAY=${overlay},PRORM_PHASE2_BASE_CONFIG=${base_config},PRORM_PHASE2_DESIGN_SHA256=${phase2_design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_config_hash},PRORM_PHASE2_OVERLAY_FILE_SHA256=${overlay_file_sha256},PRORM_PHASE2_BASE_FILE_SHA256=${base_file_sha256},PRORM_IDENTITIES_FILE_SHA256=${identities_file_sha256},PRORM_GIT_COMMIT=${git_commit},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE=${accepted_freeze},PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256=${accepted_freeze_sha256},PRORM_PHASE2_FROZEN_GLOBAL_BETA=${frozen_global_beta},PRORM_PHASE2_ATTEMPT_INDEX=${attempt_index},PRORM_PHASE2_CAMPAIGN_REGISTRY=${campaign_registry}"
mapfile -t configured_clusters < <(
  scontrol show config \
    | awk '$1 == "ClusterName" && $2 == "=" {print $3}'
)
[[ "${#configured_clusters[@]}" -eq 1 \
  && "${configured_clusters[0]}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "could not resolve exactly one safe Slurm ClusterName"
configured_cluster="${configured_clusters[0]}"
[[ "${configured_cluster}" = "hpc4" ]] \
  || die "formal Phase-2 submission must execute on Slurm cluster hpc4"
confirmatory_job_file_sha256="$(
  sha256sum -- "${repo_root}/scripts/hpc4/phase2_confirmatory.sbatch" | awk '{print $1}'
)"
exec {registry_lock_fd}> "${registry_lock}"
[[ -f "${registry_lock}" && ! -L "${registry_lock}" ]] \
  || die "formal campaign registry lock is not a non-symlink regular file"
flock -x "${registry_lock_fd}" \
  || die "failed to acquire the formal campaign registry lock"
python3 -I -S - "${registry_recoveries}" <<'PY'
import sys
from pathlib import Path


recoveries = Path(sys.argv[1])
if next(recoveries.iterdir(), None) is not None:
    raise SystemExit("formal no-retry campaign recovery registry must remain empty")
PY

campaign_plan="${campaign_registry}/campaign-plan.json"
if [[ ! -e "${campaign_plan}" && ! -L "${campaign_plan}" ]]; then
  [[ -z "$(find "${registry_submissions}" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
    || die "cannot create a campaign plan after a scheduler submission exists"
  campaign_plan_staging="$(mktemp "${registry_staging}/campaign-plan.XXXXXX")"
  python3 -I -S - \
    "${campaign_plan_staging}" "${phase2_design_sha256}" \
    "${base_config_hash}" "${git_commit}" "${accepted_freeze_sha256}" \
    "${walltime}" "${overlay_file_sha256}" "${base_file_sha256}" \
    "${identities_file_sha256}" "${PRORM_IMAGE_SHA256}" \
    "${inventory_sha256}" "${confirmatory_job_file_sha256}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


(
    output_raw,
    design_sha,
    base_hash,
    git_commit,
    freeze_sha,
    walltime,
    overlay_file_sha,
    base_file_sha,
    identities_file_sha,
    image_sha,
    inventory_sha,
    job_file_sha,
) = sys.argv[1:]
output = Path(output_raw)
seeds = list(range(20260901, 20260931))
wave_tasks = [
    list(range(0, 4)),
    list(range(4, 8)),
    list(range(8, 12)),
    list(range(12, 16)),
    list(range(16, 20)),
    list(range(20, 24)),
    list(range(24, 28)),
    list(range(28, 30)),
]
waves = [
    {
        "wave_index": index,
        "array_spec": f"{tasks[0]}-{tasks[-1]}%2",
        "array_task_ids": tasks,
        "seeds": [seeds[task] for task in tasks],
    }
    for index, tasks in enumerate(wave_tasks)
]
payload = {
    "schema_version": "prorm-phase2-fixed-wave-campaign-plan/v1",
    "status": "precommitted_before_first_slurm_submission",
    "phase2_design_sha256": design_sha,
    "base_config_hash": base_hash,
    "git_commit": git_commit,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "ordered_seeds": seeds,
    "attempt_index": 1,
    "retry_policy": "single_predeclared_attempt_no_retry",
    "replacement_seed_allowed": False,
    "optional_stopping_allowed": False,
    "max_submitted_tasks": 4,
    "max_running_tasks": 2,
    "waves": waves,
    "job_tuple": {
        "account": "sigroup",
        "partition": "gpu-l20",
        "qos": "l20_qos",
        "nodes": 1,
        "tasks": 1,
        "cpus_per_task": 8,
        "memory": "64G",
        "gpus_per_node": 1,
        "walltime": walltime,
        "no_requeue": True,
        "held_before_registry_commit": True,
        "script": "scripts/hpc4/phase2_confirmatory.sbatch",
        "script_file_sha256": job_file_sha,
    },
    "producer": {
        "overlay_file_sha256": overlay_file_sha,
        "base_file_sha256": base_file_sha,
        "identities_file_sha256": identities_file_sha,
        "image_sha256": image_sha,
        "hf_inventory_sha256": inventory_sha,
    },
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with output.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
  mv -T --no-clobber -- "${campaign_plan_staging}" "${campaign_plan}" \
    || die "failed to atomically precommit the fixed-wave campaign plan"
  python3 -I -S - "${campaign_plan}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
for target in (path, path.parent):
    flags = os.O_RDONLY
    if target.is_dir():
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(target, flags | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
fi
[[ -f "${campaign_plan}" && ! -L "${campaign_plan}" ]] \
  || die "formal fixed-wave campaign plan is missing or unsafe"

campaign_state="$(
  python3 -I -S \
    "${repo_root}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
    "${formal_design_root}" "${phase2_design_sha256}" \
    "${base_config_hash}" "${git_commit}" "${PRORM_IMAGE_SHA256}" \
    "${inventory_sha256}" --state
)" || die "formal fixed-wave campaign registry state is invalid"
mapfile -t state_info < <(
  python3 -I -S - "${campaign_state}" <<'PY'
import json
import re
import sys


value = json.loads(sys.argv[1])
status = value.get("status")
sha = value.get("campaign_plan_sha256")
if status not in {"ready", "active", "complete"}:
    raise SystemExit("invalid fixed-wave campaign state")
if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
    raise SystemExit("invalid campaign plan SHA256")
print(status)
print(sha)
print(value.get("wave_index", "-"))
print(value.get("array_spec", "-"))
print(value.get("array_job_id", "-"))
admission_sha = value.get("wave_admission_sha256")
if admission_sha is not None and (
    not isinstance(admission_sha, str)
    or re.fullmatch(r"[0-9a-f]{64}", admission_sha) is None
):
    raise SystemExit("invalid wave admission SHA256")
print("-" if admission_sha is None else admission_sha)
walltime = value.get("walltime")
created_at = value.get("plan_created_at_utc")
if (
    not isinstance(walltime, str)
    or re.fullmatch(
        r"(?:[1-9][0-9]*-)?[0-9]{2}:[0-9]{2}:[0-9]{2}",
        walltime,
    )
    is None
    or not isinstance(created_at, str)
    or re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        created_at,
    )
    is None
):
    raise SystemExit("invalid campaign plan walltime or timestamp")
print(walltime)
print(created_at)
PY
)
[[ "${#state_info[@]}" -eq 8 ]] \
  || die "could not parse the fixed-wave campaign state"
campaign_status="${state_info[0]}"
campaign_plan_sha256="${state_info[1]}"
wave_index="${state_info[2]}"
array_spec="${state_info[3]}"
committed_array_job_id="${state_info[4]}"
wave_admission_sha256="${state_info[5]}"
plan_walltime="${state_info[6]}"
plan_created_at_utc="${state_info[7]}"
printf '%s  %s\n' "${campaign_plan_sha256}" "${campaign_plan}" \
  | sha256sum --check --status || die "campaign plan changed after registry validation"
[[ "${walltime}" = "${plan_walltime}" ]] \
  || die "caller walltime differs from the immutable campaign plan"
walltime="${plan_walltime}"

if [[ "${campaign_status}" = "complete" ]]; then
  flock -u "${registry_lock_fd}" \
    || die "failed to release the formal campaign registry lock"
  printf 'COMPLETE;%s\n' "${campaign_plan_sha256}"
  exit 0
fi
[[ "${wave_index}" =~ ^[0-7]$ \
  && "${array_spec}" =~ ^(0-3|4-7|8-11|12-15|16-19|20-23|24-27|28-29)%2$ ]] \
  || die "registry returned an invalid fixed wave"

if [[ "${campaign_status}" = "ready" && "${wave_admission_sha256}" = "-" ]]; then
  admission_record="${registry_admissions}/wave-${wave_index}.json"
  admission_staging="$(
    mktemp "${registry_staging}/admission-wave-${wave_index}.XXXXXX"
  )"
  generated_admission_sha="$(
    python3 -I -S \
      "${repo_root}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
      "${formal_design_root}" "${phase2_design_sha256}" \
      "${base_config_hash}" "${git_commit}" "${PRORM_IMAGE_SHA256}" \
      "${inventory_sha256}" --admit "${admission_staging}"
  )" || die "failed to materialize the immutable wave admission receipt"
  [[ "${generated_admission_sha}" =~ ^[0-9a-f]{64}$ ]] \
    || die "wave admission materializer returned an invalid SHA256"
  printf '%s  %s\n' "${generated_admission_sha}" "${admission_staging}" \
    | sha256sum --check --status || die "staged wave admission SHA256 mismatch"
  [[ ! -e "${admission_record}" && ! -L "${admission_record}" ]] \
    || die "wave admission receipt appeared concurrently"
  mv -T --no-clobber -- "${admission_staging}" "${admission_record}" \
    || die "failed to atomically commit the wave admission receipt"
  python3 -I -S - "${admission_record}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
for target in (path, path.parent):
    flags = os.O_RDONLY
    if target.is_dir():
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(target, flags | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
  python3 -I -S \
    "${repo_root}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
    "${formal_design_root}" "${phase2_design_sha256}" \
    "${base_config_hash}" "${git_commit}" "${PRORM_IMAGE_SHA256}" \
    "${inventory_sha256}" --state >/dev/null \
    || die "committed wave admission receipt failed registry revalidation"
  wave_admission_sha256="${generated_admission_sha}"
fi
[[ "${wave_admission_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "active or ready wave lacks one immutable admission receipt"
wave_job_name="prorm-p2-${campaign_plan_sha256:0:12}-w${wave_index}"

classify_scheduler_array() {
  local array_job_id="$1" expected_name="$2" expected_spec="$3" expected_walltime="$4"
  local evidence_output="${5:-}"
  local scheduler_record=""
  scheduler_record="$(scontrol show job --oneliner "${array_job_id}")" \
    || return 3
  python3 -I -S - \
    "${scheduler_record}" "${array_job_id}" "${expected_name}" \
    "${expected_spec}" "${expected_walltime}" "${repo_root}" \
    "${evidence_output}" <<'PY'
import hashlib
import json
import os
import sys
import re
from pathlib import Path


(
    record,
    array_job_id,
    job_name,
    array_spec,
    walltime,
    repo_root,
    evidence_output,
) = sys.argv[1:]
expected_command = str(
    Path(repo_root) / "scripts" / "hpc4" / "phase2_confirmatory.sbatch"
)
records = [line for line in record.splitlines() if line]
if not records or "\r" in record:
    raise SystemExit("scontrol returned no safe scheduler record")
parsed = []
for line in records:
    fields = {}
    for token in line.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in fields:
            raise SystemExit(f"duplicate scontrol job field: {key}")
        fields[key] = value
    tres = {}
    for entry in str(fields.get("TRES", "")).split(","):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        if key in tres:
            raise SystemExit(f"duplicate TRES resource: {key}")
        tres[key] = value
    if (
        fields.get("ArrayJobId", fields.get("JobId")) != array_job_id
        or fields.get("JobName") != job_name
        or fields.get("Account") != "sigroup"
        or fields.get("Partition") != "gpu-l20"
        or fields.get("QOS") != "l20_qos"
        or fields.get("Requeue") != "0"
        or fields.get("Restarts") != "0"
        or fields.get("ArrayTaskThrottle") != "2"
        or fields.get("NumNodes") not in {"1", "1-1"}
        or fields.get("NumTasks") != "1"
        or fields.get("NumCPUs") != "8"
        or fields.get("CPUs/Task") != "8"
        or fields.get("MinMemoryNode") != "64G"
        or fields.get("TimeLimit") != walltime
        or tres.get("cpu") != "8"
        or tres.get("mem") != "64G"
        or tres.get("node") != "1"
        or tres.get("gres/gpu") != "1"
        or re.fullmatch(
            r"gres(?::|/)gpu(?::[A-Za-z0-9_.-]+)?:1",
            str(fields.get("TresPerNode", "")),
        )
        is None
        or fields.get("Command") != expected_command
        or fields.get("WorkDir") != repo_root
        or not fields.get("JobState")
    ):
        raise SystemExit("Slurm job differs from the committed fixed-wave identity")
    parsed.append(fields)
held_master = (
    len(parsed) == 1
    and parsed[0].get("ArrayTaskId") == array_spec
    and parsed[0].get("JobState") == "PENDING"
    and parsed[0].get("Reason") == "JobHeldUser"
)
if held_master:
    if evidence_output:
        output = Path(evidence_output)
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != 0
        ):
            raise SystemExit("scheduler request evidence staging path is unsafe")
        fields = parsed[0]
        normalized_tres = {
            key: value
            for key, value in (
                entry.split("=", 1)
                for entry in fields["TRES"].split(",")
                if "=" in entry
            )
            if key in {"cpu", "mem", "node", "gres/gpu"}
        }
        payload = {
            "schema_version": "prorm-phase2-held-scheduler-request/v1",
            "captured_while_held": True,
            "raw_scontrol_record": record,
            "raw_scontrol_sha256": hashlib.sha256(record.encode()).hexdigest(),
            "normalized": {
                "array_job_id": array_job_id,
                "job_name": job_name,
                "array_spec": array_spec,
                "array_task_throttle": 2,
                "account": "sigroup",
                "partition": "gpu-l20",
                "qos": "l20_qos",
                "nodes": 1,
                "tasks": 1,
                "cpus": 8,
                "cpus_per_task": 8,
                "memory": "64G",
                "gpus_per_node": 1,
                "walltime": walltime,
                "tres": normalized_tres,
                "tres_per_node": fields["TresPerNode"],
                "requeue": False,
                "restarts": 0,
                "command": expected_command,
                "work_dir": repo_root,
            },
        }
        with output.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    print("HELD")
elif any(
    fields.get("JobState") == "PENDING"
    and str(fields.get("Reason", "")).startswith("JobHeld")
    for fields in parsed
):
    raise SystemExit("fixed wave is held by an unexpected scheduler authority")
else:
    print("ALREADY_RELEASED")
PY
}

if [[ "${campaign_status}" = "active" ]]; then
  mapfile -t committed_scheduler_state < <(
    classify_scheduler_array \
      "${committed_array_job_id}" "${wave_job_name}" "${array_spec}" "${walltime}"
  ) || die "registered wave is absent from Slurm; terminalize it, never replace it"
  [[ "${#committed_scheduler_state[@]}" -eq 1 ]] \
    || die "failed to verify the committed fixed wave in Slurm"
  if [[ "${committed_scheduler_state[0]}" = "HELD" ]]; then
    scontrol release "${committed_array_job_id}" \
      || die "failed to resume-release the committed fixed wave"
  elif [[ "${committed_scheduler_state[0]}" != "ALREADY_RELEASED" ]]; then
    die "invalid committed fixed-wave scheduler state"
  fi
  flock -u "${registry_lock_fd}" \
    || die "failed to release the formal campaign registry lock"
  printf 'ACTIVE;%s;%s;%s\n' \
    "${wave_index}" "${committed_array_job_id}" "${configured_cluster}"
  exit 0
fi

[[ "${campaign_status}" = "ready" && "${committed_array_job_id}" = "-" ]] \
  || die "invalid ready fixed-wave state"
export_spec="${export_spec_base},PRORM_PHASE2_CAMPAIGN_PLAN_SHA256=${campaign_plan_sha256},PRORM_PHASE2_WAVE_ADMISSION_SHA256=${wave_admission_sha256},PRORM_PHASE2_WAVE_INDEX=${wave_index}"

# Recover the only possible crash-window array by its deterministic job name.
# An accepted scheduler identity is never cancelled and replaced.
squeue_records="$(
  squeue --noheader --user="$(id -un)" --name="${wave_job_name}" --format="%A"
)" || die "could not inspect live deterministic-name scheduler identities"
orphan_ids_raw="$(
  printf '%s' "${squeue_records}" \
    | python3 -I -S -c \
      'import re,sys; values={line.strip() for line in sys.stdin}; bad=[value for value in values if value and re.fullmatch(r"[1-9][0-9]*", value) is None]; bad and (_ for _ in ()).throw(SystemExit("squeue returned an invalid array identity")); print("\n".join(sorted((value for value in values if value), key=int)))'
)" || die "live deterministic-name scheduler identities are malformed"
mapfile -t orphan_array_ids < <(printf '%s' "${orphan_ids_raw}")
(( ${#orphan_array_ids[@]} <= 1 )) \
  || die "multiple scheduler arrays match one fixed-wave submission identity"

sacct_starttime="${plan_created_at_utc%Z}"
[[ "${sacct_starttime}Z" = "${plan_created_at_utc}" ]] \
  || die "campaign plan timestamp cannot be normalized for Slurm 22.05 sacct"
sacct_records="$(
  sacct -X --starttime="${sacct_starttime}" --name="${wave_job_name}" \
    --noheader --parsable2 \
    --format=JobIDRaw,JobID,JobName,State,Submit,Timelimit,ReqTRES,AllocTRES
)" || die "could not inspect historical deterministic-name scheduler identities"
accounted_ids_raw="$(
  python3 -I -S - \
    "${sacct_records}" "${wave_job_name}" "${walltime}" <<'PY'
import re
import sys


record, expected_name, expected_walltime = sys.argv[1:]
roots = set()
for line in record.splitlines():
    if not line or "\r" in line:
        raise SystemExit("sacct returned an unsafe historical row")
    fields = line.split("|")
    if len(fields) != 8:
        raise SystemExit("sacct historical row differs from the locked field set")
    raw_id, job_id, job_name, state, submitted, timelimit, _, _ = fields
    if (
        job_name != expected_name
        or not state
        or not submitted
        or timelimit != expected_walltime
    ):
        raise SystemExit("historical scheduler identity differs from the campaign plan")
    root = None
    for candidate in (job_id, raw_id):
        match = re.fullmatch(
            r"([1-9][0-9]*)(?:_(?:[0-9]+|\[[0-9,%\-]+\]))?",
            candidate,
        )
        if match is not None:
            root = match.group(1)
            break
    if root is None:
        raise SystemExit("sacct returned an unrecognized array identity")
    roots.add(root)
print("\n".join(sorted(roots, key=int)))
PY
)" || die "historical deterministic-name scheduler identities are malformed"
mapfile -t accounted_array_ids < <(printf '%s' "${accounted_ids_raw}")
if (( ${#orphan_array_ids[@]} == 0 && ${#accounted_array_ids[@]} != 0 )); then
  die "historical unregistered fixed-wave identity exists; resubmission is forbidden"
fi
if (( ${#orphan_array_ids[@]} == 1 )); then
  for accounted_array_id in "${accounted_array_ids[@]}"; do
    [[ "${accounted_array_id}" = "${orphan_array_ids[0]}" ]] \
      || die "ambiguous historical scheduler identity forbids crash-window recovery"
  done
fi

submission_output=""
held_array_job_id=""
submitted_cluster="${configured_cluster}"
if (( ${#orphan_array_ids[@]} == 1 )); then
  held_array_job_id="${orphan_array_ids[0]}"
  orphan_state="$(
    classify_scheduler_array \
      "${held_array_job_id}" "${wave_job_name}" "${array_spec}" "${walltime}"
  )" || die "orphan fixed-wave scheduler identity is malformed"
  [[ "${orphan_state}" = "HELD" ]] \
    || die "unregistered fixed wave was externally released; no replacement is permitted"
else
  submission_output="$(
    sbatch \
      --parsable \
      --hold \
      --job-name="${wave_job_name}" \
      --account=sigroup \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task=8 \
      --mem=64G \
      --gpus-per-node=1 \
      --chdir="${repo_root}" \
      --partition="${partition}" \
      --qos=l20_qos \
      --time="${walltime}" \
      --no-requeue \
      --signal=B:USR1@120 \
      --array="${array_spec}" \
      --output="${slurm_log_dir}/%x-%A_%a.out" \
      --export="${export_spec}" \
      "${repo_root}/scripts/hpc4/phase2_confirmatory.sbatch"
  )" || die "fixed-wave sbatch failed without consuming or replacing any seed attempt"
  [[ "${submission_output}" != *$'\n'* && "${submission_output}" != *$'\r'* ]] \
    || die "sbatch --parsable returned multiple lines"
  held_array_job_id="${submission_output%%;*}"
  if [[ "${submission_output}" = *";"* ]]; then
    submitted_cluster="${submission_output#*;}"
  fi
  [[ "${held_array_job_id}" =~ ^[1-9][0-9]*$ ]] \
    || die "sbatch did not return one numeric held fixed-wave array ID"
  [[ "${submitted_cluster}" = "${configured_cluster}" \
    || "${submitted_cluster}" = "unreported" ]] \
    || die "sbatch cluster identity differs from scontrol ClusterName"
  [[ "${submitted_cluster}" != "unreported" ]] \
    || submitted_cluster="${configured_cluster}"
fi

scheduler_request_staging="$(
  mktemp "${registry_staging}/scheduler-request-wave-${wave_index}.XXXXXX"
)"
held_state="$(
  classify_scheduler_array \
    "${held_array_job_id}" "${wave_job_name}" "${array_spec}" "${walltime}" \
    "${scheduler_request_staging}"
)" || die "held fixed wave differs from the immutable scheduler identity"
[[ "${held_state}" = "HELD" ]] \
  || die "fixed wave was not held before its registry commitment"
scheduler_request_sha256="$(
  sha256sum -- "${scheduler_request_staging}" | awk '{print $1}'
)"
[[ "${scheduler_request_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "held scheduler request evidence has an invalid SHA256"

submission_record="${registry_submissions}/array-${held_array_job_id}.json"
submission_staging="$(
  mktemp "${registry_staging}/submission-array-${held_array_job_id}.XXXXXX"
)"
python3 -I -S - \
  "${submission_staging}" "${campaign_plan}" "${campaign_plan_sha256}" \
  "${wave_admission_sha256}" "${wave_index}" \
  "${held_array_job_id}" "${submitted_cluster}" \
  "${scheduler_request_staging}" "${scheduler_request_sha256}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


(
    output_raw,
    plan_raw,
    plan_sha,
    admission_sha,
    wave_raw,
    array_job_id,
    cluster,
    scheduler_request_raw,
    scheduler_request_sha,
) = sys.argv[1:]
output = Path(output_raw)
plan_path = Path(plan_raw)
plan_bytes = plan_path.read_bytes()
if hashlib.sha256(plan_bytes).hexdigest() != plan_sha:
    raise SystemExit("campaign plan changed before wave binding")
plan = json.loads(plan_bytes)
wave_index = int(wave_raw)
wave = plan["waves"][wave_index]
admission_path = plan_path.parent / "admissions" / f"wave-{wave_index}.json"
if (
    admission_path.is_symlink()
    or not admission_path.is_file()
    or hashlib.sha256(admission_path.read_bytes()).hexdigest() != admission_sha
):
    raise SystemExit("wave admission receipt changed before submission binding")
scheduler_request_path = Path(scheduler_request_raw)
scheduler_request_bytes = scheduler_request_path.read_bytes()
if hashlib.sha256(scheduler_request_bytes).hexdigest() != scheduler_request_sha:
    raise SystemExit("held scheduler request evidence changed before submission binding")
scheduler_request = json.loads(scheduler_request_bytes)
entries = [
    {
        "seed": seed,
        "attempt_index": 1,
        "array_job_id": array_job_id,
        "array_task_id": task,
    }
    for task, seed in zip(wave["array_task_ids"], wave["seeds"], strict=True)
]
payload = {
    "schema_version": "prorm-phase2-campaign-submission/v3",
    "status": "committed_while_slurm_held",
    "campaign_plan_sha256": plan_sha,
    "wave_admission_sha256": admission_sha,
    "scheduler_request_sha256": scheduler_request_sha,
    "scheduler_request": scheduler_request,
    "wave_index": wave_index,
    "phase2_design_sha256": plan["phase2_design_sha256"],
    "base_config_hash": plan["base_config_hash"],
    "git_commit": plan["git_commit"],
    "accepted_freeze_aggregate_sha256": plan[
        "accepted_freeze_aggregate_sha256"
    ],
    "array_job_id": array_job_id,
    "submitted_cluster": cluster,
    "array_spec": wave["array_spec"],
    "attempt_index": 1,
    "entries": entries,
    "job_tuple": plan["job_tuple"],
    "producer": plan["producer"],
    "replacement_seed_allowed": False,
    "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with output.open("w", encoding="utf-8", newline="\n") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
[[ ! -e "${submission_record}" && ! -L "${submission_record}" ]] \
  || die "fixed-wave submission record already exists"
mv -T --no-clobber -- "${submission_staging}" "${submission_record}" \
  || die "failed to atomically bind the held fixed wave"
python3 -I -S - "${submission_record}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
for target in (path, path.parent):
    flags = os.O_RDONLY
    if target.is_dir():
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(target, flags | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY
python3 -I -S \
  "${repo_root}/scripts/hpc4/resolve_phase2_campaign_registry.py" \
  "${formal_design_root}" "${phase2_design_sha256}" \
  "${base_config_hash}" "${git_commit}" "${PRORM_IMAGE_SHA256}" \
  "${inventory_sha256}" --state >/dev/null \
  || die "committed fixed-wave submission failed full registry revalidation"
scontrol release "${held_array_job_id}" \
  || die "fixed-wave registry committed but held array release failed"
flock -u "${registry_lock_fd}" \
  || die "failed to release the formal campaign registry lock"
printf 'SUBMITTED;%s;%s;%s\n' \
  "${wave_index}" "${held_array_job_id}" "${configured_cluster}"
