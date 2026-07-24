#!/usr/bin/env bash
set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -lt 5 || $# -gt 6 ]]; then
  die "usage: $0 <overlay.yaml> <base.yaml> <accepted-freeze-aggregate.json> <gpu-partition> <walltime> [0-29]"
fi

overlay_input="$1"
base_input="$2"
accepted_freeze_input="$3"
partition="$4"
walltime="$5"
shift 5
array_selection=""
if (( $# == 1 )); then
  array_selection="$1"
fi

decimal_exceeds() {
  local value="$1" limit="$2"
  if (( ${#value} != ${#limit} )); then
    (( ${#value} > ${#limit} ))
    return
  fi
  [[ "${value}" > "${limit}" ]]
}

formal_array_shape_is_valid() {
  local start="$1" end="$2" count="$3"
  (( count == 30 && start == 0 && end == count - 1 ))
}

max_safe_array_integer=2147483647
array_selection_supplied=0
array_start=0
array_end=0
if [[ -n "${array_selection}" ]]; then
  array_selection_supplied=1
  if [[ "${array_selection}" =~ ^(0|[1-9][0-9]*)(-(0|[1-9][0-9]*))?$ ]]; then
    array_start_text="${BASH_REMATCH[1]}"
    array_end_text="${BASH_REMATCH[3]:-${array_start_text}}"
  else
    die "array selection must be one index or one contiguous start-end range"
  fi
  if decimal_exceeds "${array_start_text}" "${max_safe_array_integer}" \
    || decimal_exceeds "${array_end_text}" "${max_safe_array_integer}"; then
    die "array selection index exceeds safe integer limit ${max_safe_array_integer}"
  fi
  array_start=$((10#${array_start_text}))
  array_end=$((10#${array_end_text}))
  (( array_start <= array_end )) || die "array selection start exceeds its end"
fi
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
  python3 sbatch scontrol scancel flock realpath sha256sum awk git mktemp mv; do
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
if schema != "prorm-common-beta-config/v2":
    raise SystemExit("overlay is not the Phase-2 v2 schema")
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
PY
)
[[ "${#identity_info[@]}" -eq 8 ]] \
  || die "failed to resolve committed confirmatory identities"
seed_count="${identity_info[0]}"
phase2_design_sha256="${identity_info[1]}"
base_config_hash="${identity_info[2]}"
overlay_file_sha256="${identity_info[3]}"
base_file_sha256="${identity_info[4]}"
identities_file_sha256="${identity_info[5]}"
bound_freeze_sha256="${identity_info[6]}"
frozen_global_beta="${identity_info[7]}"
[[ "${seed_count}" = "30" ]] || die "invalid committed formal seed count"
for digest in \
  "${phase2_design_sha256}" "${base_config_hash}" "${overlay_file_sha256}" \
  "${base_file_sha256}" "${identities_file_sha256}" "${bound_freeze_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid committed SHA256 identity"
done

if (( ! array_selection_supplied )); then
  array_end=$((seed_count - 1))
fi
(( array_end < seed_count )) \
  || die "array selection exceeds formal seed indices 0-$((seed_count - 1))"
if ! formal_array_shape_is_valid "${array_start}" "${array_end}" "${seed_count}"; then
  die "formal campaign must submit the exact complete seed array 0-$((seed_count - 1))"
fi

if [[ -n "${PRORM_PHASE2_ARRAY_CONCURRENCY:-}" ]]; then
  concurrency="${PRORM_PHASE2_ARRAY_CONCURRENCY}"
else
  concurrency=2
fi
[[ "${concurrency}" =~ ^[12]$ ]] \
  || die "PRORM_PHASE2_ARRAY_CONCURRENCY must be 1 or 2 on HPC4 gpu-l20"
array_spec="${array_start}-${array_end}%${concurrency}"

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
if (
    value.get("schema_version") != "common-beta-pilot-selection-aggregate/v1"
    or value.get("pilot_phase") != "freeze"
    or value.get("formal_eligibility") is not False
    or value.get("supports_formal_claim") is not False
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

# This login-node check is advisory: two submissions can race after it.  The
# compute-node mkdir of attempt-N is the authoritative atomic claim.
formal_runs_root="${project_root}/runs"
formal_phase2_root="${formal_runs_root}/phase2-confirmatory"
formal_design_root="${formal_phase2_root}/${phase2_design_sha256}"
validate_existing_formal_directory "project runs root" "${formal_runs_root}"
validate_existing_formal_directory "formal Phase-2 runs root" "${formal_phase2_root}"
validate_existing_formal_directory "formal design root" "${formal_design_root}"
for (( candidate_index = array_start; candidate_index <= array_end; candidate_index++ )); do
  candidate_seed=$((20260901 + candidate_index))
  candidate_seed_root="${formal_design_root}/seed-${candidate_seed}"
  candidate_attempt_root="${candidate_seed_root}/attempt-${attempt_index}"
  validate_existing_formal_directory "formal seed root" "${candidate_seed_root}"
  [[ ! -e "${candidate_attempt_root}" && ! -L "${candidate_attempt_root}" ]] \
    || die "formal seed attempt already exists; refusing a duplicate attempt: ${candidate_attempt_root}"
done

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
registry_submissions="${campaign_registry}/submissions"
registry_executions="${campaign_registry}/executions"
registry_recoveries="${campaign_registry}/recoveries"
registry_scheduler_terminals="${campaign_registry}/scheduler-terminals"
registry_staging="${campaign_registry}/.staging"
ensure_submission_directory "formal campaign registry" "${campaign_registry}"
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
export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_REPO_ROOT=${repo_root},PRORM_PHASE2_OVERLAY=${overlay},PRORM_PHASE2_BASE_CONFIG=${base_config},PRORM_PHASE2_DESIGN_SHA256=${phase2_design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_config_hash},PRORM_PHASE2_OVERLAY_FILE_SHA256=${overlay_file_sha256},PRORM_PHASE2_BASE_FILE_SHA256=${base_file_sha256},PRORM_IDENTITIES_FILE_SHA256=${identities_file_sha256},PRORM_GIT_COMMIT=${git_commit},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE=${accepted_freeze},PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256=${accepted_freeze_sha256},PRORM_PHASE2_FROZEN_GLOBAL_BETA=${frozen_global_beta},PRORM_PHASE2_ATTEMPT_INDEX=${attempt_index},PRORM_PHASE2_CAMPAIGN_REGISTRY=${campaign_registry}"
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

# A SIGKILL after the immutable registry commit but before `scontrol release`
# must not create a second held array.  Under the campaign lock, first recover
# the one exact identity-bound submission and resume only that scheduler job.
mapfile -t committed_submission_info < <(
  python3 -I -S - \
    "${registry_submissions}" "${phase2_design_sha256}" \
    "${base_config_hash}" "${git_commit}" "${accepted_freeze_sha256}" \
    "${array_spec}" "${walltime}" "${overlay_file_sha256}" \
    "${base_file_sha256}" "${identities_file_sha256}" \
    "${PRORM_IMAGE_SHA256}" "${inventory_sha256}" \
    "${confirmatory_job_file_sha256}" <<'PY'
import json
import re
import sys
from pathlib import Path


(
    submissions_raw,
    design_sha,
    base_hash,
    git_commit,
    freeze_sha,
    array_spec,
    walltime,
    overlay_file_sha,
    base_file_sha,
    identities_file_sha,
    image_sha,
    inventory_sha,
    job_file_sha,
) = sys.argv[1:]
submissions = Path(submissions_raw)


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


paths = sorted(submissions.iterdir())
if not paths:
    print("NONE")
    raise SystemExit(0)
if len(paths) != 1 or re.fullmatch(r"array-[1-9][0-9]*\.json", paths[0].name) is None:
    raise SystemExit("campaign has an unexpected or non-unique committed submission")
path = paths[0]
if path.is_symlink() or not path.is_file():
    raise SystemExit("committed campaign submission is unsafe")
value = load_json(path)
expected_fields = {
    "schema_version",
    "status",
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "accepted_freeze_aggregate_sha256",
    "array_job_id",
    "submitted_cluster",
    "array_spec",
    "attempt_index",
    "entries",
    "job_tuple",
    "producer",
    "replacement_seed_allowed",
    "created_at_utc",
}
job_tuple = value.get("job_tuple")
expected_job_tuple = {
    "account": "sigroup",
    "partition": "gpu-l20",
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
}
expected_producer = {
    "overlay_file_sha256": overlay_file_sha,
    "base_file_sha256": base_file_sha,
    "identities_file_sha256": identities_file_sha,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
}
array_job_id = value.get("array_job_id")
expected_entries = [
    {
        "seed": 20260901 + task_id,
        "attempt_index": 1,
        "array_job_id": array_job_id,
        "array_task_id": task_id,
    }
    for task_id in range(30)
]
if (
    set(value) != expected_fields
    or value.get("schema_version") != "prorm-phase2-campaign-submission/v1"
    or value.get("status") != "committed_while_slurm_held"
    or value.get("phase2_design_sha256") != design_sha
    or value.get("base_config_hash") != base_hash
    or value.get("git_commit") != git_commit
    or value.get("accepted_freeze_aggregate_sha256") != freeze_sha
    or value.get("submitted_cluster") != "hpc4"
    or value.get("array_spec") != array_spec
    or value.get("attempt_index") != 1
    or value.get("entries") != expected_entries
    or value.get("replacement_seed_allowed") is not False
    or not isinstance(array_job_id, str)
    or re.fullmatch(r"[1-9][0-9]*", array_job_id) is None
    or path.name != f"array-{array_job_id}.json"
    or job_tuple != expected_job_tuple
    or value.get("producer") != expected_producer
    or not isinstance(value.get("created_at_utc"), str)
    or not value["created_at_utc"]
):
    raise SystemExit("committed campaign submission differs from this exact invocation")
print(array_job_id)
PY
)
[[ "${#committed_submission_info[@]}" -eq 1 ]] \
  || die "failed to classify the committed formal campaign submission"
committed_array_job_id="${committed_submission_info[0]}"
if [[ "${committed_array_job_id}" != "NONE" ]]; then
  committed_scheduler_record="$(
    scontrol show job --oneliner "${committed_array_job_id}"
  )" || die "committed held-array submission is missing from Slurm"
  mapfile -t committed_scheduler_state < <(
    python3 -I -S - \
      "${committed_scheduler_record}" "${committed_array_job_id}" \
      "${array_spec}" "${repo_root}" <<'PY'
import sys
from pathlib import Path


record, array_job_id, array_spec, repo_root = sys.argv[1:]
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
    if (
        fields.get("ArrayJobId", fields.get("JobId")) != array_job_id
        or fields.get("JobName") != "prorm-phase2-confirmatory"
        or fields.get("Account") != "sigroup"
        or fields.get("Partition") != "gpu-l20"
        or fields.get("Requeue") != "0"
        or fields.get("NumNodes") != "1"
        or fields.get("NumCPUs") != "8"
        or fields.get("Command") != expected_command
        or fields.get("WorkDir") != repo_root
        or not fields.get("JobState")
    ):
        raise SystemExit("Slurm job differs from the committed held-array identity")
    parsed.append(fields)
held_master = (
    len(parsed) == 1
    and parsed[0].get("ArrayJobId", parsed[0].get("JobId")) == array_job_id
    and parsed[0].get("ArrayTaskId") == array_spec
    and parsed[0].get("JobState") == "PENDING"
    and parsed[0].get("Reason") == "JobHeldUser"
)
if held_master:
    print("HELD")
else:
    if any(
        fields.get("JobState") == "PENDING"
        and str(fields.get("Reason", "")).startswith("JobHeld")
        for fields in parsed
    ):
        raise SystemExit("committed array is held by an unexpected scheduler authority")
    print("ALREADY_RELEASED")
PY
  )
  [[ "${#committed_scheduler_state[@]}" -eq 1 ]] \
    || die "failed to verify the committed held array in Slurm"
  if [[ "${committed_scheduler_state[0]}" = "HELD" ]]; then
    scontrol release "${committed_array_job_id}" \
      || die "failed to resume-release the committed held array"
  elif [[ "${committed_scheduler_state[0]}" != "ALREADY_RELEASED" ]]; then
    die "invalid committed held-array scheduler state"
  fi
  flock -u "${registry_lock_fd}" \
    || die "failed to release the formal campaign registry lock"
  printf '%s;%s\n' "${committed_array_job_id}" "${configured_cluster}"
  exit 0
fi
held_array_job_id=""
held_array_released=0
cleanup_held_array() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${held_array_job_id}" && "${held_array_released}" = "0" ]]; then
    scancel -- "${held_array_job_id}" >/dev/null 2>&1 || true
  fi
  flock -u "${registry_lock_fd}" >/dev/null 2>&1 || true
  exit "${status}"
}
trap cleanup_held_array EXIT INT TERM

submission_output="$(
  sbatch \
    --parsable \
    --hold \
    --account=sigroup \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --gpus-per-node=1 \
    --chdir="${repo_root}" \
    --partition="${partition}" \
    --time="${walltime}" \
    --no-requeue \
    --signal=B:USR1@120 \
    --array="${array_spec}" \
    --output="${slurm_log_dir}/%x-%A_%a.out" \
    --export="${export_spec}" \
    "${repo_root}/scripts/hpc4/phase2_confirmatory.sbatch"
)"
[[ "${submission_output}" != *$'\n'* && "${submission_output}" != *$'\r'* ]] \
  || die "sbatch --parsable returned multiple lines"
held_array_job_id="${submission_output%%;*}"
if [[ "${submission_output}" = *";"* ]]; then
  submitted_cluster="${submission_output#*;}"
else
  submitted_cluster="unreported"
fi
[[ "${held_array_job_id}" =~ ^[1-9][0-9]*$ ]] \
  || die "sbatch did not return one numeric held array job ID"
[[ "${submitted_cluster}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || die "sbatch returned an unsafe cluster identity"
if [[ "${submitted_cluster}" = "unreported" ]]; then
  submitted_cluster="${configured_cluster}"
else
  [[ "${submitted_cluster}" = "${configured_cluster}" ]] \
    || die "sbatch cluster identity differs from scontrol ClusterName"
fi

submission_record="${registry_submissions}/array-${held_array_job_id}.json"
submission_staging="$(
  mktemp "${registry_staging}/submission-array-${held_array_job_id}.XXXXXX"
)"
python3 -I -S - \
  "${registry_submissions}" "${submission_staging}" \
  "${phase2_design_sha256}" "${base_config_hash}" "${git_commit}" \
  "${accepted_freeze_sha256}" "${held_array_job_id}" "${submitted_cluster}" \
  "${array_spec}" "${walltime}" "${array_start}" "${array_end}" \
  "${attempt_index}" \
  "${overlay_file_sha256}" "${base_file_sha256}" \
  "${identities_file_sha256}" "${PRORM_IMAGE_SHA256}" \
  "${inventory_sha256}" "${confirmatory_job_file_sha256}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


(
    submissions_raw,
    output_raw,
    design_sha,
    base_hash,
    git_commit,
    freeze_sha,
    array_job_id,
    submitted_cluster,
    array_spec,
    walltime,
    start_raw,
    end_raw,
    attempt_raw,
    overlay_file_sha,
    base_file_sha,
    identities_file_sha,
    image_sha,
    inventory_sha,
    job_file_sha,
) = sys.argv[1:]
submissions = Path(submissions_raw)
output = Path(output_raw)
start = int(start_raw)
end = int(end_raw)
attempt_index = int(attempt_raw)
hex_characters = frozenset("0123456789abcdef")


def digest(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in hex_characters for character in value)
    ):
        raise SystemExit(f"{name} is not a lowercase SHA256")
    return value


def reject_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


for value, name in (
    (design_sha, "design_sha"),
    (base_hash, "base_hash"),
    (freeze_sha, "freeze_sha"),
    (overlay_file_sha, "overlay_file_sha"),
    (base_file_sha, "base_file_sha"),
    (identities_file_sha, "identities_file_sha"),
    (image_sha, "image_sha"),
    (inventory_sha, "inventory_sha"),
    (job_file_sha, "job_file_sha"),
):
    digest(value, name)
if len(git_commit) not in {40, 64} or any(
    character not in hex_characters for character in git_commit
):
    raise SystemExit("git_commit is not a full lowercase object identity")

occupied = {}
submission_paths = sorted(submissions.glob("array-*.json"))
expected_submission_fields = {
    "schema_version",
    "status",
    "phase2_design_sha256",
    "base_config_hash",
    "git_commit",
    "accepted_freeze_aggregate_sha256",
    "array_job_id",
    "submitted_cluster",
    "array_spec",
    "attempt_index",
    "entries",
    "job_tuple",
    "producer",
    "replacement_seed_allowed",
    "created_at_utc",
}
expected_job_tuple = {
    "account": "sigroup",
    "partition": "gpu-l20",
    "nodes": 1,
    "tasks": 1,
    "cpus_per_task": 8,
    "memory": "64G",
    "gpus_per_node": 1,
    "no_requeue": True,
    "held_before_registry_commit": True,
    "script": "scripts/hpc4/phase2_confirmatory.sbatch",
    "script_file_sha256": job_file_sha,
}
expected_producer = {
    "overlay_file_sha256": overlay_file_sha,
    "base_file_sha256": base_file_sha,
    "identities_file_sha256": identities_file_sha,
    "image_sha256": image_sha,
    "hf_inventory_sha256": inventory_sha,
}
for path in submission_paths:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe campaign submission record: {path}")
    value = load_json(path)
    job_tuple = value.get("job_tuple")
    producer = value.get("producer")
    if (
        set(value) != expected_submission_fields
        or value.get("schema_version") != "prorm-phase2-campaign-submission/v1"
        or value.get("status") != "committed_while_slurm_held"
        or value.get("phase2_design_sha256") != design_sha
        or value.get("base_config_hash") != base_hash
        or value.get("git_commit") != git_commit
        or value.get("accepted_freeze_aggregate_sha256") != freeze_sha
        or value.get("replacement_seed_allowed") is not False
        or not isinstance(value.get("attempt_index"), int)
        or isinstance(value.get("attempt_index"), bool)
        or value["attempt_index"] != 1
        or not isinstance(value.get("array_job_id"), str)
        or not value["array_job_id"].isdigit()
        or value["array_job_id"].startswith("0")
        or path.name != f"array-{value['array_job_id']}.json"
        or value.get("submitted_cluster") != "hpc4"
        or value.get("array_spec") not in {"0-29%1", "0-29%2"}
        or not isinstance(value.get("created_at_utc"), str)
        or not value["created_at_utc"]
        or not isinstance(job_tuple, dict)
        or set(job_tuple) != set(expected_job_tuple) | {"walltime"}
        or any(
            job_tuple.get(key) != expected_value
            for key, expected_value in expected_job_tuple.items()
        )
        or not isinstance(job_tuple.get("walltime"), str)
        or not job_tuple["walltime"]
        or producer != expected_producer
    ):
        raise SystemExit(f"incompatible or malformed campaign submission record: {path}")
    entries = value.get("entries")
    expected_entries = [
        {
            "seed": 20260901 + task_id,
            "attempt_index": 1,
            "array_job_id": value["array_job_id"],
            "array_task_id": task_id,
        }
        for task_id in range(30)
    ]
    if entries != expected_entries:
        raise SystemExit(f"campaign submission is not exact ordered tasks 0 through 29: {path}")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {"seed", "attempt_index", "array_job_id", "array_task_id"}
            or entry.get("attempt_index") != value["attempt_index"]
            or entry.get("array_job_id") != value["array_job_id"]
            or not isinstance(entry.get("array_task_id"), int)
            or isinstance(entry.get("array_task_id"), bool)
            or not 0 <= entry["array_task_id"] <= 29
            or entry.get("seed") != 20260901 + entry["array_task_id"]
        ):
            raise SystemExit(f"campaign submission entry is malformed: {path}")
        key = (entry.get("seed"), entry.get("attempt_index"))
        if key in occupied:
            raise SystemExit(f"campaign registry repeats formal attempt {key}")
        occupied[key] = (path, value)

entries = [
    {
        "seed": 20260901 + task_id,
        "attempt_index": attempt_index,
        "array_job_id": array_job_id,
        "array_task_id": task_id,
    }
    for task_id in range(start, end + 1)
]
for entry in entries:
    key = (entry["seed"], entry["attempt_index"])
    if key in occupied:
        raise SystemExit(
            f"formal attempt is already committed in campaign registry: {key}"
        )

if (
    attempt_index != 1
    or start != 0
    or end != 29
    or len(entries) != 30
    or [entry["array_task_id"] for entry in entries] != list(range(30))
):
    raise SystemExit(
        "initial campaign registry commit must reserve exact ordered tasks 0 through 29"
    )

payload = {
    "schema_version": "prorm-phase2-campaign-submission/v1",
    "status": "committed_while_slurm_held",
    "phase2_design_sha256": design_sha,
    "base_config_hash": base_hash,
    "git_commit": git_commit,
    "accepted_freeze_aggregate_sha256": freeze_sha,
    "array_job_id": array_job_id,
    "submitted_cluster": submitted_cluster,
    "array_spec": array_spec,
    "attempt_index": attempt_index,
    "entries": entries,
    "job_tuple": {
        "account": "sigroup",
        "partition": "gpu-l20",
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
  || die "held array submission record already exists"
mv -T --no-clobber -- "${submission_staging}" "${submission_record}" \
  || die "failed to atomically commit the held array campaign registry record"
python3 -I -S - "${submission_record}" <<'PY'
import os
import sys
from pathlib import Path


path = Path(sys.argv[1])
descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory)
finally:
    os.close(directory)
PY
submission_record_sha256="$(sha256sum -- "${submission_record}" | awk '{print $1}')"
[[ "${submission_record_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "committed held-array registry record has an invalid SHA256"
scontrol release "${held_array_job_id}" \
  || die "campaign registry committed but held array release failed"
held_array_released=1
flock -u "${registry_lock_fd}" \
  || die "failed to release the formal campaign registry lock"
trap - EXIT INT TERM
printf '%s\n' "${submission_output}"
