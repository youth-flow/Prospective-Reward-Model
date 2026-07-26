#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -lt 4 ]]; then
  die "usage: $0 <overlay.yaml> <base.yaml> <gpu-partition> <walltime> [<index>|<start>-<end>] [--beta-source-aggregate <json>] [--horizon-parent-aggregate <json>]"
fi

overlay_input="$1"
base_input="$2"
partition="$3"
walltime="$4"
shift 4
array_selection=""
beta_source_aggregate_input=""
horizon_parent_aggregate_input=""
while (( $# )); do
  case "$1" in
    --beta-source-aggregate)
      [[ $# -ge 2 && -z "${beta_source_aggregate_input}" ]] \
        || die "--beta-source-aggregate requires exactly one path"
      beta_source_aggregate_input="$2"
      shift 2
      ;;
    --horizon-parent-aggregate)
      [[ $# -ge 2 && -z "${horizon_parent_aggregate_input}" ]] \
        || die "--horizon-parent-aggregate requires exactly one path"
      horizon_parent_aggregate_input="$2"
      shift 2
      ;;
    --*)
      die "unknown Phase-2 pilot option: $1"
      ;;
    *)
      [[ -z "${array_selection}" ]] \
        || die "only one array index or range may be supplied"
      array_selection="$1"
      shift
      ;;
  esac
done

decimal_exceeds() {
  local value="$1" limit="$2"
  if (( ${#value} != ${#limit} )); then
    (( ${#value} > ${#limit} ))
    return
  fi
  [[ "${value}" > "${limit}" ]]
}

# Validate text before any Bash arithmetic expansion.  This prevents a very
# large decimal array index from wrapping into a valid pilot index.
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
    die "Phase-2 design is locked to HPC4 gpu-l20; refusing GPU partition: ${partition}"
    ;;
esac
[[ "${walltime}" =~ ^[0-9]+-[0-9]{2}:[0-9]{2}:[0-9]{2}$|^[0-9]{2}:[0-9]{2}:[0-9]{2}$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

# Historical v2 replay remains below unchanged.  A post-recovery v1 overlay
# is routed into the authorization-bound v3 control plane before any legacy
# identity parsing can reinterpret it as a v2 pilot.
if [[ -f "${overlay_input}" ]] \
  && grep -Eq \
    '^schema_version:[[:space:]]*prorm-common-beta-post-recovery-experiment/v1[[:space:]]*$' \
    "${overlay_input}"; then
  [[ -z "${array_selection}" || "${array_selection}" = "0-2" ]] \
    || die "post-recovery pilot must submit the exact complete array 0-2"
  repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
  submitted_base="$(realpath -e -- "${base_input}")" \
    || die "post-recovery base config cannot be resolved"
  declared_base="$(
    PYTHONPATH="${repo_root}/src" python3 - "${overlay_input}" <<'PY'
import sys
from smart_reward.phase2_config import load_phase2_config_bundle
print(load_phase2_config_bundle(sys.argv[1]).base_config_path)
PY
  )" || die "could not resolve post-recovery declared base"
  declared_base="$(realpath -e -- "${declared_base}")" \
    || die "post-recovery declared base is missing"
  [[ "${submitted_base}" = "${declared_base}" ]] \
    || die "submitted base differs from the post-recovery overlay binding"
  : "${PRORM_PROJECT_ROOT:?PRORM_PROJECT_ROOT is required}"
  authorization="${PRORM_PROJECT_ROOT}/runs/phase2-recovery-pilot/recovery-success-authorization.json"
  command=(
    bash "${repo_root}/scripts/hpc4/submit_phase2_post_recovery_pilot.sh"
    "${overlay_input}" "${authorization}" "${walltime}"
    --legacy-r2-replay
  )
  if [[ -n "${beta_source_aggregate_input}" ]]; then
    command+=(--beta-source-aggregate "${beta_source_aggregate_input}")
  fi
  if [[ -n "${horizon_parent_aggregate_input}" ]]; then
    command+=(--horizon-parent-aggregate "${horizon_parent_aggregate_input}")
  fi
  exec "${command[@]}"
fi

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
  || die "Phase-2 pilot submission requires a clean Git worktree"

overlay="$(realpath -e -- "${overlay_input}")" \
  || die "overlay does not exist or cannot be resolved: ${overlay_input}"
base_config="$(realpath -e -- "${base_input}")" \
  || die "base config does not exist or cannot be resolved: ${base_input}"
for path in "${overlay}" "${base_config}"; do
  [[ -f "${path}" ]] || die "configuration path is not a regular file: ${path}"
  case "${path}" in
    "${repo_root}"/configs/*.yaml) ;;
    *) die "Phase-2 configs must be tracked configs/*.yaml files" ;;
  esac
done
overlay_relative="$(realpath --relative-to="${repo_root}" "${overlay}")"
base_relative="$(realpath --relative-to="${repo_root}" "${base_config}")"
[[ "${overlay_relative}" = "configs/common_beta_pilot.yaml" ]] \
  || die "this entry point accepts only configs/common_beta_pilot.yaml"
[[ "${base_relative}" = "configs/common_beta_pilot_base.yaml" ]] \
  || die "this entry point accepts only configs/common_beta_pilot_base.yaml"
git -C "${repo_root}" ls-files --error-unmatch -- "${overlay_relative}" >/dev/null \
  || die "overlay is not tracked by Git"
git -C "${repo_root}" ls-files --error-unmatch -- "${base_relative}" >/dev/null \
  || die "base config is not tracked by Git"

identity_relative="configs/identities.json"
git -C "${repo_root}" ls-files --error-unmatch -- "${identity_relative}" >/dev/null \
  || die "configs/identities.json is not tracked by Git"
command -v python3 >/dev/null 2>&1 \
  || die "python3 is required to validate committed dual identities"
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
if overlay_entry["seed_count"] != 3:
    raise SystemExit("the Phase-2 pilot identity must declare exactly three seeds")
if overlay_entry["config_hash"] == base_entry["config_hash"]:
    raise SystemExit("overlay design identity and base artifact identity must differ")

# This login-node check intentionally parses only frozen scalar bindings.  The
# complete recursive YAML schema and both semantic hashes are independently
# recomputed inside the clean research container on the allocated GPU node.
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
if not re.search(r"(?m)^  stage:[ \t]*pilot[ \t]*$", overlay_text):
    raise SystemExit("overlay is not explicitly a pilot")
if not re.search(r"(?m)^  formal_eligibility:[ \t]*false[ \t]*$", overlay_text):
    raise SystemExit("pilot overlay must be formally ineligible")

print(overlay_entry["seed_count"])
print(overlay_entry["config_hash"])
print(base_entry["config_hash"])
print(overlay_file_sha)
print(base_file_sha)
print(hashlib.sha256(identity_bytes).hexdigest())
PY
)
[[ "${#identity_info[@]}" -eq 6 ]] || die "failed to resolve dual committed identities"
seed_count="${identity_info[0]}"
phase2_design_sha256="${identity_info[1]}"
base_config_hash="${identity_info[2]}"
overlay_file_sha256="${identity_info[3]}"
base_file_sha256="${identity_info[4]}"
identities_file_sha256="${identity_info[5]}"
[[ "${seed_count}" =~ ^[1-9][0-9]*$ ]] || die "invalid committed pilot seed count"
for digest in \
  "${phase2_design_sha256}" "${base_config_hash}" "${overlay_file_sha256}" \
  "${base_file_sha256}" "${identities_file_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "invalid committed SHA256 identity"
done

if (( ! array_selection_supplied )); then
  array_end=$((seed_count - 1))
fi
(( array_end < seed_count )) \
  || die "array selection exceeds configured seed indices 0-$((seed_count - 1))"

if [[ -n "${PRORM_PHASE2_ARRAY_CONCURRENCY:-}" ]]; then
  concurrency="${PRORM_PHASE2_ARRAY_CONCURRENCY}"
else
  concurrency=2
fi
[[ "${concurrency}" =~ ^[12]$ ]] \
  || die "PRORM_PHASE2_ARRAY_CONCURRENCY must be 1 or 2 for the HPC4 pilot"
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
    file) [[ -f "${resolved}" ]] || die "project path is not a file: ${resolved}" ;;
    directory) [[ -d "${resolved}" ]] || die "project path is not a directory: ${resolved}" ;;
    *) die "invalid internal path kind" ;;
  esac
  printf '%s\n' "${resolved}"
}

image="$(resolve_project_path "${PRORM_IMAGE}" file)"
hf_cache="$(resolve_project_path "${PRORM_HF_CACHE}" directory)"
[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be lowercase SHA256"
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${image}" \
  | sha256sum --check --status \
  || die "image SHA256 mismatch"

inventory_expected="${hf_cache}/inventories/${base_config_hash}.json"
inventory="$(realpath -e -- "${inventory_expected}")" \
  || die "missing base-config HF inventory: ${inventory_expected}"
[[ -f "${inventory}" ]] || die "base-config HF inventory is not a regular file"
case "${inventory}" in
  "${hf_cache}"/inventories/*) ;;
  *) die "base-config inventory escaped HF cache" ;;
esac
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
[[ "${inventory_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to hash base-config HF inventory"

beta_source_aggregate_present=0
beta_source_aggregate=""
beta_source_aggregate_sha256=""
if [[ -n "${beta_source_aggregate_input}" ]]; then
  beta_source_aggregate="$(
    resolve_project_path "${beta_source_aggregate_input}" file
  )"
  beta_source_aggregate_sha256="$(
    sha256sum -- "${beta_source_aggregate}" | awk '{print $1}'
  )"
  [[ "${beta_source_aggregate_sha256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "failed to hash beta-source aggregate"
  beta_source_aggregate_present=1
fi

horizon_parent_aggregate_present=0
horizon_parent_aggregate=""
horizon_parent_aggregate_sha256=""
if [[ -n "${horizon_parent_aggregate_input}" ]]; then
  horizon_parent_aggregate="$(
    resolve_project_path "${horizon_parent_aggregate_input}" file
  )"
  horizon_parent_aggregate_sha256="$(
    sha256sum -- "${horizon_parent_aggregate}" | awk '{print $1}'
  )"
  [[ "${horizon_parent_aggregate_sha256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "failed to hash horizon-parent aggregate"
  horizon_parent_aggregate_present=1
fi

for value in \
  "${project_root}" "${scratch_root}" "${image}" "${hf_cache}" "${repo_root}" \
  "${overlay}" "${base_config}" "${inventory}" \
  "${beta_source_aggregate}" "${horizon_parent_aggregate}"; do
  [[ "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "path contains an unsafe sbatch/export or Apptainer-bind delimiter"
done

# Close the submit/check race.  The compute job repeats these checks and then
# executes only a detached clone of this exact commit.
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" ]] \
  || die "Git HEAD changed while preparing the Phase-2 submission"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "Git worktree changed while preparing the Phase-2 submission"
printf '%s  %s\n' "${overlay_file_sha256}" "${overlay}" \
  | sha256sum --check --status || die "overlay changed during submission"
printf '%s  %s\n' "${base_file_sha256}" "${base_config}" \
  | sha256sum --check --status || die "base config changed during submission"
if (( beta_source_aggregate_present )); then
  printf '%s  %s\n' "${beta_source_aggregate_sha256}" "${beta_source_aggregate}" \
    | sha256sum --check --status \
    || die "beta-source aggregate changed during submission"
fi
if (( horizon_parent_aggregate_present )); then
  printf '%s  %s\n' "${horizon_parent_aggregate_sha256}" "${horizon_parent_aggregate}" \
    | sha256sum --check --status \
    || die "horizon-parent aggregate changed during submission"
fi

slurm_log_dir="${project_root}/slurm-logs"
mkdir -p "${slurm_log_dir}" "${scratch_root}/phase2-pilot-jobs"
export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_REPO_ROOT=${repo_root},PRORM_PHASE2_OVERLAY=${overlay},PRORM_PHASE2_BASE_CONFIG=${base_config},PRORM_PHASE2_DESIGN_SHA256=${phase2_design_sha256},PRORM_PHASE2_BASE_CONFIG_HASH=${base_config_hash},PRORM_PHASE2_OVERLAY_FILE_SHA256=${overlay_file_sha256},PRORM_PHASE2_BASE_FILE_SHA256=${base_file_sha256},PRORM_IDENTITIES_FILE_SHA256=${identities_file_sha256},PRORM_GIT_COMMIT=${git_commit},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_PRESENT=${beta_source_aggregate_present},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_PRESENT=${horizon_parent_aggregate_present}"
if (( beta_source_aggregate_present )); then
  export_spec+=",PRORM_PHASE2_BETA_SOURCE_AGGREGATE=${beta_source_aggregate},PRORM_PHASE2_BETA_SOURCE_AGGREGATE_SHA256=${beta_source_aggregate_sha256}"
fi
if (( horizon_parent_aggregate_present )); then
  export_spec+=",PRORM_PHASE2_HORIZON_PARENT_AGGREGATE=${horizon_parent_aggregate},PRORM_PHASE2_HORIZON_PARENT_AGGREGATE_SHA256=${horizon_parent_aggregate_sha256}"
fi
sbatch \
  --parsable \
  --account=sigroup \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --gpus-per-node=1 \
  --chdir="${repo_root}" \
  --partition="${partition}" \
  --time="${walltime}" \
  --array="${array_spec}" \
  --output="${slurm_log_dir}/%x-%A_%a.out" \
  --export="${export_spec}" \
  "${repo_root}/scripts/hpc4/phase2_pilot.sbatch"
