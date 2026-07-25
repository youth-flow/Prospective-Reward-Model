#!/usr/bin/env bash
set -euo pipefail
umask 027

die() { echo "error: $*" >&2; exit 2; }

readonly OVERLAY_RELATIVE="configs/common_beta_post_recovery_budgeted_end_to_end.yaml"
readonly BASE_RELATIVE="configs/common_beta_post_recovery_budgeted_end_to_end_base.yaml"
readonly MATERIALIZATION_RECEIPT_RELATIVE="configs/.common_beta_post_recovery_budgeted_end_to_end.materialized.json"
readonly AUTHORIZATION_RELATIVE="runs/phase2-recovery-pilot/recovery-success-authorization.json"
readonly SBATCH_RELATIVE="scripts/hpc4/phase2_budgeted_end_to_end.sbatch"
readonly SUBMITTER_RELATIVE="scripts/hpc4/submit_phase2_budgeted_end_to_end_once.py"
readonly ADOPTED_SCHEDULE_SHA256="46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216"

[[ $# -eq 4 ]] \
  || die "usage: $0 <overlay.yaml> <recovery-authorization.json> <accepted-freeze-aggregate.json> <walltime>"
overlay_input="$1"
authorization_input="$2"
freeze_input="$3"
walltime="$4"
[[ "${walltime}" =~ ^([1-9][0-9]*-[0-9]{2}:[0-9]{2}:[0-9]{2}|[0-9]{2}:[0-9]{2}:[0-9]{2})$ ]] \
  || die "walltime must be HH:MM:SS or D-HH:MM:SS"

for name in \
  PRORM_PROJECT_ROOT PRORM_SCRATCH_ROOT PRORM_IMAGE PRORM_IMAGE_SHA256 \
  PRORM_HF_CACHE; do
  [[ -n "${!name:-}" ]] || die "${name} is required"
done
while IFS= read -r variable; do
  case "${variable}" in
    APPTAINER*|SINGULARITY*|SBATCH_*|PRORM_BUDGETED_*|\
    PRORM_RECOVERY_AUTHORIZATION*|PRORM_OPTIMIZER_SCHEDULE*|\
    PRORM_GIT_COMMIT|PRORM_HF_INVENTORY*|PRORM_REPO_ROOT)
      die "unset ambient control variable before budgeted submission: ${variable}"
      ;;
  esac
done < <(compgen -e)

for bootstrap_command in dirname git realpath; do
  command -v "${bootstrap_command}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${bootstrap_command}"
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -d "${repo_root}" && ! -L "${repo_root}" && "${repo_root}" != / ]] \
  || die "repository root must be a canonical real directory"
entrypoint="$(realpath -e -- "${BASH_SOURCE[0]}")" \
  || die "budgeted entrypoint cannot be resolved"
[[ "${entrypoint}" = \
  "${repo_root}/scripts/hpc4/submit_phase2_budgeted_end_to_end.sh" \
  && -f "${entrypoint}" && ! -L "${BASH_SOURCE[0]}" ]] \
  || die "budgeted entrypoint must be the repository's canonical regular file"
[[ "$(git -C "${repo_root}" rev-parse --show-toplevel)" = "${repo_root}" ]] \
  || die "entrypoint is not inside the repository root"
git_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"
[[ "${git_commit}" =~ ^[0-9a-f]{40,64}$ ]] || die "Git HEAD is not a full object ID"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "budgeted submission requires a clean committed worktree"

for command_name in \
  python3 realpath sha256sum awk sbatch scontrol squeue sacct id; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

canonical_root() {
  local name="$1" raw="${!1}" resolved
  [[ "${raw}" = /* ]] || die "${name} must be absolute"
  resolved="$(realpath -e -- "${raw}")" || die "${name} cannot be resolved"
  [[ -d "${resolved}" && ! -L "${raw}" && "${resolved}" = "${raw}" \
    && "${resolved}" != / ]] || die "${name} must be a canonical real directory"
  printf '%s\n' "${resolved}"
}

project_path() {
  local raw="$1" kind="$2" name="$3" candidate resolved
  if [[ "${raw}" = /* ]]; then
    candidate="${raw}"
  else
    candidate="${project_root}/${raw}"
  fi
  resolved="$(realpath -e -- "${candidate}")" || die "${name} cannot be resolved"
  case "${resolved}" in
    "${project_root}"/*) ;;
    *) die "${name} escaped PRORM_PROJECT_ROOT" ;;
  esac
  [[ "${resolved}" = "${candidate}" ]] || die "${name} must be canonical"
  case "${kind}" in
    file)
      [[ -f "${resolved}" && ! -L "${candidate}" ]] \
        || die "${name} must be a regular non-symlink file"
      ;;
    directory)
      [[ -d "${resolved}" && ! -L "${candidate}" ]] \
        || die "${name} must be a real non-symlink directory"
      ;;
    *) die "invalid project path kind: ${kind}" ;;
  esac
  printf '%s\n' "${resolved}"
}

repo_file() {
  local expected_relative="$1" input="$2" name="$3" expected resolved
  expected="${repo_root}/${expected_relative}"
  resolved="$(realpath -e -- "${input}")" || die "${name} cannot be resolved"
  [[ "${resolved}" = "${expected}" && -f "${resolved}" && ! -L "${input}" ]] \
    || die "${name} must be exactly ${expected_relative}"
  printf '%s\n' "${resolved}"
}

reject_export_value() {
  local name="$1" value="$2"
  [[ -n "${value}" && "${value}" =~ ^[A-Za-z0-9_./+-]+$ \
    && "${value}" != *","* && "${value}" != *":"* && "${value}" != *"="* \
    && "${value}" != *$'\n'* && "${value}" != *$'\r'* ]] \
    || die "${name} is unsafe for Slurm export/container binding"
}

require_disjoint_roots() {
  local left="$1" right="$2"
  case "${left}" in
    "${right}"|"${right}"/*)
      die "repository, project, and scratch roots must be disjoint"
      ;;
  esac
}

project_root="$(canonical_root PRORM_PROJECT_ROOT)"
scratch_root="$(canonical_root PRORM_SCRATCH_ROOT)"
require_disjoint_roots "${project_root}" "${scratch_root}"
require_disjoint_roots "${scratch_root}" "${project_root}"
require_disjoint_roots "${repo_root}" "${project_root}"
require_disjoint_roots "${project_root}" "${repo_root}"
require_disjoint_roots "${repo_root}" "${scratch_root}"
require_disjoint_roots "${scratch_root}" "${repo_root}"

image="$(project_path "${PRORM_IMAGE}" file "container image")"
hf_cache="$(project_path "${PRORM_HF_CACHE}" directory "Hugging Face cache")"
authorization="$(project_path "${authorization_input}" file "recovery authorization")"
[[ "${authorization}" = "${project_root}/${AUTHORIZATION_RELATIVE}" ]] \
  || die "authorization must be the locked recovery success receipt"
aggregates_root="$(project_path "aggregates" directory "production aggregate root")"
freeze_evidence="$(project_path "${freeze_input}" file "accepted freeze aggregate")"
[[ "$(dirname -- "${freeze_evidence}")" = "${aggregates_root}" ]] \
  || die "accepted freeze must be a direct production aggregate"

overlay="$(repo_file "${OVERLAY_RELATIVE}" "${overlay_input}" "budgeted overlay")"
base="$(repo_file "${BASE_RELATIVE}" "${repo_root}/${BASE_RELATIVE}" "budgeted base")"
materialization_receipt="$(
  repo_file \
    "${MATERIALIZATION_RECEIPT_RELATIVE}" \
    "${repo_root}/${MATERIALIZATION_RECEIPT_RELATIVE}" \
    "budgeted materialization receipt"
)"
overlay_sha256="$(sha256sum -- "${overlay}" | awk '{print $1}')"
base_sha256="$(sha256sum -- "${base}" | awk '{print $1}')"
materialization_receipt_sha256="$(
  sha256sum -- "${materialization_receipt}" | awk '{print $1}'
)"
authorization_sha256="$(sha256sum -- "${authorization}" | awk '{print $1}')"
freeze_evidence_sha256="$(sha256sum -- "${freeze_evidence}" | awk '{print $1}')"

[[ "${PRORM_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "PRORM_IMAGE_SHA256 must be a lowercase SHA256"
[[ "$(sha256sum -- "${image}" | awk '{print $1}')" = "${PRORM_IMAGE_SHA256}" ]] \
  || die "container image SHA256 mismatch"
for name in \
  project_root scratch_root repo_root image hf_cache authorization \
  aggregates_root freeze_evidence overlay base materialization_receipt \
  git_commit; do
  reject_export_value "${name}" "${!name}"
done

critical_paths=(
  "${OVERLAY_RELATIVE}"
  "${BASE_RELATIVE}"
  "${MATERIALIZATION_RECEIPT_RELATIVE}"
  "pyproject.toml"
  "src/smart_reward/config.py"
  "src/smart_reward/cli.py"
  "src/smart_reward/phase2_config.py"
  "src/smart_reward/phase2_inputs.py"
  "src/smart_reward/phase2_training.py"
  "src/smart_reward/phase2_rollout.py"
  "src/smart_reward/phase2_heldout.py"
  "src/smart_reward/phase2_pilot_aggregate.py"
  "src/smart_reward/phase2_post_recovery_control.py"
  "src/smart_reward/phase2_exploratory_aggregate.py"
  "scripts/hpc4/materialize_phase2_budgeted_end_to_end.py"
  "scripts/hpc4/stage_hf_assets.py"
  "scripts/hpc4/validate_phase2_recovery_authorization.py"
  "scripts/hpc4/verify_phase2_budgeted_end_to_end_seed_output.py"
  "scripts/hpc4/submit_phase2_budgeted_end_to_end.sh"
  "${SUBMITTER_RELATIVE}"
  "${SBATCH_RELATIVE}"
)
for relative in "${critical_paths[@]}"; do
  git -C "${repo_root}" ls-files --error-unmatch -- "${relative}" >/dev/null \
    || die "required budgeted source is not tracked: ${relative}"
  worktree_digest="$(sha256sum -- "${repo_root}/${relative}" | awk '{print $1}')"
  committed_digest="$(
    git -C "${repo_root}" cat-file blob "${git_commit}:${relative}" \
      | sha256sum | awk '{print $1}'
  )"
  [[ "${worktree_digest}" = "${committed_digest}" ]] \
    || die "worktree bytes differ from HEAD: ${relative}"
done

deep_identity_output="$(
  PYTHONPATH="${repo_root}/src" python3 - \
    "${overlay}" "${base}" "${materialization_receipt}" \
    "${authorization}" "${authorization_sha256}" \
    "${freeze_evidence}" "${freeze_evidence_sha256}" \
    "${OVERLAY_RELATIVE}" "${BASE_RELATIVE}" \
    "${MATERIALIZATION_RECEIPT_RELATIVE}" <<'PY'
# BEGIN BUDGETED_DEEP_VALIDATOR
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from smart_reward.config import config_hash
from smart_reward.phase2_config import (
    PHASE2_BUDGETED_END_TO_END_BASE_CONFIG,
    PHASE2_BUDGETED_END_TO_END_CONFIG,
    PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE,
    PHASE2_BUDGETED_END_TO_END_SEEDS,
    PHASE2_BUDGETED_END_TO_END_STAGE,
    load_phase2_config_bundle,
)
from smart_reward.phase2_pilot_aggregate import (
    verify_beta_source_aggregate,
    verify_horizon_parent_aggregate,
)
from smart_reward.phase2_post_recovery_control import (
    OPTIMIZER_SCHEDULE_SHA256,
    verify_post_recovery_aggregate_success_receipt,
    verify_recovery_authorization_config_binding,
)
from smart_reward.phase2_training import compile_phase2_training_settings

(
    overlay_raw,
    base_raw,
    receipt_raw,
    authorization_raw,
    authorization_sha256,
    freeze_raw,
    freeze_sha256,
    overlay_relative,
    base_relative,
    receipt_relative,
) = sys.argv[1:]
overlay = Path(overlay_raw)
base = Path(base_raw)
receipt_path = Path(receipt_raw)
authorization = Path(authorization_raw)
freeze = Path(freeze_raw)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path, *, name: str, canonical: bool = False) -> dict[str, object]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{name} must be nonempty newline-terminated JSON")

    def reject_duplicates(
        pairs: Sequence[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    if canonical:
        expected = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if raw != expected:
            raise ValueError(f"{name} is not canonical deterministic JSON")
    return value


bundle = load_phase2_config_bundle(overlay)
config = bundle.config
design = config["design"]
run = config["run"]
common = config["objective"]["common_beta"]
maximum = config["evaluation"]["max_length"]
base_hash = config_hash(bundle.base_config)
if (
    overlay_relative != PHASE2_BUDGETED_END_TO_END_CONFIG
    or base_relative != PHASE2_BUDGETED_END_TO_END_BASE_CONFIG
    or receipt_relative
    != "configs/.common_beta_post_recovery_budgeted_end_to_end.materialized.json"
    or bundle.base_config_path != base
    or design.get("stage") != PHASE2_BUDGETED_END_TO_END_STAGE
    or design.get("pilot_phase") is not None
    or design.get("formal_eligibility") is not False
    or design.get("evidence_role") != PHASE2_BUDGETED_END_TO_END_EVIDENCE_ROLE
    or design.get("source_config") != base_relative
    or design.get("source_config_hash") != base_hash
    or tuple(run.get("seeds", ())) != PHASE2_BUDGETED_END_TO_END_SEEDS
    or run.get("confirmatory") is not False
    or run.get("formal_eligibility") is not False
    or run.get("excluded_from_confirmatory_evidence") is not True
):
    raise ValueError("overlay is not the exact fixed-five budgeted_end_to_end identity")

receipt = strict_json(
    receipt_path,
    name="budgeted materialization receipt",
    canonical=True,
)
receipt_keys = {
    "schema_version",
    "stage",
    "formal_claim_eligible",
    "git_commit_used_for_source",
    "base_relative_path",
    "base_file_sha256",
    "overlay_relative_path",
    "overlay_file_sha256",
    "phase2_design_sha256",
    "accepted_freeze_aggregate_sha256",
    "authorization_sha256",
}
source_commit = receipt.get("git_commit_used_for_source")
if (
    set(receipt) != receipt_keys
    or receipt.get("schema_version")
    != "budgeted-end-to-end-materialization-receipt/v1"
    or receipt.get("stage") != PHASE2_BUDGETED_END_TO_END_STAGE
    or receipt.get("formal_claim_eligible") is not False
    or receipt.get("base_relative_path") != base_relative
    or receipt.get("base_file_sha256") != sha256(base)
    or receipt.get("overlay_relative_path") != overlay_relative
    or receipt.get("overlay_file_sha256") != sha256(overlay)
    or receipt.get("phase2_design_sha256") != bundle.design_identity
    or receipt.get("accepted_freeze_aggregate_sha256") != freeze_sha256
    or receipt.get("authorization_sha256") != authorization_sha256
    or not isinstance(source_commit, str)
    or len(source_commit) not in {40, 64}
    or any(character not in "0123456789abcdef" for character in source_commit)
):
    raise ValueError("budgeted materialization receipt does not bind the submitted identity")

authorization_binding = verify_recovery_authorization_config_binding(
    authorization,
    overlay,
    expected_sha256=authorization_sha256,
    expected_stage=PHASE2_BUDGETED_END_TO_END_STAGE,
)
if (
    authorization_binding.get("authorization_sha256") != authorization_sha256
    or authorization_binding.get("phase2_design_sha256") != bundle.design_identity
    or authorization_binding.get("base_config_hash") != base_hash
    or authorization_binding.get("optimizer_schedule_sha256")
    != OPTIMIZER_SCHEDULE_SHA256
    or authorization_binding.get("stage") != PHASE2_BUDGETED_END_TO_END_STAGE
    or authorization_binding.get("pilot_phase") is not None
):
    raise ValueError("recovery authorization did not lock the budgeted identity")

terminal = verify_post_recovery_aggregate_success_receipt(freeze)
if (
    not isinstance(terminal, Mapping)
    or terminal.get("aggregate_sha256") != freeze_sha256
    or terminal.get("pilot_phase") != "freeze"
):
    raise ValueError("accepted freeze lacks its recursively verified production SUCCESS receipt")
freeze_value = strict_json(freeze, name="accepted freeze aggregate")
selection = freeze_value.get("selection")
horizon = freeze_value.get("horizon")
boundary = freeze_value.get("information_boundary")
if (
    freeze_value.get("schema_version") != "common-beta-pilot-selection-aggregate/v3"
    or freeze_value.get("pilot_phase") != "freeze"
    or freeze_value.get("formal_eligibility") is not False
    or freeze_value.get("supports_formal_claim") is not False
    or freeze_value.get("evidence_role") != "target_free_design_selection_only"
    or not isinstance(selection, Mapping)
    or selection.get("schema_version") != "pilot-freeze-selection/v1"
    or selection.get("selection_accepted") is not True
    or selection.get("accepted_for_confirmatory_identity") is not True
    or selection.get("all_seeds_and_arms_used_same_beta") is not True
    or selection.get("all_pre_oracle_safety_gates_passed") is not True
    or selection.get("all_length_gates_passed") is not True
    or selection.get("all_non_length_safety_gates_passed") is not True
    or selection.get("next_action") != "freeze_confirmatory_design_identity"
    or not isinstance(horizon, Mapping)
    or horizon.get("all_seed_length_gates_passed") is not True
    or not isinstance(boundary, Mapping)
    or not boundary
    or any(value is not False for value in boundary.values())
):
    raise ValueError("accepted freeze is not a fully accepted target-free production-v3 freeze")

beta_binding = verify_beta_source_aggregate(config, freeze)
horizon_binding = verify_horizon_parent_aggregate(config, freeze)
try:
    frozen_beta = float(common.get("frozen_global_beta"))
    accepted_beta = float(selection.get("frozen_global_beta"))
except (TypeError, ValueError) as error:
    raise ValueError("frozen beta is not numeric") from error
if (
    not isinstance(beta_binding, Mapping)
    or not isinstance(horizon_binding, Mapping)
    or beta_binding.get("sha256") != freeze_sha256
    or horizon_binding.get("sha256") != freeze_sha256
    or horizon_binding.get("source_pilot_phase") != "freeze"
    or common.get("beta_source_aggregate_sha256") != freeze_sha256
    or maximum.get("parent_pilot_aggregate_sha256") != freeze_sha256
    or not math.isfinite(frozen_beta)
    or frozen_beta <= 0.0
    or frozen_beta != accepted_beta
    or float(beta_binding.get("accepted_beta")) != frozen_beta
    or selection.get("next_global_beta") != selection.get("frozen_global_beta")
):
    raise ValueError("one accepted freeze did not jointly bind beta and horizon")

settings = compile_phase2_training_settings(
    {"config": config, "base_config": bundle.base_config}
)
protocol = settings.convergence.optimizer_protocol
if (
    settings.stage != PHASE2_BUDGETED_END_TO_END_STAGE
    or settings.formal_eligibility is not False
    or settings.seeds != PHASE2_BUDGETED_END_TO_END_SEEDS
    or settings.convergence.max_steps != 12760
    or settings.convergence.check_interval != 20
    or settings.convergence.consecutive_checks != 3
    or protocol is None
    or protocol.mode != "adopted"
    or protocol.schedule_sha256 != OPTIMIZER_SCHEDULE_SHA256
    or protocol.source_recovery_authorization_sha256 != authorization_sha256
    or protocol.to_dict().get("scope")
    != "every_phase2_first_order_convergence_trainer"
):
    raise ValueError("budgeted training lost the recovery-authorized adopted schedule")

print(bundle.design_identity)
print(base_hash)
print(OPTIMIZER_SCHEDULE_SHA256)
print(format(frozen_beta, ".17g"))
print(source_commit)
# END BUDGETED_DEEP_VALIDATOR
PY
)" || die "deep budgeted identity verification failed"
mapfile -t identities <<<"${deep_identity_output}"
[[ "${#identities[@]}" -eq 5 ]] || die "deep verifier returned an invalid identity record"
design_sha256="${identities[0]}"
base_hash="${identities[1]}"
schedule_sha256="${identities[2]}"
frozen_beta="${identities[3]}"
materialization_source_commit="${identities[4]}"
for digest in "${design_sha256}" "${base_hash}" "${schedule_sha256}"; do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || die "deep verifier returned an invalid digest"
done
[[ "${schedule_sha256}" = "${ADOPTED_SCHEDULE_SHA256}" ]] \
  || die "deep verifier returned a non-adopted optimizer schedule"
git -C "${repo_root}" cat-file -e "${materialization_source_commit}^{commit}" \
  || die "materialization source commit is absent"
git -C "${repo_root}" merge-base --is-ancestor \
  "${materialization_source_commit}" "${git_commit}" \
  || die "materialization source commit is not an ancestor of the submitted checkout"

inventory="${hf_cache}/inventories/${base_hash}.json"
[[ -f "${inventory}" && ! -L "${inventory}" \
  && "$(realpath -e -- "${inventory}")" = "${inventory}" ]] \
  || die "base-identity HF inventory is missing or unsafe"
python3 -I -S - "${inventory}" "${base_hash}" <<'PY' \
  || die "HF inventory does not bind the budgeted base identity"
import json
import sys
from collections.abc import Sequence
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()


def reject_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate inventory key: {key}")
        result[key] = value
    return result


value = json.loads(
    raw.decode("utf-8"),
    object_pairs_hook=reject_duplicates,
    parse_constant=lambda token: (_ for _ in ()).throw(
        ValueError(f"non-finite inventory constant: {token}")
    ),
)
if (
    not isinstance(value, dict)
    or value.get("schema_version") != "prorm-hf-assets/v1"
    or value.get("config_hash") != sys.argv[2]
):
    raise SystemExit("inventory identity is invalid")
PY
inventory_sha256="$(sha256sum -- "${inventory}" | awk '{print $1}')"
reject_export_value "inventory" "${inventory}"
reject_export_value "frozen_beta" "${frozen_beta}"

for binding in \
  "${image}:${PRORM_IMAGE_SHA256}" \
  "${inventory}:${inventory_sha256}" \
  "${authorization}:${authorization_sha256}" \
  "${freeze_evidence}:${freeze_evidence_sha256}" \
  "${overlay}:${overlay_sha256}" \
  "${base}:${base_sha256}" \
  "${materialization_receipt}:${materialization_receipt_sha256}"; do
  path="${binding%%:*}"
  expected="${binding#*:}"
  [[ "$(sha256sum -- "${path}" | awk '{print $1}')" = "${expected}" ]] \
    || die "immutable submission input changed: ${path}"
done
[[ "$(git -C "${repo_root}" rev-parse --verify HEAD)" = "${git_commit}" \
  && -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "checkout changed during budgeted submission"

# This order is an execution contract shared verbatim with
# phase2_budgeted_end_to_end.sbatch's runtime_export_spec reconstruction.
# It deliberately contains no self-hash field.
export_spec="PATH=/usr/local/bin:/usr/bin:/bin,PRORM_PROJECT_ROOT=${project_root},PRORM_SCRATCH_ROOT=${scratch_root},PRORM_REPO_ROOT=${repo_root},PRORM_IMAGE=${image},PRORM_IMAGE_SHA256=${PRORM_IMAGE_SHA256},PRORM_HF_CACHE=${hf_cache},PRORM_HF_INVENTORY=${inventory},PRORM_HF_INVENTORY_SHA256=${inventory_sha256},PRORM_BUDGETED_OVERLAY_REL=${OVERLAY_RELATIVE},PRORM_BUDGETED_BASE_REL=${BASE_RELATIVE},PRORM_BUDGETED_OVERLAY_SHA256=${overlay_sha256},PRORM_BUDGETED_BASE_SHA256=${base_sha256},PRORM_BUDGETED_DESIGN_SHA256=${design_sha256},PRORM_BUDGETED_BASE_CONFIG_HASH=${base_hash},PRORM_RECOVERY_AUTHORIZATION=${authorization},PRORM_RECOVERY_AUTHORIZATION_SHA256=${authorization_sha256},PRORM_OPTIMIZER_SCHEDULE_SHA256=${schedule_sha256},PRORM_BUDGETED_FROZEN_BETA=${frozen_beta},PRORM_BUDGETED_FREEZE_EVIDENCE=${freeze_evidence},PRORM_BUDGETED_FREEZE_EVIDENCE_SHA256=${freeze_evidence_sha256},PRORM_GIT_COMMIT=${git_commit}"

python3 "${repo_root}/${SUBMITTER_RELATIVE}" \
  --project-root "${project_root}" \
  --repo-root "${repo_root}" \
  --design-sha256 "${design_sha256}" \
  --base-config-hash "${base_hash}" \
  --authorization-sha256 "${authorization_sha256}" \
  --optimizer-schedule-sha256 "${schedule_sha256}" \
  --git-commit "${git_commit}" \
  --image-sha256 "${PRORM_IMAGE_SHA256}" \
  --inventory-sha256 "${inventory_sha256}" \
  --overlay-sha256 "${overlay_sha256}" \
  --base-file-sha256 "${base_sha256}" \
  --walltime "${walltime}" \
  --export-spec "${export_spec}" \
  --sbatch-script "${repo_root}/${SBATCH_RELATIVE}"
