#!/usr/bin/env bash
set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ "$#" -ne 3 ]]; then
  die "usage: $0 <confirmatory-overlay.yaml> <failed-job-directory> <failure-classification.json>"
fi
overlay="$1"
job_dir="$2"
classification="$3"

for command_name in git python3 realpath sha256sum mktemp mv; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ -z "$(git -C "${repo_root}" status --porcelain --untracked-files=normal)" ]] \
  || die "compute-failure terminalization requires a clean committed worktree"
for name in overlay job_dir classification; do
  resolved="$(realpath -e -- "${!name}")" || die "${name} cannot be resolved"
  printf -v "${name}" '%s' "${resolved}"
done
[[ -f "${overlay}" && ! -L "${overlay}" ]] \
  || die "overlay must be a non-symlink regular file"
[[ -d "${job_dir}" && ! -L "${job_dir}" ]] \
  || die "failed job directory must be a non-symlink directory"
[[ -f "${classification}" && ! -L "${classification}" ]] \
  || die "classification must be a non-symlink regular file"
for path in \
  "${job_dir}/FAILURE_PENDING" "${job_dir}/phase2-attempt-ledger.json"; do
  [[ -f "${path}" && ! -L "${path}" ]] \
    || die "required compute failure evidence is missing: ${path}"
done
for path in \
  "${job_dir}/SUCCESS" "${job_dir}/SCHEDULER_FAILED" \
  "${job_dir}/SUCCESS_SEALED_SYNC_ERROR" \
  "${job_dir}/phase2-success-terminal.json"; do
  [[ ! -e "${path}" && ! -L "${path}" ]] \
    || die "job directory already has conflicting terminal evidence: ${path}"
done
staging="$(mktemp -d "${job_dir}/.compute-terminal.tmp.XXXXXX")"
cleanup() {
  if [[ -d "${staging}" ]]; then
    rm -f -- "${staging}"/* 2>/dev/null || true
    rmdir -- "${staging}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 -I -S - \
  "${repo_root}/src" "${overlay}" "${job_dir}" "${classification}" \
  "${staging}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from smart_reward.cli import _run_environment_identity
from smart_reward.phase2_config import load_phase2_config_bundle

overlay = Path(sys.argv[2])
job_dir = Path(sys.argv[3])
classification_path = Path(sys.argv[4])
staging = Path(sys.argv[5])


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


classification = load(classification_path)
if not isinstance(classification, dict) or set(classification) != {
    "failure_stage",
    "failure_class",
    "failure_type",
    "failure_message_sha256",
}:
    raise SystemExit("failure classification fields differ from the locked schema")
for key in ("failure_stage", "failure_class", "failure_type"):
    if not isinstance(classification[key], str) or re.fullmatch(
        r"[a-z0-9][a-z0-9_]{0,127}", classification[key]
    ) is None:
        raise SystemExit(f"{key} must be an explicit safe token")
if classification["failure_class"] not in {
    "infrastructure",
    "scientific",
    "safety",
    "identity",
    "numerical",
    "software",
}:
    raise SystemExit("failure_class is unsupported")
if (
    not isinstance(classification["failure_message_sha256"], str)
    or re.fullmatch(r"[0-9a-f]{64}", classification["failure_message_sha256"])
    is None
):
    raise SystemExit("failure_message_sha256 is invalid")

bundle = load_phase2_config_bundle(overlay)
config = bundle.config
ledger_path = job_dir / "phase2-attempt-ledger.json"
ledger = load(ledger_path)
attempts = ledger.get("attempts") if isinstance(ledger, dict) else None
if (
    ledger.get("schema_version") != "phase2-seed-attempt-ledger/v3"
    or ledger.get("retry_policy")
    != "single_predeclared_attempt_no_retry"
    or ledger.get("replacement_seed_allowed") is not False
    or not isinstance(attempts, list)
    or len(attempts) != 1
    or attempts[0].get("attempt_index") != 1
    or attempts[-1].get("status") != "terminal_failure"
):
    raise SystemExit("compute failure does not have a terminal attempt ledger")
seed = int(job_dir.parent.parent.name.removeprefix("seed-"))
if seed not in config["run"]["seeds"]:
    raise SystemExit("failed job is not a predeclared formal seed")
attempt_index = int(job_dir.parent.name.removeprefix("attempt-"))
if len(attempts) != attempt_index:
    raise SystemExit("failed job attempt directory disagrees with its ledger")
if attempt_index != 1:
    raise SystemExit("formal Phase-2 retries are disabled")

marker_fields = {}
for line in (job_dir / "FAILURE_PENDING").read_text(encoding="utf-8").splitlines():
    if "=" not in line:
        raise SystemExit("FAILED marker contains a malformed line")
    key, value = line.split("=", 1)
    if not key or key in marker_fields:
        raise SystemExit("FAILED marker has a duplicate or empty key")
    marker_fields[key] = value
if (
    marker_fields.get("schema_version")
    != "prorm-phase2-confirmatory-run-status/v1"
    or marker_fields.get("status") != "FAILURE_PENDING"
    or marker_fields.get("seed") != str(seed)
    or marker_fields.get("attempt_index") != str(attempt_index)
    or marker_fields.get("slurm_job_id") != attempts[-1].get("slurm_job_id")
    or marker_fields.get("cluster_name") != attempts[-1].get("cluster_name")
    or marker_fields.get("array_job_id") != attempts[-1].get("array_job_id")
    or marker_fields.get("array_task_id")
    != str(attempts[-1].get("array_task_id"))
    or marker_fields.get("phase2_design_sha256") != bundle.design_identity
    or marker_fields.get("base_config_hash") != config["design"]["source_config_hash"]
    or marker_fields.get("attempt_ledger_sha256")
    != hashlib.sha256(ledger_path.read_bytes()).hexdigest()
):
    raise SystemExit("FAILURE_PENDING marker does not bind the terminal attempt ledger")
repo_root = Path(sys.argv[1]).parent
current_commit = subprocess.check_output(
    ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
    text=True,
).strip()
if marker_fields.get("git_commit") != current_commit:
    raise SystemExit("failure terminalization must use the submitted Git commit")
reveal = marker_fields.get("final_outcome_reveal_started") in {"true", "1"}
if reveal is not attempts[-1].get("final_outcome_reveal_started"):
    raise SystemExit("FAILURE_PENDING marker and ledger disagree on the outcome boundary")

run_manifest = job_dir / "run-manifest.json"
artifact_metadata = job_dir / "artifact" / "metadata.json"
for evidence_path in (run_manifest, artifact_metadata):
    if evidence_path.is_symlink():
        raise SystemExit(f"formal evidence path must not be a symlink: {evidence_path}")
if run_manifest.is_file() and not run_manifest.is_symlink():
    manifest_sha, environment = _run_environment_identity(
        run_manifest,
        expected_config_hash=config["design"]["source_config_hash"],
        expected_seed=seed,
        require_formal=True,
    )
    manifest_slot = {"status": "available", "sha256": manifest_sha}
    environment_slot = {"status": "available", "value": environment}
else:
    manifest_slot = {
        "status": "unavailable",
        "reason": "not_produced_before_failure",
    }
    environment_slot = {
        "status": "unavailable",
        "reason": "not_produced_before_failure",
    }
if artifact_metadata.is_file() and not artifact_metadata.is_symlink():
    artifact_slot = {
        "status": "available",
        "sha256": hashlib.sha256(artifact_metadata.read_bytes()).hexdigest(),
    }
else:
    artifact_slot = {
        "status": "unavailable",
        "reason": "not_produced_before_failure",
    }
evidence = {
    "compute_failure_pending_marker": hashlib.sha256(
        (job_dir / "FAILURE_PENDING").read_bytes()
    ).hexdigest(),
    "attempt_ledger": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
}
for role, name in (
    ("attempt_evidence", "attempt-evidence.log"),
    ("phase2_run_log", "phase2-run.log"),
    ("gpu_check", "gpu-check.log"),
):
    path = job_dir / name
    if path.is_file() and not path.is_symlink():
        evidence[role] = hashlib.sha256(path.read_bytes()).hexdigest()
spec = {
    "seed": seed,
    "failure_stage": classification["failure_stage"],
    "failure_class": classification["failure_class"],
    "failure_type": classification["failure_type"],
    "failure_message_sha256": classification["failure_message_sha256"],
    "final_outcome_reveal_started": reveal,
    "attempt_ledger": ledger,
    "capture_method": "compute_exit_trap",
    "evidence_availability": {
        "schema_version": "phase2-seed-failure-evidence-availability/v1",
        "run_manifest": manifest_slot,
        "artifact_metadata": artifact_slot,
        "environment_identity": environment_slot,
    },
    "evidence_sha256_by_role": evidence,
}
with (staging / "phase2-failure-spec.json").open(
    "x", encoding="utf-8", newline="\n"
) as stream:
    json.dump(spec, stream, allow_nan=False, sort_keys=True, separators=(",", ":"))
    stream.write("\n")
PY

PYTHONPATH="${repo_root}/src" PYTHONNOUSERSITE=1 \
  python3 -m smart_reward.cli phase2-failure-manifest \
  "${overlay}" "${staging}/phase2-failure-spec.json" \
  "${staging}/phase2-failure-terminal.json" \
  > "${staging}/phase2-failure-manifest.log"
terminal_sha256="$(
  sha256sum -- "${staging}/phase2-failure-terminal.json" | awk '{print $1}'
)"
while IFS= read -r line; do
  case "${line}" in
    status=FAILURE_PENDING) printf 'status=FAILED\n' ;;
    *) printf '%s\n' "${line}" ;;
  esac
done < "${job_dir}/FAILURE_PENDING" > "${staging}/FAILED"
printf 'terminal_manifest_sha256=%s\n' "${terminal_sha256}" >> "${staging}/FAILED"
python3 -I -S \
  "${repo_root}/scripts/hpc4/publish_phase2_terminal_bundle.py" \
  "${staging}" "${job_dir}" FAILED \
  phase2-failure-spec.json phase2-failure-terminal.json \
  phase2-failure-manifest.log FAILED
trap - EXIT
printf 'Compute-terminal failure recorded: %s\n' \
  "${job_dir}/phase2-failure-terminal.json"
