#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 4 ]] || {
  echo "usage: $0 RUN_ROOT ARCHIVE_ROOT AUDIT_JSON EXPECTED_AUDIT_SCHEMA" >&2
  exit 2
}

run_root="$(realpath -e "$1")"
archive_root="$(realpath -m "$2")"
audit_json="$(realpath -e "$3")"
expected_schema="$4"

case "${run_root}" in
  "/scratch/${USER}/"*) ;;
  *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;;
esac
case "${archive_root}" in
  "/project/sigroup/${USER}/"*) ;;
  *) echo "archive root must be under /project/sigroup/${USER}" >&2; exit 2 ;;
esac
[[ ! -e "${archive_root}" ]] || {
  echo "refusing to overwrite an existing archive" >&2
  exit 2
}

python3 - "${audit_json}" "${expected_schema}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    audit = json.load(stream)
if audit.get("schema") != sys.argv[2]:
    raise SystemExit("audit schema mismatch")
if audit.get("status") not in {"passed", "complete"}:
    raise SystemExit("audit status is not successful")
PY

staging="${archive_root}.incomplete.$$"
[[ ! -e "${staging}" ]] || {
  echo "archive staging path already exists" >&2
  exit 2
}
source_manifest="$(mktemp)"
trap 'rm -f -- "${source_manifest}"' EXIT

mkdir -p "${staging}/run"
rsync -a --checksum -- "${run_root}/" "${staging}/run/"
cp --preserve=mode,timestamps -- "${audit_json}" "${staging}/integrity-audit.json"

(
  cd "${run_root}"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${source_manifest}"
(
  cd "${staging}/run"
  find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "${staging}/RUN_SHA256SUMS"
cmp --silent "${source_manifest}" "${staging}/RUN_SHA256SUMS" || {
  echo "scratch and archive file manifests differ" >&2
  exit 2
}

(
  cd "${staging}"
  sed 's#  \./#  run/#' RUN_SHA256SUMS
  sha256sum integrity-audit.json
) > "${staging}/SHA256SUMS"
rm -f -- "${staging}/RUN_SHA256SUMS"
(
  cd "${staging}"
  sha256sum --check --strict SHA256SUMS
)

python3 - "${run_root}" "${archive_root}" "${expected_schema}" \
  "${staging}/SHA256SUMS" > "${staging}/ARCHIVE_RECEIPT.json" <<'PY'
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone

manifest = sys.argv[4]
with open(manifest, "rb") as stream:
    digest = hashlib.sha256(stream.read()).hexdigest()
print(json.dumps({
    "schema": "prorm-audited-run-archive-receipt/v1",
    "status": "complete",
    "source_run_root": sys.argv[1],
    "archive_root": sys.argv[2],
    "audit_schema": sys.argv[3],
    "manifest_sha256": digest,
    "host": socket.gethostname(),
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}, sort_keys=True, separators=(",", ":")))
PY

chmod -R a-w "${staging}"
mv -- "${staging}" "${archive_root}"
printf 'archive=%s\n' "${archive_root}"
printf 'manifest_sha256=%s\n' "$(sha256sum "${archive_root}/SHA256SUMS" | awk '{print $1}')"
