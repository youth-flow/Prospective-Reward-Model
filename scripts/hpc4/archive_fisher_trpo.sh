#!/usr/bin/env bash
set -euo pipefail
umask 027

[[ $# -eq 3 ]] || {
  echo "usage: $0 RUN_ROOT ARCHIVE_ROOT AUDIT_JSON" >&2
  exit 2
}

run_root="$(realpath -e "$1")"
archive_root="$(realpath -m "$2")"
audit_json="$(realpath -e "$3")"
case "${run_root}" in "/scratch/${USER}/"*) ;;
  *) echo "run root must be under /scratch/${USER}" >&2; exit 2 ;;
esac
case "${archive_root}" in /project/sigroup/"${USER}"/*) ;;
  *) echo "archive root must be under the user's project directory" >&2; exit 2 ;;
esac
[[ ! -e "${archive_root}" ]] || {
  echo "refusing to overwrite an existing archive" >&2
  exit 2
}
python3 - "${audit_json}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    value = json.load(stream)
assert value["schema"] == "prorm-fisher-trpo-integrity-audit/v1"
assert value["status"] == "passed"
PY

staging="${archive_root}.incomplete"
[[ ! -e "${staging}" ]] || {
  echo "incomplete archive staging path already exists" >&2
  exit 2
}
mkdir -p "${staging}"
rsync -a --checksum -- "${run_root}/" "${staging}/run/"
cp --preserve=mode,timestamps -- "${audit_json}" "${staging}/integrity-audit.json"

manifest="${staging}/SHA256SUMS"
source_manifest="$(mktemp)"
trap 'rm -f -- "${source_manifest}"' EXIT
(
  cd "${run_root}"
  find . -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${source_manifest}"
(
  cd "${staging}/run"
  find . -type f -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum
) > "${manifest}"
cmp --silent "${source_manifest}" "${manifest}" || {
  echo "scratch and archive file manifests differ" >&2
  exit 2
}
(
  cd "${staging}"
  sed 's#  \\./#  run/#' SHA256SUMS
  sha256sum integrity-audit.json
) > "${staging}/SHA256SUMS.final"
mv -- "${staging}/SHA256SUMS.final" "${manifest}"
(
  cd "${staging}"
  sha256sum --check --strict SHA256SUMS
)
chmod -R a-w "${staging}"
mv -- "${staging}" "${archive_root}"
printf 'archive=%s\n' "${archive_root}"
printf 'manifest_sha256=%s\n' "$(sha256sum "${archive_root}/SHA256SUMS" | awk '{print $1}')"
