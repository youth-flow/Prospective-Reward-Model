#!/usr/bin/env bash
set -euo pipefail
umask 027

die() {
  echo "error: $*" >&2
  exit 2
}

if [[ $# -ne 1 ]]; then
  die "usage: $0 <40-hex image-build Git commit>"
fi
build_commit="$1"
[[ "${build_commit}" =~ ^[0-9a-f]{40}$ ]] \
  || die "image-build Git commit must be 40 lowercase hexadecimal characters"

: "${PRORM_PROJECT_ROOT:?set PRORM_PROJECT_ROOT to the persistent project root}"
: "${PRORM_SCRATCH_ROOT:?set PRORM_SCRATCH_ROOT to the per-user scratch root}"
: "${PRORM_IMAGE:=images/prorm.sif}"
[[ "${PRORM_PROJECT_ROOT}" = /* && "${PRORM_SCRATCH_ROOT}" = /* ]] \
  || die "project and scratch roots must be absolute"
[[ "${PRORM_IMAGE}" != /* ]] \
  || die "PRORM_IMAGE must be relative to PRORM_PROJECT_ROOT"
[[ "${PRORM_IMAGE}" != *":"* && "${PRORM_IMAGE}" != *","* \
  && "${PRORM_IMAGE}" != *"\\"* \
  && "${PRORM_IMAGE}" != *$'\n'* && "${PRORM_IMAGE}" != *$'\r'* ]] \
  || die "PRORM_IMAGE contains an unsafe path delimiter"
[[ "${PRORM_IMAGE}" = *.sif ]] || die "PRORM_IMAGE must end in .sif"
image_basename="$(basename "${PRORM_IMAGE}")"
[[ "${image_basename}" != "." && "${image_basename}" != ".." ]] \
  || die "PRORM_IMAGE has an invalid basename"

for command_name in apptainer awk basename cmp cp curl dirname mktemp mv python3 realpath sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || die "required command is unavailable: ${command_name}"
done

project_root="$(realpath -e -- "${PRORM_PROJECT_ROOT}")"
scratch_root="$(realpath -e -- "${PRORM_SCRATCH_ROOT}")"
[[ "${project_root}" != "/" && "${scratch_root}" != "/" ]] \
  || die "project and scratch roots may not be the filesystem root"
case "${project_root}" in
  "${scratch_root}"|"${scratch_root}"/*) die "project and scratch roots overlap" ;;
esac
case "${scratch_root}" in
  "${project_root}"|"${project_root}"/*) die "project and scratch roots overlap" ;;
esac
[[ -w "${project_root}" && -w "${scratch_root}" ]] \
  || die "project and scratch roots must be writable"

target_parent="$(realpath -m -- "${project_root}/$(dirname "${PRORM_IMAGE}")")"
case "${target_parent}" in
  "${project_root}"|"${project_root}"/*) ;;
  *) die "PRORM_IMAGE parent escapes PRORM_PROJECT_ROOT" ;;
esac
mkdir -p -- \
  "${target_parent}" \
  "${project_root}/system-reports" \
  "${project_root}/apptainer-cache" \
  "${scratch_root}/apptainer-tmp"
target_parent="$(realpath -e -- "${target_parent}")"
case "${target_parent}" in
  "${project_root}"|"${project_root}"/*) ;;
  *) die "PRORM_IMAGE parent escapes PRORM_PROJECT_ROOT" ;;
esac
target="${target_parent}/${image_basename}"
[[ ! -L "${target}" ]] || die "refusing a symlink image target: ${target}"

package="ghcr.io/youth-flow/prospective-reward-model-hpc4"
package_path="youth-flow/prospective-reward-model-hpc4"
tag="git-${build_commit}"
work_parent="$(realpath -e -- "${scratch_root}/apptainer-tmp")"
work_dir="$(mktemp -d "${work_parent}/fetch-image.XXXXXX")"
work_dir="$(realpath -e -- "${work_dir}")"
case "${work_dir}" in
  "${work_parent}"/fetch-image.*) ;;
  *) die "temporary directory escaped its intended parent" ;;
esac
temporary_image="${target_parent}/.$(basename "${target}").${build_commit}.tmp.sif"
[[ ! -e "${temporary_image}" && ! -L "${temporary_image}" ]] \
  || die "stale temporary image already exists: ${temporary_image}"
cleanup() {
  local exit_code=$?
  trap - EXIT
  rm -r -- "${work_dir}"
  rm -f -- "${temporary_image}"
  exit "${exit_code}"
}
trap cleanup EXIT

token="$(
  curl --fail --silent --show-error \
    --get \
    --data-urlencode "service=ghcr.io" \
    --data-urlencode "scope=repository:${package_path}:pull" \
    https://ghcr.io/token \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
)"
[[ -n "${token}" ]] || die "GHCR returned an empty anonymous pull token"

headers="${work_dir}/manifest.headers"
manifest="${work_dir}/manifest.json"
curl --fail --silent --show-error \
  --dump-header "${headers}" \
  --output "${manifest}" \
  --header "Authorization: Bearer ${token}" \
  --header "Accept: application/vnd.oci.image.manifest.v1+json" \
  "https://ghcr.io/v2/${package_path}/manifests/${tag}"
manifest_digest="$(
  awk 'tolower($1)=="docker-content-digest:" {gsub("\r", "", $2); print $2}' \
    "${headers}" | tail -n 1
)"
[[ "${manifest_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "GHCR returned an invalid manifest digest: ${manifest_digest}"
observed_manifest_digest="sha256:$(sha256sum "${manifest}" | awk '{print $1}')"
[[ "${observed_manifest_digest}" = "${manifest_digest}" ]] \
  || die "manifest bytes do not match Docker-Content-Digest"

layer_digest="$(
  python3 - "${manifest}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as stream:
    manifest = json.load(stream)
if manifest.get("schemaVersion") != 2:
    raise SystemExit("unexpected OCI schemaVersion")
layers = manifest.get("layers")
if not isinstance(layers, list) or len(layers) != 1:
    raise SystemExit("expected exactly one raw SIF layer")
layer = layers[0]
digest = layer.get("digest")
if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("invalid raw SIF layer digest")
if not isinstance(layer.get("size"), int) or layer["size"] <= 0:
    raise SystemExit("invalid raw SIF layer size")
print(digest)
PY
)"

export APPTAINER_CACHEDIR="${project_root}/apptainer-cache"
export APPTAINER_TMPDIR="${scratch_root}/apptainer-tmp"
apptainer pull \
  "${temporary_image}" \
  "oras://${package}@${manifest_digest}"
observed_sif_digest="sha256:$(sha256sum "${temporary_image}" | awk '{print $1}')"
[[ "${observed_sif_digest}" = "${layer_digest}" ]] \
  || die "downloaded SIF does not match the recorded OCI layer identity"

if [[ -e "${target}" ]]; then
  existing_digest="sha256:$(sha256sum "${target}" | awk '{print $1}')"
  [[ "${existing_digest}" = "${layer_digest}" ]] \
    || die "refusing to replace a different existing image: ${target}"
  rm -f -- "${temporary_image}"
else
  mv -- "${temporary_image}" "${target}"
fi

manifest_report="${project_root}/system-reports/image-manifest-${build_commit}.json"
if [[ -e "${manifest_report}" ]]; then
  cmp --silent "${manifest}" "${manifest_report}" \
    || die "existing manifest report has different bytes: ${manifest_report}"
else
  cp -- "${manifest}" "${manifest_report}"
fi

printf 'image=%s\n' "${PRORM_IMAGE}"
printf 'image_build_commit=%s\n' "${build_commit}"
printf 'oras_manifest_digest=%s\n' "${manifest_digest}"
printf 'sif_sha256=%s\n' "${layer_digest#sha256:}"
printf 'manifest_report=%s\n' \
  "system-reports/$(basename "${manifest_report}")"

trap - EXIT
rm -r -- "${work_dir}"
