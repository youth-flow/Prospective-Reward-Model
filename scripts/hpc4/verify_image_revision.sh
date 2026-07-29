#!/usr/bin/env bash

verify_image_revision() {
  [[ $# -eq 2 ]] || {
    echo "verify_image_revision requires IMAGE and EXPECTED_COMMIT" >&2
    return 2
  }
  local image="$1"
  local expected_commit="$2"
  local observed_commit

  observed_commit="$(
    apptainer inspect --json "${image}" \
      | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
labels = payload.get("data", {}).get("attributes", {}).get("labels", {})
revision = labels.get("org.opencontainers.image.revision")
if not isinstance(revision, str):
    raise SystemExit("image revision label is missing")
print(revision)
'
  )"
  [[ "${observed_commit}" = "${expected_commit}" ]] || {
    echo "image commit ${observed_commit} does not match worktree commit ${expected_commit}" >&2
    return 2
  }
}
