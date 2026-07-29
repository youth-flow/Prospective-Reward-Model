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
      | python3 "${PRORM_REPO_ROOT}/scripts/hpc4/parse_image_revision.py"
  )"
  [[ "${observed_commit}" = "${expected_commit}" ]] || {
    echo "image commit ${observed_commit} does not match worktree commit ${expected_commit}" >&2
    return 2
  }
}
