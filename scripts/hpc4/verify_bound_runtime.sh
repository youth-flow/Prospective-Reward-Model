#!/usr/bin/env bash

verify_bound_runtime() {
  if [[ $# -ne 4 ]]; then
    echo "verify_bound_runtime requires IMAGE, IMAGE_SOURCE_COMMIT, CODE_COMMIT, and REPO" >&2
    return 2
  fi
  local image="$1"
  local image_source_commit="$2"
  local code_commit="$3"
  local repo="$4"
  [[ "${image_source_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "image source commit is malformed" >&2
    return 2
  }
  [[ "${code_commit}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "code commit is malformed" >&2
    return 2
  }
  source "${repo}/scripts/hpc4/verify_image_revision.sh"
  verify_image_revision "${image}" "${image_source_commit}"
  [[ "$(git -C "${repo}" rev-parse HEAD)" = "${code_commit}" ]] || {
    echo "bound repository HEAD differs from PRORM_GIT_COMMIT" >&2
    return 2
  }
  [[ -z "$(git -C "${repo}" status --porcelain)" ]] || {
    echo "bound repository worktree is dirty" >&2
    return 2
  }
  git -C "${repo}" diff --quiet "${image_source_commit}" "${code_commit}" -- \
    containers/prorm-hpc4.def pyproject.toml || {
      echo "runtime dependencies changed between image and bound code commits" >&2
      return 2
    }
}
