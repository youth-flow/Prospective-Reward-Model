# HPC4 Runtime

`prorm-hpc4.def` builds the CUDA runtime used by all GPU jobs. The base image and Python
packages are pinned; model and dataset weights are staged separately under the
config-hash-bound Hugging Face cache.

Build and publish the SIF through `.github/workflows/build-hpc4-image.yml`, then fetch the
immutable build on HPC4 with `scripts/hpc4/fetch_candidate_image.sh`. A formal seed must
record the Git commit, SIF SHA-256, and staged-asset inventory SHA-256 in its artifact.

The image contains the repository package but no experiment outputs or credentials.
