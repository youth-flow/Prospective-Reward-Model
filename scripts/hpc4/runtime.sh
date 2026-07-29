#!/usr/bin/env bash

prorm_apptainer_exec() {
  # HPC4 injects this stale KNEM bind on compute nodes, where its source is absent.
  apptainer exec --no-mount /opt/knem-1.1.4.90mlnx3 "$@"
}
