# HPC4 Execution

The account is `yyangjo`, the Slurm account is `sigroup`, and the frozen GPU partition is
`gpu-l20`. Login nodes are for Git, transfer, inspection, and `sbatch` only. Model loading,
Apptainer execution, CUDA checks, training, and evaluation run inside Slurm allocations.

Persistent source, images, Hugging Face assets, reports, and archives live under
`/project/sigroup/yyangjo`. Active outputs live under `/scratch/yyangjo`; inactive scratch
files may be removed after 60 days.

All compute jobs use `scripts/hpc4/runtime.sh`, which disables HPC4's stale
`/opt/knem-1.1.4.90mlnx3` bind before starting Apptainer. Keep this site workaround on every
container execution path unless HPC4 removes the injected bind and a fresh compute-node probe
proves it is no longer needed.

## One-time setup

```bash
ssh yyangjo@hpc4.ust.hk
export PRORM_PROJECT_ROOT="/project/sigroup/$USER/prorm"
export PRORM_SCRATCH_ROOT="/scratch/$USER/prorm"
mkdir -p "$PRORM_PROJECT_ROOT"/{images,hf-cache,system-reports,archive}
mkdir -p "$PRORM_SCRATCH_ROOT/runs"
mkdir -p "/project/sigroup/$USER"
cd "/project/sigroup/$USER"
git clone https://github.com/youth-flow/Prospective-Reward-Model.git
cd Prospective-Reward-Model
```

Before submissions:

```bash
squota -A sigroup
squota
sinfo -p gpu-l20
squeue -u "$USER"
```

## Immutable image

Use the 40-character commit only after both GitHub CI and `build-hpc4-image` pass.

```bash
export PRORM_IMAGE="images/prorm-hpc4.sif"
bash scripts/hpc4/fetch_candidate_image.sh <image-build-commit>
export PRORM_IMAGE="$PRORM_PROJECT_ROOT/images/prorm-hpc4.sif"
```

The fetch validates the public OCI manifest and SIF digest before installing the image.
The compute-node smoke then validates the embedded Git revision, dependency lock, CUDA
runtime, and package contract before the image is admitted to staging or experiments.

## GPU image smoke

```bash
bash scripts/hpc4/submit_gpu_smoke.sh \
  "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/system-reports" gpu-l20 00:20:00
```

Require terminal `COMPLETED`, `ExitCode=0:0`, and a report containing the L20, CUDA forward
and backward pass, package contract, config check, Git commit, and image SHA-256.

## Hugging Face staging

Stage the two config-bound inventories sequentially because they share one cache:

```bash
bash scripts/hpc4/submit_hf_stage.sh \
  configs/smoke.yaml "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/hf-cache" amd 04:00:00

bash scripts/hpc4/submit_hf_stage.sh \
  configs/main.yaml "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/hf-cache" amd 04:00:00
```

Each must finish `COMPLETED` with `ExitCode=0:0`. Inventories live in
`$PRORM_PROJECT_ROOT/hf-cache/inventories/<config-sha256>.json`.

## QOS-aware staged smoke

HPC4's `l20_qos` currently allows at most two running and four submitted GPU jobs per user.
Slurm counts every array task against the submit limit, even when `%N` caps concurrency.
Therefore `submit_pipeline.sh` submits exactly one validated stage at a time:

```text
materialize array
  -> reward array
  -> adapter array
  -> QOS-packed policy rollout workers
  -> per-seed rollout aggregate array
  -> config-wide aggregate
```

Use the same command for each stage, in the listed order. Wait for terminal `COMPLETED` and
`ExitCode=0:0`, and inspect the stage receipt before submitting the next stage:

```bash
bash scripts/hpc4/submit_pipeline.sh \
  configs/smoke.yaml \
  "$PRORM_IMAGE" \
  "$PRORM_PROJECT_ROOT/hf-cache" \
  "$PRORM_SCRATCH_ROOT/runs/smoke" \
  materialize
```

Then repeat with `reward`, `adapters`, `rollout`, `rollout-aggregate`, and `aggregate`.
The command prints the selected stage and its job ID. Monitor terminal evidence:

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES
tail -f "$PRORM_SCRATCH_ROOT/runs/smoke/logs/<stage>-<job-id>_<task>.out"
```

Re-running a stage is safe: complete stages validate and return immediately;
incomplete materialization and rollout work resumes from the last complete prompt checkpoint.
Foreign or corrupted outputs are rejected.

Do not submit `main.yaml` until smoke proves:

- every dependency job is terminal `COMPLETED` with `ExitCode=0:0`;
- MLE-RM, Pro-RM, and all three NGD solves converge;
- artifact, nine adapters, ten policy rollouts, per-seed metrics, and aggregate exist;
- one L20 has safe peak memory at configured batch sizes;
- measured stage runtimes justify the main walltimes and rollout concurrency.

## Three formal seeds

After freezing the execution fields from smoke and restaging any changed config:

```bash
bash scripts/hpc4/submit_pipeline.sh \
  configs/main.yaml \
  "$PRORM_IMAGE" \
  "$PRORM_PROJECT_ROOT/hf-cache" \
  "$PRORM_SCRATCH_ROOT/runs/main" \
  materialize
```

Again advance one verified stage at a time. Materialize/reward/adapters use three-task arrays
with at most two running tasks. Formal rollout preserves the YAML six-policy concurrency using
two Slurm jobs with three L20 GPUs each; every GPU worker processes a disjoint stride of the 30
seed-policy tasks and resumes at policy/prompt checkpoints. Aggregation is submitted only after
all upstream workers succeed.

## Archive and transfer

After verifying `aggregate.json` and all three seed receipts:

```bash
rsync -a --info=progress2 \
  "$PRORM_SCRATCH_ROOT/runs/main/" \
  "$PRORM_PROJECT_ROOT/archive/main/"
```

Generate and compare checksums before treating the archive as durable. Pull compact JSON,
JSONL, receipts, and reports locally; pull tensors and adapters only when needed. Do not
copy the SIF or Hugging Face cache back to the workstation.
