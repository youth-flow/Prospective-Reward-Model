# HPC4 Execution

The account is `yyangjo`, the Slurm account is `sigroup`, and the frozen GPU partition is
`gpu-l20`. Login nodes are for Git, transfer, inspection, and `sbatch` only. Model loading,
Apptainer execution, CUDA checks, training, and evaluation run inside Slurm allocations.

Persistent source, images, Hugging Face assets, reports, and archives live under
`/project/sigroup/yyangjo`. Active outputs live under `/scratch/yyangjo`; inactive scratch
files may be removed after 60 days.

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

The fetch validates the public OCI manifest, SIF digest, embedded Git revision, dependency
lock, and local report before installing the image.

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

## Dependency-ordered smoke

`submit_pipeline.sh` reads stage walltimes and rollout concurrency from YAML. It submits:

```text
materialize array
  -> reward array
  -> adapter array
  -> seed x policy rollout array with a concurrency cap
  -> per-seed rollout aggregate array
  -> config-wide aggregate
```

Submit the complete 24-prompt pipeline:

```bash
bash scripts/hpc4/submit_pipeline.sh \
  configs/smoke.yaml \
  "$PRORM_IMAGE" \
  "$PRORM_PROJECT_ROOT/hf-cache" \
  "$PRORM_SCRATCH_ROOT/runs/smoke"
```

The command prints all six job IDs. Monitor terminal evidence:

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES
tail -f "$PRORM_SCRATCH_ROOT/runs/smoke/logs/<stage>-<job-id>_<task>.out"
```

Re-running the same submission is safe: complete stages validate and return immediately;
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
  "$PRORM_SCRATCH_ROOT/runs/main"
```

Materialize/reward/adapters use three-task arrays. Rollout uses thirty tasks with the YAML
concurrency cap. Aggregation runs only after all upstream tasks succeed.

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
