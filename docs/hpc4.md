# HPC4 Execution

The account in the HPC4 onboarding notice is `yyangjo`, the Slurm account is `sigroup`,
and `gpu-l20` is an available partition. Login with `ssh yyangjo@hpc4.ust.hk`; off campus,
connect to the HKUST VPN first. Login nodes have no GPU and must not run model loading,
candidate generation, Apptainer validation, or training.

Persistent source, images, Hugging Face assets, reports, and archived results live under
`/project/sigroup`. Active run outputs live under `/scratch/$USER`; scratch files inactive
for 60 days may be removed by HPC4.

## One-time setup

```bash
ssh yyangjo@hpc4.ust.hk
export PRORM_PROJECT_ROOT="/project/sigroup/$USER/prorm"
export PRORM_SCRATCH_ROOT="/scratch/$USER/prorm"
mkdir -p "$PRORM_PROJECT_ROOT"/{images,hf-cache,system-reports,archive}
mkdir -p "$PRORM_SCRATCH_ROOT"/runs
cd "/project/sigroup/$USER"
git clone https://github.com/youth-flow/Prospective-Reward-Model.git
cd Prospective-Reward-Model
```

Check storage and scheduler state before large jobs:

```bash
squota -A sigroup
squota
savail
squeue -u "$USER"
```

## Image

The image workflow runs only from the committed `main` branch. After it succeeds, use its
40-character Git commit to fetch the immutable SIF. The pull itself is a login-node data
transfer; all image execution remains inside Slurm jobs.

```bash
export PRORM_IMAGE="images/prorm-hpc4.sif"
bash scripts/hpc4/fetch_candidate_image.sh <image-build-commit>
export PRORM_IMAGE="$PRORM_PROJECT_ROOT/images/prorm-hpc4.sif"
```

Submit the L20 image smoke and require `COMPLETED` with `ExitCode=0:0`:

```bash
bash scripts/hpc4/submit_gpu_smoke.sh \
  "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/system-reports" gpu-l20 00:20:00
squeue -u "$USER"
sacct -j <smoke-job-id> --format=JobID,State,Elapsed,ExitCode,AllocTRES
```

## Hugging Face staging

`smoke.yaml` and `main.yaml` have different semantic hashes, so each needs its own
config-bound inventory even though they resolve the same public repositories. Run these
jobs sequentially; wait for the first to complete before submitting the second.

```bash
bash scripts/hpc4/submit_hf_stage.sh \
  configs/smoke.yaml "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/hf-cache" amd 04:00:00

bash scripts/hpc4/submit_hf_stage.sh \
  configs/main.yaml "$PRORM_IMAGE" "$PRORM_PROJECT_ROOT/hf-cache" amd 04:00:00
```

Both jobs must finish with `ExitCode=0:0`. Their inventories are written under
`$PRORM_PROJECT_ROOT/hf-cache/inventories/`.

## Execution-equivalent smoke

This runs the complete pipeline on 24 prompts. It is the required resource and convergence
gate for the formal experiment, especially for the nested Pro-RM solve.

```bash
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml \
  "$PRORM_IMAGE" \
  "$PRORM_PROJECT_ROOT/hf-cache" \
  "$PRORM_SCRATCH_ROOT/runs/smoke" \
  gpu-l20 08:00:00
```

Require all of the following before the main submission:

- array task is `COMPLETED` with `ExitCode=0:0`;
- `reward_result.json`, nine adapters, rollout metrics, and response JSONL exist;
- MLE-RM, Pro-RM, and all three NGD solves report convergence;
- peak memory fits one L20 and measured runtime supports the main wall-time request.

```bash
sacct -j <smoke-job-id> \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES
```

## Three formal seeds

Only after the execution smoke passes:

```bash
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml \
  "$PRORM_IMAGE" \
  "$PRORM_PROJECT_ROOT/hf-cache" \
  "$PRORM_SCRATCH_ROOT/runs/main" \
  gpu-l20 48:00:00
```

The three-task array maps indices `0,1,2` to seeds `20261001,20261002,20261003`.
Monitor it without editing partial outputs:

```bash
squeue -u "$USER"
sacct -j <main-job-id> --format=JobID,State,Elapsed,ExitCode,MaxRSS,AllocTRES
tail -f "$PRORM_SCRATCH_ROOT/runs/main/logs/prorm-main-<job-id>_<task>.out"
```

## Aggregate and archive

Only after all three tasks are terminal `COMPLETED` with `ExitCode=0:0`:

```bash
bash scripts/hpc4/submit_aggregate.sh \
  configs/main.yaml "$PRORM_IMAGE" "$PRORM_SCRATCH_ROOT/runs/main" amd 01:00:00
```

After verifying `aggregate.json`, archive the immutable run directory to project storage
with `rsync`. Do not delete the scratch source until checksums and the copied files agree.

```bash
rsync -a --info=progress2 \
  "$PRORM_SCRATCH_ROOT/runs/main/" \
  "$PRORM_PROJECT_ROOT/archive/main/"
```
