# Prospective Reward Model

This repository implements one experiment only: compare exact-soft-label MLE-RM and
Pro-RM under reward-class misspecification, then evaluate the one-step policies they
induce in a fixed LoRA-B tangent.

## Main configuration

[`configs/main.yaml`](configs/main.yaml) is the formal experiment definition.

- Seeds: `20261001`, `20261002`, `20261003`
- Prompts: 4096 MultiPref prompts, split 3072/512/512
- Reference policy: Qwen2.5-1.5B-Instruct
- Candidates: 6 reference-policy responses per prompt, all 15 unordered pairs
- Oracle: Skywork-Reward-V2-Llama-3.2-3B
- Reward class: frozen Qwen response feature plus a bias-free 1536-dimensional head
- Policy tangent: final-layer `q_proj`/`v_proj`, rank-4 LoRA, fixed A and trainable B
- Learned rewards: MLE-RM and Pro-RM
- Policy families: reference, MLE-NGD, Pro-NGD, and oracle-NGD
- Common KL coefficients: `beta = [1, 2, 4]`

The oracle provides exact reward differences for this controlled experiment. The YAML,
CLI, package, tests, and HPC4 scripts expose only the workflow shown below.

## Workflow

```text
MultiPref prompts
  -> pi0 candidates on train/validation/test
  -> Skywork scores + reward features + LoRA-B score vectors
  -> exact node artifact and all pair edges
  -> MLE-RM and Pro-RM
  -> reward-fit and held-out local/tabular evaluation
  -> three beta-free NGD directions
  -> nine beta-scaled LoRA adapters
  -> ten independently resumable fresh test rollouts
  -> three-seed descriptive aggregate
```

Validate the configuration:

```bash
prorm config-check configs/main.yaml
```

Run one seed end to end after the pinned Hugging Face assets have been staged:

```bash
prorm run-seed configs/main.yaml runs/20261001 \
  --seed 20261001 --device cuda
```

The output contains:

```text
runs/20261001/
  artifact/
    prompts.jsonl
    candidates.jsonl  # prompt, response, raw oracle score, standardized r*
    edges.jsonl
    metadata.json
    tensors.safetensors
  reward_result.json
  adapters/
    metadata.json
    mle_rm__beta_1/ ... oracle__beta_4/
  policy_rollout_parts/
    pi0/ ... oracle__beta_4/
  policy_utility/
    rollouts.jsonl
    metrics.json
  stage_receipts/
    materialize.json
    reward.json
    adapters.json
```

Production runs use `scripts/hpc4/submit_pipeline.sh`. It submits dependency-ordered
Slurm arrays for materialization, reward fitting, adapter export, per-policy rollout,
per-seed rollout assembly, and final aggregation. Every completed stage is hash-bound
to the config, seed, Git commit, SIF, Hugging Face inventory, and upstream artifacts.

Aggregate the three seeds:

```bash
prorm aggregate configs/main.yaml runs/aggregate.json \
  --reward-results runs/20261001/reward_result.json \
                   runs/20261002/reward_result.json \
                   runs/20261003/reward_result.json \
  --rollout-results runs/20261001/policy_utility/metrics.json \
                    runs/20261002/policy_utility/metrics.json \
                    runs/20261003/policy_utility/metrics.json
```

## Evaluation boundary

Reward fitting is computed on the frozen reference-policy candidate pool. Local regret
and tabular utility apply the train-fitted directions to the frozen test candidate pool;
they are exact conditional on that finite pool, not population-exact.
Actual reward, forward KL, and regularized utility are Monte Carlo estimates from fresh
test rollouts. Validation is diagnostic only; test data never selects a model or beta.

See [`docs/experiment_protocol.md`](docs/experiment_protocol.md),
[`docs/theory.md`](docs/theory.md), and
[`docs/codebase_guide.md`](docs/codebase_guide.md).

## Development

```bash
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```
