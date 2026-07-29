# Codebase Guide

## Configuration

- `configs/main.yaml`: the formal three-seed experiment
- `configs/smoke.yaml`: a small execution-equivalent profiling configuration
- `src/smart_reward/config.py`: closed-schema validation and semantic hashing

## Data construction

- `prompts.py`: deterministic MultiPref prompt preparation and split serialization
- `data.py`: inspectable candidate-node JSONL with raw and standardized oracle scores
- `runtime.py`: pinned dataset/model loading, seeding, hashing, and projection diagnostics
- `hf.py`: exact generation, sequence scoring, hidden-state pooling, Skywork scoring, and fixed-A LoRA setup
- `scores.py`: sequence log probabilities and per-sample LoRA-B score vectors
- `oracle.py`: train-only median/MAD affine oracle transform
- `exact_phase.py`: end-to-end node/edge materialization
- `artifacts.py`: atomic safetensors artifact with SHA-256 verification

## Training and policy geometry

- `exact.py`: split tensors, all-pair construction, MLE-RM, Pro-RM, and reward-fit metrics
- `linear.py`: matrix-free empirical Fisher operator
- `pcg.py`: conjugate-gradient solver
- `evaluation.py`: beta-free directions, local regret, tabular utility, and rollout summaries
- `exact_run.py`: trains both reward heads and writes Evaluation 1 plus held-out local Evaluation 2
- `policy_update.py`: maps a flat direction into LoRA-B tensors
- `exact_policy.py`: loads the three train-fitted directions and exports nine common-beta adapters

## Fresh rollout and execution

- `rollout.py`: generates new test responses and writes per-response and aggregate utility
- `statistics.py`: descriptive three-seed aggregation
- `cli.py`: the only command-line interface
- `scripts/hpc4/`: staging, smoke, seed, and aggregation Slurm entry points

There is no alternate training or evaluation route in the package.
