# Phase 2 calibration recovery protocol

This document records the single recovery decision made after the first
common-beta calibration pilot terminated. It is an optimization-protocol
amendment, not an efficacy result and not a retry under the original design
identity.

The scientific recovery schedule remains one-shot. Execution revision 1
(`1648094`) stopped during immutable-asset verification, before the trainer was
entered, because Hugging Face Datasets attempted to place a runtime lock in the
read-only shared cache. Its failure evidence is retained. Execution revision 2
keeps the shared Hub cache and inventory read-only and routes only per-process
Datasets locks and derived Arrow cache files to an empty job-local scratch
directory. This is an infrastructure correction; it does not change examples,
labels, heads, optimizer state, schedule, gates, or any scientific identity.
Both submission-time and job-time gates verify the exact three revision-1
`FAILED` markers, their complete file sets, Slurm stdout/stderr hashes, and the
absence of every trainer/result file against
`configs/phase2_recovery_infrastructure_failure.json`. Revision 2 cannot start
if that pre-trainer provenance changes.

## 1. Immutable parent outcome

The original calibration design has identity
`0c8820b67b8ca85c23cd5c31b8d25001018b31a7271f6a861d88cfae2f85d7ca`.
Its three excluded pilot seeds, `20260801`, `20260802`, and `20260803`, all
terminated with exit code 2 at the same fail-closed gate:

```text
bt_mle did not satisfy the sustained first-order
gradient-ratio gate by 5760 steps
```

Candidate generation, frozen-feature materialization, Qwen3 oracle rendering,
the four-replicate label construction, GPU checks, and artifact verification
had already succeeded. The training pipeline executes BT-MLE before ProRM+,
so these failures establish only that the original BT optimizer path failed.
They do not report a ProRM+ result and contain no held-out learner comparison.

The parent `FAILED` markers, logs, manifests, and artifacts remain immutable.
The recovery receives a new config hash, design identity, run directory, and
result lineage.

## 2. Train-only diagnostic

The diagnostic was committed before execution and ran as HPC4 job `1647982`
on seed `20260801`. Its output is:

```text
diagnostics/bt-convergence/seed-20260801-commit-791c2da.json
SHA256 bd7c3d80c26500ee273b14bb1ea8bc3428f71fdb319a49c792bf4de567e2c6a9
```

The diagnostic:

- reused the immutable materialized candidates and frozen features;
- reconstructed the exact named four-replicate label stream;
- accessed only the training split;
- did not evaluate held-out reward, regret, policy utility, or learner order;
- did not serialize oracle values, labels, or reward-head vectors;
- was ineligible for the primary claim.

The exact label-stream and artifact hashes are part of the diagnostic output.

### 2.1 Constant-learning-rate result

The original optimizer was deterministic full-data AdamW. A microbatch of 64
was only a memory partition: every optimizer update accumulated the complete
training-set gradient.

The first-order gate was

$$
\rho_t =
\frac{\lVert\nabla L_{\mathrm{BT}}(w_t)\rVert_2}
{\lVert\nabla L_{\mathrm{BT}}(w_0)\rVert_2}
\le 10^{-3}
$$

for three consecutive checks, measured every 20 updates in FP64 after the
update and without gradient clipping.

With constant `lr=1e-3`, the observed ratios were:

| Update | Gradient ratio |
|---:|---:|
| 720 | 1.5917849 |
| 1,000 | 1.6161880 |
| 2,000 | 1.8148644 |
| 4,000 | 2.1015190 |
| 5,760 | 2.1762233 |

The result is not a near-threshold miss. Constant-step AdamW was oscillating
far above the required first-order accuracy, so merely extending the same
learning rate is not an admissible remedy.

### 2.2 Decay and optimization-reference results

The precommitted decay probe retained all AdamW moments and changed only the
learning rate. It first crossed the threshold at update 6,860 and completed
the required three consecutive checks at update 6,900:

| Update | Learning rate | Gradient ratio | Consecutive passes |
|---:|---:|---:|---:|
| 6,840 | `1e-4` | 0.00578663 | 0 |
| 6,860 | `1e-4` | 0.00064969 | 1 |
| 6,880 | `1e-4` | 0.00038213 | 2 |
| 6,900 | `1e-4` | 0.00032972 | 3 |

A separate zero-initialized, full-batch FP64 L-BFGS reference reached gradient
ratio `3.92696e-5`. Its objective was `0.67166348`, compared with
`0.67167752` at the selected AdamW iterate. Thus two independent optimizer
paths found finite iterates that pass the configured first-order gate. The
original failure was therefore a failure of the constant-step AdamW path, not
an inability to find any admissible finite iterate. This numerical observation
is not, by itself, a theorem that excludes logistic separation.

L-BFGS remains a diagnostic reference. It does not replace the common
first-order optimizer used for BT-MLE and ProRM+.

## 3. Frozen one-shot recovery schedule

Every Phase 2 head trained through the common first-order controller receives
the same deterministic schedule:

| Inclusive optimizer updates | Learning rate |
|---:|---:|
| 1–5,760 | `1e-3` |
| 5,761–6,760 | `3e-4` |
| 6,761–8,760 | `1e-4` |
| 8,761–10,760 | `3e-5` |
| 10,761–12,760 | `1e-5` |

The learning rate is installed immediately before the corresponding
one-indexed update. AdamW moments are never reset at a boundary. Training
starts again from an exact zero head and fresh optimizer state; no failed
checkpoint is resumed.

The unchanged primary gate is:

- relative gradient-ratio tolerance `1e-3`;
- minimum 100 updates;
- checks every 20 updates;
- three consecutive passing checks;
- FP64, full-data, post-update, unclipped audit;
- exact-zero-initialization gradient in the denominator;
- fail closed at update 12,760;
- no validation or test selection.

The selected primary head remains the first iterate completing the sustained
gate. The 720-update compute-matched state remains diagnostic only. The
5,760-update state records the legacy boundary and cannot select a head.

For every executed head, success additionally requires a per-update AdamW
state audit: the scalar optimizer step must match the controller update before
and after every call, `exp_avg` and `exp_avg_sq` must retain the required
shape/dtype/device, and the selected head plus complete optimizer state must
match byte-bound fingerprints after restoration. This is checked for all five
executed trainers, not only the two primary heads.

The shared first-order controller applies the schedule equally to primary
BT-MLE, primary ProRM+, exact-margin ProRM+, exact-soft-label BT, and the
low-dimensional control executed in this recovery job. The same recovery
settings also govern any later head-retraining sensitivity cell, although no
sensitivity cell is executed or accepted by this train-only recovery. This
keeps optimizer family, initialization, clipping, dtype, data, and convergence
requirements matched across estimators.

If this one-shot recovery fails, the protocol stops for an explicit optimizer
redesign. It does not relax the tolerance, add another adaptive tail, delete a
seed, or inspect held-out outcomes.

## 4. Materialized-artifact reuse

Recovery may reuse only the immutable materialization produced by the failed
parent Phase-2 pilot:

- prompts and Qwen2.5 candidate responses;
- canonical graph files;
- frozen reward features and policy scores;
- their immutable metadata.

It may not reuse a run manifest, repeated-label realization, reward head,
optimizer state, beta, policy update, diagnostic checkpoint, or any Phase-2
result. The new job constructs a fresh current-consumer manifest, reconstructs
the deterministic labels, and trains every head from zero.

Cross-commit reuse is allowed only through the tracked parent-failure registry.
That registry binds, per seed, the parent failure marker, run manifest,
materialization/verification receipts, artifact metadata, all six artifact
files, candidate and prompt identities, tensor bytes, old producer commit,
container image, and Hugging Face inventory.

Before training, the recovery job must verify:

1. the old producer is an ancestor of the new consumer commit;
2. all materialization-relevant source/config bytes are unchanged;
3. image and model-cache inventories are byte-identical;
4. the parent run is `FAILED`, has no `SUCCESS`, and failed at the declared BT
   gate;
5. the parent run's artifact link resolves to the exact canonical artifact
   directory and every tracked byte matches;
6. the current run manifest belongs to the new consumer identity;
7. no held-out evaluator, policy rollout, or final oracle session is opened.

Any mismatch terminates the recovery. There is no fallback that silently
regenerates data or rewrites old metadata.

Seed `20260801` is also a reproducibility anchor. Its fresh train-oracle rescore,
named-RNG initial/final states, four-replicate count/win/`h` hashes, mean-`h`
hash, derived seed, and total annotation count must exactly reproduce the
already frozen diagnostic evidence. This catches tokenizer, kernel, data-route,
or RNG drift before the recovery can be marked successful. The other two seeds
retain their own deterministic streams and are not compared with seed
`20260801` values.

## 5. Interpretation

This recovery preserves the scientific comparison. It changes neither the
BT-MLE objective nor the ProRM+ Fisher-GMM objective, repeated labels, models,
prompt population, operational oracle, common-beta design, or downstream
utility estimand. It removes an identified optimizer discretization failure
under an independently hashed, train-only engineering decision.

The three recovery seeds remain permanently excluded from confirmatory
evidence. Recovery outputs never enter a calibration aggregate and never
produce a beta candidate. An all-seed recovery pass only authorizes a new,
schedule-frozen full calibration pilot; that later pilot must independently
materialize every required calibration endpoint before any beta freeze.
