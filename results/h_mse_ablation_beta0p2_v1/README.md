# H/MSE Synthetic-Oracle Mechanism Extension

This directory retains the compact, auditable evidence for the frozen three-seed
H/MSE extension. Full adapters, annotation tensors, and rollout rows remain in the
immutable HPC4 archive; the local copy contains the results and receipts needed for
analysis and continuation without duplicating large artifacts.

## Frozen experiment

- Seeds: `20261001`, `20261002`, `20261003`
- Split: `3072 / 512 / 512` prompts
- Candidate graph: six candidates and all 15 unordered edges per prompt
- Policy update: one-step NGD with the frozen selected train Fisher and `beta = 0.2`
- Repeated annotation: `gamma = 0.9`, `N ~ Geom(0.1)`,
  `S ~ Binomial(N, sigmoid(delta_r_star))`
- H estimator: unclipped float64 randomized-series estimator; one equal-weight outer
  observation per edge
- Evaluation: test NLL, centered reward MSE, two-fold cross-product approximate
  regret, oracle reward `R`, forward KL `K = KL(pi || pi0)`, and `J = R - 0.2 K`
- Test data was evaluation-only: no hyperparameter selection, seed filtering, or
  post-hoc clipping used test results.

## Policy evaluation

Values are the mean and sample standard deviation across the three prespecified
seeds. Each policy uses 512 test prompts and six fresh responses per prompt.

| Policy | R | K | J |
|---|---:|---:|---:|
| Oracle-NGD | 0.042575 +/- 0.015709 | 0.096160 +/- 0.025721 | **0.023343 +/- 0.016486** |
| Oracle-Pro | 0.036786 +/- 0.016851 | 0.080521 +/- 0.023855 | **0.020681 +/- 0.018369** |
| H-Pro | 0.037918 +/- 0.018857 | 0.089595 +/- 0.024579 | **0.019999 +/- 0.017289** |
| H-MLE | 0.024268 +/- 0.008368 | 0.032918 +/- 0.008680 | 0.017684 +/- 0.008174 |
| H-MSE | 0.025366 +/- 0.009980 | 0.040418 +/- 0.010068 | 0.017282 +/- 0.010416 |
| Oracle-MSE | 0.025152 +/- 0.012441 | 0.039711 +/- 0.011687 | 0.017210 +/- 0.013477 |
| Oracle-MLE | 0.022266 +/- 0.009233 | 0.031385 +/- 0.009454 | 0.015989 +/- 0.009602 |
| pi0 | 0.003878 +/- 0.010156 | 0.000000 +/- 0.000000 | 0.003878 +/- 0.010156 |

## Reward evaluation

| Reward model | NLL | Centered MSE | Approximate regret |
|---|---:|---:|---:|
| Oracle-MLE | **0.683446 +/- 0.001089** | **0.302047 +/- 0.008749** | 0.006116 +/- 0.002547 |
| Oracle-MSE | 0.686504 +/- 0.001333 | 0.310552 +/- 0.008909 | 0.004687 +/- 0.003225 |
| Oracle-Pro | 0.732853 +/- 0.003183 | 0.498254 +/- 0.020681 | **0.000160 +/- 0.018767** |
| H-MLE | **0.689049 +/- 0.002166** | **0.322683 +/- 0.008559** | 0.009633 +/- 0.005899 |
| H-MSE | 0.691273 +/- 0.002438 | 0.328563 +/- 0.009059 | 0.006450 +/- 0.004649 |
| H-Pro | 0.750720 +/- 0.005865 | 0.577938 +/- 0.031509 | **0.005029 +/- 0.010524** |

## Confirmed interpretation

Within the exact-label group, Oracle-Pro improves mean downstream `J` over
Oracle-MSE by `0.003472` and over Oracle-MLE by `0.004693`. Under repeated noisy
annotations, H-Pro improves mean `J` over H-MLE by `0.002315` and over H-MSE by
`0.002717`; its mean is within `0.000683` of Oracle-Pro. In both groups, the Pro
criterion wins the downstream comparison despite materially worse NLL and MSE.
This is the intended mechanism evidence that reward-fit quality and downstream
policy utility need not agree.

The result is descriptive mechanism evidence, not a high-powered significance
claim: there are three independent seeds, the standard deviations are large, and
the third seed has low utility for every updated policy. The cross-product regret
estimator is unbiased but may be negative at finite sample size; values were not
zero-clipped.

## Evidence map

- `evidence/aggregate.json`: eight-policy three-seed result
- `evidence/reward-aggregate.json`: six-reward three-seed result
- `evidence/integrity-audit.json`: row-count, hash, writeback, and recomputation audit
- `evidence/provenance.json`: immutable-source bridge and dependency closure
- `evidence/seed-*/evaluation.json`: paired per-seed policy and reward metrics
- `evidence/seed-*/reward_result.json`: fitted heads, convergence evidence, and
  beta-free natural directions
- `evidence/seed-*/*-receipt.json`: fresh rollout receipts
- `evidence/archive-receipt.json` and `evidence/archive-sha256sums.txt`: immutable
  HPC4 archive receipt and manifest

See `RUN_RECEIPT.md` for immutable identities and retrieval verification.
