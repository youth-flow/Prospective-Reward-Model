# Real Policy Evaluation at beta = 0.2 (m = 6)

This bundle is the retained local evidence for the corrected formal three-seed
evaluation of the actually updated language-model policies. It extends the
previous rollout from four to six fresh responses per test prompt while
preserving every original row.

## Frozen protocol

- Seeds: 20261001, 20261002, 20261003.
- Policies: pi0, MLE-RM NGD, Pro-RM NGD, and oracle-r* NGD.
- Beta: 0.2, fixed before policy writeback and rollout.
- Updated policies: the validated LoRA-B adapters from the original formal run;
  material, reward heads, Fisher/moments, directions, and adapters were not
  recomputed.
- Base rollout: 512 test prompts x 4 responses per policy.
- Increment: exactly two independently seeded responses per prompt, with seed
  namespace real-rollout-extension-4-to-6-batch.
- Final rollout: 512 prompts x 6 responses = 3,072 rows per policy, or 36,864
  rows across 12 policy-seed instances.
- Reward: mean oracle reward on newly generated fixed-test responses.
- KL: Rao-Blackwellized forward sequence KL from the updated policy to pi0,
  evaluated on updated-policy trajectories.
- Utility: J = R - 0.2 K.

DPO, AuxDPO, and tabular/candidate-pool metrics are outside this evaluation.

## Three-seed results

Values are mean plus/minus sample standard deviation across the three formal
seeds.

| Policy | R | K | J |
|---|---:|---:|---:|
| oracle-r* NGD | 0.042575 +/- 0.015709 | 0.096160 +/- 0.025721 | **0.023343 +/- 0.016486** |
| Pro-RM NGD | 0.036786 +/- 0.016851 | 0.080521 +/- 0.023855 | **0.020681 +/- 0.018369** |
| MLE-RM NGD | 0.022266 +/- 0.009233 | 0.031385 +/- 0.009454 | **0.015989 +/- 0.009602** |
| pi0 | 0.003878 +/- 0.010156 | 0 | **0.003878 +/- 0.010156** |

The aggregate ordering is

~~~text
R: oracle-r* > Pro > MLE > pi0
J: oracle-r* > Pro > MLE > pi0
~~~

Seed-level regularized utilities are:

| Seed | pi0 | MLE-RM | Pro-RM | oracle-r* |
|---|---:|---:|---:|---:|
| 20261001 | 0.008395 | 0.017162 | 0.026005 | 0.031253 |
| 20261002 | 0.010992 | 0.024950 | 0.035800 | 0.034383 |
| 20261003 | -0.007752 | 0.005853 | 0.000238 | 0.004393 |

Every updated policy exceeds pi0 in every seed. Pro-RM exceeds MLE-RM in two of
three seeds and in the three-seed mean, but not in seed 20261003; with n = 3,
the result is descriptive rather than a high-power statistical claim.

This m = 6 result supersedes the m = 4 aggregate as the formal rollout estimate.
It does not discard the earlier evidence: all 24,576 original rows were matched
exactly against the immutable m = 4 archive, and the 12,288 added rows are
restricted to response indices 4 and 5.

## Integrity and storage

The formal integrity audit passed for all three seeds. It verifies four rollout
policies per seed, 3,072 rows per policy, the 4 + 2 extension boundary, source
artifact and reward hashes, adapter provenance, runtime image, and aggregate
hash.

The complete server run is archived at:

/project/sigroup/yyangjo/prorm/archives/real-policy-beta0p2-m6-2d1e5e1-20260802/

The archive contains 48 files and matched the scratch run byte for byte at
archive creation. This local bundle intentionally retains only the aggregate,
integrity audit, seed-level evaluations, and full archive checksum manifest.
Raw rollouts, metadata, receipts, and logs remain in the server archive.
