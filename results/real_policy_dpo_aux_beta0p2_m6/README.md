# DPO/AuxDPO true-policy evaluation at beta = 0.2

This directory contains the compact, auditable evidence for the three-seed
DPO/AuxDPO extension of the frozen six-policy experiment. It intentionally
does not duplicate the full rollout rows locally.

## Frozen protocol

- Seeds: `20261001`, `20261002`, `20261003`
- Test prompts: 512 per policy and seed
- Fresh responses: 6 per prompt
- New policies: `dpo__beta_0p2`, `auxdpo__beta_0p2`
- Policy metrics: `R`, forward `K = KL(pi || pi0)`, and `J = R - 0.2 K`
- Reward metrics: NLL, centered MSE, and two-fold cross-U approximate regret
- The four prior policies are referenced by the SHA-256 of their already
  audited m=6 seed evaluations; they were not regenerated.

## Three-seed policy results

| Policy | mean R | mean K | mean J |
|---|---:|---:|---:|
| pi0 | 0.00387810 | 0 | 0.00387810 |
| DPO | 0.00392438 | 0.000000587 | 0.00392426 |
| AuxDPO | 0.00391984 | 0.000000596 | 0.00391972 |
| MLE-RM | 0.02226554 | 0.03138490 | 0.01598856 |
| ProRM | 0.03678556 | 0.08052122 | 0.02068131 |
| r-star | 0.04257500 | 0.09616023 | 0.02334296 |

DPO and AuxDPO are numerically close to each other and to `pi0`. Their tiny
forward KL shows that the fitted adapters induced almost no policy movement;
the data therefore do not support a material downstream improvement for these
two baselines under this implementation and frozen protocol.

## Three-seed implicit-reward results

| Method | mean NLL | mean MSE | mean approximate regret |
|---|---:|---:|---:|
| DPO | 0.69352049 | 0.35950674 | 0.01996765 |
| AuxDPO | 0.69350395 | 0.35945078 | 0.02021568 |

## Provenance and storage

- Rollout implementation commit: `9cb524b54dde3152cd388ab9f9991688cf8b9c8b`
- Finalizer commit: `908149d3c2798b83cbb161b5e171b2a932a70473`
- Runtime image SHA-256:
  `ccabad42c29208253bb84ec8c8dfc228f64189206c4867742ca7c04b4915a7dd`
- Full read-only HPC4 archive:
  `/project/sigroup/yyangjo/prorm/archives/real-policy-dpo-aux-beta0p2-m6-908149d-20260803`
- Full archive size: 37 MB; 82 files; all entries passed
  `sha256sum -c archive-sha256.txt` before the archive was made read-only.
- Integrity audit status: `passed`; 18,432 new rollout rows in total.

`aggregate.json` is the canonical numerical summary. `integrity-audit.json`,
the three seed evaluations, adapter metadata, rollout metadata, and rollout
receipts retain the compact evidence needed to validate the result without
storing all generated text locally.
