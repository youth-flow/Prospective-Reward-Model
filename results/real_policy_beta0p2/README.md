# Real Policy Evaluation at beta = 0.2

This bundle is the retained local evidence for the formal three-seed evaluation
that writes each one-step NGD update into the model's LoRA-B parameters and then
generates new responses on the fixed test prompts. It excludes DPO, AuxDPO, and
all tabular/candidate-pool metrics.

## Frozen protocol

- Seeds: 20261001, 20261002, 20261003.
- Policies per seed: pi0, MLE-RM NGD, Pro-RM NGD, and oracle-r* NGD.
- Beta: 0.2, fixed before the formal rollout.
- Update: `LoRA-B = beta_free_natural_direction / beta`.
- Test rollout: 512 fixed prompts x 4 fresh responses = 2,048 rows per policy.
- Total: 12 policy rollouts and 24,576 generated response rows.
- Reward: mean oracle reward on the newly generated responses.
- KL: Rao-Blackwellized forward sequence KL from the updated policy to pi0,
  evaluated on updated-policy trajectories.
- Utility: `J = R - beta K`.

The adapter smoke test confirmed that an updated adapter was loaded, generation
was fresh and nonempty, and both oracle reward and forward KL were finite. Across
the three seeds, all nine updated adapters were materialized; the maximum recorded
LoRA-B writeback error is below 5e-8.

## Three-seed results

Values are mean plus/minus sample standard deviation across the three formal seeds.

| Policy | R | K | J |
|---|---:|---:|---:|
| oracle-r* NGD | 0.039797 +/- 0.019433 | 0.096179 +/- 0.025270 | **0.020561 +/- 0.019306** |
| MLE-RM NGD | 0.023927 +/- 0.008613 | 0.031421 +/- 0.009214 | **0.017643 +/- 0.008455** |
| Pro-RM NGD | 0.031558 +/- 0.015645 | 0.080526 +/- 0.023497 | **0.015453 +/- 0.016324** |
| pi0 | 0.006484 +/- 0.014937 | 0 | **0.006484 +/- 0.014937** |

The robust conclusion is that every updated policy improves J over pi0 in every
seed. The mean paired improvements are 0.014077 for oracle-r*, 0.011159 for MLE,
and 0.008969 for Pro. Pro exceeds MLE in seeds 20261001 and 20261002 but not in
seed 20261003; consequently, the three-seed mean does not support Pro > MLE.

This true rollout result supersedes any interpretation of the earlier finite
candidate-pool evaluation as evidence about the utility of the actually updated
language-model policies. The candidate-pool result remains exploratory legacy
evidence only.

## Integrity and storage

The formal integrity audit passed for all seeds with three updated adapters and
four fresh policy rollouts per seed. The complete 43 MB server run is archived at:

`/project/sigroup/yyangjo/prorm/archives/real-policy-beta0p2-ada8d6b-20260802/`

This local bundle intentionally retains only the aggregate, integrity audit,
seed-level evaluations, adapter metadata, smoke record, and full archive checksum
manifest. Raw rollouts and adapter weights remain in the server archive.
