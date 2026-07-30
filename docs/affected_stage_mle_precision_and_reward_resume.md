# Affected-stage analysis: MLE solve precision and reward substage recovery

## Trigger

Formal reward array `1675855` completed Pro-RM and all natural-direction calculations for
seeds `20261001`, `20261002`, and `20261003`, but failed the final MLE-RM convergence gate.
The failed results and logs remain preserved under the persistent failure archive.

## Root cause

`fit_mle_reward` computed complete-edge differences and the numerical-rank epsilon in the
artifact's float32 precision before converting the solve to float64. On the formal full-rank
designs the float32-scale QR threshold rejected valid directions and sent the solve through
a truncated SVD. On structurally underdetermined smoke designs, independently rounded
float32 edge differences also broke exact within-prompt cycle identities and introduced
spurious singular directions. These effects could leave the head-space gradient above its
gate even after the configured optimizer budget.

The fix converts node features and rewards to the frozen float64 solve precision before
forming edge differences, computes numerical rank in float64, and uses exact spectrally
scaled QR/SVD coordinates. The coordinate map preserves every represented logit, objective,
target, and minimum-norm head. The configured budget is counted in LBFGS optimizer
iterations, while extra strong-Wolfe closure evaluations are not misclassified as optimizer
iterations. Final convergence is checked against the original head-space gradient gate.

## Earliest affected component and recomputation closure

The earliest affected component is the MLE-RM fit inside the reward stage. The necessary
formal recomputation closure is:

```text
MLE-RM fit
  -> MLE-RM reward evaluation
  -> MLE-RM natural direction
  -> MLE-RM local/tabular policy evaluation
  -> combined reward result and reward receipt
```

Pro-RM fit, Pro-RM reward evaluation, the Pro-RM natural direction, the oracle direction,
and the Pro-RM/oracle/reference local evaluations are outside this dependency closure. They
must not be solved again. Their source result SHA-256 and original producer are bound into
the retry. The current consumer recomputes the Pro normal-equation residual, each saved
direction's damped-Fisher residual, and every saved evaluation value before preserving the
source component. Any mismatch fails closed.

## Real-artifact validation

The corrected MLE solve was exercised against all three formal materialized artifacts while
reusing the preserved Pro heads:

| seed | Slurm job | MLE gradient norm | MLE optimizer iterations | Pro gate |
| --- | --- | ---: | ---: | ---: |
| 20261001 | `1676221_0` | `2.211120559101411e-08` | 9 | previously validated |
| 20261002 | `1676221_1` | `7.668836213504435e-11` | 12 | previously validated |
| 20261003 | `1676221_2` | `1.0995348142459582e-08` | 10 | previously validated |

All three jobs completed with exit code `0:0`. These isolated diagnostics are evidence for
the final implementation fix, not formal reward outputs. The same final path reached an MLE
gradient norm of `3.73032962973098e-08` within the smoke configuration's 20-iteration
budget. Pro residuals were independently validated in the prior real-artifact diagnostic
and will be recomputed again by the formal retry's component validator.

## Scientific invariants

This change does not alter `configs/main.yaml`, seeds, data/model revisions, exact-delta
targets, Fisher estimator, damping, solver gates, beta grid, rollout budget, metrics, or the
estimand. Formal retry remains required under a new immutable producer only after local
tests, CI, image build, GPU gate, and an affected-path smoke pass.

## Adapter recovery implementation

The same change series makes each of the nine adapter exports an independent atomic
checkpoint. Every adapter now has a component receipt binding config, artifact, reward
result, seed, method, beta, direction, fixed LoRA-A/layout, producer, and all output file
hashes. A retry reuses each valid component without reloading the policy; an invalid
component is moved to a recoverable rejection directory and only that adapter is rebuilt.
Final adapter metadata binds all nine component-receipt hashes.

This implementation change has `adapters` as its earliest affected stage. No formal adapter
has yet been produced for the current three-seed experiment, so it causes no recomputation:
the nine adapters and all downstream stages will be computed for the first time after the
formal reward receipts pass.
