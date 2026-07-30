# Affected-stage analysis: MLE solve precision and reward substage recovery

## Trigger

Formal reward array `1675855` completed Pro-RM and all natural-direction calculations for
seeds `20261001`, `20261002`, and `20261003`, but failed the final MLE-RM convergence gate.
The failed results and logs remain preserved under the persistent failure archive.

## Root cause

`fit_mle_reward` computed the numerical-rank epsilon from the artifact's float32 edge design
before converting the solve to float64. On the formal full-rank designs this produced an
approximately float32-scale QR threshold, rejected valid QR directions, and sent the solve
through a truncated SVD. The discarded directions left the head-space gradient near
`2.96e-3`, independent of additional LBFGS closures.

The fix computes rank in the frozen float64 solve precision and scales exact QR/SVD
coordinates by `sqrt(num_edges)`, so their empirical Gram is identity. The coordinate map
preserves every represented logit, objective, target, and minimum-norm head. LBFGS stopping
is checked against the original head-space gradient gate.

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

| seed | Slurm job | MLE gradient norm | MLE closures | recomputed Pro residual |
| --- | --- | ---: | ---: | ---: |
| 20261001 | `1676049_0` | `1.1622365426300153e-08` | 11 | `9.876588627329863e-07` |
| 20261002 | `1676044` | `1.0271026579455525e-08` | 14 | `9.867840781765656e-07` |
| 20261003 | `1676049_1` | `5.985795165041087e-08` | 11 | `9.234327202291543e-07` |

All three jobs completed with exit code `0:0`. These isolated diagnostics are evidence for
the implementation fix, not formal reward outputs.

## Scientific invariants

This change does not alter `configs/main.yaml`, seeds, data/model revisions, exact-delta
targets, Fisher estimator, damping, solver gates, beta grid, rollout budget, metrics, or the
estimand. Formal retry remains required under a new immutable producer only after local
tests, CI, image build, GPU gate, and an affected-path smoke pass.
