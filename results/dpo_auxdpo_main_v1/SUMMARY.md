# DPO/AuxDPO controlled extension

This extension reuses Main Experiment v1's frozen candidates, exact soft-BTL
oracle, three seeds, test split, and rank-4 last-layer LoRA-B capacity. DPO and
AuxDPO use matched two-pass log-policy training; neither uses NGD, a Fisher
inverse, or TRPO. All values below are mean +/- sample standard deviation over
the three seeds.

## Policy utility

| beta | pi0 J | DPO J | AuxDPO J | MLE J | Pro J | oracle J | tabular J |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | -0.005947 +/- 0.024932 | -0.005954 +/- 0.024564 | -0.007153 +/- 0.024094 | 0.022223 +/- 0.023937 | 0.030297 +/- 0.017217 | 0.028116 +/- 0.016883 | 0.593510 +/- 0.023060 |
| 0.2 | -0.005947 +/- 0.024932 | -0.009113 +/- 0.025921 | -0.009061 +/- 0.023837 | 0.008957 +/- 0.024645 | 0.014055 +/- 0.021328 | 0.013120 +/- 0.021081 | 0.477700 +/- 0.023708 |
| 0.3 | -0.005947 +/- 0.024932 | -0.011452 +/- 0.023752 | -0.010364 +/- 0.022843 | 0.004132 +/- 0.024789 | 0.007729 +/- 0.022618 | 0.007180 +/- 0.022474 | 0.392979 +/- 0.024036 |

The original main-experiment ordering remains intact. Under the frozen direct
training budget, both DPO and AuxDPO are below pi0 in regularized utility for
all three betas. Their true reward increases slightly over pi0, but the roughly
0.021 candidate-pool KL cost is larger than that gain.

## Policy-implied reward evaluation

| beta | method | test NLL | centered reward MSE | approximate regret |
|---:|:---|---:|---:|---:|
| 0.1 | DPO | 0.693152 +/- 0.000019 | 0.358322 +/- 0.010869 | 0.413353 +/- 0.013504 |
| 0.1 | AuxDPO | 0.693201 +/- 0.000060 | 0.358532 +/- 0.011000 | 0.413369 +/- 0.013508 |
| 0.2 | DPO | 0.693520 +/- 0.000094 | 0.359507 +/- 0.010511 | 0.206661 +/- 0.006589 |
| 0.2 | AuxDPO | 0.693504 +/- 0.000199 | 0.359451 +/- 0.010516 | 0.206589 +/- 0.006465 |
| 0.3 | DPO | 0.694068 +/- 0.000191 | 0.361323 +/- 0.011139 | 0.138331 +/- 0.004652 |
| 0.3 | AuxDPO | 0.693909 +/- 0.000308 | 0.360608 +/- 0.010858 | 0.138188 +/- 0.004625 |

The test NLL is close to log(2), so the learned implicit rewards do not recover
useful held-out preference differences in this configuration. AuxDPO is not a
no-op: its mean training delta-squared is about 0.00410 and its augmented train
NLL is about 0.684 versus about 0.693 for DPO. That train improvement does not
reliably transfer to policy-implied test reward or policy utility. This is a
finding about the frozen two-pass, 1e-6 learning-rate, rank-4 last-layer setup,
not a general claim about fully tuned DPO or AuxDPO.

## Integrity

- Aggregate SHA-256: `1ed94791f344f0f7ee9972ecf65775be3df10b9c67e8782e93f9e4428a7355e3`
- Audit status: `passed`
- Maximum absolute `delta_J - beta_KL` residual: `2.220446049250313e-16`
- Immutable full server archive:
  `/project/sigroup/yyangjo/prorm/archives/dpo-auxdpo-main-v1-20260802`
