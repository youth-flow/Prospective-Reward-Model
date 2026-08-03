# DPO/AuxDPO convergence protocol

This run changes only DPO/AuxDPO fitting and its downstream real-policy evaluation. It
reuses the immutable six-candidate artifact, exact soft-BTL labels, Qwen2.5-1.5B reference,
rank-4 last-layer LoRA-B parameterization, and the three formal seeds. The comparison beta
is fixed at 0.2 before training. MLE-RM, ProRM, oracle policy, and pi0 are not retrained.

## Selection and stopping

Both methods start from the same zero LoRA-B reference policy and use matched deterministic
prompt orders and a physical prompt batch of two. The policy learning rate is `5e-6` with
one warmup epoch. A validation-plateau scheduler multiplies all optimizer-group learning
rates by 0.3 after one plateau epoch. Formal training allows at most 32 epochs.

The sole selection metric is the exact soft-BTL NLL of the policy-implied reward
`beta * log(pi/pi0)` on the frozen validation candidates. AuxDPO's prompt-candidate offset
is excluded from this metric because it is a train-only nuisance variable. A formal fit is
complete only after at least four epochs, two learning-rate reductions, five consecutive
trained epochs without an improvement greater than `1e-5`. The best checkpoint is selected
among trained epochs and restored; pi0 initializes the scheduler baseline but is not a
candidate trained checkpoint. Improvement over pi0 is reported separately with a `1e-4`
threshold and is not conflated with optimization convergence. If the plateau gate is not
met, the run fails closed and retains its resumable optimizer checkpoint.

The test labels and test metrics never affect scheduling, stopping, or checkpoint selection.
They are evaluated only after the convergence gate passes. Every epoch records train loss,
validation NLL and MSE, policy gradient norms, learning rates, plateau state, and whether the
best checkpoint changed.

## AuxDPO batching

AuxDPO's squared reference-score moment is evaluated on the actual physical prompt batch.
Gradient accumulation is disabled because independently squaring microbatch moments is not
equivalent to squaring the moment of the combined batch. The same physical batch is used by
DPO for a matched comparison. Batch size is admitted by a bounded GPU-memory smoke, not by
test performance. A batch of four passed a short control-flow smoke but failed closed on the
formal long-sequence inventory at the 44.4-GiB device limit; version 2 therefore freezes the
uniform memory-safe batch of two before any completed fit or test evaluation. Version 3
applies the standard linear-scaling rule to the halved physical batch, reducing policy and
auxiliary learning rates by two, and initializes the plateau scheduler with the epoch-zero
validation NLL so that a degraded first epoch cannot become its internal baseline.
Version 4 separates convergence from generalization: a fit may converge while validation
NLL remains worse than pi0, and that outcome is reported rather than converted into a zero
adapter. Exact train and validation NLL/MSE are recomputed at the restored best checkpoint.

## Dependency closure

The affected closure is:

```text
reference log-probability cache -> converged DPO/AuxDPO fits -> PEFT writeback
-> fresh six-response test rollout -> three-seed aggregate -> integrity audit -> archive
```

Candidate generation, oracle scoring, reward-model fitting, Fisher estimation, and the four
existing policies remain immutable and are validated by their original hashes.
