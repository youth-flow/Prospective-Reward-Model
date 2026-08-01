# DPO/AuxDPO Extension Protocol

## Frozen comparison

This extension preserves Main Experiment v1's three seeds, materialized prompt/candidate
graph, exact soft Bradley--Terry oracle, Qwen2.5-1.5B reference, fixed-A last-layer
LoRA-B rank-4 capacity, train/test split, candidate-pool reference, beta values, Fisher
evaluation geometry, and all six reported metrics. It does not regenerate candidates or
recompute MLE-RM, ProRM, Fisher selection, or the five existing policies.

The only affected stage is direct preference optimization and its downstream evaluation:

```text
immutable materialized candidates
  -> reference response-token log probabilities
  -> DPO/AuxDPO fits for 3 seeds x 3 betas x 2 methods
  -> unified seven-policy and four-reward evaluation
  -> three-seed aggregate
  -> integrity audit and immutable archive
```

Every reused artifact remains unchanged. New metadata records its original SHA-256 and
the new consumer identities.

## DPO

For a materialized response `y`, the implicit reward is

```text
r_beta(x,y) = beta * [log pi_theta(y|x) - log pi_0(y|x)].
```

Both log probabilities sum response-token conditional log probabilities on the exact
stored token sequence. Training minimizes the exact expected binary logistic loss under
the stored soft oracle labels. Each beta is trained independently; no tangent-score
linearization, Fisher inverse, NGD, or TRPO update is used.

## AuxDPO

AuxDPO jointly optimizes the same LoRA-B policy and one prompt-candidate auxiliary offset.
The offset is shared across all five edges containing that candidate. It enters the
augmented preference reward and is constrained by the paper's batchwise reference-score
moment penalty. The paper's settings are frozen: `lambda_null=1`, `lambda_amp=0.01`,
`delta_cap=1`, and auxiliary learning rate `5e-3`.

The auxiliary offset directly enters the training reward and indirectly changes the
learned policy through joint optimization. It is not a test-time policy logit. Because it
is a training-sample variable rather than a generalizing reward network, the formal test
reward row is explicitly the policy-implied reward `beta log(pi/pi0)`; the augmented
reward is limited to train diagnostics.

## Evaluation

The induced frozen candidate-pool policy is the normalized sequence likelihood-ratio
tilt of the uniform pool reference. Reward rows report test NLL, prompt-centered MSE, and
the frozen train-Fisher approximate regret. Policy rows report `R`, `K`, `J`, `delta_J`,
`beta_KL`, and `J_close`. Every seed must satisfy the two Gibbs identities to numerical
precision. Test data are used once for evaluation and never for fitting, stopping, or
hyperparameter selection.

The headline comparison is policy utility. AuxDPO's augmented training NLL, global
null-space moment norm, and auxiliary amplitude are retained as implementation diagnostics,
not substituted for held-out reward metrics.
