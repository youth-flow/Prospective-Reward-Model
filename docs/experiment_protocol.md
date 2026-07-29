# Experiment Protocol

## Symbols

- `x`: prompt
- `y`: response
- `pi0`: Qwen reference policy
- `r*`: standardized Skywork oracle reward
- `z(x,y)`: frozen last-response-token feature
- `phi`: linear reward-head parameter
- `s(x,y)`: score of the response log probability in LoRA-B coordinates at `pi0`
- `F`: policy Fisher matrix
- `beta`: coefficient multiplying forward KL in policy utility

## Data

The prompt split is fixed before model evaluation. For every prompt in all three splits,
`pi0` generates six responses. The artifact stores each response, its exact tokenization,
`r*`, `z`, and `s`. All 15 unordered response pairs are constructed deterministically.

Skywork's raw scalar is transformed as `(score - b) / tau`. Both `b` and `tau` are fit
using train nodes only and then frozen. The experiment uses the exact margin `delta r*`;
no binary label is sampled.

Every inspectable candidate node stores its split, candidate index, prompt, response,
generated token history, raw Skywork score, and standardized `r*`. High-dimensional reward
features and LoRA-B policy scores remain in the integrity-checked safetensors artifact.

## Reward training

The reward class is `r_phi(x,y) = z(x,y)^T phi`.

MLE-RM minimizes exact-soft-label Bradley-Terry negative log likelihood on all train
pairs with float64 LBFGS.

Define `g(r) = E_x Cov_{y|x}(s,r)` and `g(r_phi) = G phi`. Pro-RM minimizes

```text
0.5 * (G phi - g*)^T (F + lambda I)^-1 (G phi - g*).
```

The normal equation is solved by outer preconditioned conjugate gradient using
`diag(G^T G)` as a Jacobi proxy. Each inverse-Fisher product uses inner matrix-free
conjugate gradient. Because an inexact inner solve cannot support a tighter outer target,
the effective inner tolerance is `min(configured_inner_tolerance, 0.1 * outer_tolerance)`;
both the effective value and the final outer relative residual are serialized. These are
numerical precondition/accuracy rules for the same quadratic, not extra regularization or
a changed estimand. There is no stochastic reward-head optimizer and no auxiliary
optimization variable.

Fisher PCG preserves the low-rank-plus-damping structure without coordinate preconditioning
when the empirical score matrix has fewer rows than tangent dimensions. Once the empirical
Fisher can be full rank, it uses its exact diagonal as a Jacobi preconditioner. This adaptive
choice changes convergence speed, not the represented Fisher or its solution.

The default Fisher estimator is the raw second moment. Prompt-centered sample covariance
is a YAML-selectable sensitivity setting.

## Reward evaluation

Metrics are averaged over edges within each prompt, then over test prompts:

- pair Bernoulli KL, equivalently excess soft NLL
- soft Bradley-Terry NLL
- preference-probability MSE
- pairwise sign accuracy
- prompt-centered reward NMSE

Validation reports the same metrics for diagnostics only. It does not select an epoch,
head, beta, or damping value.

## Policy update

For each source in `{MLE-RM, Pro-RM, r*}`, use only the train split to solve one
beta-free direction

```text
d_r = (F + lambda I)^-1 g(r).
```

For every common `beta` in `{1, 2, 4}`, set LoRA-B to `d_r / beta`. LoRA-A and the Qwen
backbone stay frozen. The resulting inventory is one shared reference policy plus nine
updated adapters. Actual `KL(updated || reference)` is measured afterward and never used
to rescale one method independently.

## Policy evaluation

Held-out local metrics apply those fixed train-fitted directions to the frozen test
candidate pool. Test rewards and test geometry are used only for evaluation; no direction
is refitted on validation or test.

- local quadratic regret
- Fisher cosine with the oracle direction
- local target utility
- finite-candidate tabular optimum
- finite-candidate tabular regret

Fresh-rollout metrics use four new responses from every actual policy on each test prompt:

- mean standardized oracle reward
- reward improvement over `pi0`
- forward KL
- `J_beta = reward - beta * KL`
- utility improvement over `pi0`
- regret to oracle-NGD at the same beta

The fresh-rollout sampling unit is the prompt. All response-level values remain in
`rollouts.jsonl` for inspection.

Each of the ten policy instances writes an independent, resumable rollout. Prompt batches
use deterministic batch-index-derived random streams shared across policy instances.
Changing execution order or resuming a completed batch therefore cannot change the output
for a fixed config and seed. Batch size is operational and explicitly config-bound.

## Aggregation

Every seed runs the complete pipeline independently. The aggregate reports every seed,
the three-seed mean, and sample standard deviation. This is descriptive evidence; three
seeds do not support strong asymptotic significance claims.

## Execution integrity

Materialization checkpoints candidate generation at fixed prompt intervals. Reward fitting,
adapter export, each policy rollout, per-seed rollout assembly, and the final aggregate are
separate stages. A stage is reusable only when its receipt and files validate against the
same config hash, seed, Git commit, SIF SHA-256, Hugging Face inventory SHA-256, and upstream
artifact hashes. Partial or foreign outputs fail closed.
