# Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret

[![CI](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/ci.yml)
[![HPC4 image](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/build-hpc4-image.yml/badge.svg)](https://github.com/youth-flow/Smart-Reward-Model/actions/workflows/build-hpc4-image.yml)

## 当前执行路线（先读）

当前实际采用的是 **Phase 2 budgeted end-to-end fixed-three exploratory**，不是
exact-30 正式实验：

```text
recovery 3 seeds（工程修复证据，永久排除）
  -> fresh post-recovery calibration 3 seeds（target-free）
  -> accepted freeze（唯一全局 beta + response horizon）
  -> fresh fixed-three E2E seeds 20261001..20261003
  -> seed-level strict verification
  -> fixed-three descriptive aggregate
```

这条路线中的数据、候选、重复标签、reward heads、optimizer state 和 policy rollouts
均按阶段重新生成；recovery 或 pilot head 不会被带入 E2E。三-seed aggregate 只报告
`ProRM+ - BT-MLE` 的描述性效果、异质性和 paired-seed descriptive interval，
明确禁止 p-value、显著性标签和正式 claim。exact-30 只保留为未来、需要重新冻结和
预注册的协议，不能由本轮结果事后激活。

完整理论—工程—HPC4 契约见
[Phase 2 预算版端到端实验](docs/phase2_budgeted_end_to_end.md)。
历史 [Phase 1 结果](docs/phase1_results.md)、完整
[Phase 2 正式设计](docs/phase2_design_decisions.md) 与
[post-recovery runbook](docs/phase2_post_recovery_hpc4.md) 继续保留，但不应被误读为
当前正在执行 exact-30。实时队列状态只以 HPC4 的 Slurm 与不可变运行证据为准。

Preference likelihood asks whether a reward model explains past labels. **Prospective Reward Modeling
(ProRM)** instead asks what the downstream policy optimizer will do with that reward model. The method
therefore trains for the reward error that changes the next policy update, rather than for every pointwise
reward error.

The paper title is fixed as:

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

This repository implements **ProRM+**, the observable repeated-label Fisher–GMM realization of the ideal
ProRM objective. Its formal comparison is repeated-label Bradley–Terry maximum likelihood (BT-MLE).

## Name and claim contract

The two names refer to different mathematical levels:

| Name | Meaning | Observable/trainable? |
|---|---|---|
| **ProRM** | Ideal population loss: local downstream policy regret measured under the target reward | No; it contains the unobserved target reward |
| **ProRM+** | Repeated-label identification plus Fisher–GMM dual training, implemented with ridge and PCG | Yes, under the stated data contract |

The “+” means that the unobserved ProRM target has been turned into a trainable moment problem. ProRM is
therefore an ideal target, not a separately implemented baseline or an ablation stage. The repository and
Python package retain `Smart-Reward-Model` and `smart_reward` as compatibility infrastructure; public
method terminology is ProRM/ProRM+.

## Estimand contract

The repository separates three questions that must not be collapsed into one metric:

| Role | Estimand | What it preserves |
|---|---|---|
| **Primary** | Fixed-`beta` ProRM regret, with the same `beta` for every reward model | Direction angle **and** natural-gradient norm calibration |
| **Secondary** | Fixed-`K` constrained regret and Fisher cosine | Direction angle after each method is normalized to the same local KL radius |
| **Transfer endpoint** | Finite policy rollout under a declared update rule | Whether local geometry survives an actual policy update |

For `u_r=F_0^\dagger A_0r`, the primary estimand is

$$
\mathcal R_\beta(r_\phi)
=\frac{1}{2\beta}\|u_{r_\phi}-u_*\|_{F_0}^2.
$$

At a fixed local KL radius `K`, the constrained update normalizes
`u_r` by its own Fisher norm, and its target-reward regret is proportional to
`1-cos_F(u_r,u_*)`. Fixed-`K` matching therefore discards norm-calibration error and is a useful
secondary diagnostic, not a replacement for the fixed-`beta` ProRM target.

## Current status

| Item | Status |
|---|---|
| Mathematical specification, numerical core, real-model pipeline, immutable artifacts, aggregation | Implemented in the working tree; the next HPC4 submission still requires a commit-bound release audit |
| Automated verification | Historical test and HPC4 `bash -n` records remain provenance for the snapshots that produced them, not certification of a later worktree. Every budgeted identity must bind one clean commit and pass the Python/static suite, shell checks, input/materialization checks and per-seed output verifier before descriptive aggregation |
| Slurm/Apptainer probe, staging, submission and runtime control plane | Implemented |
| HPC4 account/preflight and host-driver gate | Passed on `gpu-l20`, job `1640437`: NVIDIA L20, driver `570.211.01` |
| Driver-selected image definition and exact Python version lock | Implemented; digest-locked PyTorch 2.7.1/CUDA 12.6 |
| HPC4 GPU environment smoke | **Passed**, job `1640778`; image-build commit `b057bc9e134f1844248d655ed0f6c340af03099f`; validated SIF SHA256 `d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb` |
| Offline Hugging Face snapshots | Cached and offline-validated; main inventory SHA256 `095d5dc5e5a952be53ce07279aa7b5f1eda57a7a8b5745a1e4afa545a1f11f7c` |
| Historical pre-fix controlled smoke | **Passed**, job `1641475` on NVIDIA L20 (`00:03:14`), but only under the superseded FP32-solver identity |
| Superseded main attempt | Seed `20260722`, job `1641489`, failed the mandatory initial ProRM+ PCG gate: true relative residual `2.717e-5 > 1e-5` after 2048 iterations |
| Frozen Phase-1 numerical design | Main config `ae5d628e…a0df6`; FP64 policy geometry and 8192-iteration main ceiling |
| Phase-1 five-seed accepted experiment | **Completed**; five NVIDIA L20 jobs, `14:55:11` total GPU time |
| Phase-1 formal aggregation | **Completed**, job `1645205`; source validation and atomic publication passed |
| “ProRM+ outperforms BT-MLE” result | **Not supported** under the locked Phase-1 setting; preregistered status `not_passed` |
| Current post-Phase-1 route | Recovery 3（排除且不得复用其 head/data/optimizer state）→ fresh calibration/freeze（排除）→ accepted global `beta`/horizon → exactly three fresh `budgeted_end_to_end` exploratory E2E seeds；exact-30 不是当前执行目标 |
| Post-Phase-1 repair | Pilot calibration/global-`beta` freeze, fresh `R=4` heads, exact/direct/low-dimensional controls, updated-policy KL, real Qwen/Skywork runtime and strict seed normalization/description core implemented; live Phase 2 state is intentionally read from HPC4 evidence, not this README |
| First Phase-2 calibration pilot | All three excluded pilot seeds fail-closed at the BT first-order gate under constant-`1e-3` AdamW; a train-only diagnostic established a deterministic decay path that passes without held-out access. The one-shot recovery has a separate identity and can only authorize a new full calibration pilot; it cannot produce or enter a beta aggregate |
| Phase-2 recovery execution | Execution revision 1 (`1648094`) stopped before training when Hugging Face Datasets attempted a runtime lock in the read-only shared cache. Revision 2 is authorized only after exact marker/file/log hashes and absent trainer outputs pass at submission and job time; frozen assets remain read-only, only derived Datasets cache files are isolated per job, and the scientific recovery schedule/identity are unchanged |

The failed attempt produced no accepted comparison, rollout or scientific metric. Its `FAILED` marker,
manifest and log are retained as numerical-amendment evidence; it cannot be mixed with the replacement
five-seed campaign. The replacement campaign completed without numerical failures; see the
[formal Phase 1 result](docs/phase1_results.md).

## Formal Phase 1 result

All differences below are `ProRM+ − BT-MLE`. The intervals are paired-bootstrap engineering decision
intervals over the five locked seeds, not population confidence intervals or p-values.

| Preregistered metric | Estimand role | Favorable sign | Paired mean | 95% engineering interval | Gate |
|---|---|---:|---:|---:|---|
| Held-out local regret | Fixed-`beta` primary | `< 0` | `-0.0091789` | `[-0.0765277, 0.0615383]` | Fail |
| Squared Fisher direction error | Fixed-`beta` geometry | `< 0` | `-0.0183622` | `[-0.1529629, 0.1230318]` | Fail |
| Fisher cosine | Fixed-`K` secondary | `> 0` | `+0.0416380` | `[-0.0529131, 0.1316971]` | Pass under the locked mean-sign rule |
| Matched-KL rollout improvement | Fixed-`K` transfer | `> 0` | `-0.0037335` | `[-0.0074192, -0.0006188]` | Fail |

The campaign is an engineering success and a valid negative scientific result. ProRM+ has slightly
favorable mean local-regret and Fisher-error estimates, but they are heterogeneous across seeds and their
intervals cross zero. Its matched-KL rollout difference is unfavorable and the entire engineering interval
lies below zero. The repository therefore does **not** claim that ProRM+ outperforms BT-MLE under this
locked Phase-1 setting. Exact identities, per-seed values, sensitivity evidence and interpretation are in
[docs/phase1_results.md](docs/phase1_results.md).

This estimand clarification does not reopen the experiment: the authoritative Phase-1 status remains
`not_passed`.

One additional finite-sample audit is decisive. Under the locked
`gamma=0.9` estimator and `p in [0.25,0.75]`,
`sd(h | p)` ranges from approximately `0.841` to `0.935`; the Phase-1
train-only node-centered oracle RMS is only about `0.24`. Unbiasedness is
therefore intact, but one randomized estimate per edge has low signal-to-noise.
The next design includes an exact-margin positive control and averages four
independent `gamma=0.9` estimates per pair. The average remains exactly
unbiased and halves conditional standard deviation; labels are never clipped
or silently truncated.

The original `128`-token horizon was also active for roughly `74%–80%` of
accepted candidates. Those are valid samples from the declared capped action
space, but they limit external validity. `256` tokens is currently only a
candidate horizon, not a frozen formal choice. A pilot excluded from the
formal campaign will select a horizon using only maximum-length/EOS
diagnostics; every policy arm will then report its length-limit rate and
token-length distribution under the frozen choice.

## 1. From future policy utility to a reward-model loss

For a candidate reward `r`, let the downstream optimizer return

$$
\theta_r\in\arg\max_\theta
\left\{
\mathbb E_{x\sim\rho,y\sim\pi_\theta}[r(x,y)]
-\beta\mathbb E_{x\sim\rho}
D_{\mathrm{KL}}(\pi_\theta(\cdot|x)\Vert\pi_0(\cdot|x))
\right\}.
$$

The theoretical regularizer is explicitly **policy-to-reference**,
`KL(pi_theta || pi_0)`. The locked Phase-1 line search instead measures
**reference-to-updated** `KL(pi_0 || pi_updated)` on fixed reference histories. The two orientations
share the same Fisher expansion at `pi_0`, but they are not equal at a finite step; the Phase-1 quantity
is therefore an operational fixed-`K` budget, not the exact finite-step theoretical regularizer.

The globally correct reward-model criterion is the target-reward utility lost because the optimizer was
given `r_phi` rather than `r*`. That definition is prospective but bilevel and unobservable. ProRM is its
local, closed-form counterpart around the reference policy.

Fix the prompt distribution, `pi_0=pi_{theta_0}`, and the exact tangent coordinates that the next policy
update may change. Define

$$
s_0(x,y)=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0},\qquad
A_0r=\mathbb E[s_0r(x,y)],\qquad
F_0=\mathbb E[s_0s_0^\top].
$$

The ideal population ProRM loss is

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|A_0(r_\phi-r^*)\right\|_{F_0^\dagger}^{2}
}.
$$

In the local quadratic policy problem this is exactly the regret of the update induced by `r_phi` under
the target reward. Prompt-only shifts and reward errors in the score null space are not penalized because
they cannot change that update.

## 2. From pairwise labels to ProRM+

Sample a natural pair from

$$
Q_0(dx,dy,dy')=\rho(dx)\pi_0(dy|x)\pi_0(dy'|x),
$$

and define

$$
z_0=s_0(x,y)-s_0(x,y'),\qquad
\Delta r_\phi=r_\phi(x,y)-r_\phi(x,y').
$$

The score identity gives

$$
A_0r=\frac12\mathbb E_{Q_0}[z_0\Delta r].
$$

A single Bernoulli preference cannot provide a per-edge unbiased estimate of a BTL logit. ProRM+ obtains
conditionally iid repeated labels for the same edge and constructs a randomized U-statistic `h` satisfying

$$
\mathbb E[h\mid e]=\operatorname{logit}(p^*(e))=\Delta r^*(e).
$$

Consequently,

$$
\boxed{
m_\phi=\frac12\mathbb E[z_0(\Delta r_\phi-h)]
=A_0(r_\phi-r^*)
}.
$$

The two data streams have separate roles:

```text
Fisher stream:          (x,y) ~ rho*pi_0       -> s_0 -> F_0
Repeated-label stream:  e ~ Q_0, labels -> h   -> z_0 -> m_phi
                                                   |
                                                   v
                                         Fisher-GMM ProRM+
```

At population level and without damping,

$$
\boxed{
\min_\phi\max_v\frac1\beta
\left[v^\top m_\phi-\frac12v^\top F_0v\right]
=\min_\phi\mathcal L_{\mathrm{ProRM}}(\phi)
}.
$$

This identity requires natural `Q_0` pairs and the repeated-label assumptions. The three-edge
[closed-form example](docs/closed_form_example.md) establishes a population ordering reversal between
BT-MLE and the ideal ProRM target; it does **not** by itself establish the ProRM+ identification theorem.
That theorem uses the natural `Q_0` expectation above.

## 3. Empirical ridge ProRM+

With all on-policy node scores in `S` and canonical labeled-edge differences in `Z`, the implementation
uses

$$
\widehat F_0=\frac1{n_F}S^\top S,
\qquad
\widehat m_\phi=\frac1{2n_E}Z^\top(\Delta r_\phi-h),
$$

and trains the explicitly damped empirical objective

$$
\boxed{
\min_\phi\max_v\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right]
},
$$

equivalently,

$$
\widehat L_\lambda(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F_0+\lambda I)^{-1}\widehat m_\phi,
\qquad
\lambda=c\,\operatorname{mean}(\operatorname{diag}\widehat F_0)>0.
$$

| Level | Exact claim |
|---|---|
| Population, `lambda=0`, `F_0^dagger` | ProRM+ inner optimum equals local ProRM regret |
| Finite sample, `lambda>0` | Ridge-regularized empirical surrogate |
| `c in {1e-4,1e-3,1e-2}` | Preregistered damping sensitivity, not post-hoc tuning |

PCG solves `(F_hat + lambda*I)v=m_hat` without forming a dense Fisher. Because this operator is a
low-rank empirical Fisher plus isotropic damping, the controlled path deliberately uses unpreconditioned
CG: coordinate-wise Jacobi scaling destroys the repeated damping eigenvalue. Stored scores remain FP32,
but moment construction, damping, Fisher matvecs, Krylov state, held-out geometry and rollout directions
use the config-locked FP64 policy-geometry workspace. The reward head, autograd and AdamW remain FP32;
FP64 envelope weights cross to FP32 exactly once at that boundary.

Convergence is accepted only from an explicitly recomputed true residual `rhs-Ax` at relative tolerance
`1e-5`. Periodic checks do not replace the recursive residual while retaining an incompatible Krylov
direction; a false recursive crossing explicitly restarts from the true residual. The formal main ceiling
is `8192` iterations, while smoke retains `2048`; both remain fail-closed ceilings, not forced iteration
counts. The reported quadratic and detached envelope surrogate differ by a factor of two in value but
yield the correct gradient; see [theory.md](docs/theory.md).

## 4. Controlled Phase 1 experiment

The fixed question is:

> Under the same restricted reward class and training budget, does ProRM+ recover the operational-oracle
> policy-update direction more accurately than repeated-label BT-MLE, and does that advantage survive
> equal measured-KL policy optimization?

The target `r*` in Phase 1 is a train-calibrated transformation of frozen Skywork scores. It is an
**operational oracle**, not human utility. BT-MLE and ProRM+ share candidates, repeated labels, features,
zero initialization, optimizer, step count, GPU and stopping rule; only the training objective changes.

MultiPref supplies **prompts only** in this controlled experiment; its historical human preference labels
are not training targets. Qwen generates the four candidate responses. For canonical candidate pair
`0-1`, frozen Skywork defines $p^*=\sigma(\Delta r^*)$; a named seed then generates conditionally iid
Bernoulli repeats and the randomized estimator `h`. Thus the Phase-1 “annotator” is a reproducible
Skywork-defined BTL simulator, not a new human-labeling round.

Formal offline jobs resolve the pinned MultiPref snapshot locally and load its sorted
`data/train-*.parquet` shards through the Parquet builder. They do not call
`load_dataset("allenai/multipref", ...)`: Datasets 3.6 may still query Hub metadata on that path even
with offline flags set.

```text
MultiPref prompts
    -> pi_0: four exact-token candidates per prompt
       -> fixed-A LoRA-B scores --------> Fisher geometry
       -> frozen hidden features -------> zero-init linear reward class
       -> frozen operational oracle ----> train-only calibration
                                           -> repeated BTL labels
                                                  |          |
                                               BT-MLE      ProRM+
                                                  \          /
                                           held-out re-solve geometry
                                                    |
                                  deployed train direction + rollouts
```

| Component | Locked design |
|---|---|
| Prompts | MultiPref pinned revision; local `data/train-*.parquet`; `1536/256/256` prompt-level split |
| Reference policy | Pinned Qwen2.5-0.5B-Instruct, FP32 |
| Candidates | Four independent base-distribution samples per prompt from the frozen length-eligible prompt pool; no post-generation filtering or deduplication |
| Policy tangent | Last four `q_proj/v_proj` modules, rank-4 fixed-A LoRA-B |
| Oracle | Pinned Skywork-Reward-V2-Qwen3-0.6B, FP32 |
| Repeated labels | Canonical candidate `0-1`; geometric continuation `gamma=0.9`, hence `E[N]=10` |
| Reward class | Frozen final-response-token feature plus bias-free linear head |
| Model execution and storage | Qwen, Skywork, frozen features and stored score tensors in FP32 |
| Fisher/GMM geometry | Moment, damping, Fisher matvec, PCG, held-out metrics and rollout direction in FP64 |
| Reward optimization | FP32 head/gradient/AdamW; one explicit FP64-to-FP32 envelope-weight cast |
| PCG gate | True relative residual `1e-5`; main cap `8192`, smoke cap `2048` |
| Training | 720 fixed steps; identical optimization budget |
| Evaluation | Held-out re-solved Fisher geometry plus deployed train-direction, measured sequence-KL `0.01 ± 5%` rollout |
| Statistics | Five paired seeds; fixed main damping plus two sensitivity settings |

The capacity bottleneck does not logically guarantee misspecification. The immutable artifact therefore
records a train-only, prompt-centered linear projection residual under
`train_reward_class_projection`. It is descriptive mechanism evidence and cannot select a checkpoint,
damping or conclusion.

Held-out geometry and rollout do not use the same solved vector. With a frozen learned head, held-out
metrics recompute both predicted and target directions from that split's moment, Fisher and damping.
The policy rollout instead deploys the direction solved only from train moment/Fisher; test geometry
never re-solves or modifies it. A difference between held-out ordering and rollout ordering is therefore
a transfer failure, not an inconsistency in one direction.

## 5. Evidence required for a positive result

Pairwise prediction is descriptive. Held-out BTL NLL and oracle-probability MAE measure preference fit;
they are not success gates. `aggregate.json` may report `passed` only if all preregistered policy evidence
passes:

| Evidence | Fixed five-seed criterion |
|---|---|
| Main-damping held-out ridge local-regret proxy | `ProRM+-BT-MLE` mean `<0`, bootstrap upper `<0` |
| Squared Fisher direction error | `ProRM+-BT-MLE` mean `<0`, bootstrap upper `<0` |
| Fisher cosine | `ProRM+-BT-MLE` mean `>0`; both direction norms nonzero |
| Matched-KL rollout improvement | Both methods meet KL tolerance; `ProRM+-BT-MLE` mean `>0`, bootstrap lower `>0` |
| Damping sensitivity | Both secondary local-regret means `<0`; all required PCG solves converge |
| Identity and numerical integrity | PCG/KL convergence plus identical Git/image/GPU/manifest identities |

The percentile-bootstrap interval over five preregistered paired seeds is an engineering decision interval,
not a population confidence interval or p-value.

| Observed pattern | Permitted conclusion |
|---|---|
| Geometry, rollout and sensitivity all pass | Supports the preregistered prospective reward-modeling mechanism claim |
| Geometry passes but rollout fails | Local surrogate improved; downstream transfer not established |
| Geometry fails | Core mechanism not supported |
| Sensitivity fails or reverses | Failure remains in evidence; status is `not_passed` |
| Only NLL/accuracy/probability MAE improves | Not evidence that ProRM+ succeeded |

The completed campaign has preregistered status `not_passed`; no positive mechanism claim is made.

### Next experiment: common-beta deployment

The complete post-audit decision record is
[docs/phase2_design_decisions.md](docs/phase2_design_decisions.md). A Phase 2
confirmatory identity is admissible only after the target-free pilot acceptance
chain is complete and its accepted freeze evidence is hash-bound.

Phase 2 also freezes a single operational-oracle coordinate system across
seeds: `b_0=-4.500244140625` and `tau_0=2.7715682983398438`. They are the
componentwise medians of the train-only robust transforms from the five
Phase-1 seeds, all of which are excluded from Phase 2. The config binds those
five artifact-metadata hashes, and materialization reuses the resulting
`RobustOracleTransform` without fitting the current pilot/formal seed. Thus
both `r*` and `beta_0` are global quantities in the confirmatory estimand.

The fixed-A LoRA tangent basis is global too. Phase 2 always uses initialization
seed `946081152281754541` from excluded Phase-1 seed `20260722` and verifies
the expected A fingerprint
`a2b5804109396f76b96cde98d1e2060f175a47724b1ca9fef317c7a10cb9a838`
during both materialization and policy reload. Formal seeds therefore vary
data and stochastic streams without silently changing the policy class.

The excluded pilot computes, for each pilot seed, the train-only damped oracle
natural direction

$$
u_{*,s}^{\mathrm{tr}}
=(F_{\mathrm{tr}}+\lambda I)^{-1}g_*^{\mathrm{tr}},
$$

and the calibration candidate

$$
\widetilde\beta_s
=
\sqrt{
\frac{(u_{*,s}^{\mathrm{tr}})^\top F_{\mathrm{tr}}u_{*,s}^{\mathrm{tr}}}
{2K_{\mathrm{cal}}}
},
\qquad K_{\mathrm{cal}}=0.003.
$$

The calibration pilot publishes these train-only candidates plus convergence,
rank, response-length/EOS, and on-policy KL diagnostics. A strict aggregate
over permanently excluded seeds `20260801`, `20260802`, and `20260803`
selects their maximum. A second, separately hashed freeze-pilot
identity binds that aggregate SHA-256 and deploys the same beta to every
seed/arm. Neither stage invokes the
held-out evaluator, opens a final oracle-scoring session, or computes/serializes
reward, utility, regret, head vectors, prompt/response text, token IDs, or
learner ordering.

The calibration base and the freeze grid are:

$$
\beta_{\mathrm{base}}
=
\max_{s\in\mathcal S_{\mathrm{pilot}}}\widetilde\beta_s,
\qquad
\beta^{(k)}=2^k\beta_{\mathrm{base}},\quad k=0,1,\ldots.
$$

If pilot-only worst-arm KL safety requires a larger value, the only allowed
adjustment is a new freeze identity at the next member of
`{beta_base, 2 beta_base, 4 beta_base, ...}`. The first freeze binds the calibration
aggregate; every later freeze binds the immediately preceding non-length
safety failure and its exact `next_global_beta=2*previous_beta`. It cannot skip
a grid point. The horizon-parent hash remains a separate binding to the
calibration aggregate that accepted that horizon. Mean/p95/p99/prompt-max/sequence-max KL
caps are `0.02/0.02/0.05/0.10/0.20`; the all-arm maximum-length-rate cap is
`0.05`. Horizons follow `[256,512,1024]`: a length failure requires a new
calibration identity at the next horizon, bound to the failed aggregate hash,
and then a complete freeze rerun. Only an accepted freeze aggregate may source
the confirmatory identity. If

$$
k_*=\min\{k\ge 0:\text{the freeze at }\beta^{(k)}
\text{ passes every frozen gate}\},
\qquad
\boxed{\beta_0=2^{k_*}\beta_{\mathrm{base}}},
$$

then `beta_0` is the single confirmatory scalar. If no permitted grid point
passes, the campaign stops and no confirmatory `beta_0` is defined.
Confirmatory sensitivity is restricted to the frozen multiples
`beta in {0.5 beta_0, 2.0 beta_0}` for every sensitivity seed and arm.
Seed-specific `K_cal` calibration remains a pilot-only train diagnostic and
cannot choose any confirmatory step size.

The confirmatory chain is:

1. train BT-MLE and ProRM+ from the same zero head until each passes its own
   frozen full-gradient gate; retain step 720 only as a compute-matched
   secondary snapshot;
2. deploy `u_BT/beta_0`, `u_ProRM+/beta_0`, and the oracle-step control directly,
   with no learner- or seed-specific line search or norm normalization;
3. measure `KL(pi_updated || pi_0)` on each updated policy's own histories and
   evaluate `J*=E[r*]-beta_0*KL(pi_updated || pi_0)`; retain fixed-history
   `KL(pi_0 || pi_updated)` only as a secondary diagnostic;
4. fail closed on frozen mean/tail KL, optimization, rank, identity, numerical,
   horizon, or positive-control gates;
5. require the intersection of held-out fixed-`beta_0` regret, ProRM+ versus
   BT-MLE utility, ProRM+ versus zero-B utility, and oracle-step versus zero-B
   utility—not preference accuracy alone.

Candidate values are averaged within prompt, and every seed contributes one
paired scalar. The five Phase 1 seeds and all pilot/design-development seeds
are permanently excluded. The confirmatory campaign is the exact ordered
30-seed list `20260901` through `20260930`; outcome-dependent early stopping is
forbidden. Its formal estimand is the RNG expectation of each paired contrast
conditional on the frozen eligible prompt pool, models, oracle, and design—not
an unrestricted human-prompt population. The four required positive contrasts
form one intersection-union test: a two-sided 95% paired-seed percentile
interval must have lower endpoint above zero for every component. This is an
effective one-sided component level of `0.025`; no Bonferroni correction is
needed for the single conjunctive claim, while separate endpoint claims are
forbidden without multiplicity control. With 30 seeds, the prospective
normal-approximation 80%-power threshold is about `0.53` paired standard
deviations for one component; the weakest component determines conjunctive
power. The noisy primary arm averages four independent
`gamma=0.9` unbiased `h` replicates per edge, while BT-MLE receives all
underlying Bernoulli labels. Because `E[N]=10`, this costs 40 labels per
canonical edge in expectation and has an unbounded geometric tail; exact
unbiasedness is statistically useful but annotation-expensive.

This stream is finite-variance but not light-tailed. At the locked probability
endpoints, the second-moment tail ratio is `0.75/0.9 < 1`, whereas the
fourth-moment ratio is `0.75/0.9^3 > 1`; a single replicate therefore has an
infinite fourth moment there. Averaging `R=4` divides conditional variance by
four but does not change the tail exponent. Post-recovery and confirmatory
artifacts consequently bind a scalar-only
`repeated-label-tail-diagnostics/v1` record into `label_stream_sha256`. Its
nearest-rank `p50/p90/p95/p99/max` summaries are descriptive only: they may not
clip or select samples, choose beta or seeds, gate acceptance, or authorize a
retry. The project makes no sub-Gaussian or finite-fourth-moment claim.

Phase 2 also fails closed on prompt semantics. Qwen2.5 renders the complete
original user prompt with its own tokenizer/chat template and
`truncation=False`. A reproducible local audit of all 5,323 unique prompts in
the pinned MultiPref snapshot found 88 over the frozen 1024-policy-token cap
(`1.65%`), leaving 5,235 eligible prompts. The three pilot seeds would have
selected 39, 34, and 36 over-limit prompts under the old
shuffle-before-length-check order, so silent truncation and late failure are
both removed: Phase 2 first constructs the `<=1024` eligible pool, then applies
the seeded shuffle/split.

Materialization records total/eligible/excluded/selected counts and prompt-ID
list hashes, while each selected record binds the raw-text hash, policy token
count, prefix hash, cap, and `truncated=false`. The declared Phase 2 prompt
population is consequently the length-eligible MultiPref subset. Skywork Qwen3
receives the same raw prompt plus assistant response and independently rerenders
them with its pinned Qwen3 tokenizer/template; Qwen2.5 tokens are never reused
by the oracle. These counts are an input precheck, not an experiment result.

The experimental evidence architecture borrows from AuxDPO: an analytic
misspecification example, matched data/compute, capacity and sample-size stress,
exactly 30 preregistered paired formal seeds, controls, and separate OOD/human
evaluation. It does not copy
AuxDPO's null-space parameterization or promote IPO/DPOP to the primary
reward-model baseline. The scientific object here is reward-model training for
downstream regret, so repeated-label BT-MLE remains the primary comparator.
Capacity, all-six-pair, frozen-global-beta sensitivity, and OOD studies are
secondary to the locked common-`beta_0` mechanism test. Seed-specific beta is
forbidden in the confirmatory estimand. None of these changes rewrites Phase 1:
its authoritative status remains `not_passed`.

## 6. Local verification

```bash
python -m pip install -e ".[dev]"
prorm config-check configs/smoke.yaml
prorm config-check configs/main.yaml
prorm phase2-config-check configs/common_beta_pilot.yaml
prorm closed-form-check --output outputs/closed-form.json
prorm synthetic-check --seed 0 --output outputs/synthetic.json
# With downloaded Phase-1 artifacts:
# prorm estimand-audit CONFIG ARTIFACT COMPARISON ROLLOUT OUTPUT --seed SEED
# prorm optimization-audit CONFIG ARTIFACT COMPARISON OUTPUT --seed SEED
pytest -q
ruff check .
ruff format --check .
```

`closed-form-check` is marked `population_example_only=true`; it verifies the analytic ordering reversal
without presenting the three-edge distribution as ProRM+ training data. `synthetic-check` is always marked
`benchmark_only=true`; it validates identities and integration and does not assert that ProRM+ must beat
BT-MLE. Real Hugging Face execution additionally needs:

```bash
python -m pip install -e ".[llm,dev]"
```

`common_beta_pilot.yaml` is deliberately a three-seed,
outcome-blind, non-confirmatory identity. Passing `phase2-config-check`
establishes only that its source binding and design contract are valid; it is
not evidence that the optimization/KL/horizon pilot has run or passed.

`prorm` is the public CLI name. The historical `smart-reward` executable and `smart_reward` import package
remain compatibility surfaces while artifacts and scripts migrate.

The Phase 2 control-plane commands have the following positional contracts;
`--help` lists the identity-bound parent and device flags:

```text
phase2-config-check OVERLAY
phase2-run OVERLAY ARTIFACT MANIFEST OUTPUT --seed SEED
phase2-pilot-aggregate OVERLAY OUTPUT RESULT...
  --aggregator-git-commit COMMIT --producer-git-commit COMMIT
  --aggregation-image-sha256 SHA256
  --aggregation-hf-inventory-sha256 SHA256
  --validator-source-sha256 SHA256
phase2-aggregate OVERLAY OUTPUT RESULT...
phase2-sensitivity-run OVERLAY ARTIFACT MANIFEST PRIMARY_RESULT OUTPUT --seed SEED
phase2-sensitivity-aggregate OVERLAY PRIMARY_AGGREGATE OUTPUT RESULT...
phase2-mechanism-run OVERLAY ARTIFACT MANIFEST PRIMARY_RESULT OUTPUT --seed SEED
phase2-mechanism-aggregate OVERLAY PRIMARY_AGGREGATE OUTPUT RESULT...
phase2-failure-manifest OVERLAY SPEC OUTPUT
phase2-campaign-finalize OVERLAY OUTPUT AGGREGATE_OUTPUT TERMINAL...
```

The Slurm wrappers remain the required HPC4 entry points. Direct CLI invocation
is for tests, detached compute jobs, and forensic replay—not login-node model
execution.

## 7. HKUST HPC4 entry

Repository inputs are relative to the checkout. Only the cross-node project and scratch anchors must be
absolute:

| Content | Persistent or temporary location |
|---|---|
| Git checkout | `$HOME/Smart-Reward-Model` |
| Qwen, Skywork and raw MultiPref snapshots | `$PRORM_PROJECT_ROOT/hf-cache/hub` |
| Processed Hugging Face/Arrow dataset cache | `$PRORM_PROJECT_ROOT/hf-cache/datasets` |
| Image, build/staging/GPU evidence | `$PRORM_PROJECT_ROOT/{images,system-reports}` |
| Generated candidates, labels, scores, features and Fisher data | `$PRORM_PROJECT_ROOT/artifacts/...` |
| Learned linear RM heads, rollouts and aggregate | `$PRORM_PROJECT_ROOT/runs/...` |
| Per-job working copy | `$PRORM_SCRATCH_ROOT/jobs/$SLURM_JOB_ID` |

The experiment does not create another full Qwen checkpoint. Base weights stay in the pinned HF cache;
the learned bias-free linear reward heads are serialized in `comparison.json`. The local LoRA-B policy
update is reconstructed for evaluation and is not exported as a production adapter checkpoint.
Each persistent run contains a relative `artifact` symlink to its content-addressed Phase-1 artifact, so
serialized POSIX path references remain valid after scratch cleanup. Heavy assets and results are ignored
by Git.

The first SSH connection is the only interactive identity step: enter the ITSO password and complete
Duo/2FA in the SSH client. Never send a password, 2FA response, private key or recovery code to Codex, put
one in this repository, or place one in a Slurm log. After that private login:

```bash
ssh YOUR_ITSO@hpc4.ust.hk
git clone https://github.com/youth-flow/Smart-Reward-Model.git
cd Smart-Reward-Model
test "$(git remote get-url origin)" = \
  "https://github.com/youth-flow/Smart-Reward-Model.git"
git rev-parse --verify HEAD

# Private, ignored path configuration; never edit the tracked example.
test -e .env.hpc4 || cp scripts/hpc4/env.example .env.hpc4
source .env.hpc4
mkdir -p \
  "${PRORM_PROJECT_ROOT}"/{images,hf-cache,system-reports,slurm-logs,artifacts,runs} \
  "${PRORM_SCRATCH_ROOT}/jobs"
bash scripts/hpc4/preflight.sh

# No image is needed for this first GPU/driver observation.
bash scripts/hpc4/submit_host_gpu_probe.sh gpu-l20
```

The completed gate is job `1640437`. It observed one NVIDIA L20 (46,068 MiB), driver `570.211.01` and
maximum supported CUDA 12.8. The resulting candidate is therefore the digest-locked
PyTorch 2.7.1/CUDA 12.6 definition in
[`containers/prorm-hpc4.def`](containers/prorm-hpc4.def), with the exact Python package lock in
[`containers/requirements-hpc4.lock`](containers/requirements-hpc4.lock).

HPC4 cannot build the definition locally because its Apptainer installation has no SUID builder or
subuid/subgid mapping, and user namespaces are disabled on the login node. The login node is therefore
limited to Git, file checks and Slurm submission; it must not run `apptainer exec` for HF staging or
aggregation. The dedicated GitHub workflow builds the raw SIF, records build evidence and publishes it
through public GHCR ORAS. Pull the validated artifact by its immutable **image-build commit**, not by the
current source
`HEAD`; the source checkout may legitimately contain later staging/control-plane changes. The fetcher
resolves and verifies the OCI manifest digest and requires the local SIF SHA256 to equal the manifest's
SIF-layer digest:

```bash
image_build_commit=b057bc9e134f1844248d655ed0f6c340af03099f
bash scripts/hpc4/fetch_candidate_image.sh "${image_build_commit}"

export PRORM_IMAGE=images/prorm.sif
export PRORM_HF_CACHE=hf-cache
export PRORM_IMAGE_SHA256=d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb
printf '%s  %s\n' \
  "${PRORM_IMAGE_SHA256}" "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}" \
  | sha256sum --check
```

This exact SIF passed the HPC4 GPU environment smoke in job `1640778`; its persistent report is
`$PRORM_PROJECT_ROOT/system-reports/gpu-smoke-1640778.txt`. A file with any other SHA256 remains an
unvalidated candidate.

HF model and dataset staging is a separate, mandatory gate. Because login-node user namespaces are
disabled, do not run `stage_hf_assets.py` or `apptainer exec` directly there. The two configs share one
HF cache, so their first downloads must be serialized on an allowed CPU compute partition (`amd` or
`intel`). Submit the smoke stage first:

```bash
export PRORM_HF_STAGE_WALLTIME=04:00:00
cache_root="${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"
smoke_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/smoke.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
smoke_stage_job="${smoke_stage_job%%;*}"
test -n "${smoke_stage_job}"
squeue -j "${smoke_stage_job}"
```

After the smoke stage leaves the queue, require `COMPLETED`, `ExitCode=0:0` and an exact
`status=passed` report before submitting the main stage:

```bash
sacct -j "${smoke_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
smoke_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${smoke_stage_job}.log"
tail -n 20 "${smoke_stage_report}"
grep -Fx 'status=passed' "${smoke_stage_report}"

main_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/main.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
main_stage_job="${main_stage_job%%;*}"
test -n "${main_stage_job}"
squeue -j "${main_stage_job}"
```

After the main stage leaves the queue, apply the same acceptance check:

```bash
sacct -j "${main_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
main_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${main_stage_job}.log"
tail -n 20 "${main_stage_report}"
grep -Fx 'status=passed' "${main_stage_report}"
sha256sum "${cache_root}"/inventories/*.json
```

`04:00:00` is the staging default. Change it only if an administrator-enforced partition limit requires
an approved lower value.

Staging downloads only the public pinned snapshots. Its offline proof resolves snapshot revisions,
configs and tokenizers, then reads MultiPref directly from the pinned snapshot's sorted
`data/train-*.parquet` shards. This avoids the Datasets 3.6 Hub-metadata path; it does not claim to have
instantiated model weights.
Actual Qwen/Skywork weight loading is tested by the controlled model smoke. Each config-specific inventory
digest is reverified offline and bound into the run manifest, artifact producer identity and final
aggregate.

Only after the validated-image GPU smoke has passed, both CPU staging jobs are `COMPLETED` with
`ExitCode=0:0`, both config-specific inventories exist, the Git checkout is clean and `HEAD` equals the
reviewed remote commit may `submit_controlled.sh` be used:

```bash
git fetch origin main
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

export PRORM_SMOKE_WALLTIME=REPLACE_WITH_APPROVED_PILOT_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml gpu-l20 "${PRORM_SMOKE_WALLTIME}"

# The accepted formal campaign used a 12-hour allocation and averaged 02:59:02/seed.
export PRORM_ARRAY_CONCURRENCY=2
export PRORM_MAIN_WALLTIME=12:00:00
# HPC4 l20_qos currently allows at most four submitted jobs per user.
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}" 0-3
# Submit index 4 only after at least one task above reaches a terminal state.
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}" 4
```

The optional fourth argument is a zero-based configured seed index or one contiguous inclusive range.
Omitting it submits all configured seeds. Splitting an array changes only Slurm scheduling: every task
still resolves its seed from the same committed config and records its own numeric `SLURM_JOB_ID`.
`PRORM_ARRAY_CONCURRENCY=2` fills the current `l20_qos MaxJobsPU=2` allowance. The QoS remains the global
limit even after index 4 is submitted, so the campaign never runs more than two GPU tasks concurrently.

After all five main seeds have been individually accepted, map every configured seed to exactly one
successful controlled job and submit aggregation to a CPU partition. Replace each value below only with
the corresponding `COMPLETED`, `ExitCode=0:0` job ID whose run directory contains a valid `SUCCESS`
marker:

```bash
job_20260722=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260723=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260724=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260725=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260726=REPLACE_WITH_ACCEPTED_JOB_ID

aggregate_job="$(
  bash scripts/hpc4/submit_aggregate.sh \
    configs/main.yaml amd 01:00:00 \
    "20260722=${job_20260722}" \
    "20260723=${job_20260723}" \
    "20260724=${job_20260724}" \
    "20260725=${job_20260725}" \
    "20260726=${job_20260726}"
)"
aggregate_job="${aggregate_job%%;*}"
test -n "${aggregate_job}"
squeue -j "${aggregate_job}"
```

Aggregation is never run by direct login-node `apptainer exec`. The CPU job publishes the no-overwrite,
atomic result at
`$PRORM_PROJECT_ROOT/runs/controlled-main/<main-config-hash>/aggregate/`. Its `SUCCESS` marker means the
aggregation pipeline and evidence validation completed; it does **not** mean the scientific criterion
passed. The conclusion is exclusively `pre_registered_evidence.status` in `aggregate.json`, mirrored as
`pre_registered_evidence_status=passed` or `not_passed` in `SUCCESS`.

By default, the clean submission `HEAD` is both the aggregation control-plane commit and the
source/producer commit. A later wrapper-only hotfix may instead place
`--source-commit <full-producer-commit>` immediately after the walltime. The source must be an ancestor of
the control-plane `HEAD`, and the worktree config bytes must still equal the source blob. Aggregation
Python, config identities, seed manifests, and artifacts are then all validated against and executed from
that detached source commit; `aggregation-manifest.json` records the distinct control-plane and
aggregation-source commits.

Formal jobs never use `--allow-download`. They bind the submission Git commit and config-specific cache
inventory before allocation work begins. The run manifest records that **source Git SHA** separately from
the validated **SIF SHA256**; it does not require the source commit to equal image-build commit
`b057bc9e134f1844248d655ed0f6c340af03099f`. Wall time, GPU-hours and storage budgets come from the
accepted smoke record. See [hpc4.md](docs/hpc4.md) for exact aggregation acceptance and scratch-retention
commands.

### Legacy pre-recovery Phase 2 pilot — historical replay only

The commands in this subsection document the original v2
`common_beta_pilot.yaml` campaign. That calibration identity has already
terminated at its frozen optimization gate and must **not** be invoked to
continue the current experiment. The only admissible continuation is:

```text
one-shot recovery revision 2 reaches a valid three-seed terminal state
  -> recovery-success authorization is built and verified
  -> a fresh authorization-bound post-recovery calibration is materialized
  -> the post-recovery pilot/aggregate control plane is used
```

The authoritative transition documents are
[the recovery protocol](docs/phase2_recovery_protocol.md),
[the recovery authorization contract](docs/phase2_recovery_authorization.md),
and the
[post-recovery HPC4 runbook](docs/phase2_post_recovery_hpc4.md).
Do not use `submit_phase2_pilot.sh`, `submit_phase2_pilot_aggregate.sh`, or the
historical `common_beta_pilot.yaml` identity as a substitute for that chain.
They remain available only for byte-compatible historical replay.

The current post-recovery control plane is crash-resumable and
evidence-bearing. GPU arrays and CPU aggregation attempts are created held,
their exact scheduler requests are fsynced to immutable ledgers, and only then
released. For CPU aggregation, the wrapper reads the declared
`commit:scripts/hpc4/phase2_post_recovery_aggregate.sbatch` with binary
`git cat-file blob` and passes those exact bytes to `sbatch` on stdin; no
mutable script pathname is submitted. While the job remains held, the
controller-accepted script is read back, compared byte-for-byte, durably bound
to the attempt, and checked once more immediately before release. A CPU
aggregation job writes a persistent `aggregate.json`, `evidence/`, and `READY`
under its attempt namespace; it never publishes the production aggregate
itself. After Slurm records the unique registered attempt as
`COMPLETED/0:0/0:0`, an external terminalizer executes the crash-resumable,
no-overwrite sequence
`ATTEMPT -> evidence -> aggregate -> PUBLISHED -> TERMINAL -> SUCCESS`.
Because the HPC4 project filesystem lacks atomic directory
`RENAME_NOREPLACE`, evidence publication uses an atomic `mkdir` claim,
`EVIDENCE_CLAIM.json`, and per-file hard-link create-if-absent from a complete
fsynced sibling tree. A partial tree is resumable only when it is an exact
claim-bound prefix; `.PUBLISHED`, not directory existence or `READY`, is the
publication-completeness gate.
The resulting exact-tree bundle contains the committed script, every
controller readback, the complete attempt/failure chain, and raw
scheduler-authority evidence. Live publication resolves the declared Git
object again; later offline verification validates the self-contained byte
and digest chain without depending on Slurm or the live submission registry.

In that historical v2 path, Phase 2 used a dedicated outcome-blind entry point
and the **base** config for HF inventory staging:

```bash
bash scripts/hpc4/submit_hf_stage.sh \
  configs/common_beta_pilot_base.yaml amd 02:00:00

export PRORM_PHASE2_ARRAY_CONCURRENCY=2
bash scripts/hpc4/submit_phase2_pilot.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  gpu-l20 1-00:00:00
```

The executable Phase 2 pilot and confirmatory campaign are hardware-locked to
`gpu-l20` (NVIDIA L20). Their dedicated submit entries reject every other GPU
partition; pilot aggregation and formal campaign finalization remain CPU-only
`amd|intel` jobs.

The three-seed pilot writes only convergence/rank, train-only beta candidate,
response-length/EOS and on-policy KL evidence under
`$PRORM_PROJECT_ROOT/runs/phase2-pilot/<design-sha>/`. Pilot evidence is never a
formal result; exact acceptance paths and forbidden outcome fields are defined
in [docs/hpc4.md](docs/hpc4.md).

After all three declared seeds have immutable `SUCCESS` directories, aggregate
them on a CPU node; never run Apptainer on the login node:

```bash
design_sha=REPLACE_WITH_COMMITTED_DESIGN_SHA256
aggregate_dir="${PRORM_PROJECT_ROOT}/runs/phase2-pilot/${design_sha}/aggregate"
mkdir -p "${aggregate_dir}"

bash scripts/hpc4/submit_phase2_pilot_aggregate.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  "${aggregate_dir}/calibration.json" \
  amd 01:00:00 \
  REPLACE_WITH_20260801_SUCCESS_DIR \
  REPLACE_WITH_20260802_SUCCESS_DIR \
  REPLACE_WITH_20260803_SUCCESS_DIR
```

The wrapper defaults the producer commit to the current aggregation commit.
When a validator-only follow-up commit consumes already immutable seed outputs,
pass `--producer-commit <exact-full-seed-producer-commit>`. The wrapper permits
this only when that commit is an ancestor and the overlay, base config, and
identity bytes are identical in both commits. The aggregate v2 record preserves
the producer and aggregator identities separately and verifies each source
result, diagnostic sidecar, run manifest, output-verification receipt, artifact
metadata, and `SUCCESS` receipt before publication.

For freeze and retry identities, both the GPU pilot submission and CPU
aggregate submission bind the same required parent evidence. The first freeze
uses the accepted calibration aggregate as both `--beta-source-aggregate` and
`--horizon-parent-aggregate`. A beta-grid retry uses the immediately preceding
failed freeze as its beta source while retaining the accepted calibration as
its horizon parent. Exact commands are in [docs/hpc4.md](docs/hpc4.md).

The historical commands above end here. The exact-30 machinery described
below is a **future formal protocol only**. It is inactive in the current
budgeted fixed-three route, and the observed three-seed result must not be used
to decide whether to activate it. If a later study independently preregisters
and refreezes that protocol, retries belong only to outcome-blind pilot design
selection. Its formal campaign has a different, stricter contract: the exact ordered seeds
`20260901` through `20260930`, exactly `attempt-1` for every seed, and an
immutable `campaign-plan.json` committed before the first Slurm submission.
There is no formal retry, requeue, replacement seed, or optional stopping.
The scheduler realizes that one scientific campaign as eight predeclared
waves: `0-3%2`, `4-7%2`, `8-11%2`, `12-15%2`, `16-19%2`, `20-23%2`,
`24-27%2`, and `28-29%2`. This keeps at most four submitted tasks and two
running tasks under the observed HPC4 `l20_qos MaxSubmitJobsPU=4`; it does not
change the exact-30 estimand or any scientific configuration.

Every wave also has an immutable `admissions/wave-<index>.json` committed and
fsynced before its `sbatch`. Wave 0 binds an empty predecessor; every later
receipt hash-binds the preceding admission, submission, and the exact ordered
terminal-manifest/marker snapshot. The submission v3 record then binds that
receipt plus the raw and normalized held `scontrol` request. Caller walltime,
`l20_qos`, `%2`, CPU, memory, node, and GPU resources must equal the plan.
Before a new submit, both `squeue` and historical `sacct` are checked under the
deterministic wave name; any unregistered historical identity fails closed and
is never replaced.

### Future exact-30 Phase 2 execution — inactive

Do not run the commands in this subsection for the present study. They are
retained to preserve and test the separately versioned future protocol.

Do not copy an identity from this README. First commit the accepted
confirmatory overlay/base identities and accepted freeze aggregate, then use
their real paths on HPC4. With the required `PRORM_*` environment already
exported and a clean committed checkout:

```bash
wave_submission="$(
  bash scripts/hpc4/submit_phase2_confirmatory.sh \
    configs/REPLACE_WITH_CONFIRMATORY_OVERLAY.yaml \
    configs/REPLACE_WITH_CONFIRMATORY_BASE.yaml \
    /ABS/PATH/TO/accepted-freeze-aggregate.json \
    gpu-l20 \
    REPLACE_WITH_WALLTIME
)"
printf '%s\n' "${wave_submission}"
```

The command accepts no caller-selected range or concurrency override. On its
first invocation it durably commits the complete ordered seed/attempt/wave
plan before submitting wave 0 held; each later identical invocation derives
the only admissible next action from that plan and the immutable registry.
A next wave becomes eligible only after every task in every preceding wave has
a valid terminal bundle, regardless of whether those bundles record success or
failure. If the shell is interrupted after Slurm accepts a held wave but before
its registry record or release, rerun the **identical command**: the
deterministic wave identity is recovered and released without creating a
replacement job.

Every task must publish one terminal head. A canonical job directory with
`FAILURE_PENDING` is finalized by
`terminalize_phase2_compute_failure.sh`. A terminal non-success scheduler
record with no canonical job directory is finalized by
`terminalize_phase2_scheduler_failure.sh`, using one raw `sacct -X -n -P` root
row. These paths classify the single attempt; they never authorize another
attempt.

After all 30 heads are terminal, submit the CPU finalizer:

```bash
design_root="${PRORM_PROJECT_ROOT}/runs/phase2-confirmatory/REPLACE_WITH_DESIGN_SHA256"

bash scripts/hpc4/submit_phase2_campaign_finalize.sh \
  configs/REPLACE_WITH_CONFIRMATORY_OVERLAY.yaml \
  configs/REPLACE_WITH_CONFIRMATORY_BASE.yaml \
  "${design_root}/campaign-final/phase2-campaign-terminal.json" \
  "${design_root}/campaign-final/phase2-primary-aggregate.json" \
  amd \
  REPLACE_WITH_WALLTIME
```

The finalizer resolves the exact 30 terminal heads from the immutable registry;
callers do not select result directories. If any seed failed, the campaign
terminal status is `not_passed_due_to_seed_failure` and no primary aggregate
or CI is produced. The complete copy-paste runbook, failure-classification
schemas, and monitoring commands are in [docs/hpc4.md](docs/hpc4.md).

## 8. Documentation and code map

| Goal | Entry point |
|---|---|
| Global-to-local derivation, assumptions and contribution boundary | [docs/theory.md](docs/theory.md) |
| Three-edge closed-form population ordering reversal | [docs/closed_form_example.md](docs/closed_form_example.md) |
| Fixed Phase 0–1 design, metrics and artifacts | [docs/experiment_protocol.md](docs/experiment_protocol.md) |
| Formal five-seed results and scientific conclusion | [docs/phase1_results.md](docs/phase1_results.md) |
| **Current Phase 2 fixed-three budgeted route, methods, endpoints and claim boundary** | [docs/phase2_budgeted_end_to_end.md](docs/phase2_budgeted_end_to_end.md) |
| Phase 2 pilot boundary, global-beta decision and formal gates | [docs/phase2_design_decisions.md](docs/phase2_design_decisions.md) |
| First Phase-2 failure, optimizer diagnosis and one-shot recovery boundary | [docs/phase2_recovery_protocol.md](docs/phase2_recovery_protocol.md) |
| Recovery terminal evidence and success-authorization boundary | [docs/phase2_recovery_authorization.md](docs/phase2_recovery_authorization.md) |
| Authorization-bound post-recovery pilot and promotion runbook | [docs/phase2_post_recovery_hpc4.md](docs/phase2_post_recovery_hpc4.md) |
| HPC4 environment closure and Slurm execution | [docs/hpc4.md](docs/hpc4.md) |
| Base and recovery config identities | [configs/main.yaml](configs/main.yaml), historical [configs/common_beta_pilot.yaml](configs/common_beta_pilot.yaml), [configs/common_beta_recovery_pilot.yaml](configs/common_beta_recovery_pilot.yaml), [configs/identities.json](configs/identities.json) |

Selected implementation map:

```text
Smart-Reward-Model/             # retained repository name
├── configs/                    # Phase 1 identities and Phase 2 base/overlay designs
├── containers/                 # digest-locked HPC4 definition and exact runtime lock
├── docs/                       # theory, examples, protocol, HPC4 runbook
├── scripts/hpc4/               # preflight, driver probe, staging, GPU smoke, arrays
├── src/smart_reward/           # retained compatibility package
│   ├── annotations.py          # randomized repeated-label estimator
│   ├── objective.py            # moment, reported value, envelope gradient
│   ├── training.py             # paired BT-MLE / ProRM+ trainers
│   ├── phase1.py               # immutable real-model materialization
│   ├── rollout.py              # natural directions and measured-KL updates
│   ├── repeated_label_diagnostics.py # randomized-estimator tail diagnostics
│   ├── phase2_training.py      # objective-specific convergence and controls
│   ├── phase2_rollout.py       # global-beta one-step deployment
│   ├── phase2_aggregate.py     # formal paired-seed gates
│   ├── phase2_pilot_aggregate.py # target-free calibration/freeze decisions
│   ├── phase2_recovery_aggregate.py # head-free recovery authorization
│   ├── phase2_post_recovery_control.py # crash-safe post-recovery control plane
│   ├── phase2_post_recovery_aggregate.py # calibration/freeze aggregation
│   ├── phase2_post_recovery_output.py # post-recovery output verification
│   ├── phase2_exploratory_aggregate.py # fixed-three descriptive-only aggregation
│   ├── phase2_campaign.py      # exact-30 terminal-slot finalization
│   ├── phase2_sensitivity.py   # frozen ridge/beta sensitivity grid
│   ├── phase2_mechanism.py     # exact-target and low-dimensional qualifiers
│   ├── statistics.py           # paired-seed aggregation
│   └── cli.py                  # fail-closed control plane
└── tests/
```

## 9. Current claim boundary and execution order

1. Preserve the completed Phase 1 aggregate and its `not_passed` conclusion.
2. Freeze and verify the Phase 2 implementation, image, base inventory, and
   dual base/overlay identities.
3. Preserve the failed original calibration and recovery execution revision 1
   as immutable history. Finish recovery execution revision 2; only three
   successful seed receipts plus exact terminal scheduler evidence may produce
   the head-free recovery-success authorization.
4. From that authorization, materialize and run a fresh three-seed
   post-recovery calibration and its strict aggregate. Recovery heads, labels,
   beta values, optimizer state, and policy state do not cross this boundary.
   If the length gate fails, rerun calibration under a new identity at the next
   horizon in `[256, 512, 1024]`, binding the failed parent aggregate SHA-256.
5. Run a second target-free freeze pilot with one global beta for every
   seed/arm. Start at the maximum calibration candidate; failures may only
   issue a new identity at the next `beta*=2` grid point, byte-bound to the
   immediately preceding failed freeze aggregate.
6. Bind the accepted freeze aggregate SHA-256, unique global `beta_0`, response
   horizon, optimizer schedule, implementation, image and data inventory into
   one new `budgeted_end_to_end` identity.
7. Audit and commit the fixed-three materializer, exactly-once held-array
   submitter, GPU wrapper, per-seed verifier, terminal-evidence capture and
   descriptive publication layer before allocating the E2E jobs.
8. Submit exactly fresh seeds `20261001`–`20261003` as one `0-2%2` array. There is no
   adaptive seed selection, outcome-dependent stopping or substitution.
9. Require all three canonical run directories and exact Slurm
   `COMPLETED/0:0/0:0` evidence, then publish only the three frozen
   `ProRM+ - BT-MLE` endpoints with mean/SD/min/median/max and the fixed paired
   descriptive bootstrap interval. No p-value, significance label, efficacy
   gate or formal population claim is permitted.
10. Preserve exact-30 and the broader sensitivity/robustness matrix only as
    future, separately authorized studies. The fixed-three outcome cannot
    trigger them.

With fixed finite labels, CoVal identifies only a truncated logit series. It must be reported as
**candidate-restricted truncated ProRM+ robustness** and cannot inherit the exact unbiasedness or human-
utility interpretation of the controlled experiment.

Primary engineering dependencies and data/model assets:

- [PyTorch](https://docs.pytorch.org/docs/stable/index.html)
- [Transformers chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)
- [MultiPref](https://huggingface.co/datasets/allenai/multipref)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Skywork Reward V2 Qwen3 0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B)
- [CoVal](https://huggingface.co/datasets/openai/coval)
