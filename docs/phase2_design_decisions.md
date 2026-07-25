# Phase 2 design decisions

> **Current route:** this document preserves the complete future formal
> design, including exact-30. The experiment currently being executed is the
> non-formal fixed-five budgeted route documented in
> [phase2_budgeted_end_to_end.md](phase2_budgeted_end_to_end.md). Nothing in a
> five-seed exploratory result activates the exact-30 protocol.

This document records the decisions made after auditing the completed Phase 1
campaign. It is intentionally separate from the frozen Phase 1 result. Nothing
in this document changes the authoritative Phase 1
`pre_registered_evidence_status=not_passed`.

The paper title remains:

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward
> Models by Downstream Policy Regret**

## 1. Status

The population/local ProRM estimand and the repeated-label identifying moment
remain the scientific core:

$$
\mathcal R_\beta(r_\phi)
=
\frac{1}{2\beta}
\left\|A_0(r_\phi-r^*)\right\|_{F_0^\dagger}^2,
\qquad
m_\phi
=
\frac12\mathbb E[z(\Delta r_\phi-h)]
=
A_0(r_\phi-r^*).
$$

Phase 2 is not a rerun of Phase 1. It uses fresh candidates, labels, heads,
policy rollouts, seeds, and a new design identity. A confirmatory identity is
admissible only after the target-free optimization/KL/response-horizon pilot
described below has produced an accepted, hash-bound freeze aggregate. This
design document intentionally does not assert live HPC4 queue or campaign
status.

## 2. Decisions that are frozen in principle

The following choices will not be selected from held-out reward, utility, or
learner-ordering outcomes:

1. BT-MLE and ProRM+ receive the same candidate graph, frozen reward features,
   underlying Bernoulli annotations, zero head initialization, and policy
   tangent.
2. The noisy primary arm uses four independent `gamma=0.9`
   randomized-truncation estimates per canonical edge. ProRM+ uses their
   arithmetic mean; BT-MLE uses every underlying Bernoulli label.
3. The confirmatory campaign uses one scalar `beta_0` shared by every formal
   seed and every policy arm. Learner-specific and formal-seed-specific
   calibration, line search, or norm normalization is forbidden.
4. The finite-policy endpoint is

   $$
   J_\ell^*
   =
   \mathbb E_{\pi_\ell}[r^*]
   -
   \beta_0
   D_{\rm KL}(\pi_\ell\Vert\pi_0),
   $$

   with histories sampled from the updated policy.
5. `D_KL(pi_0 || pi_updated)` is diagnostic only.
6. The train-oracle arm is an algorithmic local reference, not a global
   optimum.
7. The Skywork reward model defines a controlled operational oracle. The
   experiment does not identify human utility.
8. Failed optimization, numerical, identity, safety, or positive-control gates
   fail the complete seed. Failed seeds cannot be deleted or replaced.

The oracle coordinate system is global as well. Phase 2 never refits the
robust transform on a pilot or formal seed. It freezes

$$
b_0=-4.500244140625,\qquad
\tau_0=2.7715682983398438,
\qquad
r^*=\frac{\log 3}{2}\tanh\!\left(\frac{R_{\rm Skywork}-b_0}{\tau_0}\right).
$$

These two values are the predeclared componentwise medians of the five
train-only transforms from Phase-1 seeds `20260722` through `20260726`. Those
seeds are excluded from every Phase-2 analysis. The base and overlay configs
bind the Phase-1 semantic config hash and each source artifact
`metadata.json` SHA-256; materialization records the complete provenance and
fails if it attempts a current-seed refit. This makes `r*`, not merely the
rule used to construct `r*`, identical across pilot and formal seeds.

The policy tangent basis is also global. Every Phase-2 materialization and
policy reload initializes LoRA-A with named seed `946081152281754541`, taken
from the excluded minimum Phase-1 seed `20260722`, and must reproduce
`A_SHA256=a2b5804109396f76b96cde98d1e2060f175a47724b1ca9fef317c7a10cb9a838`.
The config binds that seed, fingerprint, Phase-1 config identity, and source
artifact metadata hash. Current-run seeds still control prompt selection,
candidate generation, labels, heads, minibatches, and rollouts, but they do
not change the fixed-A policy class. Phase 1 retains its historical
per-seed-A semantics.

The excluded pilot seed `s` may compute only the train-only calibration
candidate

$$
\widetilde\beta_s
=
\sqrt{\frac{\kappa_s}{2K_{\rm cal}}},
\qquad
\kappa_s
=
(u_{*,s}^{\rm tr})^\top F_{{\rm tr},s}u_{*,s}^{\rm tr}.
$$

The pilot is deliberately two-stage over permanently excluded seeds
`20260801`, `20260802`, and `20260803`. Stage A
(`pilot_phase=calibration`) produces only the three seed-specific train candidates. Its strict pilot
aggregate verifies all three result/JSONL hashes and defines the first
global-beta grid point by

$$
\beta_{\rm base}
=\max_{s\in\mathcal S_{\rm pilot}}\widetilde\beta_s,
\qquad
\beta^{(k)}=2^k\beta_{\rm base}.
$$

Stage B (`pilot_phase=freeze`) is a new design identity. It binds the Stage-A
aggregate byte SHA-256 and one finite positive `frozen_global_beta`, then uses
that exact scalar for all three seeds and all updated arms. It still stops
before held-out evaluation or final-oracle scoring. If any non-length
pre-oracle gate fails, the only permitted adjustment is a new freeze identity
at the next point in `{beta_base, 2 beta_base, 4 beta_base, ...}`. The first freeze's
`beta_source_aggregate_sha256` binds Stage A. For grid index `k>0`, that field
instead binds the immediately preceding freeze aggregate, which must show:
the same accepted horizon, a non-length-only safety failure,
`selection_accepted=false`,
`next_action=issue_new_pilot_freeze_identity_at_double_beta`, and
`next_global_beta` equal to the new scalar. The new index must be exactly
`k-1 -> k`; direct jumps are rejected. The independent
`parent_pilot_aggregate_sha256` continues to bind the calibration aggregate
that accepted the current horizon. Only an all-seed pass may source the
confirmatory identity. Formally,

$$
k_*=\min\{k\ge0:\beta^{(k)}\text{ passes every frozen freeze gate}\},
\qquad
\beta_0=2^{k_*}\beta_{\rm base}.
$$

The confirmatory scalar `beta_0` exists only when this set is nonempty;
otherwise the campaign stops without a confirmatory identity.

The frozen pre-oracle thresholds are: mean updated-to-reference KL `0.02`,
prompt-mean KL `p95=0.02`, `p99=0.05`, prompt maximum `0.10`, per-sequence
maximum `0.20`, and maximum-length rate `0.05`. Both pilot stages publish these
as measure-only target-free diagnostics; only the Stage-B aggregate makes the
engineering selection decision. Per-seed calibration remains a pilot-only
diagnostic. Formal sensitivity is separately frozen as `beta=c*beta_0` for
`c in {0.5, 2.0}`; formal-seed curvature never changes that scalar.

The closed config/runtime contract makes this distinction executable:

- calibration pilot: `pilot_phase: calibration`,
  `frozen_global_beta: null`,
  `beta_source_aggregate_sha256: null`,
  `calibration_split: train`,
  `sensitivity_k_cal: [0.001, 0.01]`, and
  `sensitivity_frozen_global_beta_multipliers: null`;
- freeze pilot: `pilot_phase: freeze`,
  `frozen_global_beta: <finite positive grid point>`,
  `beta_source_aggregate_sha256: <Stage-A SHA at k=0; immediate failed-freeze SHA at k>0>`,
  `calibration_split: excluded_pilot_calibration`, and both sensitivity
  fields null;
- confirmatory: `frozen_global_beta: <finite positive beta_0>`,
  `beta_source_aggregate_sha256: <accepted Stage-B aggregate SHA-256>`,
  `calibration_split: excluded_pilot`,
  `sensitivity_k_cal: null`, and
  `sensitivity_frozen_global_beta_multipliers: [0.5, 2.0]`.

The confirmatory result records current-seed oracle curvature only to report
the predicted local KL at the already frozen beta. It is not an input to step
size selection.

## 3. Optimization pilot

Equal optimizer steps are not the primary scientific comparison. BT-MLE is a
convex logistic objective and ProRM+ is a convex quadratic objective, but they
have different curvature and conditioning. Comparing the two heads after the
same arbitrary AdamW step count would mix estimator error with optimization
error.

The primary formal rule will be objective-specific first-order convergence:

$$
\rho_\ell
=
\frac{\|\nabla_w L_\ell(w_{\rm final})\|_2}
{\max(\|\nabla_w L_\ell(w_0)\|_2,\epsilon)}.
$$

The accepted gradient is recomputed after an optimizer update and is:

- full-data rather than minibatch-local;
- unclipped;
- evaluated on the saved head;
- for ProRM+, based on a fresh cold-start FP64 Fisher solve and the exact
  envelope gradient of the reported quadratic objective.

The optimization pilot uses seeds that are permanently excluded from the
formal campaign. It may inspect train-only optimization trajectories to freeze:

- relative-gradient tolerance;
- minimum and maximum optimizer steps;
- audit interval and required consecutive successful checks;
- the deterministic zero-initialized algorithmic tie-breaking contract and
  reward-head rank evidence. AdamW from zero is not described as a Euclidean
  minimum-norm solver; if the objective is rank deficient, that limitation is
  reported or a separately frozen explicit tie-break is introduced before the
  confirmatory design.

The pilot may not invoke the held-out evaluator, open a final oracle-scoring
session, or compute reward, utility, regret, or learner ordering. Failure to
reach the frozen gate in a formal run is a hard seed failure.

The 720-step head remains a compute-matched secondary checkpoint. It is never
substituted for the converged primary head.

### 3.1 Pilot information boundary

The pilot is an outcome-blind engineering stage, not a small efficacy study.
Its public result and JSONL sidecar may contain only:

- train-only convergence, numerical, rank, and positive-control diagnostics;
- the train-only `widetilde beta_s` calibration candidate;
- arm-wise response length, EOS, maximum-length, and on-policy KL summaries;
- immutable source, config, artifact, environment, and output hashes.

It must not serialize prompt or response text, token IDs, reward-head weights
or direction vectors, held-out targets, final oracle scores, rewards, utilities,
regrets, or pairwise learner comparisons. The source bridge artifact can
contain tensors created by the earlier materialization contract; pilot code
must not read or republish those fields. This is an auditable operational
information boundary, not a cryptographic claim that the source bytes never
existed.

## 4. Response-horizon pilot

The Phase 1 horizon of 128 tokens produced excessive maximum-length
termination. Merely doubling it is not evidence that truncation is controlled.
The pre-formal pilot must measure the maximum-length rate for every policy arm
and select a horizon without computing or reading learner ordering.

The response horizon follows the preregistered sequence `[256, 512, 1024]`.
The initial identity uses 256. If the all-arm maximum-length-rate gate fails,
the next run must be a new calibration identity at the immediately following
horizon. It must bind the failed parent pilot aggregate SHA-256 and set
`previous_horizon_failed_length_gate=true`; changing the horizon inside an
identity is invalid. Because the horizon changes trajectories and geometry,
calibration and freeze are both rerun at the accepted horizon. Exhausting 1024
without a pass stops the protocol for redesign rather than silently expanding
the grid.

The formal design freezes both:

- one response horizon; and
- one maximum-length-rate acceptance threshold.

The same pilot reports prompt-level mean on-policy KL tails (`p95`, `p99`, and
maximum) for every arm, in addition to the global mean. A finite number of
large prompt-level updates can violate the local regime even when the mean is
small. Any tail threshold used formally must be frozen under a new identity;
the present pilot records it measure-only and never uses it to rank learners.

Prompt semantics are fail-closed rather than another pilot-selected quantity.
The Qwen2.5 policy renders every original MultiPref user message with its own
chat template with `truncation=False`. The frozen cap is 1024 rendered policy
prompt tokens. A deterministic local precheck over all 5,323 unique prompts in
the pinned MultiPref snapshot found 88 over the cap (`1.65%`), leaving 5,235
eligible prompts. Under the old shuffle-before-length-check order, the three
pilot seeds would select 39, 34, and 36 over-limit prompts respectively and
therefore fail closed before useful work.

The corrected sampling contract first renders and counts every unique prompt,
constructs the `<=1024` eligible pool, and only then applies the seeded
shuffle/split. No prompt is truncated. Artifact metadata records total unique,
eligible, excluded, and selected counts; hashes of the corresponding prompt-ID
lists; and per selected prompt the raw-text hash, policy token count,
prompt-token-prefix hash, cap, and `truncated=false`. Phase 2 revalidates those
records against the candidate graph, and each rollout trajectory carries the
same evidence. The formal prompt population is therefore the declared
length-eligible MultiPref subset, not all MultiPref prompts.

The Skywork Qwen3 oracle receives that identical raw prompt plus the assistant
response and independently renders them with its pinned Qwen3 tokenizer and
chat template. Qwen2.5 template tokens are never reused as Qwen3 input. The
5,323/88/5,235 audit is a reproducible input precheck, not a pilot or learner
effect result.

If a practical horizon cannot satisfy the threshold, the estimand will be
renamed explicitly as a capped-action-space utility. It will not be presented
as the unrestricted generation-policy utility.

## 5. Positive-control gates

The aggregate must derive every gate from serialized numerical evidence. A
constant `all_controls_passed=true` is forbidden.

Required gates are:

| Control | Required evidence | What is not required |
|---|---|---|
| R=4 label stream | four independent streams, preserved replicate boundaries, no clipping, ProRM mean-`h`, BT all raw labels | a favorable finite-sample learner ordering in every seed |
| Direct oracle identity | complete-pair and all-node moment identity within a frozen tolerance; converged direction solve | a trained reward head |
| Exact-margin head | exact `h=Delta r*`, objective decrease from zero, full-gradient convergence | equality to the oracle direction under a misspecified reward class |
| Exact soft-label BT | train on the exact `p*=sigmoid(Delta r*)` rather than sampled Bernoulli outcomes | use as a primary learner |
| Low-dimensional tangent | dimension and numerical rank 256, orthonormal projection, Moore-Penrose residual, projection/scatter score identity, full-gradient convergence | use in the primary full-tangent claim |
| Oracle finite step | positive paired utility improvement over zero-B | global optimality |

The trained exact-margin oracle-direction gap is retained as a decomposition of
reward-class and optimizer error. It is not forced to zero.

The pilot may exercise only the train/local controls and their target-free
diagnostics. The oracle finite-step utility gate is evaluated for the first time
inside the fresh confirmatory campaign; it is not computed, serialized, or
reported by the pilot.

The exact soft-label BT and exact-margin ProRM+ heads form a separate
misspecification diagnostic: they remove annotation noise from both objectives
and show whether the restricted reward class gives the two population
objectives room to select different solutions. This control uses no additional
generation or oracle forward pass. It must be bound before the confirmatory
identity is issued, but its favorable ordering is not substituted for the noisy
primary comparison.

This variance reduction is not free. With `gamma=0.9`, one randomized estimate
uses `E[N]=10` Bernoulli annotations. The `R=4` canonical-edge arm therefore
costs 40 annotations per prompt in expectation and has an unbounded geometric
tail; the all-six-pairs arm costs 240 in expectation. No hard cap, clipping, or
silent retry is permitted. This is the main practical annotation-cost
limitation of the exact unbiased construction and must be exposed in both
runtime accounting and the paper.

Finite variance is the strongest tail-moment guarantee used here. For this
estimator's endpoint tail calculation, the locked second-moment ratio is
`max(p*, 1-p*) / gamma = 0.75 / 0.9 < 1`, while the fourth-moment ratio is
`max(p*, 1-p*) / gamma^3 = 0.75 / 0.9^3 > 1`. Thus the single-replicate
estimator has a finite second moment but an infinite fourth moment at the
probability-range endpoints. Averaging `R=4` independent replicates divides
conditional variance by four but does not change that tail exponent. We
therefore make no sub-Gaussian or finite-fourth-moment claim.

Before post-recovery calibration and confirmatory execution, the evidence
schema is frozen to emit `repeated-label-tail-diagnostics/v1` under
`label_stream`. It reports only nearest-rank empirical `p50/p90/p95/p99/max`
for replicate counts, `abs(replicate_h)`, and `abs(mean_h)`, with their sample
sizes and source-tensor SHA256s. Nearest rank means ascending order statistic
`x_(ceil(q*n))` with one-based indexing and no interpolation. The canonical
diagnostic SHA is itself bound into `label_stream_sha256`. The object is
scalar-only and descriptive-only; it is explicitly forbidden from clipping,
selection, gating, beta calibration, seed exclusion, or retry decisions.

## 6. Formal endpoints and decision rule

After the pilot freezes the single global `beta_0`, the formal claim is
supported only by the intersection of:

1. held-out fixed-`beta_0` local regret favors ProRM+ over BT-MLE;
2. the paired-seed interval for
   `utility(ProRM+) - utility(BT-MLE)` has lower endpoint above zero;
3. the paired-seed interval for
   `utility(ProRM+) - utility(zero-B)` has lower endpoint above zero;
4. the paired-seed interval for
   `utility(oracle-step) - utility(zero-B)` has lower endpoint above zero;
5. every optimization, positive-control, KL-safety, provenance, and numerical
   gate passes.

If ProRM+ beats BT-MLE but both are worse than zero-B, the result is not a
successful downstream policy-improvement claim.

Candidate values are averaged within prompt. Each seed then contributes one
paired scalar. Formal uncertainty is computed over paired seeds; candidates or
prompts are not treated as independent training replicates.

For endpoint `k`, the formal estimand is

$$
\mu_k
=
\mathbb E_{\mathrm{RNG}}\!\left[
\Delta_k
\mid
\text{frozen eligible MultiPref pool, models, oracle, and design}
\right].
$$

The single positive paper claim is an intersection-union test:

$$
H_0=\bigcup_k\{\mu_k\le 0\},
\qquad
H_1=\bigcap_k\{\mu_k>0\}.
$$

Each component uses a two-sided 95% paired-seed percentile interval and must
have lower endpoint strictly above zero, an effective one-sided component
level of `0.025`. No Bonferroni adjustment is required for this one conjunctive
claim; the repository does not make separately unadjusted endpoint claims.
The interval is a frequentist RNG interval conditional on the frozen
experimental system, not a confidence interval for an unrestricted human
prompt population. At `n=30`, the prospective normal approximation gives an
80%-power minimum detectable effect of about `0.53` paired standard deviations
for one component; conjunctive power is controlled by the weakest component.

The formal paper campaign uses exactly 30 preregistered paired seeds,
`20260901` through `20260930` in that order, all with the same `beta_0`. Pilot
seeds, the five Phase 1 seeds, and any seed observed while
changing the design are excluded. Seed-conditional `K_cal` calibration is a
pilot-only scale diagnostic and is forbidden in confirmatory execution.
Every one of the 30 seed slots must end in either one admissible result or one
immutable terminal failure manifest. The formal ledger policy is
`single_predeclared_attempt_no_retry`: each seed has exactly `attempt-1`, the
registry `recoveries/` directory must remain empty, and neither retry nor
replacement seed is admissible; optional stopping is also forbidden. Before
the first Slurm submission, the submitter immutably commits
`campaign-plan.json`, binding the ordered task-to-seed map and the eight fixed
waves `0-3%2`, `4-7%2`, `8-11%2`, `12-15%2`, `16-19%2`, `20-23%2`,
`24-27%2`, and `28-29%2`. The plan permits at most four submitted tasks and
two running tasks, matching the observed HPC4 `l20_qos MaxSubmitJobsPU=4`.
A wave is submitted only after every task in all earlier waves has a strict
terminal bundle, irrespective of its scientific or scheduler outcome. Thus
the waves are a capacity-safe realization of the frozen exact-30 campaign,
not an adaptive design and not a change to its estimand. The GPU publisher,
compute and scheduler terminalizers, registry resolver, and CPU finalizer
jointly enforce one terminal head per slot. A failed slot produces
`not_passed_due_to_seed_failure` with no primary CI; valid negative effects
retain their intervals and produce `not_passed`.

Wave eligibility is itself immutable evidence. Before each `sbatch`, the
submitter fsyncs `admissions/wave-<index>.json`; for later waves it hash-binds
the preceding admission and submission plus all ordered predecessor terminal
manifest and marker bytes. Submission v3 binds this receipt and a canonical
record of the raw held `scontrol` response and normalized scheduler request.
The resolver recomputes the complete chain. Walltime and all Slurm resources
are plan-bound, while `squeue` plus historical deterministic-name `sacct`
queries make an unregistered accepted job a fail-closed condition rather than
a license to resubmit.

The terminal ownership rule is structural. The GPU job builds its complete
bundle in a hidden staging directory and atomically renames it to the canonical
job directory only after validation and durable sync. If a hard termination
occurs before that rename, no canonical job exists and the scheduler
terminalizer may bind the terminal `sacct` root record. If the canonical
directory exists with `FAILURE_PENDING`, only the compute terminalizer may
complete it. A published success or failure is immutable and idempotently
recognized. None of these recovery operations creates another scientific
attempt.

The post-recovery CPU aggregate uses a different directory-publication
primitive because the HPC4 project filesystem rejects
`renameat2(RENAME_NOREPLACE)`. GNU `mv --no-clobber` is not an equivalent
fallback: without native support it has a check-then-rename race. The
terminalizer instead claims the evidence directory with atomic `mkdir`,
installs a claim binding the outer attempt owner and payload manifest, and
hard-links each already-fsynced staged file with create-if-absent semantics.
Only an exact claim-bound prefix may resume, and only the later `.PUBLISHED`
receipt authorizes consumption. Thus a crash can expose an incomplete
directory but cannot expose it as a valid aggregate or overwrite an occupied
name.

## 7. Ridge, scale, and efficiency experiments

The full LoRA-B tangent is rank deficient at the available sample size. Its
ridge objective is a regularized algorithmic realization, not the exact
unregularized population pseudoinverse theorem.

The experiment hierarchy is:

1. central full-tangent ridge and the globally frozen `beta_0` form the
   confirmatory finite-policy experiment;
2. all configured ridge scales retrain ProRM+ and report held-out local
   geometry;
3. formal beta sensitivity uses only the prespecified multipliers
   `c in {0.5, 2.0}` and deploys every sensitivity seed/arm at
   `beta=c*beta_0`; it never recomputes beta from that seed's curvature and
   cannot select a favorable primary scale;
4. a 256-dimensional, full-rank, ridge-free Moore-Penrose control tests the
   theorem's identifiable geometry;
5. the all-six-pairs prompt U-statistic is a separate efficiency experiment,
   clustered by prompt and ineligible to replace the canonical-edge result.

An ordering reversal across mandatory ridge scales prevents a robustness claim
about the population mechanism, even when the central finite-policy contrast is
positive.

## 8. External validity

The controlled experiment deliberately creates BTL labels from a transformed
Skywork operational oracle and evaluates against that same frozen target. This
is a mechanistic estimator test, not proof of alignment with humans.

External evaluation must therefore be reported separately:

- human-labeled original response pairs may test reward-model ordering under
  distribution shift;
- newly generated Qwen responses require labels on those exact responses before
  they can support a human downstream-utility claim;
- no MultiPref annotation attached to a different response may be reused as a
  label for a newly generated response.

## 9. Experimental lessons taken from AuxDPO

The AuxDPO paper is useful here as an experimental-design precedent, not as the
method being implemented. Phase 2 adopts four parts of its evidence
architecture:

1. an analytic low-action misspecification example before the large-model
   result;
2. matched data and compute for the primary estimator comparison;
3. capacity/sample-size stress tests that ask when the misspecification effect
   becomes visible;
4. many paired seeds and separate in-distribution, out-of-distribution, and
   control evaluations.

For this project, “many” means exactly 30 preregistered paired formal seeds, not a
five-seed point estimate. OOD or human-labeled evaluations remain external
validity endpoints and cannot replace the controlled operational-oracle
mechanism test.

AuxDPO's auxiliary null-space parameterization, IPO/DPOP baseline hierarchy,
and direct-policy objective are not transplanted. They address DPO
misspecification, whereas this project asks how to train a reward model for
downstream policy regret. BT-MLE is therefore the primary reward-model
baseline; exact-soft BT, exact-margin ProRM+, zero-B, and oracle-step controls
isolate label noise, reward-class misspecification, and local-to-finite
transfer.

## 10. Claim boundary

The exact claim remains narrow and testable:

- the theorem is population, local, one-reference, and undamped; the main
  full-tangent experiment is its ridge-regularized finite-sample realization;
- the policy experiment is one frozen LoRA-B update, not PPO, multi-step RLHF,
  or a guarantee for arbitrary large updates;
- Skywork is an operational oracle, not human utility;
- preference NLL/accuracy are diagnostics, not the primary success criterion;
- the low-dimensional ridge-free arm tests the identifiable theorem, while
  capacity, all-six-pair, frozen-global-beta sensitivity, and OOD studies are
  secondary; seed-specific beta is forbidden in the confirmatory estimand;
- Phase 1 remains authoritatively `not_passed`; Phase 2 does not retroactively
  change it.

## 11. Execution order

```text
finish and verify the pilot-capable implementation
  -> run target-free calibration pilot and strict three-seed aggregate
  -> if needed, rerun calibration at the next identity-bound horizon
  -> run target-free fixed-global-beta freeze pilot on the beta*=2 grid
  -> freeze accepted beta_0, horizon, numerical gates, and formal identity
  -> precommit the exact-30 campaign plan and run its eight fixed waves
  -> run mandatory ridge and frozen-global-beta multiplier sensitivity
  -> run all-six and capacity/sample-size secondary experiments
  -> run separate human/external robustness evaluation
  -> publish the immutable aggregate and final report
```

No formal result is accepted until every upstream arrow has authoritative
artifact, manifest, hash, and numerical-gate evidence.
