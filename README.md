# Prospective Reward Model

This repository contains the controlled ProRM experiment: learn a reward model for
the one-step policy update it induces, rather than only for preference likelihood.

## Formal real-policy evaluation

The current formal result fixes beta = 0.2, writes the MLE-RM, Pro-RM, and
oracle-r* one-step NGD updates into the model's LoRA-B parameters, and performs
fresh generation on 512 fixed test prompts for each of four policies and three
seeds. The retained evidence and full protocol are in
[Real Policy Evaluation at beta = 0.2](results/real_policy_beta0p2/README.md).

All three updated policies improve regularized test utility over pi0 in every
seed. The three-seed mean ordering is oracle-r* > MLE > Pro > pi0. Therefore the
real-policy rollout does not support the stronger Pro > MLE claim; the older
finite candidate-pool result below must not be interpreted as evaluation of the
actually updated language-model policies.

## Exploratory finite-pool Main Experiment v1

The first complete main experiment is the **Fisher-corrected, common-beta, one-step
NGD evaluation** with three formal seeds:

- seeds: **20261001**, **20261002**, **20261003**;
- report betas: **0.1**, **0.2**, **0.3**;
- nominal condition: **beta = 0.2**;
- policies: pi0, MLE-NGD, Pro-NGD, oracle-NGD, and the closed-form finite-pool
  tabular optimum;
- Fisher: train-node raw second moment with train-only five-fold cross-fit;
- selected relative damping: **10**;
- test use: frozen candidate-pool evaluation only.

The three-seed means satisfy

~~~text
R: tabular > oracle > Pro > MLE > pi0
J: tabular > Pro > oracle > MLE > pi0
~~~

Pro-NGD has higher regularized utility than both MLE-NGD and oracle-NGD in all
3 betas x 3 seeds. At beta = 0.2, the mean gaps are
J(Pro)-J(MLE) = 0.005098 and J(Pro)-J(oracle) = 0.000935.

The reward-level result is deliberately reported as a limitation: ProRM has worse
held-out NLL, centered reward MSE, and approximate regret than MLE-RM. The train
profiled objective was optimized successfully, so the current diagnosis is
reward-moment overfitting / insufficient held-out generalization, not a failed
PCG solve or an invalid NGD update.

The beta subset was chosen after inspecting a dense test sweep. Main Experiment v1
is therefore complete and auditable **exploratory evidence**, not a new held-out
confirmatory test. A future confirmation should freeze beta = 0.2 before opening a
new test split.

## Results and report

- [Machine-readable summary](results/main_experiment_v1/summary.json)
- [Complete retained evidence](results/main_experiment_v1/README.md)
- [Academic report source](reports/main_experiment_v1/ProRM_main_experiment_v1.tex)
- [Report build instructions](reports/main_experiment_v1/README.md)

The retained evidence includes all three seed-level evaluation JSON files, the
three-seed aggregate, integrity audit, and immutable archive manifest. The full
server archive remains content-addressed and unchanged.

## Scientific workflow

~~~text
validated material + reward heads + train Fisher
  -> three frozen train directions (MLE / Pro / oracle)
  -> common-beta NGD candidate-pool evaluation
  -> five-policy metrics on frozen test candidates
  -> three-seed aggregate
  -> integrity audit
  -> immutable HPC4 archive
  -> local Git evidence bundle and report
~~~

The dense sweep is retained in the raw evidence for audit and future analysis. The
main report exposes only beta = {0.1, 0.2, 0.3}. Unaffected upstream artifacts are
reused by immutable receipt and hash; a change only recomputes its earliest affected
stage and scientific downstream closure.

## Core definitions

For a frozen train direction:

~~~text
d_m = F_train,lambda^-1 g_m,
delta_m(beta) = d_m / beta.
~~~

On the six-candidate test pool:

~~~text
R(pi) = E_pi[r*]
K(pi) = KL(pi || pi0)
J(pi) = R(pi) - beta K(pi)
DeltaJ(pi) = J(pi_tabular) - J(pi)
betaKL(pi) = beta KL(pi || pi_tabular)
~~~

The implementation verifies the exact finite-pool identities
J(pi_tabular) = J_close and DeltaJ = betaKL.

See [theory](docs/theory.md), the
[Fisher/TRPO ancestor protocol](docs/fisher_trpo_protocol.md), and the
[codebase guide](docs/codebase_guide.md) for the broader derivation and provenance
history. Main Experiment v1 itself uses common-beta NGD, not TRPO scaling.

## Development

~~~bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
~~~
