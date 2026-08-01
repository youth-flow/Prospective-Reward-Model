# Main Experiment v1 evidence

This directory is the compact, version-controlled evidence bundle for the first
complete ProRM main experiment.

## Frozen interpretation

- Report conditions: beta = {0.1, 0.2, 0.3}.
- Nominal condition: beta = 0.2.
- Reward diagnosis: ProRM overfits the train policy-moment objective and has worse
  held-out NLL, centered reward MSE, and approximate regret than MLE-RM.
- Downstream result: Pro-NGD has higher test-candidate regularized utility than
  MLE-NGD and oracle-NGD in all nine seed x beta conditions.
- Evidence class: exploratory main result, because the displayed beta range was
  selected after the dense test sweep.

The raw JSON retains the full dense grid
{0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 4} so later work can audit
the selection or formulate a new preregistered run. summary.json contains only
the frozen report subset and is regenerated deterministically.

## Files

~~~text
evidence/
  seed-20261001-evaluation.json
  seed-20261002-evaluation.json
  seed-20261003-evaluation.json
  aggregate.json
  integrity-audit.json
  ARCHIVE_SHA256SUMS
summary.json
~~~

## Provenance

~~~text
producer commit:
  0f51c6a59454e7ad7e4fa9fd09f022feb0598d68

HPC4 run:
  /scratch/yyangjo/prorm/runs/fisher-ngd-beta-sweep-v4-0f51c6a

immutable archive:
  /project/sigroup/yyangjo/prorm/archives/
  fisher-ngd-beta-sweep-v4-0f51c6a-20260801

Slurm jobs:
  evaluation 1685687
  aggregate  1685688
  audit      1685689
~~~

The audit status is passed. The archive is preserved; no intermediate archive
was deleted or overwritten.

## SHA-256

~~~text
fb6967abf4a8b0ca679c6a022c9537a6d1dc471a90187580e110b751605ac930  aggregate.json
fac001daf941a365d6db4541219e24b26d01e028d9a158e111a092040d097e85  ARCHIVE_SHA256SUMS
b1df1781fe71518557908f9ad1d986918ca6c5c0a01ca214fddf5c1ffe6c2874  integrity-audit.json
8e926b64effe9285a7a40b8b6f05133cec8e38fb8984bc8d2769520cea3ca653  seed-20261001-evaluation.json
52c58457b2eef9f0de3d6a4c01ffc238409b64319d1326d7ad86db2002693b53  seed-20261002-evaluation.json
6c9ad5cea0d424ec9b23ad9a72d4874d2a15ac88ac07280a212f544c36d92795  seed-20261003-evaluation.json
~~~

These are scientific outputs, not fixtures. Do not edit them manually; regenerate
only summary.json and the LaTeX tables with the reporting script.
