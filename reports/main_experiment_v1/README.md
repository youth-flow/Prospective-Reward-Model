# Main Experiment v1 report

ProRM_main_experiment_v1.tex is the canonical academic report. The result tables
in ProRM_main_results.tex are generated from the audited JSON evidence and should
not be edited by hand.

From the repository root:

~~~bash
python scripts/reporting/build_ngd_main_report.py \
  --aggregate results/main_experiment_v1/evidence/aggregate.json \
  --audit results/main_experiment_v1/evidence/integrity-audit.json \
  --seed-result results/main_experiment_v1/evidence/seed-20261001-evaluation.json \
  --seed-result results/main_experiment_v1/evidence/seed-20261002-evaluation.json \
  --seed-result results/main_experiment_v1/evidence/seed-20261003-evaluation.json \
  --summary results/main_experiment_v1/summary.json \
  --latex reports/main_experiment_v1/ProRM_main_results.tex

cd reports/main_experiment_v1
xelatex -interaction=nonstopmode -halt-on-error ProRM_main_experiment_v1.tex
xelatex -interaction=nonstopmode -halt-on-error ProRM_main_experiment_v1.tex
~~~

The report deliberately separates:

1. the theory and analytic example;
2. the negative reward-level generalization result;
3. the strong downstream policy result;
4. the post-hoc beta-selection limitation and external-validity boundary.
