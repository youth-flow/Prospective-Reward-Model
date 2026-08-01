# DPO/AuxDPO Main Experiment v1

- Formal run root: `/scratch/yyangjo/prorm/runs/dpo-auxdpo-main-45e83c0`
- Seeds: `20261001`, `20261002`, `20261003`
- Betas: `0.1`, `0.2`, `0.3`
- Methods: `dpo`, `auxdpo`
- Source configuration SHA-256: `0ff8cb872c5bdecc33bd2e5ded7d9c3adcbc43d7b6c355b40f8d34a1ae95ce92`
- Extension configuration SHA-256: `b28b15a57e58df975a9806d912dad40db8e0817217752c8d76011c1e01d003ac`
- Hugging Face inventory SHA-256: `701c43fe0fa376b172f95fa9004b15a39d3faa3a526dc2d7dc9b317cf88f514b`

## Producers

- Reference and training commit: `45e83c0397b34b014d338e703551de90fe0c8c89`
- Reference and training image SHA-256: `82dddf2ec53914df628faec34a68052ab908b0242c7429b9441f5940385c57e3`
- Evaluation commit: `00bba94b211721d9076c729ff48a80f300572e26`
- Evaluation image SHA-256: `88e0fcc3b2c497b506b83930d31e97fffa27ef7325034b5383556b928988349b`
- Final aggregate and audit commit: `41189b9e4741a0856845f73122179b806256861b`
- Final aggregate and audit image SHA-256: `6b9d8742f0f7ea1af05092a363260a1e7a894f477085e3101a7f446e0d89fc48`

The downstream-only commit changes preserve all reference and fit artifacts. They
normalize CLI paths during evaluation and pass the complete producer identity to
aggregation; neither change alters an estimator, training trajectory, or metric.

## Slurm jobs

- Formal reference: `1686639`
- Formal training: `1686699`
- Successful evaluation: `1686779`
- Successful final aggregate: `1686792`
- Successful final integrity audit: `1686793`

## Final evidence

- Aggregate SHA-256: `1ed94791f344f0f7ee9972ecf65775be3df10b9c67e8782e93f9e4428a7355e3`
- Integrity audit status: `passed`
- Maximum absolute Gibbs-identity residual across seeds: `2.220446049250313e-16`
- The pre-fix aggregate is retained as `aggregate-pre-provenance-fix.json`; its
  SHA-256 is `857c2ccd7a712e588fdcebea31698fb9406e31ad254700827208fdf8592f7b82`.

The server archive contains all tensors and logs. This local directory deliberately
contains compact JSON metadata, evaluations, and logs only.
