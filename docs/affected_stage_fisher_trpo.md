# Fisher-TRPO affected-stage analysis

The immutable `e01359d` run remains the source of legacy evidence. Its result,
receipts, and archive are not modified.

## Unaffected reusable components

- frozen train and validation prompt identities;
- their six pi0 candidates per prompt;
- policy-score and reward-feature tensors;
- raw oracle scores and train-fitted oracle affine transform;
- fixed LoRA-A state and LoRA-B layout;
- converged MLE-RM head, after a fresh train-gradient and head-digest gate.

Every reuse records the source artifact, JSONL, tensor, component, receipt, and
producer identities. A mismatch fails closed.

## Earliest affected components

- the confirmatory test split is replaced because legacy test was inspected;
- inverse-Fisher regularization is selected by train-only cross-fit;
- Pro-RM depends on the inverse Fisher and is therefore refit;
- all MLE, Pro, and oracle directions depend on the selected rule and are recomputed;
- policy scaling changes from fixed beta to matched-KL TRPO.

Therefore every adapter, KL calibration, fresh rollout, seed aggregate, and
three-seed aggregate is recomputed.
