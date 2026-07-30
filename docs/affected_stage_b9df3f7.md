# Affected-stage analysis for `b9df3f7`

## Change

Commit `b9df3f7b6649e4ecb12c3355655004db0bcc5cbd` changes:

1. the numerical parameterization used by the full-rank MLE reward-head optimizer; and
2. the domain and coverage reporting of prompt-centered reward NMSE.

It does not change prompt loading or sampling, policy generation, oracle scoring, train-only
oracle standardization, frozen reward features, policy score features, split assignment,
random-number seeding, tensor precision, or materialized artifact serialization.

## Dependency conclusion

The earliest affected stage is `reward`. The required recomputation closure is:

```text
reward -> adapters -> rollout -> rollout-aggregate -> aggregate
```

The completed `materialize` artifacts produced at commit
`27eb42b691b13fcb3963d004a70da56a911a99f2` are scientifically unaffected and may be reused
only when their original receipts and artifact hashes validate and a provenance bridge records
the original producer and the new consumer without rewriting the original receipt.

The failed reward results from job `1675136` are not reusable and must remain failure evidence.
