# Phase-2 recovery success authorization

This is the fail-closed bridge between the one-shot train-only recovery and a
new full common-beta calibration pilot. It is not a calibration aggregate,
does not choose beta, and does not authorize policy optimization or held-out
evaluation.

The bridge is campaign-specific. Its production root is exactly
`/project/sigroup/smart-reward-model`; relocating or copying the evidence tree
does not create another valid authorization source.

## Locked candidate execution

The authorization validator is identity-locked to the following candidate
execution. This list fixes which run may be evaluated; it does **not** assert
that the array has completed or that authorization exists. The builder may
publish the authorization only after all three run receipts and the later
terminal scheduler capture satisfy every condition below:

- recovery design:
  `9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4`;
- optimizer schedule:
  `46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216`;
- Slurm array: `1648125`, execution revision `2`;
- ordered task/seed mapping:
  `0 -> 20260801`, `1 -> 20260802`, `2 -> 20260803`;
- retry reason: `pretrainer_hf_datasets_runtime_lock`;
- recovery producer commit:
  `ad7613b7cef3ff536ec62f6f80608ee29e927b1c`.

Each run must have the exact `SUCCESS` receipt schema, zero workload/final exit
codes, no `FAILED` marker, no recovery-failure evidence, the same locked
design/Git/parent identities, a formal single-L20 run manifest, and an exact
train-only `information_boundary`.

The authorization builder does not trust the
`five_head_recovery_protocol_verified` boolean. It independently deep-validates
the complete serialized `phase2-fresh-head-training/v3` object. The recovery
configuration bytes must equal the `ad7613b7` Git blob and SHA256; the loaded
Phase-2 config validator, training/tensor-hash implementation, full aggregate
gate, and this authorization validator are separately Git-bound in the
aggregation identity.

For all five heads, the builder requires the exact arm, learner, objective and
FP32 schema. It reconstructs each serialized vector as a `torch.float32`
tensor and recomputes the producer's dtype/shape/storage SHA256, including the
exact zero-head initialization hash. This is stronger than checking hashes
that are merely repeated inside the result. It also reuses the full Phase-2
identifiability, PCG, exact-margin, exact-soft-BT and low-dimensional geometry
gates.

The `ad7613b7` producer predates the later `final_pcg` projection: for the
primary and exact-margin ProRM+ heads it serialized the complete eight-field
fresh-inner record (`method`, `dtype`, `cold_start`, `warm_start_used`,
`iterations`, `residual_norm`, `relative_residual`, `converged`). The
authorization boundary requires exactly that historical schema and exact
equality to the restored final-gate inner-solver record. Only after this raw
evidence passes does an in-memory deep-copy adapter project those two records
to the five-field `training_final` view expected by the newer shared gate.
The source result is never mutated, and neither a pre-projected five-field
historical record nor a mixed schema is accepted.

For each first-order trace, checks must be exactly `20, 40, ...` through the
first iterate that completes three consecutive passing checks. The builder
recomputes every gradient ratio, eligibility flag, threshold flag, consecutive
counter, selected step and scheduled learning rate. It strictly validates the
initial, restored-final, fixed-720 and legacy-5760 measurement schemas and
their objective-specific inner solver (`null`, cold FP64 PCG, or truncated
Moore-Penrose pseudoinverse). Executed updates must equal exactly
`max(selected_step, 720, 5760)`. The complete AdamW execution flags,
per-update counters, crossed LR boundaries, selected/restored hashes, and
selected-checkpoint composite hash are rederived. Finally, the builder
rederives the training-instance hash and canonical
`recovery-output-verification.json` and requires equality with the published
receipt.

The frozen parent registry and validator must equal their `ad7613b7` Git blobs.
For each seed the builder reruns that validator against the production project
root, rehashes all six parent artifact files and all five parent-run evidence
files, checks both before/after snapshots, and requires `parent-artifact` to be
a relative symlink resolving to that seed's exact registry artifact. Result
artifact hashes must equal the freshly verified registry entry.

The aggregate records hashes for every evidence file but does not copy trained
reward-model parameters, optimizer states, selected training steps, labels,
rewards, responses, or policy state.

## Supplementary live control receipt

The no-overwrite, mode-`0440` submission/live receipt is fixed at:

```text
/project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/
9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4/
execution-2/scheduler-control-live-20260725T153801+0800/
scontrol-array-1648125.txt
```

Its SHA256 is
`cb61484f435747d6705ff4567257afff2c447faa16144b697e9f9dcc03f83a5e`.
It binds the real, non-consecutive Slurm allocation IDs:

```text
task 0 -> 1648126, RUNNING
task 1 -> 1648203, RUNNING
task 2 -> 1648125, PENDING
```

All three entries have `Requeue=0`, `Restarts=0`, account `sigroup`,
partition `gpu-l20`, one task, eight CPUs, and one requested GPU. The two
running entries additionally prove an allocated L20. The task-2 `PENDING`
state is historical live-control evidence and **never** counts as terminal
success. Terminal status comes only from the later three-row `sacct` capture
plus each run's `SUCCESS` receipt. The exact `ad7613b7` bootstrap code,
`#SBATCH --no-requeue`, its runtime restart check, and this live receipt form
the no-restart evidence chain; the run manifest itself does not contain
`SLURM_RESTART_COUNT`.

## Scheduler capture

Scheduler success is not inferred from the run receipts. Capture it directly
on HPC4 with the checked-in command:

```bash
PYTHONPATH=src python scripts/hpc4/capture_phase2_recovery_terminal.py \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/recovery-1648125-terminal.json
```

The capture command executes the single locked HPC4 query:

```text
sacct -X -n -P -j 1648125 --format=JobID,JobIDRaw,State,ExitCode,DerivedExitCode,Cluster,Account,Partition,NNodes,NCPUS,ReqTRES,AllocTRES
```

HPC4 does not support the proposed `sacct --array` or `Restarts` field for this
query, so neither appears in the command. Restart/requeue evidence instead
comes from the immutable live `scontrol` receipt, `#SBATCH --no-requeue`, and
the recovery bootstrap's runtime gate. The real HPC4 `-X` terminal output was
also cross-checked on a completed array: it expands to the three task
allocations with no parent, range or step rows, and has no trailing `|`.

The parser therefore accepts exactly 12 fields and exactly the three ordered
rows `1648125_0..2`. Every row must have `COMPLETED`, both exit codes `0:0`,
cluster `hpc4`, account `sigroup`, partition `gpu-l20`, one node, eight CPUs,
and these exact resources:

```text
ReqTRES=billing=8,cpu=8,gres/gpu=1,mem=96G,node=1
AllocTRES=billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1
```

Each `JobIDRaw` must be a unique positive decimal allocation ID and equal the
corresponding fixed live-receipt ID. Allocation IDs are not assumed
consecutive; task 2 legitimately uses the array master ID. Each `JobIDRaw`
must also equal the run manifest's `SLURM_JOB_ID`. The manifest must use the
real allowlisted Slurm schema and bind the same one-node/eight-CPU/one-GPU
allocation. Its Torch record must identify exactly one NVIDIA L20
(47,676,129,280 bytes, compute capability 8.9), CUDA 12.6 and
`torch==2.7.1+cu126`; the independent `gpu-check.json` must agree. Parent
rows, ranges, steps, missing/duplicate tasks, trailing fields, resource drift,
and unrecognized manifest environment fields are rejected.

The raw PSV bytes are preserved beside the canonical JSON and bound by
filename, size, and SHA256. Both files are no-overwrite, mode `0440` on POSIX,
and directory-fsynced. If JSON publication fails after raw publication, the
raw file is removed and that removal is directory-fsynced.

## Build the authorization

After all three tasks and the scheduler capture pass:

```bash
PYTHONPATH=src python -m smart_reward.phase2_recovery_aggregate \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/recovery-success-authorization.json \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4/execution-2/seed-20260801/job-1648125_0 \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4/execution-2/seed-20260802/job-1648125_1 \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4/execution-2/seed-20260803/job-1648125_2 \
  --scheduler-evidence \
  /project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/recovery-1648125-terminal.json \
  --aggregator-git-commit "$(git rev-parse HEAD)"
```

The output uses schema
`prorm-phase2-recovery-success-authorization/v1`, canonical compact JSON, and
atomic no-overwrite publication. The canonical bytes and SHA256 are computed
once before publication. A hard link publishes the new inode; the builder then
opens that published name without following symlinks and verifies by file
descriptor that its device/inode, mode, size, bytes and SHA256 equal the
temporary inode and precomputed canonical bytes. The held parent-directory
descriptor and parent device/inode are checked across link/unlink, and the
directory is fsynced. The CLI reports the precomputed digest; it never reopens
the output path to derive the authoritative digest after publication.

The CLI also requires `HEAD` to equal the declared aggregator commit, a clean
worktree, and all four validator/dependency sources to equal their Git blobs at
that commit. It repeats those checks immediately before publication. The
output path, run paths, terminal scheduler paths, live-control path, and
registry namespace are all exact; there is no production root override. Its
external file SHA256 is the value that a new calibration configuration and its
submit/job control plane must bind and reverify.

Both the new calibration submission control plane and its detached execution
must call the same consumer verifier:

```python
from smart_reward.phase2_recovery_aggregate import (
    verify_phase2_recovery_authorization,
)

authorization = verify_phase2_recovery_authorization(
    authorization_path,
    expected_authorization_sha256,
)
```

It rechecks the external SHA256, canonical bytes, exact schema and namespace,
the complete embedded live-receipt projection, terminal scheduler rows,
ordered seed evidence, all recorded hash relationships, and the absence of
any recursive `head`, `vector`, or `path` field. The builder and the
pre-publication gate read the raw live receipt from its fixed production path;
an offline consumer checks the embedded fixed SHA256, byte length, capture
time, exact three allocation/resource rows, and its non-terminal evidence
role, so the post-recovery container needs to mount only the authorization
artifact. The consumer also proves that the claimed aggregation commit
exists, that its validator Git blob equals the recorded and currently loaded
validator bytes, and that the aggregation commit is an ancestor of the
consumer checkout. It returns the fully validated JSON object and permits no
compatibility fallback.

## Trust boundary

These controls are fail-closed integrity and reproducibility checks, not a
cryptographic attestation service. A same-UID process with write authority can
potentially alter files or interfere with the filesystem namespace. POSIX mode
`0440` is an operational write barrier, not protection against the file owner
or a privileged process. Likewise, the historical producer `SUCCESS` marker
does not itself embed the SHA256 of `recovery-result.json`; success status
alone is therefore not a content signature.

The design narrows this boundary by freezing the terminal raw evidence and its
canonical hash, using no-overwrite files, mode `0440`, descriptor/inode checks,
directory fsync, exact source inventories, independent result rederivation,
frozen Git blobs, and an externally pinned authorization SHA256. These steps
substantially reduce accidental drift and ordinary substitution attacks, but
they do not make same-UID historical evidence cryptographically
tamper-proof. Stronger provenance would require an external append-only store,
signed attestations, or a separately administered identity.

The only positive authorization is:

```text
issue_schedule_frozen_full_common_beta_calibration_pilot
```

Only the optimizer schedule may cross that boundary. Recovery beta, trained
reward-model parameters, and policy state are all non-reusable. Validation,
test, held-out evaluation, policy optimization, final-oracle access, downstream
utility computation, confirmatory claims, and formal efficacy claims remain
unauthorized.
