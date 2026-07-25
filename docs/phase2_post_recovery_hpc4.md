# Phase-2 post-recovery pilot control plane on HPC4

This protocol starts a new three-seed calibration after the train-only
recovery has succeeded and implements the complete v3 lineage: calibration,
horizon escalation, frozen-beta rehearsal, sequential doubled-beta retries,
and accepted-freeze to confirmatory promotion. The recovery run contributes
exactly one thing: its frozen optimizer schedule. It does **not** contribute
trained heads, optimizer state, artifacts, beta values, responses, labels,
rewards, or policy state.

The calibration is still a formally excluded pilot. It selects a candidate
common beta and checks target-free locality and length gates; it does not
produce confirmatory efficacy evidence.

## Immutable inputs

The submission is accepted only when all of the following agree:

- the canonical recovery-success authorization and its external SHA256;
- the authorization projection embedded in
  `configs/common_beta_post_recovery_calibration.yaml`;
- the adopted five-stage AdamW schedule SHA256
  `46e0b0fdc70c507b0325c445068326ba7bc30326d70b65fa33803cc3c876c216`;
- a clean committed Git checkout containing the materialized overlay;
- the pinned Apptainer image and base-identity HF inventory;
- the fixed seed order `20260801`, `20260802`, `20260803`;
- one L20 allocation per seed on cluster `hpc4`, account `sigroup`, partition
  `gpu-l20`, with restart count zero.

The historical v2 implementation remains available for byte-compatible replay.
The shared pilot entrypoints inspect the submitted schema and route only
post-recovery v1 identities to the separate v3 scripts, namespaces, receipts,
and aggregate schema.

## 1. Materialize the authorization-bound overlay

Start from a clean checkout after
`recovery-success-authorization.json` has been produced. Compute its byte
SHA256, then run:

```bash
authorization=/project/sigroup/smart-reward-model/runs/phase2-recovery-pilot/recovery-success-authorization.json
authorization_sha256="$(sha256sum "${authorization}" | awk '{print $1}')"

PYTHONPATH=src python scripts/hpc4/materialize_phase2_post_recovery_calibration.py \
  "${authorization}" \
  --expected-sha256 "${authorization_sha256}" \
  --repo-root "$(pwd -P)"
```

This command creates exactly
`configs/common_beta_post_recovery_calibration.yaml` and refuses to overwrite
it. The new overlay preserves the full common-beta pilot design, replaces the
one-shot recovery mode with the adopted deterministic decay protocol, and
embeds an exact scientific projection of the authorization.

Review and validate the new file:

```bash
PYTHONPATH=src python scripts/hpc4/validate_phase2_recovery_authorization.py \
  "${authorization}" \
  configs/common_beta_post_recovery_calibration.yaml \
  --expected-sha256 "${authorization_sha256}"

python -m pytest -q \
  tests/test_phase2_config.py \
  tests/test_phase2_training.py \
  tests/test_phase2_aggregate.py \
  tests/test_phase2_post_recovery_control.py \
  tests/test_phase2_post_recovery_hpc4.py \
  tests/test_phase2_post_recovery_aggregate.py
```

Commit and push this exact overlay and the control-plane code. Sync that commit
to HPC4 before submission. The submit script will reject an untracked,
uncommitted, dirty, or byte-divergent overlay.

## 2. Submit the fixed three-seed calibration

Set the same canonical environment roots used by the existing HPC4 workflow:

```bash
export PRORM_PROJECT_ROOT=/project/sigroup/smart-reward-model
export PRORM_SCRATCH_ROOT="/scratch/${USER}/smart-reward-model"
export PRORM_IMAGE=/project/sigroup/smart-reward-model/images/prorm.sif
export PRORM_IMAGE_SHA256=<64-lowercase-hex>
export PRORM_HF_CACHE=/project/sigroup/smart-reward-model/hf-cache
```

Submit one fixed array:

```bash
bash scripts/hpc4/submit_phase2_post_recovery_calibration.sh \
  "${authorization}" \
  2-00:00:00
```

The calibration-specific command delegates to the generic post-recovery pilot
launcher, so both entrypoints reach the same exactly-once submitter. The
submitter first commits `intent.json`, creates the array held, captures and
fsyncs its exact `scontrol` request as `submission.json`, rechecks the
deterministic job name for collisions, and only then releases it. Rerunning
either entrypoint adopts the same registered array; it cannot create a
replacement.

Record the returned array job ID. The scheduler request is always
`--array=0-2%2`: all three seeds are fixed in one array, at most the HPC4
`l20_qos` limit of two run concurrently, and the task-to-seed mapping cannot
change. Its immutable registry is
`runs/phase2-post-recovery-<phase>/<design-sha256>/submission-registry/`.

Each task performs, in order:

1. verify authorization, Git, image, inventory, scheduler, and config identity;
2. clone the submitted commit into a detached job-local checkout;
3. mount only the job directory writable, with HF assets and authorization
   read-only;
4. materialize a fresh controlled artifact;
5. run `prepare_phase2_inputs`;
6. train all five fresh zero-initialized heads with the adopted schedule;
7. run target-free policy rollouts and safety diagnostics;
8. deep-validate the full five-head audit and publish two verification
   receipts;
9. publish the fresh artifact and immutable `SUCCESS` marker.

There is no branch that mounts or reuses a recovery output.

Every pilot receipt uses the generic
`prorm-phase2-post-recovery-pilot-{terminal,run-status,output-verification}/v1`
schemas and carries an explicit `pilot_phase`. A seed `SUCCESS` receipt records
both the real numeric Slurm allocation as `slurm_job_id` and
`allocation_job_id_raw`, and the array-task display identity separately as
`slurm_array_task_job_id`. The strict JSON receipt repeats the same identities.
Aggregation requires both raw fields to equal the matching terminal `JobIDRaw`,
requires the composite field to equal `JobID`, and rejects duplicate raw
allocation IDs across seeds.

## 3. Monitor and capture terminal scheduler evidence

Operational monitoring is read-only:

```bash
squeue -M hpc4 -j <array-job-id>
sacct -M hpc4 -X -j <array-job-id> \
  --format=JobID,JobIDRaw,State,ExitCode,DerivedExitCode,Cluster,Account,Partition,NNodes,NCPUS,ReqTRES,AllocTRES
```

After all three tasks are terminal, capture the exact unfiltered allocation
rows:

```bash
terminal=/project/sigroup/smart-reward-model/runs/phase2-post-recovery-calibration/terminal-<array-job-id>.json

PYTHONPATH=src python scripts/hpc4/capture_phase2_post_recovery_terminal.py \
  capture <array-job-id> "${terminal}" --pilot-phase calibration
```

The command writes a canonical JSON envelope and sibling
`terminal-<array-job-id>.sacct.psv`. It accepts only three ordered,
newline-terminated, exactly twelve-column rows with no trailing delimiter.
Every row must be `COMPLETED/0:0/0:0`, report one node and eight CPUs, and
match the exact locked requested and allocated TRES:
`billing=8,cpu=8,gres/gpu=1,mem=96G,node=1` and
`billing=8,cpu=8,gres/gpu:l20=1,gres/gpu=1,mem=96G,node=1`. Parent, step,
duplicate, missing, coalesced, or extra rows are rejected.

Do not aggregate a partial array, a requeued task, or a run with `FAILED`.

## 4. Submit the target-free calibration aggregate

The producer commit is the exact commit used by the three GPU tasks. The
aggregator commit may be the same commit or a clean descendant.

```bash
design_sha256=<phase2-design-sha256-returned-by-the-validator>
producer_commit=<gpu-task-producer-commit>
array_job_id=<array-job-id>

bash scripts/hpc4/submit_phase2_post_recovery_aggregate.sh \
  "${authorization}" \
  "${terminal}" \
  "${array_job_id}" \
  /project/sigroup/smart-reward-model/aggregates/phase2-post-recovery-calibration-aggregate.json \
  amd \
  02:00:00 \
  "${producer_commit}" \
  "/project/sigroup/smart-reward-model/runs/phase2-post-recovery-calibration/${design_sha256}/seed-20260801/job-${array_job_id}_0" \
  "/project/sigroup/smart-reward-model/runs/phase2-post-recovery-calibration/${design_sha256}/seed-20260802/job-${array_job_id}_1" \
  "/project/sigroup/smart-reward-model/runs/phase2-post-recovery-calibration/${design_sha256}/seed-20260803/job-${array_job_id}_2"
```

The command returns
`<job-id>;hpc4;<attempt-index>;<intent-sha256>;<attempt-ledger-sha256>`.
Keep the numeric job ID. The `--sbatch-script` argument used inside the
wrapper identifies the repository-relative script; it is not the submitted
source. The submitter reads
`<aggregator-commit>:scripts/hpc4/phase2_post_recovery_aggregate.sbatch`
with binary `git cat-file blob`, binds its byte length, SHA256, and Git blob
SHA-1 in the immutable intent, and durably stores the exact bytes as
`submission-registry/script.sbatch`. The `sbatch` command has no positional
script path: those committed bytes are its binary standard input. Therefore a
worktree edit after the Git read cannot change the submitted program.

Every CPU attempt is created with `--hold`. While it is still held, the
submitter requires the locked request fields, including `Command=(null)` and
`BatchFlag=1`, then obtains the controller's accepted script with
`scontrol write batch_script <job-id> -`. The raw response is required to
equal the committed bytes and is fsynced as
`controller/attempt-<index>.sbatch`. Its query, path, byte length, digest,
stdin transport, and submission command are bound into the attempt ledger.
After rereading that ledger, the submitter repeats both the deterministic-name
collision snapshot and the controller byte comparison immediately before
release. A mismatch fails closed while the job remains held. A later CPU
attempt is admissible only after every earlier registered attempt has one
exact terminal-failure `sacct` row plus its unmodified raw bytes; historical
attempt validation uses its durable controller copy rather than assuming that
Slurm retains old batch scripts indefinitely.

The CPU aggregation job mounts the three run directories, their exact
published artifact targets, the authorization, and the terminal JSON/raw pair
read-only. It rebuilds the pilot selection through the native post-recovery
reader and computes all relative references against the eventual production
path. It does **not** write the final aggregate namespace. Instead it durably
publishes one attempt bundle at

```text
runs/phase2-post-recovery-aggregate-attempts/
  <aggregate-filename>/
    submission-registry/
    job-<cpu-job-id>/
      aggregate.json
      evidence/
      READY
```

`READY` is written only after the aggregate, every copied config, both pilot
submission ledgers, the complete CPU attempt/failure chain, the committed
aggregate script, every per-attempt controller readback, and all raw prior
failure rows have passed validation and a recursive fsync barrier. This live
validation resolves the bound `commit:path` through Git again before accepting
the copies.

After the CPU job is absent from `squeue`, inspect its exact root row. Only
`COMPLETED|0:0|0:0` with the locked CPU resources is publishable:

```bash
aggregate=/project/sigroup/smart-reward-model/aggregates/phase2-post-recovery-calibration-aggregate.json
aggregate_job_id=<numeric-job-id-returned-by-the-submitter>

sacct -M hpc4 -X -n -P -j "${aggregate_job_id}" \
  --format=JobID,JobIDRaw,State,ExitCode,DerivedExitCode,Cluster,Account,Partition,NNodes,NCPUS,ReqTRES,AllocTRES

PYTHONPATH=src python \
  scripts/hpc4/capture_phase2_post_recovery_aggregate_terminal.py \
  capture "${aggregate}" --attempt-job-id "${aggregate_job_id}"
```

The terminalizer verifies that this is the unique current registered success,
that no same-name job is still live, and that the complete deterministic-name
`sacct` history is exactly the registered attempt chain. It copies that raw
scheduler-authority snapshot into the evidence bundle and then resumes the
following no-overwrite sequence under one lock:

```text
ATTEMPT
  -> evidence directory
  -> aggregate JSON
  -> PUBLISHED
  -> raw selected-job sacct row
  -> TERMINAL
  -> SUCCESS
```

HPC4's project filesystem does not provide
`renameat2(RENAME_NOREPLACE)`, and GNU `mv --no-clobber` cannot supply the
same race-free directory semantics when that primitive is absent. The
terminalizer therefore does not publish the evidence tree with a fallback
directory rename. It first builds and fsyncs a complete sibling staging tree,
atomically claims the final evidence name with `mkdir`, and installs
`aggregation-attempt/EVIDENCE_CLAIM.json` as the first hard-linked file.
That claim binds the outer `.ATTEMPT` receipt SHA256, CPU job ID, `READY`
SHA256, aggregate SHA256, and the exact payload-tree manifest SHA256. Every
remaining directory is created without replacement and every file is
hard-linked create-if-absent from the complete sibling tree. An existing file
is accepted only when its type, size, SHA256, and bytes are all exact.

A crash may therefore leave a visible but incomplete evidence directory. It
is not a result: before the internal claim, only an empty root or an empty
`aggregation-attempt/` prefix is adoptable; after the claim, only an exact
subset of the bound tree is adoptable. Hidden, extra, linked, special, or
wrong-byte entries fail closed. The full tree is recursively fsynced and
re-read before the aggregate is installed, and `.PUBLISHED` remains the only
publication-completeness gate. The authority record uses the outer
`.ATTEMPT.created_at_utc` as its deterministic sealing time so reconstruction
after a crash produces identical bytes; fresh `squeue` and `sacct` bytes must
still match.

Each namespace claim and file install is atomic and idempotent. If the
operator or login process stops at any point, rerun the identical `capture`
command. A valid completed publication can later be verified without the live
submission registry:

```bash
PYTHONPATH=src python \
  scripts/hpc4/capture_phase2_post_recovery_aggregate_terminal.py \
  verify "${aggregate}"
```

This protocol covers process crashes, duplicate cooperating terminalizers,
accidental occupancy, and byte-level tampering. It does not claim
cryptographic authenticity against a malicious process running as the same
Unix UID: that process can rewrite ordinary files and recompute unkeyed
receipts. Protecting against that stronger threat requires a separate
principal or signed evidence and is outside the HPC4 single-user execution
model.

The completed no-overwrite publication consists of:

- `phase2-post-recovery-calibration-aggregate.json`;
- `phase2-post-recovery-calibration-aggregate.json.evidence/`, containing the
  committed configs, pilot and CPU submission ledgers, all earlier CPU
  failure receipts/raw rows, the submitted Git-blob script, every controller
  script readback, the selected `READY`, `EVIDENCE_CLAIM.json`, and
  scheduler-authority query bytes;
- `.ATTEMPT`, `.PUBLISHED`, `.TERMINAL.sacct.psv`, and `.TERMINAL.json`
  sibling receipts;
- `phase2-post-recovery-calibration-aggregate.json.SUCCESS`.

The post-publication `verify` path needs neither Slurm nor the live submission
registry: it validates the embedded script/readback bytes, their SHA256,
Git-blob SHA-1 computation, every attempt binding, and the exact evidence-tree
manifest. The earlier live validator is the step that additionally proves
those bytes resolve from the declared Git `commit:path`.

The aggregate schema is `common-beta-pilot-selection-aggregate/v3`. Its generic
control envelope records `pilot_phase`, `pilot_terminal_evidence`, and
`pilot_array_job_id`, with portable references and SHA256 bindings for both the
recovery authorization and terminal scheduler evidence. It also binds the
overlay's raw SHA256, repository-relative path, producer commit, Git blob
SHA-1, normalized semantic config and design SHA256, plus all three validator
source digests.
A downstream freeze reader must reopen those files, verify the Git objects,
reparse the overlay, and recompute the source lineage; the JSON dictionary
alone is not sufficient evidence.

The aggregate itself must be published under the exact semantic name
`/project/sigroup/smart-reward-model/aggregates/phase2-post-recovery-calibration-aggregate.json`.
Only the committed config copies live inside
`<aggregate>.evidence/configs/`. All other relative references deliberately
resolve to immutable campaign evidence under
`/project/sigroup/smart-reward-model`. The v3 reader does not trust an arbitrary
relative path: it requires the authorization at the one recovery path, the
phase-specific terminal envelope at
`runs/phase2-post-recovery-<pilot_phase>/terminal-<array>.json`, each seed file
under the exact design/seed/job namespace, and artifact metadata under its
matching artifact namespace. A temporary mirror or another project root is
rejected even when every outer hash has been recomputed.

## 5. Materialize the next pilot identity

The v3 aggregate is the only object that authorizes a transition. Do not edit
beta, horizon, phase, or parent SHA256 fields by hand. From a clean committed
checkout, run the stage materializer with the exact source overlay and its
production predecessor:

```bash
PYTHONPATH=src python scripts/hpc4/materialize_phase2_post_recovery_stage.py \
  next-pilot \
  configs/<current-post-recovery-overlay>.yaml \
  /project/sigroup/smart-reward-model/aggregates/<current-aggregate>.json \
  --repo-root "$(pwd -P)" \
  --authorization "${authorization}" \
  --authorization-sha256 "${authorization_sha256}" \
  [--horizon-parent-aggregate \
    /project/sigroup/smart-reward-model/aggregates/<accepted-calibration>.json]
```

The allowed transition is determined exactly:

| Predecessor decision | New identity | Required parents |
|---|---|---|
| calibration or freeze failed only the length gate | next-horizon calibration | failed pilot aggregate as horizon parent |
| calibration accepted | first frozen-beta rehearsal | accepted calibration as beta source and horizon parent |
| freeze failed a non-length safety gate | next sequential retry at exactly `2 × beta` | immediately preceding failed freeze as beta source; accepted calibration as horizon parent |
| freeze accepted | no further pilot; materialize confirmatory | accepted freeze |

The retry sequence is unbounded by the control plane but strictly sequential:
`beta_base`, `2 beta_base`, `4 beta_base`, and so on. A retry cannot skip an
index, change horizon, use a non-immediate beta predecessor, or follow a length
failure. Every transition recursively reopens the v3 aggregate, its committed
config evidence, terminal scheduler evidence, three source results, sidecars,
receipts, and parent lineage before writing a new file.

The materializer uses exclusive creation and emits one of these exact paths:

- `configs/common_beta_post_recovery_calibration_horizon_<i>.yaml` and its
  horizon-specific base config;
- `configs/common_beta_post_recovery_freeze.yaml`;
- `configs/common_beta_post_recovery_freeze_retry_<i>.yaml`.

Review the generated identity, update `configs/identities.json`, run the
validation tests, then commit, push, and sync the exact commit to HPC4. The
next submission rejects a dirty checkout or config bytes that differ from the
submitted commit.

## 6. Run and aggregate any post-recovery pilot

Use the generic launcher for every generated calibration or freeze identity:

```bash
bash scripts/hpc4/submit_phase2_post_recovery_pilot.sh \
  configs/<new-post-recovery-overlay>.yaml \
  "${authorization}" \
  2-00:00:00 \
  [--beta-source-aggregate /project/sigroup/smart-reward-model/aggregates/<beta-parent>.json] \
  [--horizon-parent-aggregate /project/sigroup/smart-reward-model/aggregates/<horizon-parent>.json]
```

The launcher always submits `0-2%2` on one L20 per task with `--no-requeue`.
The compute job repeats the recursive predecessor verification inside the
container before model materialization. It then creates a new artifact, five
zero-initialized heads, and fresh AdamW state. No recovery or earlier-pilot
head, optimizer state, policy state, output, or artifact is reused.

After the three tasks terminate, capture evidence using the phase declared by
the overlay:

```bash
PYTHONPATH=src python scripts/hpc4/capture_phase2_post_recovery_terminal.py \
  capture <array-job-id> \
  /project/sigroup/smart-reward-model/runs/phase2-post-recovery-<phase>/terminal-<array-job-id>.json \
  --pilot-phase <calibration-or-freeze>
```

Then call the generic aggregate launcher. Pass the same parents used by the
GPU array:

```bash
bash scripts/hpc4/submit_phase2_post_recovery_aggregate.sh \
  "${authorization}" \
  /project/sigroup/smart-reward-model/runs/phase2-post-recovery-<phase>/terminal-<array-job-id>.json \
  <array-job-id> \
  /project/sigroup/smart-reward-model/aggregates/<semantic-output-name>.json \
  amd 02:00:00 <producer-commit> <run-0> <run-1> <run-2> \
  --overlay configs/<new-post-recovery-overlay>.yaml \
  [--beta-source-aggregate /project/sigroup/smart-reward-model/aggregates/<beta-parent>.json] \
  [--horizon-parent-aggregate /project/sigroup/smart-reward-model/aggregates/<horizon-parent>.json]
```

Save the returned CPU job ID, wait for its exact successful terminal root row,
and run `capture_phase2_post_recovery_aggregate_terminal.py capture` exactly
as in Section 4. The CPU job's `READY` is not a scientific result and does not
authorize the next stage. Only the final aggregate plus its deeply verified
`SUCCESS`, `PUBLISHED`, terminal scheduler evidence, evidence-tree manifest,
and CPU submission-authority chain may be consumed.

The only accepted publication names are:

- initial calibration:
  `phase2-post-recovery-calibration-aggregate.json`;
- horizon index `i > 0`:
  `phase2-post-recovery-calibration-horizon-<i>-aggregate.json`;
- initial freeze:
  `phase2-post-recovery-freeze-aggregate.json`;
- beta-grid index `i > 0`:
  `phase2-post-recovery-freeze-retry-<i>-aggregate.json`.

Both the pilot and aggregate launchers invoke
`verify_beta_source_aggregate` and `verify_horizon_parent_aggregate` before
submission; both compute jobs repeat those checks before model or aggregation
work. Only target-free pilot gates are eligible to authorize a transition.

## 7. Promote an accepted freeze to confirmatory

Only a freeze v3 aggregate with every target-free gate accepted may create the
formal identity:

```bash
PYTHONPATH=src python scripts/hpc4/materialize_phase2_post_recovery_stage.py \
  confirmatory \
  configs/<accepted-freeze-overlay>.yaml \
  /project/sigroup/smart-reward-model/aggregates/<accepted-freeze-aggregate>.json \
  --repo-root "$(pwd -P)" \
  --authorization "${authorization}" \
  --authorization-sha256 "${authorization_sha256}"
```

This exclusively creates:

- `configs/common_beta_post_recovery_confirmatory_base.yaml`;
- `configs/common_beta_post_recovery_confirmatory.yaml`.

The overlay binds the accepted freeze SHA256, its frozen beta, the recovery
authorization SHA256, and the adopted optimizer schedule. It declares exactly
the ordered seeds `20260901` through `20260930`. After review, identity
registration, commit, push, and HPC4 sync, submit:

```bash
bash scripts/hpc4/submit_phase2_confirmatory.sh \
  configs/common_beta_post_recovery_confirmatory.yaml \
  configs/common_beta_post_recovery_confirmatory_base.yaml \
  /project/sigroup/smart-reward-model/aggregates/<accepted-freeze-aggregate>.json \
  gpu-l20 <walltime>
```

Before its first Slurm submission, the submitter immutably commits
`campaign-plan.json` with ordered seeds `20260901..20260930`, one
`attempt-1` per seed, and fixed waves `0-3%2`, `4-7%2`, `8-11%2`,
`12-15%2`, `16-19%2`, `20-23%2`, `24-27%2`, and `28-29%2`. The plan fixes
at most four submitted tasks and two running tasks, matching the observed
HPC4 `l20_qos MaxSubmitJobsPU=4`; callers cannot override the range,
concurrency, or wave.

Before each wave `sbatch`, the submitter atomically publishes and fsyncs
`admissions/wave-<index>.json`. Later receipts hash-bind the prior admission,
prior submission, and every ordered predecessor terminal manifest and marker.
Submission v3 binds that receipt and the raw plus normalized held `scontrol`
request. Walltime and scheduler resources must exactly match the plan. Both
`squeue` and historical deterministic-name `sacct` are queried before a fresh
submission; any unregistered historical identity blocks replacement.

Rerun the identical command after each wave is fully terminal. The registry
resolver makes the next predeclared wave eligible only after every task in all
prior waves has a valid terminal bundle, regardless of success or failure.
Thus scheduling is outcome-independent: there is no retry, replacement, or
optional stopping, and a failed early wave does not suppress the remaining
predeclared attempts. Confirmatory uses `attempt-1`, `--no-requeue`, and no
prior head, artifact, optimizer state, or policy output. The accepted freeze
is consumed only as a recursively verified target-free design decision. The
compute job repeats that deep verification before any model work.

The legacy v2 confirmatory path remains readable for historical replay; a
post-recovery confirmatory overlay is always verified against the native v3
lineage.

## Acceptance conditions

Each three-seed pilot stage is complete only when:

- all three GPU tasks have strict `SUCCESS` markers;
- the terminal JSON and raw PSV verify;
- every result and receipt hash matches;
- the deep validator accepts all five head-training audits, including learning
  rate arithmetic, boundary transitions, per-update AdamW state, restored
  checkpoint state, sustained convergence, identifiability, and vector
  redaction;
- one registered CPU aggregate attempt has a durable `READY`, exact
  `COMPLETED/0:0/0:0` terminal row, and unique scheduler-authority history;
- the terminalizer has atomically published the semantic v3 aggregate,
  exact evidence-tree manifest, `PUBLISHED`, raw/parsed terminal evidence, and
  self-contained aggregate `SUCCESS` receipt;
- the aggregate's `next_action` authorizes the next transition.

The complete pilot lineage is finished only when an accepted freeze v3 has
materialized the exact 30-seed confirmatory identity. Confirmatory policy
optimization and held-out preference/utility evaluation then run under that
frozen identity; no pilot outcome evaluation may be used to change it.
