# Formal Run Receipt

## Identity

- Formal code commit: `2019f1e9b1d29be7bcbd2331fafa61d153201344`
- Formal extension config SHA-256:
  `78a429f80caa180461d08f697d3429f9fcc7e04b6ab0ea95173305fa5aea1f9c`
- Frozen source config SHA-256:
  `0ff8cb872c5bdecc33bd2e5ded7d9c3adcbc43d7b6c355b40f8d34a1ae95ce92`
- Runtime image SHA-256:
  `ccabad42c29208253bb84ec8c8dfc228f64189206c4867742ca7c04b4915a7dd`
- HF inventory SHA-256:
  `701c43fe0fa376b172f95fa9004b15a39d3faa3a526dc2d7dc9b317cf88f514b`
- Immutable source run:
  `/project/sigroup/yyangjo/prorm/archives/fisher-trpo-main-014cf45-20260731/run`
- Immutable source m=6 run:
  `/project/sigroup/yyangjo/prorm/archives/real-policy-beta0p2-m6-2d1e5e1-20260802/run`

## Formal outputs

- Scratch run:
  `/scratch/yyangjo/prorm/runs/h-mse-ablation-beta0p2-2019f1e-20260804`
- Immutable archive:
  `/project/sigroup/yyangjo/prorm/archives/h-mse-ablation-beta0p2-2019f1e-20260804`
- Integrity audit SHA-256:
  `06119cff5d2bf5627847115a4ff5bd8abbe2855c7c7eb874f25364d3cdfdb3c0`
- Policy aggregate SHA-256:
  `adbf2f7363fb684a6acfd639a927730afca18ed857b80320487c1ac11bd54536`
- Reward aggregate SHA-256:
  `81179860e6affaa99fdab88e160d61580913833c257d916a19120cda583c8100`
- Provenance bridge SHA-256:
  `31c2f3dc0bf0d6c44a52ab00c6a3b35de69bef5db96929334c14d5953efbdfea`
- Archive manifest SHA-256:
  `db7ee09eacf889e3ea18ab5235e3765f0dc8db970c4ece97d8389877d760ba72`
- Smoke result SHA-256:
  `35b8cdc4c5c2d422b8ec797826a9e52900bc9e962d9ba07ad383ceccc0219932`

The integrity audit status is `passed`; it covers exactly three seeds, four new
reward models, six learned reward models in evaluation, four new policies, eight
policies in the combined evaluation, 36,864 new rollout rows, and 73,728 combined
rollout rows.

## Local retrieval verification

The compact local evidence contains 28 files mapped to the immutable archive.
Every local file was SHA-256 checked against `evidence/archive-sha256sums.txt` with
zero mismatches. The downloaded manifest hash equals the hash recorded in
`evidence/archive-receipt.json`. The remaining local files are the separately
audited smoke result and the archive receipt/manifest themselves. Large rollout
rows, adapters, and annotation tensors remain recoverable from the immutable HPC4
archive and were intentionally not duplicated locally.

`report111.tex` was not changed by this extension; its retained SHA-256 is
`64D97850249E9F902EECFBF32CDC456512C74CA4A238F468C41846CCF31A22B0`.

The formal finalize job completed seed aggregation and the integrity audit, then
encountered host execute permission on the archive helper. The identical helper
was invoked explicitly through `bash`; it verified every archived file and created
the immutable archive above. The pipeline now invokes that helper through `bash`
so future finalization does not repeat the operational failure.
