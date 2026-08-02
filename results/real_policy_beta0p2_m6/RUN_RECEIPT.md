# Formal m = 6 Run Receipt

- Scratch run: /scratch/yyangjo/prorm/runs/real-policy-beta0p2-m6-2d1e5e1
- Immutable archive: /project/sigroup/yyangjo/prorm/archives/real-policy-beta0p2-m6-2d1e5e1-20260802
- Immutable m = 4 base: /project/sigroup/yyangjo/prorm/archives/real-policy-beta0p2-ada8d6b-20260802/run
- Source upstream archive: /project/sigroup/yyangjo/prorm/archives/fisher-trpo-main-014cf45-20260731/run
- Extension code commit: 2d1e5e1352a1d77bb43dbf9a8feeda45d294575e
- Runtime image: /project/sigroup/yyangjo/prorm/images/prorm-hpc4-861b2c1.sif
- Runtime image SHA-256: ccabad42c29208253bb84ec8c8dfc228f64189206c4867742ca7c04b4915a7dd
- Runtime image source commit: 861b2c11698ae2aaf65913548efea961878529a9
- HF inventory SHA-256: 701c43fe0fa376b172f95fa9004b15a39d3faa3a526dc2d7dc9b317cf88f514b
- Source config SHA-256: 0ff8cb872c5bdecc33bd2e5ded7d9c3adcbc43d7b6c355b40f8d34a1ae95ce92
- Extension config SHA-256: 64e0e2ea13c2a8dfb8d399f9feed4fdf5bfebd0a42af84f777c156c515794bc1
- Aggregate SHA-256: 7ae7654bb807535c2d71ecd6f03aad8ed16ae48d9adbc53fda0c76dfb9a371de
- Integrity audit SHA-256: 78b057b77aeff135a59daa8ad39864b52b575490f77fc96cdeb0dad6b0ca3c8a
- Archive manifest SHA-256: 1d7bcf2c480eaa3d99bc44c6878407b2707c2d8452a4e1ff661bcf9d7e589119
- Source validation job: 1692601
- Incremental rollout jobs: 1692617_0, 1692618_0
- Seed aggregate jobs: 1693073[0-2]
- Three-seed aggregate job: 1693076
- Integrity audit job: 1693077

All formal jobs completed with exit code 0. Before aggregation, an independent
gate compared every response-index 0--3 row to the immutable m = 4 archive,
verified that only indices 4 and 5 were new, confirmed 3,072 rows for each of 12
policy-seed instances, and found no residual work directories.

At archive creation, all 48 source files and 48 archived files had identical
relative paths, SHA-256 digests, and total byte count (48,302,937 bytes).
