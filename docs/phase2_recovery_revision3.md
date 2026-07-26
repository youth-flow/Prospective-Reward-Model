# Phase 2 recovery revision 3：权威设计与放行边界

## 0. 文档地位

本文是 Phase 2 optimizer recovery revision 3（以下简称 **R3**）的权威设计
合同。它只规定下一轮实现、证据和放行条件，不是作业提交许可，也不声称 R3
已经实现或运行。

当前仓库仍实现 revision 2（以下简称 **R2**）的 five-head recovery 和锁定
`1648125` 的旧 authorization。该实现不得以 R3 名义提交。只有本文要求的代码、
配置、验证器、receipt schema、测试和 campaign identity 全部物化并审查通过后，
才可提交新的 HPC4 作业。

本文与既有文档的关系如下：

- [recovery protocol](phase2_recovery_protocol.md) 和
  [recovery authorization](phase2_recovery_authorization.md) 保留为 R1/R2
  的历史协议与证据解释；
- 本文仅取代它们对未来 recovery execution 的设计权限，不改写历史事实；
- [budgeted end-to-end route](phase2_budgeted_end_to_end.md) 的 fresh
  calibration、target-free freeze 和 fixed-three 顺序不变；
- 若旧文档与本文对 R3 的要求冲突，以本文为准；不得做兼容回退。

## 1. R2 冻结为失败证据

### 1.1 2026-07-26 operator live audit 观察

2026-07-26 的 HPC4 operator live audit 观察到 R2 array `1648125` 如下；Gate 0
尚未把原始 scheduler 字节与 run inventories 发布成当前 checkout 可独立验证的
immutable bundle，因此本表在 Gate 0 前是待冻结的外部观察，不是本地自足证明：

| Array task | Seed | Scheduler state | Elapsed | Run evidence |
|---:|---:|---|---:|---|
| `0` | `20260801` | `TIMEOUT` | `12:00:04` | 仅 `FAILED`；无 `SUCCESS`、无 recovery result |
| `1` | `20260802` | `TIMEOUT` | `12:00:04` | 仅 `FAILED`；无 `SUCCESS`、无 recovery result |
| `2` | `20260803` | `CANCELLED` | `08:58:42` | 仅 `FAILED`；无 `SUCCESS`、无 recovery result |

该次 live audit 时数组已无排队或运行中的任务。按这些观察，R2 不能满足旧
authorization 要求的三个 `COMPLETED 0:0`，因此：

1. `1648125` 永久不能生成 recovery-success authorization；
2. 任何单 seed 的部分计算都不能补救该三-seed campaign；
3. 不得重写、删除或用新结果覆盖 R2 的 `FAILED`、Slurm 或 run-directory
   证据；
4. 不得把 R2 的内存态、零字节训练日志或缺失文件解释成 checkpoint、结果或
   scientific failure；
5. R2 只证明当前执行架构和 12 小时资源预算不合格，不支持关于 BT-MLE 与
   ProRM+ 相对效果的结论。

### 1.2 R3 前必须补齐的 R2 冻结包

本地 checkout 目前没有可独立验证上述终态的完整 immutable terminal bundle。
在任何 R3 profiling 或训练提交前，必须在 HPC4 生产根目录一次性发布并绑定：

- `sacct -X` 的原始字节和 canonical 三行解析；
- 三个 task 的 state、elapsed、exit fields、allocation identity 和资源字段；
- 每个 run directory 的完整文件 inventory、`FAILED` 字节与 SHA256；
- `SUCCESS`、recovery result、可恢复 checkpoint 均不存在的显式审计；
- task `2` 的取消事实；
- 捕获工具 Git blob、执行 commit、容器和 parent registries；
- no-overwrite publication、文件 mode、inode/descriptor 检查和目录 `fsync`。

该冻结包只能成为 R3 的 **failure parent**。它不授权复用 head、optimizer、
step、RNG、PCG、beta 或任何效用结果。

## 2. R2 工作量风险与待验证失败诊断

R2 把下列五个 trainer 顺序放入每个 recovery seed：

1. primary BT-MLE；
2. primary ProRM+；
3. low-dimensional ProRM+ control；
4. exact-margin ProRM+ control；
5. exact-soft-label BT control。

每个 trainer 的上限均为 `12,760` updates，单 seed 的静态 worst-case 上限为
`63,800` updates。相比 Phase 1 的 `2 × 720 = 1,440` updates，这个上限约为
`44.3` 倍，而 R2 只申请 12 小时。该上限比较说明资源风险，但不证明实际执行到
哪个 trainer、哪一步，也不证明瓶颈或因果。零字节训练日志和缺少 progress receipt
尤其不能用于排除模型、数据、HF、PCG 或其他运行时故障。

当前最强的证据支持结论是：

> R2 没有产生可接受的 recovery 结果；现有串行 worst-case 工作量、12 小时墙钟和
> 无 durable progress/checkpoint 构成首要工程假设，必须由 Gate 0 与 100-update
> profiling 检验。它不是修改 scientific objective、数据或 gate 的证据。

## 3. 新 campaign identity

R3 是新 campaign，不是对 R2 run directory 的重试或 execution revision
覆盖。其语义 identity 固定为：

```text
campaign_kind = phase2_recovery_revision3_primary_only
execution_revision = 3
campaign_role = train_only_optimizer_recovery
ordered_seeds = [20260801, 20260802, 20260803]
primary_heads = [bt_mle, prorm_plus]
```

R3 必须另行物化 canonical identity object，并以其完整字节的 SHA256 作为
新的 `design_sha256`。当前文档不虚构该 SHA；只有包含本文的已提交 Git blob、
实现和配置均冻结后，机器生成的 hash 才有效。该 object 至少绑定：

- 本文 Git blob、producer/validator/submitter Git blobs 和 clean source
  commit；
- 新 primary-only config 的完整字节与 SHA256；
- R2 failure terminal bundle 及既有 parent/infrastructure registries；
- container SIF、Python lock、模型、数据、HF inventory 和 immutable
  materialization identities；
- 固定 seed 顺序和 task mapping；
- 两个 primary objective、初始化、dtype、optimizer、schedule 和 convergence
  gate；
- throughput-profile authorization 的 SHA256；
- checkpoint、progress、signal 和 continuation policy；
- Slurm account、partition、GPU type、CPU、memory、walltime、array concurrency
  和最大 scheduler segments；
- information boundary 和允许发布的文件 schema。

以下对象必须使用不同 namespace 和不同 identity hash，不得共用 run
directory 或伪装成 R3 primary result：

```text
phase2-recovery-r3-throughput-profile/v1
phase2-recovery-r3-primary/v1
phase2-recovery-r3-mechanism-controls/v1
phase2-recovery-r3-success-authorization/v1
```

R2 design
`9602b0f00a73880545fd57ce1886ec65d7901385cce2b919fd72f3efec4592d4`
不得作为任何 R3 对象的 design identity。

### 3.1 HKUST HPC4 operational boundary

用户提供的 2026-07-14 HKUST IT Service Desk account notice 记录了以下运行边界：

- Slurm account 为 `sigroup`，可用 partition 包括 `gpu-l20`；
- `/home` 为每用户 200 GB，`/scratch` 为每用户 500 GB SSD NFS，
  `/project/sigroup` 为组共享 10 TB NFS；
- `/scratch` 的 inactive files 在 60 天后清理，active checkpoint/work data
  应放在 `/scratch`，终态证据与需长期保留的产物应原子复制到 `/project/sigroup`；
- Singularity/Apptainer 受支持；
- GPU job 应首先只申请 primary GPU resource；CPU 与 memory flags 只有在 profiling
  给出测量依据时才加入。

该邮件证明账号/组资源授予，不证明当前 partition availability、QoS、最大 walltime、
GPU 空闲量或 quota 余量。Gate P 提交前仍须在登录后只读捕获 `squota -A sigroup`、
`squota`、`savail`、partition/QoS 配置和容器 runtime；不得从邮件推测墙钟上限。
校外登录可能需要 VPN 与 2FA。认证若过期，只由用户完成交互登录；自动化不得记录
密码、MFA token 或私钥。

## 4. 不可改变的科学合同

R3 只可改变 execution architecture、资源预算和持久化机制。下列内容保持
R2 已冻结定义：

- primary learners 仅为 BT-MLE 与 ProRM+；
- 两者的 objective、R=4 label semantics、`gamma=0.9`、named RNG streams、
  prompt/candidate graph、train split、frozen features、oracle transform、
  model revisions 和 templates；
- 两个 head 均 exact-zero initialization，使用 fresh AdamW state；
- 同一 deterministic learning-rate schedule：

| Inclusive update | Learning rate |
|---:|---:|
| `1..5760` | `1e-3` |
| `5761..6760` | `3e-4` |
| `6761..8760` | `1e-4` |
| `8761..10760` | `3e-5` |
| `10761..12760` | `1e-5` |

- schedule boundary 不重置 AdamW moments；
- first-order relative-gradient tolerance `1e-3`；
- minimum `100` updates、每 `20` updates 检查、连续 `3` 次通过；
- full-data、FP64、post-update、unclipped gate；
- denominator 为 exact-zero head gradient；
- 最晚在 update `12,760` fail closed；
- selected primary head 是首次完成 sustained gate 的 iterate；`720` 和
  `5,760` checkpoint 只作已冻结诊断，不能选择 head；
- recovery seeds 固定为 `20260801..20260803`，不得删除、替换或追加。

### 4.1 禁止 outcome-driven 修改

Throughput profile、R3 primary recovery 和 mechanism controls 均不得打开或
计算：

- validation/test 或 held-out learner comparison；
- held-out local regret、preference fit 或 learner ordering；
- policy rollout、finite-policy utility 或 final-oracle outcome；
- beta candidate、learner-specific beta、seed-specific beta 或 beta freeze。

尤其不得根据 held-out、rollout、utility、preference accuracy、任何 learner
ordering 或“是否支持 idea”来改变 objective、tolerance、schedule、maximum
steps、数据、labels、seeds、response horizon 或 beta。

Profiling 后唯一允许调整的是 HPC resource/segmentation 参数、checkpoint
I/O cadence 的工程实现和 signal lead time；这些调整必须只由 runtime、memory
和 I/O measurements 推导，在正式 R3 identity 生成前冻结。若必须改变任何
科学项，R3 停止并进入新的、显式理论/优化设计评审，不能继续沿用本文 identity。

## 5. 执行拆分

### 5.1 R3 primary-only recovery

每个 R3 seed 只训练：

1. primary BT-MLE；
2. primary ProRM+。

R3 primary job 不得实例化、训练或验证 exact-margin、exact-soft-label BT 或
low-dimensional trainer。它也不执行 beta calibration、policy optimization、
held-out evaluation 或 final-oracle scoring。

三个 seed 必须全部成功；不能以 `2/3`、平均值或替换 seed 放行。R3
authorization 只证明冻结 optimizer schedule 能使两个 primary objective 在
三个排除 seed 上通过原 gate。它必须是 head-free 的，不携带可供 calibration
复用的参数或 optimizer state。

### 5.2 独立 mechanism controls

下列 controls 保留，但移到独立、永久排除的 diagnostic campaign：

- exact-margin ProRM+；
- exact-soft-label BT；
- low-dimensional ProRM+。

每个 control 使用独立 job/run namespace 和显式 control identity。任何
scheduler allocation 都不得同时顺序执行 primary recovery 与 controls。
Controls：

- 只访问 train/local mechanism evidence；
- 使用同源 immutable artifacts，保留各 control 已冻结的 target/label semantics，
  共享冻结的 optimizer schedule/common first-order gate，并使用各自
  objective-specific positive-control gates；
- 不读取或复用 R3 primary head、optimizer state 或 checkpoint；
- 如需 reference head，必须在 control namespace 内按冻结规则独立重建并绑定，
  不得从 recovery authorization 携带参数；
- 不进入 primary efficacy aggregate，不产生 beta，不评价 learner outcome；
- 任一 required control failure 都阻止 fresh calibration，但不能反向改变 R3
  primary objective 或 gate。

该拆分降低单 allocation 的串行 trainer 数量，不删除 positive controls，也不
降低它们在正式实验前的 fail-closed 作用。

## 6. Gate P：先做短 train-only throughput profiling

正式 R3 primary array 前必须先完成一个独立 profile identity。初始 profile
固定：

```text
seed = 20260801
head_order = [bt_mle, prorm_plus]
completed_updates_per_head = 100
information_boundary = train_only_runtime_measurement
result_reusable_for_training = false
```

选择 `100` 是因为它等于现有 gate 的 minimum-update boundary；profile
仍不得因 gradient ratio 提前停止或扩展。两个 head 均从零和 fresh optimizer
开始，profile 结束后其 head、optimizer 和 checkpoint 永久不可供 R3 primary
恢复或选择。

这一隔离必须机械实现：profile result schema 禁止 head/optimizer/RNG state，
profile I/O benchmark 使用独立 namespace 和不可被 primary validator 接受的 binding；
primary runner/validator 必须拒绝任何 profile-role checkpoint 或 artifact。仅写在
文档中、隐藏输出字段或不把 head 放入 aggregate 都不构成隔离。

Profile receipt 必须记录并 hash-bind：

- setup、artifact verification、label reconstruction 和 trainer-enter elapsed；
- 每 update wall time；BT/ProRM+ 分开统计；
- 每个 ProRM+ update 的 PCG iterations、true residual、convergence reason 和
  elapsed；
- convergence-audit elapsed，但不把 ratio 用作设计选择；
- checkpoint serialization、`fsync` 和 verification latency；
- peak GPU memory、CPU memory、GPU identity/utilization sampling；
- source/config/container/artifact/label hashes和 scheduler raw evidence；
- exact stop reason：`predeclared_profile_update_cap`。

Profile aggregate 只能发布 runtime/resource plan，不得发布可消费的 head、
optimizer、gradient direction、beta 或 outcome。Resource plan 必须：

1. 给出从实测 setup、BT update、ProRM+ update、audit 和 checkpoint I/O 到
   `12,760` worst-case 的透明投影公式；
2. 在查看 profile 前预声明正的 walltime safety margin；
3. 证明申请 walltime 能覆盖该投影；若单 allocation 超出 HPC4 上限，则预先
   固定 state-complete continuation segment 数、每段边界和最大总 segments；
4. 固定 Slurm resources、并发度、advance-signal lead time 和 checkpoint
   cadence；
5. 若测量缺失、计时异常、OOM、PCG error 或投影不能被资源上限覆盖，则
   profile gate 失败，不提交 primary array。

如需第二次 profile，必须仅因 instrumentation/runtime 证据提出，使用新 profile
identity，并在运行前冻结范围；不得依据 gradient、held-out 或 outcome 决定。

## 7. Durable checkpoint、progress 与 signal receipts

R3 不允许训练状态只存在于进程内存。

### 7.1 Checkpoint

每个 primary head 的 full state checkpoint cadence 是 **Gate P 后冻结的
operational policy**，不是 first-order selection rule。Gate P 必须分别测量
full-data audit 与完整 checkpoint I/O，并在查看任何 primary outcome 前固定一个
`20` 的正整数倍 cadence。无论该 cadence 为何，至少还必须在以下 safe boundary
发布 checkpoint：

- 每个 learning-rate boundary；
- head selected 后、恢复验证前和 head transition 前；
- 收到预告终止信号后的下一个可验证 update boundary；
- 每个 scheduler segment 的计划终止边界。

first-order check 仍严格每 `20` 个完整 updates 执行：它是一次真实的 full-data
FP64 objective/gradient（以及 ProRM+ solver）审计，并不是“只保存日志”。审计完成后
另行发布轻量 progress receipt；该 receipt 才是日志式证据，而且不能作为恢复状态。
降低 full-state 写盘频率不得改变 audit、连续通过计数、selected iterate 或
fail-closed maximum；它只允许减少重复 I/O。该区分落实第
4.1 节已经冻结的规则：profiling 后只可依据 runtime/I/O evidence 调整 checkpoint
cadence，不可调整科学 gate。

Checkpoint 必须使用版本化 state schema、内容哈希绑定、atomic generation publication
和 no-overwrite 语义，并执行文件和父目录 `fsync`；HPC4 上任何 durability syscall
失败都必须 fail closed。PyTorch serialization bytes 可以被哈希绑定，但在没有
canonical encoder 证明时不得称为 canonical encoding。每个 checkpoint 至少包含：

- campaign/design/profile/config/source/container/artifact/label identity；
- seed、task、logical run、scheduler segment、head name 和 objective；
- 已完成 update、下一 update、schedule stage 和当前 learning rate；
- exact FP32 head tensor bytes、shape、dtype 和 SHA256；
- 完整 AdamW `state_dict`，包括 scalar step、`exp_avg`、`exp_avg_sq` 和
  parameter groups；
- convergence history、连续通过计数、zero-gradient denominator 和 selected
  status；
- Python、NumPy、Torch CPU/CUDA 及全部仍会被后续 update 消费的 named RNG states；
- ProRM+ 所有跨 update 延续的 solver/dual state 和 warm/cold-start policy；
- 上一 checkpoint metadata hash。整个 generation 由 state bytes、外部 metadata
  及 external `COMMITTED` receipt 共同绑定；state envelope 不要求自包含自己的
  文件 hash。

不得尝试恢复半个 optimizer update 或半个 PCG solve。若信号在 update 内到达，
进程只能完成当前 update 并发布新 safe checkpoint，或退回最近一个已验证
checkpoint；不能序列化不可验证的中间态。

Primary labels 在 trainer checkpoint 前一次性由专用 generator 生成。Gate 1 必须
绑定该 generator 的初末状态与 label hashes，并机械验证 trainer 阶段没有未登记的
sampler/DataLoader/augmentation RNG；若存在任何 active named generator，则必须显式
序列化并恢复，不能只依赖 global RNG。

### 7.2 State-complete safe-boundary continuation

Continuation 是同一 logical run 的预声明 scheduler segment，不是重新抽 seed
或从零重跑。恢复前必须逐字节验证 identity、artifact、labels、head、optimizer、
step、RNG、schedule、history 和 predecessor chain。恢复后的首次审计必须证明：

- head/optimizer hashes 与 checkpoint 完全一致；
- next update 没有跳过或重复；
- AdamW moments 没有重置；
- RNG stream 没有重放或前移；
- gradient measurement 与保存的 safe-boundary audit 在冻结的数值容差内一致。

safe-boundary replay 的离散字段（step、pass/fail、连续通过数、schedule stage、
solver convergence flag 等）必须完全相等；仅对重新计算得到的有限浮点算术使用
`rel_tol = 1.0e-10`、`abs_tol = 1.0e-14`。这沿用 R2 aggregate 对同一
gradient-ratio/gate arithmetic 已冻结的 replay 容差，而不是在看到 R3 结果后
新调出的阈值。head、optimizer、trainer state、identity、artifact 和 predecessor
绑定仍以 SHA-256 精确相等验证，不能用浮点容差替代。

任何 mismatch 立即 fail closed。达到 identity 中冻结的最大 segment 数后仍未
完成，整个 seed 失败；不得临时增加 segment 或从零再跑。

这里的 “state-complete” 只表示在完整 update 边界保存继续所需的显式状态，并不
自动声称跨 CUDA driver、device、kernel 或非确定性算法的 bitwise uninterrupted
trajectory 等价。若要使用 “exact continuation” 一词，Gate 1 还必须在冻结 HPC4
环境中通过 interrupted-versus-uninterrupted 下一步及完整轨迹等价故障注入，并绑定
device UUID/order、driver、Torch/CUDA、deterministic-algorithm、TF32/cuBLAS、
thread/worker 和 AMP 状态；否则只允许上述较弱且可验证的 state-continuity claim。

### 7.3 Progress receipts

每个 `20`-update first-order check 以及每个额外 checkpoint boundary 之后都发布
不可覆盖的 `progress/head-<name>-step-<step>.json`，并形成 hash chain。若同一
boundary 同时产生 checkpoint，则 checkpoint generation 是第一提交点；若进程在
checkpoint commit 与 progress event 之间崩溃，恢复器必须验证 committed generation
并追加 `checkpoint_discovered_after_crash`，不得假装两者原子提交。只有 progress
而没有 full-state checkpoint 的普通 check 不能被恢复器当作 continuation state。
Progress contract 至少记录：

- monotonic 和 wall-clock timestamps、累计有效训练/审计/I/O elapsed；
- completed/next step、learning rate、gradient ratio 和 consecutive passes；
- ProRM+ PCG iterations、true relative residual、reason；
- peak/current memory、last checkpoint hash、previous progress hash；
- signal state、scheduler segment 和 remaining allocation time；
- information boundary 为 `train_only`。

标准输出不能作为唯一进度证据；buffered log 为零字节也不能替代 progress
receipt。

### 7.4 Signal 和 terminal receipts

SBATCH 必须请求预告信号。其 lead time 必须由 profile 中的最大完整-update
时间、checkpoint flush/verify 时间和预声明安全余量推导并写入 identity。
handler 对 `USR1`、`TERM`、`INT` 等实际收到的信号发布 no-overwrite receipt，
记录：

- signal 名称、接收时间和 scheduler identity；
- 当时 head、in-flight update、最后完整 update；
- 最后 durable checkpoint/progress hash；
- 是否成功到达 safe boundary、flush 和 verify；
- planned exit/continuation reason。

Signal receipt 或 checkpoint 都不是 `SUCCESS`。`TIMEOUT`、`CANCELLED`、
nonzero exit 也不能被 result 文件覆盖。只有同一 identity 允许的 state-complete
continuation 可以消费最后一个完整 checkpoint，且必须保留前一 scheduler
segment 的真实终态。

## 8. 分阶段 gates

所有 gate 均 fail closed，必须按顺序满足：

### Gate 0 — R2 evidence frozen

- R2 三行 scheduler terminal bundle 和三个 run failure inventories 完整；
- 明确无 `SUCCESS`、result 和 durable checkpoint；
- R2 evidence hash 已进入 R3 parent identity。

失败：不得 profile。

### Gate 1 — R3 implementation materialized

- primary-only runner 不会实例化三个 controls；
- profile、primary、controls 和 authorization 使用独立 schemas/namespaces；
- checkpoint/progress/signal/state-complete continuation 全部实现并有故障注入测试；
- 新 config validator 锁住第 4 节全部科学常量；
- submitter、SBATCH、terminal capture 和 authorization validator 均 Git-bound；
- clean commit、容器、inventory 和 shell/Python tests 通过。

失败：不得提交任何 R3 HPC4 作业。

### Gate P — throughput/resource authorization

- 固定 100-update-per-head profile 成功；
- receipts 完整且未越过 train-only boundary；
- resource projection、walltime、signal lead、checkpoint cadence 和最大 segments
  已冻结；
- profile heads/state 不可消费。

失败：不得提交 primary array。

### Gate R — primary-only recovery

每个固定 seed 的 BT-MLE 和 ProRM+ 均：

- 从 zero/fresh state 开始或从同一 logical run 的 exact checkpoint 续接；
- 完成未变的 sustained first-order gate；
- checkpoint/progress/signal/terminal chain 完整；
- final head 与 optimizer restore audit 通过；
- 无 held-out、policy、final-oracle 或 beta access。

三个 seed 和全部 scheduler segments 均通过后，才可发布 head-free
`phase2-recovery-r3-success-authorization/v1`。任一 seed 失败则 Gate R
关闭，不能以剩余 seed 聚合。

### Gate C — separated mechanism controls

三个 control families 在独立 campaign 中完成其冻结 numerical/identity/
positive-control gates；所有 seed、jobs、receipts 和 terminal evidence 完整。
Controls 的失败不得用 primary outcome 解释或调参。

失败：不得 fresh calibration。

### Gate F — fresh post-recovery calibration

只有 Gate R 和 Gate C 的 authorization 都通过，新 calibration identity 才可
物化。Fresh calibration：

- 重新物化/验证其规定的数据和标签；
- 重新从零训练，不复用 recovery/control heads 或 optimizer state；
- 使用相同固定排除 seeds `20260801..20260803`；
- 只访问 train/target-free convergence、curvature、KL、length、rank 和
  numerical diagnostics；
- 不打开 held-out evaluator、policy outcome 或 learner ordering。

其严格三-seed aggregate 只可授权 target-free beta/horizon freeze 的下一
identity。

### Gate B — accepted target-free freeze

只有 fresh calibration aggregate 通过，才能按既有顺序执行 frozen-global-beta
rehearsal。Beta 只可沿预注册 target-free grid 选择；response horizon 只可沿
预注册 length-gate 路径变化。任何 held-out/outcome 都不得参与。

只有全 seed、全 arm、全 numerical/KL/length/provenance gates 通过的 accepted
freeze aggregate 才能锁定：

- 唯一 `frozen_global_beta`；
- 唯一 accepted response horizon；
- optimizer schedule identity；
- accepted-freeze aggregate bytes 和 SHA256。

### Gate E — budgeted fixed-three

只有 Gate B 完成且 fixed-three config/receipt/campaign identity 已物化，才允许
固定 fresh seeds `20261001..20261003` 的 budgeted end-to-end array。该阶段是第一次
允许 policy rollout、held-out geometry、preference fit 和 finite-policy
utility。

Fixed-three：

- 每 seed 的主 reward heads 仍仅为 BT-MLE 与 ProRM+；
- 使用同一个 accepted global beta 和 horizon；
- 不得按当前 seed curvature、learner 或 outcome 改 beta；
- 任一 seed inadmissible 时 withholding 全部 effect summary；
- 只产生 exploratory/descriptive 结果，不是 exact-30 正式推断，也不能用其
  outcome 决定补 seed、换 beta 或修改设计。

## 9. 失败后的唯一允许动作

| 失败位置 | 允许动作 | 禁止动作 |
|---|---|---|
| Gate 0/1 | 补证据或修实现，生成新 commit/identity | 改写 R2、直接提交 |
| Gate P | 仅修 instrumentation/resources/segmentation，再建 profile identity | 看 gradient/outcome 调 objective 或 gate |
| Gate R | 冻结失败并做显式 optimizer/engineering redesign review | 删除 seed、放宽 `1e-3`、增加 adaptive tail、偷看 held-out |
| Gate C | 冻结对应 mechanism failure，做独立理论/实现诊断 | 把 control 删除后继续 calibration |
| Gate F/B | 仅按既有 horizon/beta target-free transition 规则处理 | 用 learner ordering、utility 或 held-out 选 beta/horizon |
| Gate E | 发布完整 exploratory 成功或失败边界 | 补 seed、重跑到有利结果、反向修改 freeze |

## 10. 当前放行状态

截至 2026-07-27 的代码与 HPC4 证据审计：

- R2 已知终态为两次 `TIMEOUT` 和一次 `CANCELLED`，旧 success
  authorization 永久不可签发；
- R2 immutable terminal bundle 已由 HPC4 Gate-0 failure-parent receipt 绑定；
- R3 science config、verified train-materialization attestation、Gate-P admission/
  terminal/resource authorization、primary logical-run/segment identity、正式
  primary orchestrator、controls identity、SBATCH、terminal finalizer，以及
  Gate-R + Gate-C combined authorization bridge 已在代码中实现；
- immutable-generation checkpoint、predecessor audit、deferred RNG restore、
  strict progress/signal schemas、safe-boundary controller 与 CPU fault-injection
  tests 已接入对应 R3 控制面；
- 100-update BT→ProRM+ profile 保持固定工作量、原始 PCG stop reason、CUDA
  同步计时、memory 和匿名 checkpoint I/O probe，并且不会把训练状态带入 Gate R；
- 第一次正式 Gate-1 job 未生成 source-test receipt 或 implementation closure；
  后续在同一已提交快照上的 CPU/GPU 精确全量诊断均通过，但它们明确是
  non-authorizing evidence，不能替代正式 Gate 1；
- 当前修订仍须形成新的 clean、pushed commit，再重新通过正式 Gate 1；
  在此之前 Gate P、Gate R、Gate C 和 Gate F 均未获授权。

因此当前状态是：

```text
R3_HPC_SUBMISSION_AUTHORIZED = false
FRESH_CALIBRATION_AUTHORIZED = false
FIXED_THREE_AUTHORIZED = false
```

下一步只能先对新的 clean commit 运行并验证正式 Gate 1；通过后严格进入
Gate P，不得跳过 profiling，也不得提前启动 Gate R、Gate C、fresh
calibration 或 fixed-three。
