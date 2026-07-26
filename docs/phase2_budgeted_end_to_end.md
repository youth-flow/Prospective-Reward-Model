# Phase 2 预算版端到端实验：权威路线与工程契约

本文是当前实际执行路线的权威说明。它描述的是
`budgeted_end_to_end` 三-seed 探索性实验，不是未来的 exact-30 正式实验。
实时 Slurm 队列以 HPC4 上的 `squeue`、`sacct` 和不可变运行证据为准，不写入静态文档。

## 1. 一句话结论

当前实验要回答的确定问题是：

> 在同一批新生成候选、同一组重复标签、同一冻结 reward class、同一全局
> `beta` 和同一单步 LoRA-B policy optimizer 下，ProRM+ 相比重复标签
> BT-MLE 是否产生更小的局部下游 regret、更高的有限步 policy utility，
> 以及怎样的 operational-oracle preference fit。

当前路线固定为：

```text
one-shot recovery：3 seeds，工程修复证据，永久排除
  -> fresh post-recovery calibration：3 seeds，只看 train/target-free 诊断
  -> target-free freeze：接受一个全局 beta 与 response horizon
  -> budgeted_end_to_end：固定 3 个 fresh seeds，完整 RM -> policy -> held-out evaluation
  -> fixed-three descriptive aggregate：只做描述统计，不做正式推断
```

三个端到端 seed 固定且有序：

```text
20261001, 20261002, 20261003
```

这三个 seed 的结果永远满足：

- `formal_eligibility=false`；
- `formal_claim_eligible=false`；
- `evidence_role=budgeted_end_to_end_fixed_three_exploratory_only`；
- 不进入 confirmatory evidence；
- 不产生 p-value、显著性标签或正式论文 claim。

exact-30 仅保留为未来协议。只有重新冻结并预注册正式 identity、代码、数据、镜像、
全局 `beta`、30 个 seed 和 no-retry ledger 后，才能启动该协议；本轮三-seed
结果不能被事后升级为正式实验。

## 2. 为什么必须按这条链执行

### 2.1 Recovery 的 3 个 seed 只修工程，不估计效果

第一次 post-Phase-1 calibration 在固定学习率下未通过 BT-MLE 的一阶收敛门。
one-shot recovery 用三个排除 seed `20260801..20260803` 验证授权后的确定性
AdamW decay schedule。
它可以证明“训练器能在不访问 validation/test 的条件下稳定收敛”，但不能：

- 选择或训练可复用的正式 reward head；
- 贡献 preference、regret 或 utility 结果；
- 进入 calibration/freeze aggregate；
- 直接授权三-seed efficacy 结论。

recovery 成功后只产生 head-free authorization。其 head、数据物化、标签、
checkpoint 和 optimizer state 均不得复用；后续阶段重新物化数据、重新生成标签、
从零初始化 reward heads，并创建新的 optimizer state。

### 2.2 Fresh calibration 的 3 个 seed 只选择全局尺度

post-recovery calibration 使用三个新鲜、永久排除的 seed。它只允许读取：

- train-only oracle natural-direction curvature；
- optimization convergence；
- updated-policy KL mean/tails；
- EOS、长度上限和 numerical safety；
- provenance、rank 和 positive-control 诊断。

它不得打开最终 held-out evaluator，不得输出 learner ordering、regret、finite-policy
utility 或 preference-fit 结果。其 aggregate 只决定后续 target-free freeze identity。

### 2.3 Accepted freeze 同时锁住 `beta` 与 horizon

freeze 阶段把同一个候选 `beta` 施加给所有 seed 和所有 policy arm，只依赖
target-free KL/length/numerical gates。只有完整通过的 freeze aggregate 才能成为
`budgeted_end_to_end` 的共同父证据，并同时绑定：

- 唯一全局 `frozen_global_beta`；
- 唯一 response horizon；
- accepted-freeze aggregate 的原始字节 SHA256；
- recovery-authorized optimizer schedule；
- Phase 2 design/runtime identity。

端到端 seed 无权重估 `beta`。当前 seed 的 curvature 只能作为诊断，不能改变步长。

### 2.4 三-seed E2E 是第一次允许读取最终效果的阶段

每个端到端 seed 都执行完整闭环：

```text
fresh prompts/candidates/labels/artifact
  -> fresh BT-MLE and ProRM+ heads
  -> frozen-beta natural-gradient directions
  -> four policy arms and updated-policy rollouts
  -> final Skywork scoring
  -> deferred held-out local geometry and preference fit
  -> strict seed verification
```

任何 seed 未通过 optimization、PCG、KL、length、identity、prompt semantics 或
positive-control gate，整个 seed 都是 inadmissible。不能删掉失败 seed 后只汇总剩余结果。

## 3. 数据、模型、标签与模板隔离

### 3.1 Prompt 来源

实验只从固定 revision 的 `allenai/multipref` 提取原始 user prompts，不使用
MultiPref 附带的历史 chosen/rejected 作为本实验训练标签。

Phase 2 先用 Qwen2.5 的 policy chat template 在 `truncation=False` 下检查全部 prompt，
构造 `<=1024` policy-token 的冻结 eligible pool，再进行 seeded shuffle 和
`1536/256/256` 的 train/validation/test prompt-level split。旧审计中：

- unique prompts：`5,323`；
- 超过冻结 policy prompt cap：`88`；
- eligible prompts：`5,235`。

排除发生在随机 split 之前；不存在 silent truncation。

### 3.2 Reference/SFT policy

reference policy 是固定 revision 的
`Qwen/Qwen2.5-0.5B-Instruct`。这里的 “SFT” 指现成的 Instruct checkpoint；
本项目不再训练一个额外 SFT 模型。

每个 prompt 从该冻结 reference policy 采样四个候选回答。候选生成后不按 oracle
分数筛选、不去重、不替换。policy 与 reward-feature backbone 都使用 Qwen2.5
自己的 tokenizer/chat template。

### 3.3 Reward class

BT-MLE 与 ProRM+ 使用完全相同的 reward class：

- 冻结 Qwen2.5 backbone；
- final response token 的冻结 hidden feature；
- 无 bias 的线性 reward head；
- 相同的零初始化、数据、优化协议和一阶收敛门。

比较对象是 reward-model objective，不是 backbone capacity、候选数据或训练预算。

### 3.4 Operational oracle 与标签生成

固定 revision 的 `Skywork/Skywork-Reward-V2-Qwen3-0.6B` 是
**operational oracle**，不是人类效用。它对 Qwen 新生成的每个回答打分，再使用冻结的
全局 robust transform 得到 `r*`。

对同一 prompt 的 canonical candidate pair `0-1`：

$$
p^*(e)=\sigma\!\left(\Delta r^*(e)\right).
$$

随后由命名随机流生成条件 iid Bernoulli 标签。每个 Phase 2 primary edge 有四个独立的
`gamma=0.9` randomized-truncation replicate：

- BT-MLE 使用四条流中的全部 raw wins/totals；
- ProRM+ 分别构造四个无偏 `h`，再使用它们的算术平均；
- raw labels 不被裁剪、挑选或按结果重试。

因此“标注者”是可复现的 Skywork-defined BTL simulator，不是人工重新标注。
如需人类外部有效性评估，必须对这些**确切的新回答**重新取得人类标签，不能把
MultiPref 中属于其他回答的标签移植过来。

### 3.5 Qwen2.5 与 Qwen3 模板绝不混用

两套模型看到相同的语义内容，但输入分别重渲染：

- Qwen2.5：用自身 tokenizer/chat template 渲染完整 raw prompt 并生成回答；
- Skywork Qwen3：用自身 tokenizer/chat template 重新渲染
  `raw prompt + assistant response`；
- Qwen2.5 token IDs 不作为 Qwen3 输入；
- 每条 rollout 保存 raw-text hash、policy prompt-token count、prefix hash、
  token cap 和 `truncated=false` 证据。

这消除了 Qwen2.5 policy 与 Qwen3 reward model 的模板家族错配。

## 4. BT-MLE 与 ProRM+ 实际训练什么

### 4.1 BT-MLE

对 canonical edge 的 reward margin
`Delta r_phi = r_phi(x,y_0)-r_phi(x,y_1)`，BT-MLE 最小化所有重复 Bernoulli
标签的 logistic negative log-likelihood。四个 replicate 的 wins/totals 全部进入
同一目标，不先压缩成一个硬标签。

BT-MLE 回答的是：

> 当前 reward class 能否最好地拟合已经观察到的 pairwise labels？

### 4.2 ProRM+

令 LoRA-B policy tangent score difference 为 `z_0`，重复标签构造的无偏 logit
估计为 `h`。ProRM+ 的经验 moment 是

$$
\widehat m_\phi
=
\frac{1}{2n_E}
\sum_e z_0(e)\left(\Delta r_\phi(e)-h(e)\right).
$$

它训练下列 ridge Fisher-GMM saddle objective：

$$
\min_\phi\max_v
\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right].
$$

对固定 reward head，dual maximizer 满足

$$
(\widehat F_0+\lambda I)v=\widehat m_\phi.
$$

因此 ProRM+ 只惩罚会改变下游 policy natural-gradient update 的 reward error。
它不是让所有 pointwise rewards 都逼近 `r*`。

### 4.3 “minibatch saddle + PCG” 的准确工程含义

当前实现不是抽一个随机 minibatch、对 `phi` 和 `v` 同时做有噪 SGD。实际顺序是：

1. 在全部 train edges 上计算当前 reward margins；
2. 由完整 empirical moment 和 Fisher operator 构造 dual linear system；
3. 在 FP64 policy-geometry workspace 中用 PCG 求解 `v`；
4. 显式重算 true residual；未达到冻结 tolerance 就 fail closed；
5. detach 已求解的 `v`，使用 envelope theorem 得到 reward-head gradient；
6. 按固定 canonical edge 顺序分 microbatch 累积**完整 full-batch gradient**；
7. 对 FP32 reward head 做一次 AdamW update；
8. 下一步重新计算 moment 并重新求 dual；上一步 `v` 最多作为 PCG warm start，
   不能作为 stale dual 直接复用。

所以 `microbatch_size` 是显存分块，不改变 full-batch estimand，也不引入随机
minibatch sampling。BT-MLE 同样按 microbatch 累积完整目标梯度。

primary head 不是按相同步数强行停止，而是必须通过冻结的一阶门：

- post-update；
- full-data；
- unclipped；
- 相对零初始化 full-gradient norm；
- 连续多次检查通过；
- 不访问 validation/test 选择 checkpoint。

## 5. Reward model 如何进入 policy optimization

### 5.1 冻结 policy tangent

policy tangent 只包含 Qwen2.5 最后四层 `q_proj/v_proj` 的 rank-4 LoRA-B。
LoRA-A 是跨 seed 冻结且 hash-bound 的全局基底，reference policy 对应 `B=0`。

对任一 reward signal `r`，先在 train split 计算

$$
g_r=A_0r,\qquad
u_r=(\widehat F_0+\lambda I)^{-1}g_r.
$$

然后使用 accepted freeze 给出的同一个全局 `beta_0`：

$$
\Delta\theta_r=\frac{u_r}{\beta_0}.
$$

没有 learner-specific line search、fixed-K renormalization 或 seed-specific beta。

### 5.2 四个 policy arms

每个 seed 固定四个 arm，顺序不可改变：

| Arm | LoRA-B displacement | 作用 |
|---|---|---|
| `zero_b` | `0` | reference/control |
| `bt_mle` | `u_BT / beta_0` | BT reward induced update |
| `prorm_plus` | `u_ProRM+ / beta_0` | ProRM+ reward induced update |
| `oracle_step` | `u_* / beta_0` | train-oracle local reference，不是全局最优 policy |

这是一次直接 natural-gradient LoRA-B update，不是 PPO、DPO，也不是多轮 RLHF。

### 5.3 Rollout、KL 与 oracle 的时间顺序

四个 arm 对每个 test prompt 使用完全相同的命名随机流：

```text
SeedBundle(seed).rollout
  -> derive_seed(..., "phase2-test-prompt:<prompt_id>")
```

每个 arm 都在自身 updated-policy histories 上计算

$$
D_{\mathrm{KL}}(\pi_{\mathrm{updated}}\Vert\pi_0).
$$

在 heads、`beta_0`、directions 和 pre-oracle safety 全部冻结并通过后，才允许：

1. Skywork 对最终 policy rollouts 打分；
2. 打开 deferred validation/test oracle targets；
3. 计算 held-out PCG geometry、regret、utility 和 preference fit。

这保证最终 oracle 信息不反馈进 reward-head 训练、beta 选择或 policy direction。

## 6. 五个固定端点及方向

所有 effect 都定义为：

$$
\Delta=\text{ProRM+}-\text{BT-MLE}.
$$

| Endpoint | 定义 | 越优方向 | 有利的 `Delta` |
|---|---|---|---|
| `heldout_local_regret` | test split、固定 `beta_0` 下的局部下游 regret | 越低越好 | `< 0` |
| `finite_policy_utility` | `mean(r*) - beta_0 * mean KL(pi_updated || pi_0)` | 越高越好 | `> 0` |
| `oracle_pairwise_cross_entropy` | 对 test prompt 内全部 unordered candidate pairs 的 operational-oracle CE，先 prompt 内平均再跨 prompt 平均 | 越低越好 | `< 0` |
| `oracle_probability_mae` | 预测 pairwise probability 与 oracle probability 的 MAE，同样先 prompt 内平均 | 越低越好 | `< 0` |
| `pairwise_order_accuracy` | oracle pairwise ordering accuracy；tie 计 `0.5` | 越高越好 | `> 0` |

前两个直接回答 prospective/downstream 问题；后三个描述传统 preference fit。
后三个改善不能代替 local regret 或 finite-policy utility。

`finite_policy_utility` 的 operational oracle 仍是 Skywork 变换后的 `r*`，因此结论只能是：

> 在冻结的 MultiPref eligible prompt pool、Qwen2.5 policy、Skywork operational
> oracle、LoRA-B tangent 和单步 optimizer 条件下的受控机制结果。

它不是开放域人类 utility 结论。

## 7. Seed 输出为何必须先验证

每个 seed 的 `phase2-result.json` 和 rollout JSONL 在进入聚合前必须通过
`verify_phase2_budgeted_end_to_end_seed_output.py`。验证器会重新检查：

- result schema、stage、非正式 claim markers、seed/design/base/freeze/beta；
- accepted freeze 与 runtime contract 的 canonical hash；
- rollout 原始字节 hash、exact row count、arm/prompt/candidate 顺序；
- 每个 prompt 的精确命名 rollout seed，防止跨 seed splice；
- Qwen prompt semantics、token prefix 和 cross-arm common random numbers；
- run manifest 的 Git/image/HF inventory/GPU/Slurm identity；
- artifact metadata 与 fresh materialization receipt；
- 通过生产 `prepare_phase2_inputs` 重新打开 tensors、prompts、candidates 和语义证据；
- numerical event sequence 与 information boundary；
- `normalize_budgeted_end_to_end_seed_result(...).admissible == true`；
- 正好五个 endpoint，且每个 endpoint 正好含 `bt_mle` 与 `prorm_plus`。

verification JSON 使用
`prorm-phase2-budgeted-seed-output-verification/v1`，canonical serialization、
`O_EXCL` no-overwrite，并绑定所有输入 SHA256。它只能声明 `status=verified`，
不能声明科学结果 “passed”。

## 8. Fixed-three 聚合与 claim 边界

聚合单位是 seed，不是 prompt、candidate 或 pair。只有三个固定 seed 全部存在、顺序正确、
共享同一 design/runtime/freeze/beta identity 且全部 admissible 时，才输出 effect summaries。
任一 seed 缺失或 inadmissible，聚合器输出
`effect_summaries_withheld`，不会对剩余 seed 计算效果。

完整描述性汇总包含：

- 每个 seed 的 BT、ProRM+ 和 `ProRM+ - BT`；
- `n=5`；
- mean、sample SD、min、median、max；
- seed-level paired percentile-bootstrap **descriptive interval**。

严格禁止：

- p-value；
- `significant` / “统计显著”；
- 正式 hypothesis-test verdict；
- `passed` / `not_passed` efficacy gate；
- population confidence interval 表述；
- 用三个 seed 决定是否补 seed、换 beta、重跑或启动 exact-30。

允许的论文语言是“在该冻结系统的三个 exploratory seed 中观察到的方向、大小和异质性”。
即使三个点方向一致，也不能称为正式证据或显著结果。

## 9. HPC4 已落地的执行入口

以下命令只说明仓库中已经存在的入口。所有路径与 SHA256 必须替换为真实、
canonical、non-symlink、hash-bound 对象。

### 9.1 从 accepted freeze 物化 budgeted identity

```bash
PYTHONPATH=src python scripts/hpc4/materialize_phase2_budgeted_end_to_end.py \
  SOURCE_FREEZE_OVERLAY \
  ACCEPTED_FREEZE_AGGREGATE \
  --repo-root REPO_ROOT \
  --authorization RECOVERY_SUCCESS_AUTHORIZATION \
  --authorization-sha256 RECOVERY_SUCCESS_AUTHORIZATION_SHA256
```

该入口只创建候选文件：

- `configs/common_beta_post_recovery_budgeted_end_to_end_base.yaml`；
- `configs/common_beta_post_recovery_budgeted_end_to_end.yaml`；
- `configs/.common_beta_post_recovery_budgeted_end_to_end.materialized.json`。

它们必须 review、commit、push 并同步到 HPC4；未提交或 dirty worktree 不能提交实验。

### 9.2 exactly-once 提交 fixed-three array

在已配置 `PRORM_PROJECT_ROOT`、`PRORM_SCRATCH_ROOT`、`PRORM_IMAGE`、
`PRORM_IMAGE_SHA256` 和 `PRORM_HF_CACHE` 的干净提交上：

```bash
bash scripts/hpc4/submit_phase2_budgeted_end_to_end.sh \
  configs/common_beta_post_recovery_budgeted_end_to_end.yaml \
  RECOVERY_SUCCESS_AUTHORIZATION \
  ACCEPTED_FREEZE_AGGREGATE \
  WALLTIME
```

wrapper 会验证 materialization receipt、accepted freeze、authorization、Git、image、
inventory 和 committed script bytes，再调用
`submit_phase2_budgeted_end_to_end_once.py`。固定 array 是 `0-2%2`：
三个 task，最多两个并发；不是 adaptive seed selection。

### 9.3 每个 seed 的 output verifier

`phase2_budgeted_end_to_end.sbatch` 自动调用：

```text
verify_phase2_budgeted_end_to_end_seed_output.py
  OVERLAY RESULT ROLLOUTS OUTPUT
  --seed SEED
  --design-sha256 SHA256
  --base-config-hash SHA256
  --git-commit COMMIT
  --image-sha256 SHA256
  --hf-inventory-sha256 SHA256
  --artifact-metadata-sha256 SHA256
  --freeze-evidence-sha256 SHA256
  --slurm-job-id-raw ID
  --array-job-id ID
  --array-task-id INDEX
```

正常运行不应绕过 sbatch 手工伪造这些参数。

### 9.4 终态聚合

三个 array task 全部离开队列后，先捕获一次严格的 Slurm 终态快照。该命令只接受
task `0..2` 全部为 `COMPLETED/0:0/0:0`，并锁定 `hpc4/sigroup/gpu-l20/l20_qos`
和 seeds `20261001..20261003`；原始 `sacct` 字节与 canonical JSON 会一起
no-replace 发布：

只有三个 fresh seed 都从数据物化完整运行到 RM 训练、policy 更新、rollout、
最终评估，且三个 seed 的逐项验证、Slurm 终态和描述性聚合全部通过，当前预算内
工程才算完成；任何 `2/3` 部分结果都不得发布效果汇总。

```bash
campaign_root="${PRORM_PROJECT_ROOT}/runs/phase2-budgeted-end-to-end/${DESIGN_SHA256}"
terminal_dir="${campaign_root}/terminal-evidence"
repo_root="$(git rev-parse --show-toplevel)"
install -d -m 0750 "${terminal_dir}" "${PRORM_PROJECT_ROOT}/aggregates"

terminal_json="${terminal_dir}/array-${ARRAY_JOB_ID}.json"
PYTHONPATH=src python \
  scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py \
  capture "${ARRAY_JOB_ID}" "${terminal_json}"

terminal_sha256="$(sha256sum "${terminal_json}" | awk '{print $1}')"
PYTHONPATH=src python \
  scripts/hpc4/capture_phase2_budgeted_end_to_end_terminal.py \
  verify "${terminal_json}" \
  --expected-sha256 "${terminal_sha256}" \
  --array-job-id "${ARRAY_JOB_ID}"
```

随后只从三个 canonical `SUCCESS` run directory 发布描述性汇总，run 参数顺序必须
严格对应 task `0..4`：

```bash
aggregate_json="${PRORM_PROJECT_ROOT}/aggregates/phase2-budgeted-${DESIGN_SHA256}.json"

PYTHONPATH=src python \
  scripts/hpc4/aggregate_phase2_budgeted_end_to_end.py \
  "${aggregate_json}" \
  "${campaign_root}/seed-20261001/job-${ARRAY_JOB_ID}_0" \
  "${campaign_root}/seed-20261002/job-${ARRAY_JOB_ID}_1" \
  "${campaign_root}/seed-20261003/job-${ARRAY_JOB_ID}_2" \
  --project-root "${PRORM_PROJECT_ROOT}" \
  --repo-root "${repo_root}" \
  --terminal-evidence "${terminal_json}" \
  --terminal-evidence-sha256 "${terminal_sha256}" \
  --array-job-id "${ARRAY_JOB_ID}"
```

publication layer 会重新执行唯一 seed normalizer，并逐一绑定三个 `SUCCESS`、
result、rollouts、manifest、artifact、verification、submission intent/ledger、
terminal evidence、producer commit 和 clean checkout。bootstrap 固定为 seed
`20260801`、`10000` 次、`95%` descriptive interval，CLI 不允许事后选择。
aggregate 与其 `.evidence.json` receipt 均为 canonical、no-replace；若进程恰在
aggregate 已精确写入而 receipt 尚未写入时中断，相同命令只能按相同字节恢复。
任一冲突、交叉绑定、失败 seed 或非终态任务都会拒绝发布。

## 10. 与历史协议的关系

- [Phase 1 结果](phase1_results.md) 保持 `not_passed`，Phase 2 不重写历史。
- [Phase 2 设计决策](phase2_design_decisions.md) 保存完整的未来正式设计与
  exact-30 规格。
- [Recovery 协议](phase2_recovery_protocol.md) 与
  [recovery authorization](phase2_recovery_authorization.md) 解释为何 recovery
  结果只能授权新鲜下游阶段。
- [Post-recovery HPC4 runbook](phase2_post_recovery_hpc4.md) 覆盖
  calibration/freeze 控制面。
- 本文覆盖 accepted freeze 之后当前实际采用的 fixed-three budgeted E2E 路线。

优先级是明确的：当前执行与结果解释以本文为准；未来 exact-30 只有在新的正式 identity
被显式启用后，才由正式协议接管。
