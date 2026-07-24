# Formal Phase 1 result

本文记录锁定的五 seed HPC4 Phase 1 实验。实验协议见
[experiment_protocol.md](experiment_protocol.md)，数学对象与适用边界见
[theory.md](theory.md)。这里报告观察到的结果，不反向修改协议、指标或成功判据。

论文标题固定为：

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

## 1. 一句话结论

五个预注册 seed 均通过身份、数值、PCG、measured-KL 和产物完整性验收，因此本次运行是
**工程成功的正式实验**；权威聚合的预注册科学状态为 **`not_passed`**。

ProRM+ 相对 BT-MLE 在 held-out local regret 和 squared Fisher direction error 上取得了略有利的
平均点估计，平均 Fisher cosine 也更高，但前两项的 paired-bootstrap 工程区间跨越零，没有通过
预注册 geometry gates。更关键的是，在相同实测 KL 预算下，ProRM+ 的
transformed-operational-oracle rollout improvement 平均比 BT-MLE 低 `0.0037335072`，工程区间
完全位于零以下。因此，这个锁定的 Phase 1 配置不支持“ProRM+ 的局部几何优势能够稳定传递到
downstream policy optimization”的预注册主张。

这个结果不改变 population、local、undamped 条件下的 ProRM/ProRM+ identity；它表明该有限
样本、ridge、受限线性 reward class、operational oracle 和 measured-KL `0.01` 配置下的
预注册经验主张未获支持。

### 1.1 结果对应的 estimand

Phase 1 的 `beta=1` held-out local regret 是 **fixed-beta ProRM 主 estimand**。它评价
同一个下游 KL penalty 下 reward-induced natural direction 的角度和范数校准；squared
Fisher direction error 是同一 fixed-beta 几何的配套指标。

Fisher cosine 与 matched-KL rollout 回答另一个问题：各 learner direction 被独立缩放到
共同 KL 半径后，哪个方向更好。局部 fixed-K constrained regret 与
`1-cos_F` 成正比，因此它们是**次级 fixed-K estimand/transfer endpoint**，不能替代
fixed-beta 主 estimand。Phase 1 同时要求这些预注册 gates，是证据链设计；不表示这些量在
数学上是同一个 estimand。

还必须区分两种 direction：

- held-out geometry 冻结 reward head，但用 held-out moment、held-out Fisher 和
  split-specific damping **重新求解** predicted/target directions；
- rollout 部署的是仅由 train moment/Fisher 求出的 train direction，test geometry 不进入
  求解或步长选择。

因此 held-out re-solve 的轻微有利点估计没有传递到 deployed train-direction rollout，并非
同一个向量出现了自相矛盾的评价，而是两个不同 operational functionals 的 ordering 没有
一致。

## 2. 冻结身份与权威证据

| Object | Locked identity |
|---|---|
| Run | `controlled-main` |
| Producer / aggregation-source Git commit | `738401d15ff97c941f3066bab1973aab37a60bc6` |
| Aggregation control-plane hotfix commit | `535f5abf7a858e35b33d5ac461d8c8e58add41fa` |
| Semantic config hash | `ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6` |
| Raw `configs/main.yaml` SHA256 | `722dae181bf39ddb162d65d9797c2bd7f584098fc0bd3a4cdef355299a5d9a08` |
| Validated SIF SHA256 | `d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb` |
| HF inventory SHA256 | `095d5dc5e5a952be53ce07279aa7b5f1eda57a7a8b5745a1e4afa545a1f11f7c` |
| GPU producer environment | `sigroup / gpu-l20 / NVIDIA L20` |
| Formal aggregate job | `1645205`, `amd/cpu13`, `COMPLETED (0:0)`, `00:00:33` |
| `aggregate.json` SHA256 | `97c2caf7790caf30d3a1108a71b766a9efb9f432312d22a72552cbf911eb15c3` |
| `aggregation-manifest.json` SHA256 | `c1fd5a9399be0fe39ed97d1e3409c9d26919bac2ff173ae078db198d17f07c38` |
| Scientific status | `not_passed` |

权威目录是：

```text
$PRORM_PROJECT_ROOT/runs/controlled-main/
  ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6/
  aggregate/
```

其中只有四个 regular、non-symlink 文件：
`SUCCESS`、`aggregate.json`、`aggregate.json.sha256` 和
`aggregation-manifest.json`。`SUCCESS` 的 `status=SUCCESS` 表示聚合流水线成功；科研结论只由
`pre_registered_evidence_status=not_passed` 决定。

聚合前两次提交 `1645152` 和 `1645166` 分别在 `amd/cpu26`、`intel/cpu07` 的容器创建阶段失败：
HPC4 的系统级 Apptainer bind 指向不存在的 KNEM 路径。两次均未创建 aggregate 或科学输出。
control-plane commit `535f5ab` 显式关闭未审计的 system `bind-paths`，同时从 producer commit
`738401d` 的 detached checkout 执行原始 aggregation Python/config。最终 manifest 分开记录
control-plane 与 aggregation-source commit，因此这次运行没有把事后运维修复冒充成训练源码。

## 3. 数据、标签、reward model 与 policy update

这不是 MultiPref 人工偏好训练，也不是 Skywork 自蒸馏：

1. 只从固定 revision `12910233a0238a997ebe425656e9dfed7b0ff031` 的
   `allenai/multipref` 提取 `2048` 个 prompt；每 seed 按 prompt 划分
   `1536/256/256` train/validation/test。
2. 固定 revision `7ae557604adf67be50417f59c2c2f167def9a775` 的
   `Qwen/Qwen2.5-0.5B-Instruct` 为每个 prompt 独立生成四个候选回答。
3. 固定 revision `8c14a4e9e6321deaf572544339b16b8d6bbe8886` 的
   `Skywork/Skywork-Reward-V2-Qwen3-0.6B` 使用自己的 Qwen3 tokenizer/chat template，为
   `prompt + assistant response` 定义 operational-oracle score。
4. 重复标签是在固定 canonical candidate edge 上，依据 transformed Skywork margin 生成的
   条件独立 BTL Bernoulli observations；它们不是人工重复标注。
5. BT-MLE 与 ProRM+ 学到的 reward model 都是冻结 Qwen2.5 last-response-token feature 加
   bias-free linear head。两者共享 candidate graph、标签、初始化、训练步数和优化预算，只改变
   reward-model objective。
6. downstream policy optimization 不是 PPO、DPO 或全参数微调。它在 Qwen2.5 最后四层
   `q_proj/v_proj` 的 fixed-A rank-4 LoRA-B tangent 中分别构造一次更新，并为每个 learner
   独立线搜索到实测 sequence KL `0.01 ± 5%`。

每 seed 的不可变 artifact 含 `2048` prompts、`8192` candidates、`1536` training edges 和
`512` held-out edges。每个成功 run 的 `updated_rollouts.jsonl` 含 `3072` 行：
reference、BT-MLE、ProRM+ 各 `256 × 4` 个 test responses；文件不保存 raw oracle score。

Phase 1 的 `max_response_tokens=128` 是该实验 action space 的一部分，因此达到上限的回答
仍是有效样本，不是需要删除的数据。但接受的 artifacts 中约 `74%–80%` candidates 达到该
上限，说明 horizon 在实际约束大量回答；这限制了结果向更长回答的外推。下一 design 固定
`256` tokens，并逐 policy arm 报告 EOS、length-limit rate 与 response-token 分布。

## 4. 接受的五个 seed

| Seed | Controlled job | Node | Elapsed |
|---:|---:|---|---:|
| `20260722` | `1642737` | `gpu19` | `02:57:49` |
| `20260723` | `1642741` | `gpu21` | `03:11:52` |
| `20260724` | `1643282` | `gpu19` | `02:49:53` |
| `20260725` | `1642736` | `gpu19` | `03:01:11` |
| `20260726` | `1643747` | `gpu21` | `02:54:26` |

五个 job 均为 `COMPLETED (0:0)`，使用一张 NVIDIA L20、16 CPUs 和 120 GiB host allocation。
总 GPU 时间为 `14:55:11`，平均约 `02:59:02/seed`，范围为
`02:49:53–03:11:52`，campaign 峰值并发为 `2`。

## 5. 预注册主指标

下表所有差值统一定义为 `ProRM+ − BT-MLE`。local regret 和 squared Fisher error 越低越好；
Fisher cosine 与 rollout improvement 越高越好。区间是五个固定 paired seeds、`10000` 次
resamples、bootstrap seed `20260722` 的 percentile engineering interval；它不是 population
confidence interval、p-value 或显著性检验。

| Metric | Favorable sign | Paired mean | Sample SD | SE | 95% engineering interval | Gate |
|---|---:|---:|---:|---:|---:|---|
| Held-out local regret | `< 0` | `-0.0091788985` | `0.0883603528` | `0.0395159511` | `[-0.0765276830, 0.0615382558]` | Fail |
| Squared Fisher direction error | `< 0` | `-0.0183621838` | `0.1766131019` | `0.0789837803` | `[-0.1529629244, 0.1230318023]` | Fail |
| Fisher cosine | `> 0` | `+0.0416379998` | `0.1188229946` | `0.0531392586` | `[-0.0529130971, 0.1316970727]` | Pass under the locked mean-sign rule |
| Matched-KL rollout improvement | `> 0` | `-0.0037335072` | `0.0042905189` | `0.0019187784` | `[-0.0074192179, -0.0006187576]` | Fail |

Rollout 是明确的负向工程证据：五个 seed 中四个差值为负，而且聚合区间完全位于零以下。准确表述是
“在本次工程决策区间内，锁定配置的 rollout 结果整体有利于 BT-MLE”；不能把它写成
“ProRM+ 没有显著改善”，也不能把工程区间解释为 population significance。

## 6. Seed-level heterogeneity

| Seed | Local regret Δ | Fisher error Δ | Fisher cosine Δ | Rollout improvement Δ |
|---:|---:|---:|---:|---:|
| `20260722` | `-0.0655430` | `-0.1310105` | `+0.1466878` | `+0.0000789` |
| `20260723` | `-0.0388817` | `-0.0777654` | `+0.0731103` | `-0.0101898` |
| `20260724` | `+0.0606812` | `+0.1211339` | `+0.1459997` | `-0.0059158` |
| `20260725` | `+0.1041844` | `+0.2083452` | `-0.1264787` | `-0.0020300` |
| `20260726` | `-0.1063353` | `-0.2125141` | `-0.0311291` | `-0.0006108` |

ProRM+ 在三个 seed 上降低 local regret，在两个 seed 上升高；方向误差呈相同的明显异质性。
因此，有利的平均 local 点估计不能被描述为稳定 geometry improvement。

## 7. Damping、PCG 与 measured-KL 完整性

| Damping multiplier | Mean local-regret Δ | 95% engineering interval | PCG | Mean-sign non-reversal |
|---:|---:|---:|---|---|
| `0.1` | `-0.0371851373` | `[-0.0951122180, 0.0392747366]` | All converged | Yes |
| `1.0` | `-0.0091788985` | `[-0.0765276830, 0.0615382558]` | All converged | Yes |
| `10.0` | `-0.0093321666` | `[-0.0721854986, 0.0674549481]` | All converged | Yes |

所有要求的 ProRM+ Fisher solves 与两种 learner 的 rollout-direction PCG 都达到 true relative
residual `≤ 1e-5`。所有 policy line searches 均收敛，BT-MLE/ProRM+ 的实测 KL 都落在
`[0.0095, 0.0105]`。这里匹配的参数顺序明确为固定 reference histories 上的
$D_{\mathrm{KL}}(\pi_0\Vert\pi_{\alpha d})$；理论正则项则是
$D_{\mathrm{KL}}(\pi_{\alpha d}\Vert\pi_0)$。二者在 reference 处共享二阶 Fisher，但有限
更新下不相等。因此可以确定的是：`not_passed` 不是数值失败、PCG 失败或**锁定 operational
KL gate 内**的 line-search 不收敛造成的状态降级；不能进一步声称另一 KL 方向或全
held-out distribution 也被精确匹配。

权威 criteria 为：

| Criterion | Result |
|---|---|
| `main_local_regret_negative_with_ci` | `false` |
| `main_direction_error_negative_with_ci` | `false` |
| `main_fisher_cosine_positive` | `true` |
| `matched_kl_rollout_positive_with_ci` | `false` |
| `sensitivity_local_regret_nonreversal` | `true` |
| `all_pcg_converged` | `true` |
| `all_measured_kl_updates_converged` | `true` |

## 8. Preference diagnostics

这些指标只描述 frozen operational oracle 下的 preference fit，不属于成功判据：

| Metric | Paired mean Δ | 95% engineering interval |
|---|---:|---:|
| Pairwise accuracy | `+0.0136745453` | `[-0.0188696980, 0.0405840874]` |
| Oracle pairwise NLL | `-0.0025069714` | `[-0.0225379109, 0.0175239682]` |
| Oracle probability MAE | `-0.0004975230` | `[-0.0111485064, 0.0101599510]` |

三个区间都跨零。即使其中某个预测指标稳定改善，也不能替代 downstream geometry 与 rollout
证据。

## 9. Repeated-label finite-sample audit

`gamma=0.9` 的 randomized estimator 在 locked probability range 内条件无偏且二阶矩有限，
因此 Phase 1 没有标签构造错误。但高精度条件枚举显示
`sd(h|p)=0.840942` at `p=0.5`，在 `p=0.25/0.75` 上升到 `0.934657`。作为对照，五个
artifact 的 train-only node-centered oracle RMS 约为 `0.24`；全六 pair-margin RMS 约为
`0.392`。所以单 edge 单份 `h` 是高噪声 estimator。

在 worst endpoint，`P(|h-Delta r*|>5)≈2.99e-4`；若粗略按 1536 个独立 edges 计算，
至少出现一个该级别 deviation 的概率约 `36.8%`。这不能证明它造成了 rollout reversal，
但足以否定“无偏所以 1536 edges 必然足够”的推理。

下一 design 不 clip、不固定 `N`、也不删除极端值。它加入：

- `h=Delta r*` 的 zero-noise oracle-margin positive control；
- 每 edge 四份独立 `gamma=0.9` estimates 的均值，保持严格无偏并把 conditional SD 减半；
- 独立的 all-six natural-pairs prompt U-statistic arm；六条 pair 共享四个 nodes，因此按
  prompt 聚类，不当作六个独立样本；
- BT-MLE 在每个 arm 使用完全相同的 underlying Bernoulli annotation budget。

## 10. 科学解释边界

已由本实验确定的结论只有：

- 当前受限配置没有通过预注册 ProRM+ 机制主张；
- local/Fisher mean 的轻微改善不稳定，且没有传递为 matched-KL rollout 改善；
- Phase 1 使用 transformed Skywork operational oracle，不能据此声称 human preference utility；
- 结果不能归因于 source/config 混用、Qwen2.5/Qwen3 chat-template 混用、PCG 未收敛、
  锁定 reference-to-updated KL gate 未收敛、漏 seed 或事后换 run。

有限样本方差、ridge、随机截断标签、冻结线性 reward class、local quadratic approximation 与
有限 KL rollout 的差距，都是与 population identity 不同的边界条件；本实验没有单独识别哪一项
造成了 rollout reversal，因此不能把任何一个写成已证实的原因。

## 11. 下一步的确定工程顺序

当前 Phase 1 已经结束，不因 `not_passed` 换 seed、改阻尼或重跑同一配置。后续工作必须建立新
design identity，并按以下顺序进行：

1. **先完成零 GPU estimand 与优化审计。** 对现有五 seed 直接评价 saved train direction、
   applied alpha、隐式 `beta_eff`、final objective/gradient 和 policy-relevant approximation
   error；不重训、不改写 Phase 1 结果。
2. **再过 exact-margin/noisy-margin local controls。** 在同一 candidates/Fisher/reward class
   上比较 `h=Delta r*`、单份 `h` 与 `R=4` 独立 `h` 均值。只有 exact-margin ProRM+ 能到达其
   policy-relevant reward-class optimum，才进入 finite-policy 主链。
3. **运行 common-beta 主实验。** 只用 train operational oracle、train Fisher 和冻结的
   train candidate graph 构造
   $u_*^{tr}=(F_{\rm train}+\lambda I)^{-1}g_*^{tr}$，以主尺度 `K_cal=0.003` 解析设定
   $\beta_{\rm common}=\sqrt{(u_*^{tr})^\top F_{\rm train}u_*^{tr}/(2K_{\rm cal})}$，
   随后在该 seed 内对 BT-MLE、ProRM+ 与 oracle-step 共同冻结；
   两 learner 直接部署
   $d_\ell=(F_{\mathrm{train}}+\lambda I)^{-1}g_\ell^{train}/\beta_{\mathrm{common}}$，
   不再各自 line-search 或按自身 norm 重标。held-out oracle 只在 heads、beta 和 directions
   冻结后用于评价。主效用使用 updated-policy trajectories 上的
   $J^*=E[r^*]-\beta_{\rm common}KL(\pi_{\rm updated}\Vert\pi_0)$；固定 reference histories
   上的反方向 KL 只作 secondary diagnostic。policy-to-reference KL 超过 `0.02` 则 fail
   closed，不回调 beta。
4. **用 fixed-K 诊断定位 local-to-rollout gap。** 在冻结 heads、candidate graph 和
   common-random rollout seeds 上保留较小的 matched-KL grid，例如
   `{0.001, 0.003, 0.01}`，并加入 train-only oracle local direction 的 finite-rollout
   calibration。它是次级尺度/角度诊断，不取代 common-beta 主结果。
5. **再运行 all-six 与 capacity/sample arms。** all-six natural pairs 与 `R=4` 首轮分开；
   若更小 KL 仍不恢复 ordering，则用更大 prompt budget
   和更高容量但相同输入/标签预算的 reward parameterization，重新比较 BT-MLE 与 ProRM+；不能
   只扩大 ProRM+ 容量。
6. **最后才做 human robustness。** 当前 `not_passed` 不激活 CoVal 作为本协议的 confirmatory
   Phase 2。任何 CoVal 或大模型实验都必须标为 exploratory，或以新的冻结协议重新预注册。

这条顺序继续检验同一个 prospective reward-modeling insight，同时避免用规模扩大掩盖本次未
通过的预注册机制判据。以上是新 design 的规格概要，不修改 Phase 1 权威
`pre_registered_evidence_status=not_passed`。
