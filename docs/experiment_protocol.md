# ProRM+ 固定实验协议

本文是实验执行与结果判定规格。数学定义以 [theory.md](theory.md) 为准。
本文在观察正式结果前冻结，不因结果改写；执行记录与权威结果见
[phase1_results.md](phase1_results.md)。
第 1–12 节保留已冻结的 Phase 1 协议与历史执行语义；第 13 节是 Phase 2
预实验/confirmatory 规格，不能反向改变 Phase 1 的 `not_passed` 判定。本文不声明
HPC4 的实时队列或 campaign 状态。
论文标题固定为：

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

本协议比较 repeated-label BT-MLE 与 ProRM+。ProRM 是含目标 reward 的 ideal
population/local regret，不是可单独运行的 baseline；ProRM+ 才是 repeated-label
Fisher–GMM/PCG 训练实现。“+”表示把不可观测 ProRM target 转换成可训练 moment method。

- **design identity**：`configs/main.yaml` 的 canonical hash，锁定 prompt、revision、tangent、
  optimizer、seed、metric 与 config 中的停止规则；
- **run identity**：design identity + selected seed + Git commit + image SHA256 + Slurm
  account/partition/GPU + manifest/artifact/comparison hashes。

改变 config 必须使用新 design identity；改变 code-locked numerical rule 必须使用新 Git
identity。两者都要作为新实验报告，不能与旧结果合并。

本文按“研究问题 → 数据生成 → 训练 → held-out 几何 → downstream rollout → 判定”排序。
执行者不应从中挑选单独步骤；每个正式 seed 是一条不可拆分的证据链。

## 1. 研究问题与唯一主链

主问题是：**在受限且可能 misspecified 的 reward class 中，ProRM+ 能否比 repeated-label
BT-MLE 更准确地恢复 operational-oracle 局部 policy update，并把该优势传递到相同实测 KL
预算下的 policy optimization？**

第一阶段只识别这条链：

```text
同一 pi_0 candidate graph
  -> 同一 repeated BTL observations
  -> 同一 frozen feature linear reward class
  -> BT-MLE vs ProRM+（只改变训练目标）
  -> held-out Fisher geometry
  -> same measured-KL downstream policy rollouts
```

高容量 reward model、主动选边、多轮 RLHF 和人类 CoVal robustness 均不得替代该 controlled
experiment。代码和 synthetic benchmark 通过只说明实现自洽，不构成 ProRM+ 效果结论。

### 1.1 阶段契约

| 阶段 | 冻结输入 | 新输出 | 失败语义 |
|---|---|---|---|
| Phase 0 | source tree、两份 config | tests/config checks | 任一失败即停止 GPU 实验 |
| Phase 1A/B | config、pinned snapshots、named seed | immutable artifact | 任何 schema/hash/model 错误均硬失败 |
| Phase 1C/D | 同一 artifact、同一 zero head | comparison | 主阻尼失败为硬失败；sensitivity 失败保留证据 |
| Phase 1E | 主阻尼两个 head、同一 policy geometry | matched-KL rollout | direction PCG 或 KL search 失败为硬失败 |
| Aggregate | 五 seed comparison/rollout/manifest | aggregate | 身份不一致拒绝；sensitivity failure 产生 `not_passed` |

“保留 sensitivity failure”不表示忽略失败：它必须写入 `damping_evidence`，并使预注册主结论
不通过。这样既不丢失负面数值证据，也不把失败 seed 静默排除后聚合。

### 1.2 Estimand 与 KL 方向契约

本协议涉及三个相关但不同的对象：

| 层级 | 定义 | 作用 |
|---|---|---|
| 主 estimand | 所有 reward model 共用同一 `beta` 的 ProRM local regret | 同时评价 natural direction 的角度和范数校准 |
| 次级 estimand | 每个 direction 归一化到同一局部 KL 半径 `K` 后的 constrained regret / Fisher cosine | 只评价 fixed-K 边界上的方向角度 |
| finite-update endpoint | 对真正更新后 policy 做 operational-oracle rollout | 检验局部几何能否传递到有限更新 |

固定 `beta` 的主 estimand 是

$$
\mathcal R_\beta(r_\phi)
=\frac{1}{2\beta}
\|A_0(r_\phi-r^*)\|_{F_0^\dagger}^2.
$$

固定 `K` 时，每个 natural direction 都被自身 Fisher norm 归一化；对应 constrained regret
与 `1-\cos_F` 成正比。因此 fixed-K/cosine 是有意义的次级机制指标，但不能替代 ProRM
主 estimand。

理论 policy objective 中的正则项方向是
$D_{\mathrm{KL}}(\pi_\theta\Vert\pi_0)$。锁定 Phase 1 line search 实测的是固定
reference histories 上的
$D_{\mathrm{KL}}(\pi_0\Vert\pi_{\alpha d})$。它们在 zero-B reference 处共享同一 Fisher
二阶项，有限步长下并不相等。Phase 1 的 matched-KL rollout 因此严格解释为 code-locked
fixed-K transfer test；这一澄清不改变冻结协议、已有数值或 `not_passed` 判定。

## 2. 预注册身份

### 2.1 Main config

- run name：`controlled-main`
- paired seeds：`20260722, 20260723, 20260724, 20260725, 20260726`
- prompts：2,048；train/validation/test = `1536/256/256`
- candidates：每 prompt 4 个
- model/storage dtype：Qwen、Skywork、reward feature 与 artifact score tensor 为 FP32
- reward optimization dtype：linear head、autograd、gradient 与 AdamW state 为 FP32
- policy-geometry dtype：moment、damping、Fisher matvec、PCG、held-out geometry 与 rollout
  direction 为 FP64
- reward optimizer：720 outer steps，AdamW，lr `1e-3`，weight decay `0`，
  microbatch `64`，max grad norm `1`
- objective：`beta=1`，PCG true relative-residual tolerance `1e-5`，main fail-closed ceiling
  `8192` iterations
- main relative damping：`c=1e-3`
- mandatory sensitivity：multiplier `0.1, 1, 10`，即 `c=1e-4,1e-3,1e-2`
- measured sequence forward-KL：target `0.01`，relative tolerance `0.05`
- paired percentile bootstrap：10,000 resamples，seed `20260722`
- current semantic config hash：`ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6`
- current raw config SHA256：`722dae181bf39ddb162d65d9797c2bd7f584098fc0bd3a4cdef355299a5d9a08`

所有 model/dataset revision 是 config 中的 40 位 commit SHA。不得把浮动 `main`、本地目录
mtime 或下载时间当作 revision。

### 2.2 Smoke config

`configs/smoke.yaml` 使用 64 prompts、48/8/8 split 和 10 steps，但保留 main 的 rank-4、
最后四层 `q_proj/v_proj` tangent 与 16-candidate KL probe。oracle batch 上限同为 16；reward
head 只消费已冻结 feature，因此 smoke 的 head microbatch 16 足以覆盖 backbone 峰值。它验证
真实 snapshot、显存、CUDA/Apptainer、I/O 和端到端命令，禁止与 main 结果合并。smoke 同样
锁定 `pcg_dtype=float64` 与 `1e-5` true-residual gate，但保留 `2048` iteration ceiling。

### 2.3 数值修订身份

旧 main design `7b3f12ba…f7b2`、source `f16edb12…d29e` 在任何 accepted scientific result
产生前，于 seed `20260722` / job `1641489` 的 mandatory initial ProRM+ solve 硬失败：
2048 iterations 后 true relative residual 为 `2.717e-5 > 1e-5`。该失败没有读取或产生
downstream scientific metric；修订只处理被预先规定的数值门揭示的 FP32 residual floor。

旧 `FAILED` marker、Slurm log、manifest 与已完成 artifact 必须保留；comparison、rollout 与
`SUCCESS` 不存在，该 run 不是可聚合 seed，也不是科研 `not_passed`。`pcg_dtype`、ceiling 与
solver code 改变后形成新 config/Git identity；新身份下五个 seeds 必须全部重跑，禁止与旧
campaign 混合。

## 3. Phase 0：CPU 数值门槛

占用 GPU 前必须运行：

```bash
python -m pip install -e ".[llm,dev]"
prorm config-check configs/smoke.yaml
prorm config-check configs/main.yaml
prorm closed-form-check --output outputs/closed-form.json
prorm synthetic-check --seed 0 --output outputs/synthetic.json
pytest -q
ruff check .
ruff format --check .
python -m compileall -q src tests
```

LLM environment 必须满足 `transformers>=4.52.3,<5` 并暴露
`Qwen3ForSequenceClassification`；这是 pinned Skywork Qwen3 oracle 的硬兼容门槛。

测试必须覆盖以下数值与安全不变量：

1. `closed-form-check` 重现[三边四响应解析例](closed_form_example.md)中的 population
   ProRM/BT-MLE ordering reversal，并明确标记
   `population_example_only=true`；它不冒充 natural-`Q_0` ProRM+ 训练；
2. randomized estimator `h` 的 Monte Carlo 均值匹配 `logit(p)`，并检查有限二阶矩；
3. PCG 相对残差达到门槛，并与小矩阵 direct solve 一致；
4. primal/dual value 和 envelope gradient 与解析/finite-difference 结果一致；
5. reward 加 prompt-level 常数后 local metric 不变；
6. node/pair 两种 moment/Fisher identity 在模拟中一致；
7. microbatch 与 full-batch 外层梯度等价，dual 每一步刷新；
8. train schema 无法接收 true/oracle reward，split 必须 disjoint；
9. config、JSONL、artifact、comparison identity 不匹配时 fail closed；
10. KL line-search 每次从 zero-B 覆盖，异常或不收敛恢复原点；
11. synthetic output 标记 `benchmark_only=true`，测试不要求 ProRM+ 胜过 BT-MLE。

任一门槛失败时停止真实模型实验，不得忽略失败继续提交 HPC job。

## 4. Phase 1A：不可变 candidate graph

### 4.1 Prompt

从固定 revision 的 MultiPref 读取 `prompt_id` 与 `text`，按 `prompt_id` 去重后，再用
named seed 做 deterministic prompt-level split。输入行顺序变化不能改变去重选择与 split。
train、validation、test prompt ID 必须两两不交。

### 4.2 Reference policy 与采样

使用固定 revision 的 `Qwen/Qwen2.5-0.5B-Instruct`。tokenizer 必须提供非空 chat
template；prompt 左截断至 384 tokens，response 最多 128 new tokens。

每个 prompt 从 reference distribution 独立返回 4 个 response：

- `do_sample=true`、temperature `1`、top-p `1`、top-k `0`；
- `min_new_tokens=0`、repetition penalty `1`；
- 禁止 beam、top-k/top-p 截断、质量过滤、oracle 筛选、文本去重；
- 不使用改变分布的 logits processor；
- 保留完整 input token IDs、response mask、EOS/达到长度上限状态；
- candidate 生成与 score 提取必须是同一个 FP32、eval、fixed-A/zero-B model instance。

重复文本是合法的独立样本，不得删除。正常终止时 EOS 属于 response；达到长度上限时以
最后一个生成 token 结束。

### 4.3 Fixed-A LoRA policy tangent

main tangent 是 Qwen 最后四层（20–23）`q_proj/v_proj` 的 rank-4 LoRA，
`alpha=rank`、dropout 0：

- A 只随机初始化一次后冻结；B 精确置零且是唯一 `requires_grad=True` 的坐标；
- 加 adapter 前后 probe logits 必须满足 no-op 门槛；
- 保存 A SHA256，以及每个 B 的参数名、shape、flatten offset 和总维度；
- policy score 是完整 response sequence log-probability 之和对 B 的 per-sample gradient；
- 不做长度归一化，不得在 BF16 采样后换 FP32 实例重算 score；
- `S`、feature 和 oracle assembly tensor 存为 float32、detached CPU tensor。

存储精度不等于求解精度：consumer 构造任何 Fisher/GMM policy geometry 时，先把固定
`S/Z/h` 提升到 config-locked FP64，再计算 moment、damping、matvec 与 Krylov solve。

每 prompt 四个 node 全部进入 Fisher：

$$
\widehat F=\frac{1}{PM}\sum_{i=1}^P\sum_{j=1}^M s_{ij}s_{ij}^\top,
\qquad M=4.
$$

训练 edge 只取 canonical candidate `0 - 1`，其 score difference 为
`z=s_0-s_1`。如果 UI 随机交换展示顺序，label 必须映射回 canonical left-win；存储层禁止
只写 `chosen/rejected`。

### 4.4 Frozen reward feature

reward learner 使用同一 Qwen zero-B forward 的最后一层 hidden state，并只 pool 最后一个
response token。正常结束时是 EOS，长度截断时是最后一个生成 token；prompt 和 padding 不得
参与 pooling。backbone 完全冻结，linear scalar head 无 bias、全零初始化。

这个受限 class 施加可审计 capacity bottleneck，但不逻辑保证 oracle 一定不可表示。对
train nodes 按 prompt 中心化 feature 与 transformed oracle reward，再做一次不参与训练的
最优线性投影；artifact 的
`metadata.json.evidence.train_reward_class_projection` 记录 `fit_split`、`centering`、`solver`、
`target_centered_rms`、`residual_rmse` 与 `relative_residual`，不暴露 fitted weight 或 train true
rewards。该 diagnostic 不参与调参、checkpoint 或成功判据；若 residual 接近数值零，只能说明
train candidates 上没有观察到线性不可表示证据，不能排除 held-out 或 population
misspecification。当前 CPU float64 `lstsq` 未固定 LAPACK driver/rcond，因此末位跨平台差异不作
判据。高容量 LoRA RM 只能在主链通过后作为 scale-up。

## 5. Phase 1B：oracle、标签与泄漏边界

### 5.1 Controlled oracle

使用固定 revision 的 `Skywork/Skywork-Reward-V2-Qwen3-0.6B` sequence-classification
logit。policy 从 GPU 释放后才加载 oracle；两者不同时驻留。

本文把其冻结变换后的输出记为 `r*`，含义仅是 controlled Phase 1 的 **operational ground
truth**。它不是人类 utility，也不证明 Skywork 对目标人群无偏；Phase 1 的结论只能解释为
对该冻结 oracle 所定义局部 update 的恢复能力。

只用 **train 的全部 node raw score** 拟合

$$
b=\operatorname{median}(R),\qquad
\tau=\max\{1.4826\operatorname{median}|R-b|,10^{-6}\}.
$$

冻结后对全部 split 应用

$$
r^*(x,y)=\frac{\log 3}{2}\tanh((R_{oracle}(x,y)-b)/\tau).
$$

因此每条 edge 的 $|\Delta r^*|\le\log 3$，BTL probability 位于 `[0.25,0.75]`。不得
用 validation/test 重新拟合 `b,tau`。

### 5.2 Repeated BTL observations

每个 split 使用由 base seed 派生的独立 annotation stream；held-out 数量变化不能改变
train labels。对每条 edge：

1. 独立采 `N ~ Geometric(0.1)`，支撑从 1 开始；
2. 以 `p*=sigmoid(r^*_0-r^*_1)` 采 `N` 个条件 iid Bernoulli label；
3. 用 `gamma=0.9` 和完整 label sequence 构造 randomized-truncation `h`；
4. 不得硬截断 `N`，不得按 `N` 给 ProRM+ edge 加权。

BT-MLE 使用全部原始 Bernoulli label，等价于每个 label 等权；ProRM+ 每个 edge 对 moment
贡献一次。两者的 weighting 不可互换。

### 5.3 物理隔离

`TrainingTensorData` 只允许：

```text
prompt_ids, policy_scores, reward_features, h, left_wins, num_annotations
```

`true_rewards` 只允许存在于 validation/test `EvaluationTensorData`。训练 JSONL 只含
`raw_labels,N,left_wins,h` 等 observable fields；`true_margin` 只进入 evaluation JSONL。
任何 true/oracle field 出现在 train schema 都必须硬报错。

## 6. Phase 1C：固定预算训练

### 6.1 BT-MLE

对 canonical edge margin `t_phi`，使用 count-compressed、与逐 label 完全等价的 repeated
Bernoulli negative log-likelihood：每个原始 label 权重相同。

### 6.2 ProRM+

ProRM+ 的经验 moment 与显式阻尼 Fisher–GMM target 固定为

$$
\widehat m_\phi=\frac{1}{2n_E}Z^\top(t_\phi-h),
\quad
\widehat L_\lambda
=\frac{1}{2\beta}\widehat m_\phi^\top
(\widehat F+\lambda I)^{-1}\widehat m_\phi.
$$

等价的实际训练问题必须写出 ridge：

$$
\boxed{
\min_\phi\max_v\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F+\lambda I)v
\right]
}.
$$

只有 population、`lambda=0`、`F_0^dagger` 的内层最优值与 ideal ProRM local regret 精确
相等。这里的 finite-sample、`lambda>0` objective 是 ridge empirical surrogate。

每一步执行：

```text
full margins -> FP64 m_hat -> warm-start FP64 PCG -> true-residual gate -> detach v
             -> one AdamW step -> repeat
```

其中 `v=(F_hat+lambda I)^-1 m_hat`。microbatch 只允许累积一个 full moment 对应的 outer
gradient；禁止 batch-local `m/F`、stale `v` 跑多个 step、动态 edge-weight normalization。

每步记录的 ridge objective 是 `m_hat^T v/(2*beta)`；用于 autograd 的 detached envelope
surrogate 在同一 full batch 上数值为 `m_hat^T v/beta`。两者相差 2，但后者才产生二次型的
完整 envelope gradient。不得用 surrogate 数值替代 reported objective；完整推导见
[theory.md](theory.md) 第 7 节。

PCG 不使用 coordinate-wise preconditioner：`S^T S/n + lambda I` 是低秩加 isotropic damping，
unpreconditioned CG 保留重复的 `lambda` 特征值，而 Jacobi 会破坏该 Krylov 结构。main cap
为 `8192`，smoke cap 为 `2048`；config validator 还要求 cap 至少覆盖 train Fisher node
rank bound `n_F+1`。recursive residual 只服务 recurrence；每 20 次及其首次达到 threshold 时
显式验证 true `rhs-Ax`，但不周期替换 residual。只有 true relative residual `<=1e-5` 才
converged；recursive 假性过门时必须从 true residual 显式 restart。最终 evidence 始终保存
true residual；改变 dtype、preconditioner、上限或验证周期必须使用新 Git/config/run identity。

FP64 direction 产生 FP64 edge envelope weights；weights 只在进入 FP32 reward-head surrogate
前显式转换一次。reward feature/head、gradient 与 AdamW 不因此变成 FP64。

### 6.3 公平性

BT-MLE 与 ProRM+ 必须共享：artifact、seed、candidate、label、feature、canonical edge、零初始化
head、optimizer type、lr、step 数、microbatch、gradient clip、weight decay、GPU 分区/型号和
停止规则。validation 只作描述，不能选 checkpoint、early stop 或调 hyperparameter。

主 run 在三档 damping 各自从同一零 head 完整重训；`comparison.json` 必须含唯一
`damping_multiplier=1` 主结果、head bytes 对应的 SHA256 和 final PCG evidence。

## 7. Phase 1D：held-out metric

对每个 held-out prompt 的四 candidate，用无偏、严格 gauge-invariant covariance moment：

$$
\widehat g_r
=\frac{1}{P(M-1)}\sum_{i=1}^P\sum_{j=1}^M
(s_{ij}-\bar s_i)(r_{ij}-\bar r_i).
$$

注意 moment 分母是 `P(M-1)`，held-out node Fisher 分母是 `PM`。每个 split 独立解析
$\lambda=c\operatorname{mean}(\operatorname{diag}F)$，并继承同一
`pcg_dtype/tolerance/cap/true-residual` contract。主指标为：

1. **fixed-beta 主 estimand**：held-out ridge local regret

   $$
   \frac1{2\beta}m_{error}^\top(F+\lambda I)^{-1}m_{error};
   $$

2. predicted/target damped natural direction 之间的 undamped-Fisher squared error；
3. **fixed-K 次级 estimand**：同两方向的 Fisher cosine。

Fisher cosine 定义为

$$
\cos_F(u,v)=\frac{u^\top Fv}
{\sqrt{(u^\top Fu)(v^\top Fv)}}.
$$

若任一方向的 Fisher norm 为零，该指标未定义；内部 NaN 在 JSON 中记录为 `null`，聚合器随后
拒绝该输入。不得加 epsilon 伪造 cosine，对应 seed/criterion 必须失败。

这些 metric 是 **held-out re-solve**：reward head 保持冻结，但 evaluator 分别用 validation
或 test 的 moment、node Fisher 和 split-specific damping 重新求解 predicted/target
directions，

$$
u_\phi^H=(F_H+\lambda_HI)^{-1}g_\phi^H,\qquad
u_*^H=(F_H+\lambda_HI)^{-1}g_*^H.
$$

它们不是下一阶段真正写入 policy 的 train direction。不得把 test re-solved direction
用于 rollout 或反向调整训练 head。

Prediction diagnostics 对四 candidates 的全部无序 pair 计算。令

$$
q_{ijk}=\sigma(r_\phi(x_i,y_{ij})-r_\phi(x_i,y_{ik})),
\qquad
p^*_{ijk}=\sigma(r^*(x_i,y_{ij})-r^*(x_i,y_{ik})).
$$

固定报告：

1. oracle-expected BTL NLL

   $$
   -\operatorname{mean}_{i,j<k}
   \left[p^*_{ijk}\log q_{ijk}+(1-p^*_{ijk})\log(1-q_{ijk})\right];
   $$

2. probability MAE `mean_{i,j<k}|q_ijk-p*_ijk|`；
3. pairwise ordering accuracy：真实 tie 排除，预测 tie 计 0.5。

这些值只描述 preference-probability fit，不参与 checkpoint、damping 或成功判据，也不能替代
local regret/direction。它们用于检验“preference fit 与 downstream policy geometry 是否发生
分离”，不能单独支持 ProRM+。

## 8. Phase 1E：matched measured-KL rollout

只使用主阻尼 `c=1e-3` 的两个训练后 head。由 train 的全部四 candidate 在同一 FP64
policy geometry 中构造实际部署方向

$$
d_\phi^{deploy}
=\beta^{-1}(F_{train}+\lambda I)^{-1}\widehat g_{r_\phi}^{train}.
$$

该方向在读取任何 test moment/Fisher 之前已经由 train quantities 唯一确定。Phase 1E
不会在 held-out split 上重新求解 direction；上一节的 held-out re-solve 只是一组 geometry
diagnostics。

重新加载相同 revision、相同 named seed 的 fixed-A/zero-B Qwen，并验证 A SHA256、B layout
和 chat-template SHA256 与 artifact 一致。KL probe 是 train 保存 candidate 的共享、
输入顺序无关的确定性子集。

Fisher approximation 以 FP64 `d/F` 给出 `sqrt(2*kappa/(d^T F d))` 作为 line-search 初值，
但不得用于接受更新。direction 仅在真正写入 FP32 LoRA-B parameter 时按 parameter dtype
转换。每个 trial 从 zero-B 坐标原点覆盖 `alpha*d`，并在保存的完整 token history 上
计算全 vocabulary 的 **reference-to-updated sequence-level KL**：

$$
\widehat{KL}=\frac1B\sum_{b=1}^{B}\sum_{t\in response_b}
KL\!\left(\pi_0(\cdot\mid h_t)\,\|\,
\pi_{\alpha d}(\cdot\mid h_t)\right).
$$

参数顺序明确为 reference-to-updated：
$D_{\mathrm{KL}}(\pi_0\Vert\pi_{\alpha d})$。它不是理论正则项
$D_{\mathrm{KL}}(\pi_{\alpha d}\Vert\pi_0)$ 的有限步长无偏估计；两者只在 reference
附近共享二阶 Fisher。

即每个 response 内先对 token 求和，再对 batch 中 sequence 求均值；禁止除以总 response
token 数。这样 `kappa=0.01` 与 sequence log-prob score/Fisher 的尺度一致。

line search 的 code-locked 停止规则为：先测 zero-B 与 Fisher quadratic 初值；若初值 KL
低于 target，步长反复乘 2 直到形成上下 bracket；随后在 step-size 区间二分。最多进行 30
次 measured-KL evaluations，只有 relative error `<=0.05` 才收敛。每个 trial 都从同一
zero-B 原点覆盖参数，而不是在上一次 trial 上累加；耗尽预算或出现非有限值时恢复 zero-B。
该算法假设所搜正向 ray 上的 measured KL 在目标附近足以单调形成 bracket；代码不声称验证
全局单调性，最终接受标准始终是实测 KL 容差。改变该规则必须使用新 Git/run identity。

BT-MLE 与 ProRM+ 分别达到 `0.01 ± 5%` 才能进入 rollout；不收敛或异常时恢复 zero-B 并使该 seed
失败。test prompt 每个重新采样 4 candidates；zero-B、BT-MLE update、ProRM+ update 对每个 prompt
重置相同派生 seed，使 candidate index 成为严格 common-random pair。policy 全部卸载后只
加载一次 oracle；raw logit 不落盘，只保存冻结 transform 后的 reward。

结果报告各 learner 的 measured KL、transformed-oracle mean，以及相对**本次同 seed zero-B
rollout** 的 paired improvement。Phase 1 artifact 的原 test reward 使用不同 candidate-
generation stream，只作未配对 descriptive sanity，不能作为 updated rollout 的 paired
reference。experimental unit 是 prompt：对 learner $\ell$，先计算

$$
\Delta_i^{(\ell)}=\frac1M\sum_{j=1}^M
\left(r_{ij}^{(\ell)}-r_{ij}^{(0)}\right),
$$

再跨 $P_{test}$ 个 prompts 报告 $\bar\Delta^{(\ell)}$ 和 sample
`SE=sd(Delta_i)/sqrt(P_test)`；不得把同 prompt 的四 candidates 当作四个独立实验单位。
最终比较 `ProRM+ improvement - BT-MLE improvement`。

## 9. 结果判定与统计

每个 seed 必须完整配对。正式统计以每 seed scalar 的 `ProRM+ - BT-MLE` 为单位，报告配对均值、
样本标准差、标准误，以及在 5 个预注册 paired seeds 上计算的 deterministic 95%
percentile-bootstrap **工程判定区间**；不得把 candidate 或 prompt 当作独立 seed。该区间
不是 population confidence interval，聚合器也不输出 p-value 或“显著”标签。

主结论仅在以下条件全部满足时通过：

1. `c=1e-3` 的 test local regret：配对均值 `<0` 且 interval upper `<0`；
2. `c=1e-3` 的 test squared Fisher error：配对均值 `<0` 且 interval upper `<0`；
3. test Fisher cosine：配对均值 `>0`；
4. 两 learner 每 seed measured KL 都在容差内；rollout improvement 的 `ProRM+-BT-MLE` 配对均值
   `>0` 且 interval lower `>0`；
5. 两个 sensitivity damping 的 `ProRM+-BT-MLE` local-regret 配对均值均严格 `<0`，所有正式
   PCG/KL search 收敛且无数据完整性失败；exact zero 是 inconclusive/`not_passed`。

如果只提高 pairwise accuracy，主想法没有得到验证。如果 local metric 改善而 downstream
rollout 不改善，结论限定为“局部 surrogate 改善但未建立 downstream transfer”。任一主条件
失败后不得更换 seed、挑 checkpoint 或事后改变 primary metric；后续诊断必须标注 exploratory。

结果解释固定如下：

| Held-out geometry | Matched-KL rollout | Sensitivity/solver | 允许的结论 |
|---|---|---|---|
| 通过 | 通过 | 通过 | 支持预注册的 policy-aware mechanism claim |
| 通过 | 未通过 | 任意 | 只支持局部 surrogate 改善，不支持 downstream transfer |
| 未通过 | 任意 | 任意 | 核心机制未获支持 |
| 任意 | 任意 | sensitivity failure/reversal | 主结论 `not_passed`，保留失败证据 |
| 仅 prediction NLL/accuracy/probability MAE 改善 | 任意 | 任意 | 不构成 ProRM+ 成功证据 |

## 10. 实际命令链与产物

单 seed：

```bash
seed=20260722
run_dir="outputs/main/seed-${seed}"
mkdir -p "${run_dir}"

prorm env-report configs/main.yaml \
  --seed "${seed}" --repo-root . --output "${run_dir}/run-manifest.json"

prorm controlled-materialize configs/main.yaml \
  "${run_dir}/artifact" --seed "${seed}" --device cuda

prorm controlled-compare configs/main.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" \
  --seed "${seed}" --device cuda \
  --run-manifest "${run_dir}/run-manifest.json"

prorm controlled-rollout configs/main.yaml \
  "${run_dir}/artifact" "${run_dir}/comparison.json" "${run_dir}/rollout.json" \
  --seed "${seed}" --device cuda
```

所有写操作都新建/原子替换受控目标；materialization 和 rollout 拒绝覆盖现有完整产物。
`controlled-materialize` 默认离线，只有非正式 staging 时才可显式加 `--allow-download`。
`env-report --seed` 把 manifest 锁到一个 declared seed；CUDA comparison 会校验该 manifest
的 config/selected seed/SHA256、clean Git、`PRORM_GIT_COMMIT`、image SHA、Slurm account
`sigroup`、partition 和唯一 GPU model，并要求 artifact producer Git/image 与它一致。这里的
旧 `SRM_GIT_COMMIT` 只作为现有 Slurm script 接受的 compatibility environment key。
正式运行使用 HPC 脚本，
不得在登录节点手工执行这组 CUDA 命令。

五 seed comparison 聚合：

```bash
prorm aggregate-results configs/main.yaml outputs/main/aggregate.json \
  outputs/main/seed-20260722/comparison.json \
  outputs/main/seed-20260723/comparison.json \
  outputs/main/seed-20260724/comparison.json \
  outputs/main/seed-20260725/comparison.json \
  outputs/main/seed-20260726/comparison.json \
  --rollouts \
  outputs/main/seed-20260722/rollout.json \
  outputs/main/seed-20260723/rollout.json \
  outputs/main/seed-20260724/rollout.json \
  outputs/main/seed-20260725/rollout.json \
  outputs/main/seed-20260726/rollout.json
```

`aggregate-results` 同时聚合 main-damping held-out metric 与 prompt-level
`test_rollout_improvement`。它要求两组输入的 seed 与 config 完全相同，并验证 artifact
metadata SHA、comparison bytes SHA、rollout JSONL SHA；还会从每个 comparison 同目录重新
读取 `run-manifest.json`，核对 manifest SHA、config、`selected_seed` 与 comparison 记录的
environment identity。任何交叉 artifact/comparison/manifest、缺失或篡改都硬失败。

五个 seed 的 formal identity 必须逐字段相同：Git commit、image SHA256、Slurm account/partition、
唯一 GPU model。聚合器不允许把不同 commit、image、partition 或 GPU 型号的结果放入同一
paired table，并把这份共享 identity 写入 `aggregate.json.environment_identity`。

同一个命令还遍历 config 声明的**每个** damping multiplier，对五 seed test local regret
做配对聚合并写入 `damping_evidence`。每档记录：

- `status=ok|incomplete`、所有 ProRM+ PCG 是否收敛；
- 完整时的 paired local-regret summary 与 `local_regret_nonreversal`；该字段只在 paired
  mean 严格 `<0` 时为 true，exact zero 为 false；
- 不完整时保留逐 seed failure record；solver exception 含 `failure_type/message`，已产出结果
  则保留 `pcg_converged`，不得丢弃失败 seed 后聚合。

随后按第 9 节已经固定的规则写出：

```text
pre_registered_evidence.status = passed | not_passed
pre_registered_evidence.supports_pre_registered_claim = true | false
pre_registered_evidence.criteria = {
  main_local_regret_negative_with_ci,
  main_direction_error_negative_with_ci,
  main_fisher_cosine_positive,
  matched_kl_rollout_positive_with_ci,
  sensitivity_local_regret_nonreversal,
  all_pcg_converged,
  all_measured_kl_updates_converged
}
```

rollout direction PCG、measured-KL convergence 或 KL tolerance 不满足时，输入在写 aggregate
前即被拒绝；sensitivity PCG failure 则保留在 evidence 中并令状态 `not_passed`。`passed`
只表示这组结果满足预注册工程判据，不是 p-value、“统计显著”标签或 population theorem 的
证明。不得根据 `damping_evidence` 事后选择新主阻尼；改变判据/config 必须开启新实验。

Artifact 目录包含 `metadata.json`、`tensors.safetensors`、`prompts.jsonl`、
`candidates.jsonl`、`training_edges.jsonl` 和 `evaluation_edges.jsonl`。rollout 额外生成
`matched-kl-rollout/v2` JSON 和同目录 `updated_rollouts.jsonl`；后者写出
`updated-rollout/v2`，包含 zero-B reference、BT-MLE 和 ProRM+ 三路
candidate-index-aligned records。旧 v1 result 中的 `srm_plus` 仅是兼容标识，读取时会归一化为
`prorm_plus`；论文和结果解释统一使用 ProRM+。

## 11. 每个 run 必存证据

- Git commit 与 dirty flag、完整 normalized config、config SHA256；
- `selected_seed`、base seed 和 prompt split/candidate generation/LoRA-A/annotation/
  reward-head/minibatch/rollout named seeds；
- dataset/model/tokenizer ID、commit revision、chat-template hash；
- LoRA-A state SHA256、B 参数 layout、zero-B no-op error；
- `train_reward_class_projection`：train-only prompt-centered reward-class projection 的
  `target_centered_rms`、`residual_rmse`、`relative_residual`、centering/solver identity；不保存
  fitted weight 或 oracle target；
- prompt/candidate/edge JSONL hash、safetensors hash、split prompt IDs；
- artifact producer Git/image digest；formal environment 提供该身份时，consumer 必须逐字节
  匹配，不能跨 commit/image 复用；
- Python、PyTorch、Transformers、PEFT、Datasets、CUDA/cuDNN、GPU 信息；
- GPU smoke 的 Transformers `==4.52.3` / Qwen3 class 验收、`pip check` 和排序后
  `pip freeze`；
- Slurm job/account/partition/node、镜像路径与 SHA256；
- comparison 与 rollout 绑定的 `run-manifest.json` bytes-level SHA256 与 formal environment
  identity；rollout 还会将当前执行进程与 comparison identity 逐字段匹配；
- semantic config hash、raw config file SHA256、`pcg_dtype`、iteration ceiling 与 residual
  verification interval；
- Fisher mean diagonal、relative/absolute damping、PCG iterations、true residual norm/
  relative residual，以及 schema 已序列化处的 convergence reason；训练 evidence 还固定
  FP64-to-FP32 envelope boundary；
- head init/final SHA256、固定 step 数、validation/test policy metrics 与描述性的 prediction
  NLL/probability-MAE/accuracy；
- shared KL probe IDs、每 learner line-search 轨迹摘要、实际 KL、rollout seed；
- artifact `metadata.json`、`comparison.json`、`updated_rollouts.jsonl` 的 bytes-level SHA256。

跨 seed aggregate 还必须保存并复核共享 Git commit、image SHA256、account、partition 和 GPU model；
其中任一不一致都不得产生 aggregate。

Manifest 只读取明确 allowlist，不得序列化完整 environment。HF/GitHub/W&B credential 不得
进入 config、metadata、stdout、Slurm log 或 artifact evidence。

## 12. 历史外部鲁棒性提案：CoVal（非当前 Phase 2）

早期协议曾把这个条件性 CoVal extension 称为“Phase 2”；该名称现已废止，避免与第 13
节正在执行的 common-global-beta Phase 2 冲突。只有 Phase 1 主链通过后才启动固定 revision
的 CoVal world-ranking 实验。CoVal 的四个
candidate 是有限支持、非 on-policy 样本；固定有限 label 数只能识别 logit series 的截断，
因此实验必须称为 **candidate-restricted truncated ProRM+ robustness**。

保留 annotator identity 仅用于防止重复计数，不作画像。ties、最低 label 数筛选、保留率和
selection analysis 必须完整报告。若给四 candidate 定义 policy probability
$\bar\pi(j\mid x)$，无序边 $\{j,k\}$ 的权重是
$2\bar\pi(j\mid x)\bar\pi(k\mid x)$；不得无权枚举六条边后仍称为原 candidate-policy
objective。该阶段检验现实鲁棒性，不证明 Phase 1 的 population theorem。

锁定的 Phase 1 结果为 `not_passed`，因此 CoVal 不作为本协议的 confirmatory continuation
启动。后续 CoVal、容量扩大或新 KL 预算实验必须明确标为 exploratory，或以新的 design identity
重新预注册。

## 13. 下一实验规格概要：global-beta + outcome-blind pilot

下一实验必须建立新 design identity；它不追加、覆盖或重新聚合 Phase 1。核心变化是把
fixed-beta ProRM 对齐为主 deployment，而把 learner-specific matched-KL 保留为次级分析。

Phase 2 的 operational-oracle 坐标也必须跨 seed 固定，不再对当前 seed 重新拟合。锁定

$$
b_0=-4.500244140625,\qquad
\tau_0=2.7715682983398438,\qquad
r^*=\frac{\log 3}{2}\tanh((R_{\rm oracle}-b_0)/\tau_0).
$$

`(b_0,tau_0)` 是完全排除于 Phase 2 的五个 Phase-1 seeds
`20260722`–`20260726` 各自 train-only transform 的 componentwise median。base/overlay
config 同时绑定 Phase-1 semantic config hash 与五份 artifact `metadata.json` SHA256；
materializer 直接构造同一个 `RobustOracleTransform`，不得读取当前 seed 的 raw scores
重新决定 `b` 或 `tau`。artifact metadata 必须记录 `mode=frozen_global`、数值、聚合规则、
source split/seeds/hashes，以及 `oracle_transform_fitted_on_current_seed=false`。

Phase 2 的 policy tangent 同样跨 seed 固定。所有 materialization 与 rollout reload 都使用
排除的最小 Phase-1 seed `20260722` 对应的 `policy_lora_a` named seed
`946081152281754541`，并硬校验
`A_SHA256=a2b5804109396f76b96cde98d1e2060f175a47724b1ca9fef317c7a10cb9a838`。
config 和 artifact metadata 绑定 source seed、named stream、Phase-1 config hash、source
artifact metadata hash、effective seed 与 observed A hash；当前实验 seed 派生出的 LoRA-A
seed 只记录为未使用的诊断。这样跨 seed 改变的是抽样与标注，而不是 policy class。
Phase 1 main 配置不增加该字段，继续保留历史 per-seed-A 语义。

### 13.1 Pilot 的唯一信息边界

当前三个永久排除 seed `20260801,20260802,20260803` 上运行两个
`confirmatory=false` 的 target-free 阶段：
`pilot_phase=calibration` 只产生 seed-wise beta candidates；
`pilot_phase=freeze` 在新 design identity 中给所有 seeds/arms 部署同一个
`frozen_global_beta`。两个阶段都只允许产生工程诊断：

- train-only optimization、数值收敛、rank 与局部正控；
- train-only beta calibration candidate；
- 各 arm 的 response-token、EOS、达到长度上限比例，以及 on-policy KL 的 mean/p95/p99/max；
- source/config/artifact/environment/output hashes。

Pilot 不调用 held-out evaluator，不开启 final oracle-scoring session，不计算或发布 reward、
utility、regret 或 learner ordering，也不序列化 prompt/response text、token IDs、head weights
或 policy direction vectors。它不是三 seed 的小型效果实验；其 seeds 永久排除在正式统计之外。

### 13.2 Pilot candidate 与正式 global beta

pilot seed `s` 只用 train operational-oracle rewards、train candidate graph 与 train Fisher
构造

$$
u_{*,s}^{tr}=(F_{{\rm train},s}+\lambda I)^{-1}g_{*,s}^{tr},
\qquad
\widetilde\beta_s
=
\sqrt{
\frac{(u_{*,s}^{tr})^\top F_{{\rm train},s}u_{*,s}^{tr}}
{2K_{\rm cal}}
},
\qquad K_{\rm cal}=0.003.
$$

正式实验不再使用 seed-conditional beta。calibration pilot 先给出

$$
\beta_{\rm base}
=\max_{s\in\mathcal S_{\rm pilot}}\widetilde\beta_s,
\qquad
\beta^{(k)}=2^k\beta_{\rm base}.
$$

calibration aggregate 严格校验三个 result/sidecar 的 schema、SHA256、无 outcome 泄漏和
完整 seed 集，再取上述最大值。随后必须建立新的 freeze identity，绑定 calibration
aggregate 的 bytes SHA256，并在三个 seeds 的全部 arms 上使用完全相同的 beta。若
worst-arm 非长度 KL safety 不通过，只能建立新的 freeze identity，在预先声明的序列
`{beta_base, 2 beta_base, 4 beta_base, ...}` 中取下一个值；不得在同一个
job/identity 中重调。
首个 freeze 的 `beta_source_aggregate_sha256` 绑定 calibration aggregate；第 `k>0`
个 freeze 必须改为绑定**紧邻的上一个 freeze aggregate**，且后者必须证明同一 horizon
下只有非长度 safety 失败、`selection_accepted=false`、`next_action` 为 double-beta，
并满足 `next_global_beta=当前 beta` 与 grid index `k-1 -> k`。不得从 calibration
直接跳到 `2 beta_base` 或跨过任何 grid point。horizon 的 parent hash 独立保留为接受该
horizon 的 calibration aggregate。
预注册阈值为 mean KL `0.02`、prompt-mean KL p95 `0.02`、p99 `0.05`、
prompt maximum `0.10`、per-sequence maximum `0.20` 与 maximum-length rate `0.05`。
response horizon 只能沿 `[256, 512, 1024]` 递增。初始 identity 使用 256；若任一
seed/arm 的长度门槛失败，必须从下一个 horizon 重新运行 calibration 与 freeze。新
calibration identity 必须绑定上一个失败 aggregate 的 SHA256，并声明
`previous_horizon_failed_length_gate=true`。若 1024 仍失败则停止并修改协议，不能静默扩展。
令

$$
k_*=\min\{k\ge0:\beta^{(k)}\text{ 的 freeze 通过全部冻结 gate}\},
\qquad
\beta_0=2^{k_*}\beta_{\rm base}.
$$

只有该集合非空时，strict freeze aggregate 才定义可用于 confirmatory 的
`beta_0`；否则停止并修改协议。随后才把这个单一 scalar、response
horizon、最大长度率门槛、KL tail 门槛、优化 tolerance 与全部 hashes 写入新的 confirmatory
config。所有正式 seeds 和 BT-MLE、ProRM+、zero-B、oracle-step arms 共用同一个
`beta_0`；禁止 learner-specific 或 formal-seed-specific line search、norm normalization
与 beta calibration。正式 beta sensitivity 只能使用预注册的
`c in {0.5, 2.0}`，并对所有 sensitivity seeds/arms 直接部署
`beta=c*beta_0`；不得读取该 seed 的 curvature 重新定标。seed-conditional `K_cal`
曲线仅属于 pilot train-only 尺度诊断，不能进入正式实验。

### 13.3 正式训练与直接部署

BT-MLE 与 ProRM+ 从相同 zero head 出发，但分别通过各自 full-data、post-update、unclipped
gradient 相对 zero-init gradient 的连续一阶门槛。ProRM+ 的正式检查使用 cold-start FP64
PCG。720-step head 仅作为 compute-matched secondary checkpoint；validation/test 不得选择
stopping time 或 checkpoint。

正式 train direction 直接部署：

$$
d_\ell^{deploy}
=\frac1{\beta_0}
(F_{\mathrm{train}}+\lambda I)^{-1}g_{\ell}^{\mathrm{train}},
\qquad \ell\in\{\mathrm{BT},\mathrm{ProRM+},\mathrm{oracle}\}.
$$

test Fisher/moment 不进入该式。每个 updated policy 在自身 trajectories 上计算

$$
J_\ell^*
=\mathbb E_{\pi_\ell}[r^*]
-\beta_0D_{\mathrm{KL}}(\pi_\ell\Vert\pi_0).
$$

fixed-history $D_{\mathrm{KL}}(\pi_0\Vert\pi_\ell)$、fixed-K constrained regret、Fisher
cosine 与 learner-specific matched-KL rollout 均为 secondary diagnostics。正式值超过冻结的
mean/tail KL safety gate 时整 seed fail closed，不得缩放或重选 beta。

### 13.4 标签方差、成本与控制臂

noisy primary 对每条 canonical edge 独立生成 `R=4` 份 `gamma=0.9` randomized estimates
并取均值。均值严格无偏且 conditional variance 为单份的四分之一；BT-MLE 使用四份 replicate
的全部 raw Bernoulli labels。不得合并后重算一个 `h`、硬截断、clip 或静默 retry。

`gamma=0.9` 时每份 `E[N]=10`，因此 canonical-edge `R=4` arm 每 prompt 平均需要 40 个
Bernoulli annotations，且 geometric tail 无上界；all-six-pairs arm 平均需要 240 个。这是
精确无偏方案的主要实际成本，必须单独报告 annotation counts 与 tail。exact
`h=Delta r*` ProRM+、exact-soft BT、direct oracle identity 与 `d=256` ridge-free tangent
用于区分 label noise、reward-class misspecification、数值误差和 full-tangent ridge 影响。

这里不能把“有限方差”写成“轻尾”。对这个 estimator 在 `p*=0.25/0.75` 边界的 tail
计算，二阶矩比例是 `max(p*,1-p*)/gamma=0.75/0.9<1`，而四阶矩比例是
`max(p*,1-p*)/gamma^3=0.75/0.9^3>1`；因此单份 estimator 二阶矩有限但四阶矩发散。
`R=4` 只把方差除以 4，不改变 tail exponent。项目不作 sub-Gaussian 或
finite-fourth-moment 声称。

post-recovery 与正式运行必须在 `label_stream` 中写入预注册的
`repeated-label-tail-diagnostics/v1`。对 replicate counts、`abs(replicate_h)`、
`abs(mean_h)` 分别记录 sample size 及 empirical `p50/p90/p95/p99/max`；quantile 固定为
nearest-rank：升序后取一基 `ceil(q*n)`，不插值。record 只能含标量与源 tensor SHA256，
其 canonical SHA 必须进入 `label_stream_sha256`。这些值严格为 descriptive-only：
不得用于 clipping、head/beta/seed selection、acceptance gate、retry 或样本删除。正式统计
单位仍是 paired seed，而不是这些 label-level order statistics。

all-six-pairs arm 是 prompt-level U-statistic：复用四个 iid candidates，不增加 generation/oracle
forward，但六条边共享 nodes，必须按 prompt 聚类，不能当作六倍独立样本。

### 13.5 Prompt/template 契约

Phase 2 的 Qwen2.5 policy 对完整 raw prompt 使用自己的 tokenizer/chat template，
`truncation=False`。对 pinned MultiPref 的 5,323 个 unique prompts 做 deterministic local
precheck 后，88 个超过 1024 policy tokens（`1.65%`），剩余 eligible pool 为 5,235。旧的
“先 seeded shuffle/split、后 fail-closed length check”会使三个 pilot seeds 分别选中
39、34、36 个超限 prompt，无法进入有效 pilot。

因此 Phase 2 必须先对全部 unique prompts 完整渲染计数，建立 `<=1024` eligible pool，再做
seeded shuffle/split；不得再沿用 Phase 1 的 384-token 左截断，也不得在 selected prompt 上
静默截断。metadata 必须记录 unique/eligible/excluded/selected counts、对应 prompt-ID list
hashes；每条 selected metadata 与 trajectory 还要绑定 raw-text hash、policy token count、
prompt-prefix hash、cap 和 `truncated=false`。consumer 必须复核这些值。Phase 2 prompt
population 明确定义为 length-eligible MultiPref subset，而不是完整 MultiPref。

Skywork Qwen3 RM 对**同一份 raw prompt 加 assistant response**使用自己的 pinned Qwen3
tokenizer/chat template 独立重渲染。Qwen2.5 的 template/token IDs 不得传给 Qwen3。这样允许
policy 与 RM 属于不同基座家族，同时避免模板混用或语义内容错位。

上述 5,323/88/5,235 与 39/34/36 是固定输入/tokenizer 下的本地可复现 preflight evidence，
不属于 pilot outcome，也不能支持任何 learner 效果结论。

### 13.6 Endpoint、统计与外部有效性

正式正面结论要求以下证据的交集：

1. held-out fixed-`beta_0` local regret 支持 ProRM+；
2. `utility(ProRM+) - utility(BT-MLE)` paired interval 下界大于 0；
3. `utility(ProRM+) - utility(zero-B)` paired interval 下界大于 0；
4. `utility(oracle-step) - utility(zero-B)` paired interval 下界大于 0；
5. 所有 optimization、rank、positive-control、KL/horizon、identity 与 numerical gates 通过。

每个 prompt 先平均 candidates，每个 seed 只贡献一个 paired scalar；candidate/prompt 不充当
独立 seed。正式 campaign 使用顺序锁定的 30 个全新 paired seeds
`20260901`–`20260930`，排除五个 Phase-1 seeds、全部 pilot seeds 与任何参与设计修改的 seed。

对每个 endpoint `k`，正式 estimand 固定为

$$
\mu_k
=
\mathbb E_{\mathrm{RNG}}\!\left[
\Delta_k
\mid
\text{冻结的 eligible MultiPref pool、models、oracle、design}
\right].
$$

四个正向 endpoint 构成一个 intersection-union test：

$$
H_0=\bigcup_k\{\mu_k\le0\},
\qquad
H_1=\bigcap_k\{\mu_k>0\}.
$$

每个 component 使用 two-sided 95% paired-seed percentile interval，要求下界严格大于
0；其 effective one-sided level 为 `0.025`。对这个单一合取主张不需要 Bonferroni，
但不得把四个 endpoint 另行解释成未经 multiplicity control 的独立正面主张。该区间针对
冻结系统下的 RNG expectation，不外推为任意人类 prompt population 的 CI。`n=30`
时，normal approximation 下单个 component 的 80% power MDE 约为 `0.53` 个 paired
standard deviations；合取检验的 power 由最弱 endpoint 决定。

exact 30 中每个 seed 必须有一个 terminal slot：合法 result 或 immutable failure
manifest。正式 ledger policy 是 `single_predeclared_attempt_no_retry`：每个 seed 只有
`attempt-1`，`recoveries/` 必须为空，不允许 retry、requeue、replacement seed 或 optional
stopping。submitter 在第一次 `sbatch` 前将任务 `0..29`、seeds
`20260901..20260930`、单次 attempt 与完整固定波次原子写入 immutable
`campaign-plan.json`。波次依次为 `0-3%2,4-7%2,...,24-27%2,28-29%2`，固定最多 4
个 submitted tasks、2 个 running tasks，以满足实测 HPC4
`l20_qos MaxSubmitJobsPU=4`。只有先前所有 wave tasks 都有合法 terminal bundle 才能提交
下一 wave，且判定与 success/failure、效果值无关；因此失败 wave 不停止后续预注册
attempt。重复执行同一提交命令只会恢复同一确定性 held wave 或提交唯一合格的下一 wave，
不能产生 retry/replacement。任一失败 slot 令 campaign
`not_passed_due_to_seed_failure` 且不计算 primary CI。若 30 个结果都合法但效果未通过，
仍计算并保留区间，科学状态为 `not_passed`。

每次 `sbatch` 前还必须原子发布并 fsync 当前 `admissions/wave-<index>.json`。wave 0
绑定空前驱；后续 receipt 按序 hash-bind 前一 admission、前一 submission，以及前一
wave 全部 terminal manifest 与 marker。submission v3 再绑定该 receipt、原始 held
`scontrol` 记录及其规范化资源身份。caller walltime、`l20_qos`、`%2`、CPU、内存、节点与
GPU 必须和 plan 精确一致。确定性 job name 在新提交前同时查询 `squeue` 与历史 `sacct`；
若存在未注册历史 identity，则 fail closed，绝不补提或替换。

正式 CPU finalizer 必须调用统一终态入口，而不是绕过 terminal-slot 检查直接运行
`phase2-aggregate`：

```bash
bash scripts/hpc4/submit_phase2_campaign_finalize.sh \
  configs/REPLACE_WITH_CONFIRMATORY_OVERLAY.yaml \
  configs/REPLACE_WITH_CONFIRMATORY_BASE.yaml \
  "${DESIGN_ROOT}/campaign-final/phase2-campaign-terminal.json" \
  "${DESIGN_ROOT}/campaign-final/phase2-primary-aggregate.json" \
  amd \
  REPLACE_WITH_WALLTIME
```

若任一 terminal 是 failure manifest，`PRIMARY_AGGREGATE_JSON` 必须保持不存在且不计算 CI；
若 30 个 terminal 全是合法 success result，finalizer 才在该保留路径发布 primary aggregate。
已实现的 CPU wrapper 会在 submission 与 compute 两侧绑定 commit、image、inventory 和
base/overlay identities，并由 registry resolver 自动选择且验证 exact-30 terminal heads；
调用者不得手选 30 个输入。直接在 login node 运行底层 Python CLI 不是正式流程。

终态所有权也固定：canonical job directory 含 `FAILURE_PENDING` 时，只能调用
`terminalize_phase2_compute_failure.sh`；若作业在原子 rename 前被 scheduler 硬终止、
canonical job directory 不存在，则用一条 terminal non-success `sacct` root record 调用
`terminalize_phase2_scheduler_failure.sh`。两者都只终结 `attempt-1`，不授权新 attempt。
完整命令、classification schema 与验收步骤见 [HPC4 runbook](hpc4.md)。

借鉴 AuxDPO 的是证据架构：analytic misspecification example、matched data/compute、
capacity/sample-size stress、exactly 30 preregistered paired formal seeds、
base/oracle controls，以及单独的 ID/OOD/人类 evaluation。不得照搬其 auxiliary
null-space parameterization，也不把 IPO/DPOP 设为本项目
reward-model 主 baseline；这里的主比较始终是 repeated-label BT-MLE vs ProRM+。capacity、
all-six、frozen-global-beta multiplier sensitivity 与 OOD/human evaluation 都是
secondary/external-validity
experiments，不能替代上述 common-`beta_0` controlled mechanism test。

pilot evidence 不构成正式结果。pilot 通过后才创建新的 confirmatory config；其必须绑定
`beta_0`、完整有序的 30-seed 列表 `20260901`–`20260930`、源 artifact identity、
聚合判据、全部数值正控及
response-horizon/KL gates。任何值都不能从 confirmatory held-out 或 rollout outcomes
反向选择。Phase 1 的权威状态继续是 `not_passed`。
