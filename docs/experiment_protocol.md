# ProRM+ 固定实验协议

本文是实验执行与结果判定规格。数学定义以 [theory.md](theory.md) 为准。
本文在观察正式结果前冻结，不因结果改写；执行记录与权威结果见
[phase1_results.md](phase1_results.md)。
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

## 12. Phase 2：CoVal 人类 robustness

只有 Phase 1 主链通过后才启动固定 revision 的 CoVal world-ranking 实验。CoVal 的四个
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

## 13. 下一实验规格概要：common-beta + train-only oracle

下一实验必须建立新 design identity；它不追加、覆盖或重新聚合 Phase 1。核心变化是把
fixed-beta ProRM 对齐为主 deployment，而把 matched-KL 保留为次级分析。

1. **Train-only calibration。** 只用 train operational-oracle rewards、train candidate
   graph 与 train Fisher 构造
   $u_*^{tr}=(F_{\rm train}+\lambda I)^{-1}g_*^{tr}$。主尺度固定
   `K_cal=0.003`，并解析设定

   $$
   \beta_{\rm common}
   =
   \sqrt{
   \frac{(u_*^{tr})^\top F_{\rm train}u_*^{tr}}
   {2K_{\rm cal}}
   }.
   $$

   这是“每个 seed 一个、所有 policy arms 共享”的 scalar；校准规则跨 seed 固定，实际值允许
   随 train sample 改变。选择过程不得读取 validation/test oracle reward、held-out geometry
   或 rollout。
2. **一个共同 beta。** `beta_common` 在 learner 训练结果和 held-out endpoint 打开前冻结，
   并对 BT-MLE、ProRM+ 和 oracle calibration direction 完全相同。禁止为两个 learner
   分别 line-search、分别选择 beta 或按各自 norm 归一化。
3. **Train direction 直接部署。**

   $$
   d_\ell^{deploy}
   =\frac1{\beta_{\mathrm{common}}}
   (F_{\mathrm{train}}+\lambda I)^{-1}g_{\ell}^{\mathrm{train}},
   \qquad \ell\in\{\mathrm{BT},\mathrm{ProRM+}\}.
   $$

   test Fisher/moment 不进入该式。
4. **KL 只测量，不回调。** 每个 updated policy 生成自己的 trajectories，在这些 histories
   上逐 token 计算完整 vocabulary
   $D_{\mathrm{KL}}(\pi_{d_\ell}(\cdot|h)\Vert\pi_0(\cdot|h))$，按 sequence 求和并保留
   per-sequence 值；它进入

   $$
   J_\ell^*=\mathbb E[r^*]-\beta_{\rm common}
   D_{\rm KL}(\pi_{d_\ell}\Vert\pi_0).
   $$

   固定 reference histories 上的
   $D_{\mathrm{KL}}(\pi_0\Vert\pi_{d_\ell})$ 继续作为 secondary diagnostic。主 safety cap
   固定为 policy-to-reference KL `0.02`；超过即 fail closed，不得根据 learner 身份重调
   `beta_common`。
5. **Endpoint 层级。** fixed-beta held-out regret 与 common-beta rollout 是主 endpoint；
   fixed-K constrained regret、Fisher cosine 和 learner-specific matched-KL rollout 是
   次级机制/尺度诊断。held-out oracle 只能在 `beta_common`、heads 和 deployed directions
   全部冻结后用于 evaluation。
6. **正控与统计。** 同时部署 train-oracle direction 作为 local-to-finite-policy positive
   control；其 gap 只能称为 oracle-step reference gap，不是 global optimum regret。候选先在
   prompt 内平均，prompt 才是 rollout 实验单位。`K_cal={0.001,0.01}` 只作 sensitivity。
   已观察过的五个 Phase-1 seeds 只作 estimand audit；confirmatory 主张必须使用新冻结的十
   seed campaign。
7. **Repeated-label variance control。** 新 noisy-label 主臂对每条 edge 独立生成
   `R=4` 份 `gamma=0.9` randomized estimates 并取均值。该均值仍严格无偏，conditional
   variance 为单份的四分之一；BT-MLE 使用对应的全部 raw Bernoulli labels。不得把四份
   labels 合并后按一个 truncation level 重算 `h`。同时运行 `h=Delta r*` 的
   zero-noise positive control。
8. **Natural-pair efficiency arm。** 四个现有 iid candidates 给出六条无序 pair。all-six
   arm 与 `R=4` 首轮分开运行，在不增加 generation/oracle forward 的前提下使用完整
   prompt-level U-statistic；annotation 数约六倍，reward-head edge work 约六倍，
   Fisher nodes/PCG 维度不变。六条边共享 nodes，统计上必须按 prompt 聚类，不能宣称六倍
   独立样本。
9. **Response horizon。** 新 design 的 candidate 与 rollout 上限固定为
   `max_response_tokens=256`，并逐 arm 报告 EOS、达到上限的比例和 response-token
   分布。旧 Phase 1 的 `128` token action space 保持原样，不用新上限重写旧结果。
10. **Ridge positive control。** 主 full tangent 继续共享同一
    $H_\lambda=F_{\rm train}+\lambda I$；另加一个 `d<n_F` 的低维 tangent arm，检验
    $\lambda\to0$ ordering。有限 rollout 的 oracle arm 只称 `oracle-step reference`，
    因为主 utility 不含显式 parameter-space ridge。

以上 calibration、KL 方向、主尺度和 safety cap 已冻结。正式提交前还需把新的十个 seed、
源 artifact identity、聚合判据和 positive-control gate 写入新 config；
这些值不能从新 campaign 的 test rollout 反向选择。
