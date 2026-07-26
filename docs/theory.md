# ProRM/ProRM+ 理论规格：从全局 policy utility 到可训练 Fisher–GMM

论文标题固定为：

> **Prospective Reward Modeling, Then Policy Optimization: Training Reward Models by Downstream Policy Regret**

本文档是仓库的数学源规范。它固定 ProRM 与 ProRM+ 的含义、全部常数因子、population
identity、有限样本正则化以及工程近似的边界。实验设计见
[experiment_protocol.md](experiment_protocol.md)，三边四响应解析构造见
[closed_form_example.md](closed_form_example.md)。

## 0. 名称与结论层级

Prospective Reward Modeling 的出发点是：

> Reward model 应按它将诱导的未来 policy optimization 来训练，而不应只按它解释过去
> preference labels 的能力来训练。

两个名称严格对应两个数学层级：

| 名称 | 固定含义 |
|---|---|
| **ProRM** | 含不可观测目标 reward 的理想 population/local downstream-regret loss |
| **ProRM+** | 用 repeated labels 识别该目标 moment，并以 Fisher–GMM dual、ridge 与 PCG 实现的可训练方法 |

“+”表示从不可观测理想 target 到可训练 moment method 的识别与优化闭环。ProRM 不是一个
单独实现的 baseline。正式训练对照始终是 repeated-label BT-MLE 与 ProRM+。

整条推导是：

```text
global downstream utility
  -> local quadratic policy problem
  -> ideal ProRM regret in Fisher geometry
  -> natural-pair moment identity
  -> repeated-label unbiased margin h
  -> observable ProRM+ Fisher-GMM dual
  -> damped empirical objective + PCG
  -> held-out geometry + matched measured-KL rollout
```

最常用符号如下。除非另行说明，node expectation 包含
`x ~ rho, y ~ pi_0(.|x)`。

| 符号 | 含义 |
|---|---|
| `rho` | 固定的目标 prompt 分布 |
| `pi_0=pi_{theta_0}` | 生成候选并定义局部几何的 reference policy |
| `theta` | 下一次 policy optimization 真正允许改变的 tangent 坐标 |
| `s_0(x,y)` | $\nabla_\theta\log\pi_\theta(y\mid x)$ 在 `theta_0` 的 sequence score |
| `A_0 r` | reward 对局部 policy update 的一阶 moment |
| `F_0` | reference policy 在同一 tangent 坐标中的 Fisher |
| `r*`, `r_phi` | 目标/operational-oracle reward 与学习到的 reward model |
| `Q_0` | natural pair law `rho*pi_0*pi_0` |
| `z_0`, `Delta r_phi`, `h` | pair score difference、预测 margin、真实 margin 的无偏估计 |
| `m_phi` | `A_0(r_phi-r*)` 的可观测 repeated-label moment 表达 |
| `beta` | downstream KL regularization 系数，严格为正 |
| `lambda` | finite-sample Fisher ridge，严格为正 |

## 1. 全局 Prospective Reward Modeling 问题

### 1.1 先定义未来 policy utility

给定 reward `r`，下游 KL-regularized policy optimization 定义为

$$
J(\theta;r)
=\mathbb E_{x\sim\rho,y\sim\pi_\theta(\cdot|x)}[r(x,y)]
-\beta\mathbb E_{x\sim\rho}
D_{\mathrm{KL}}\!\left(
\pi_\theta(\cdot|x)\,\Vert\,\pi_0(\cdot|x)
\right),
\qquad \beta>0.
$$

这里的理论正则项方向固定为 **policy-to-reference**：
$D_{\mathrm{KL}}(\pi_\theta\Vert\pi_0)$。后文 Phase 1 line search 实测的是固定
reference histories 上的 **reference-to-updated**
$D_{\mathrm{KL}}(\pi_0\Vert\pi_{\theta_0+\delta})$。两者在 `theta_0` 处具有同一个二阶
Fisher 展开，但在有限步长下不是同一个量。为避免“forward/reverse KL”的命名歧义，本文
凡涉及有限更新都显式写出两个分布的顺序。

全局定义把 downstream optimizer 本身也视为决策问题的一部分。对所讨论的每个 reward，
假设下式的 `argmax` 非空，并预先固定一个确定的 selection map
$\mathsf S(r)\in\arg\max_\theta J(\theta;r)$，其中包括 optimizer initialization 与多解时的
tie-break。令

$$
\theta_r:=\mathsf S(r)\in\arg\max_\theta J(\theta;r)
$$

表示把 `r` 交给下游 optimizer 后得到的 policy。reward model 的真实价值不能由它自己的
训练 loss 定义，而应由该 policy 在目标 reward `r*` 下的效用定义：

$$
U(r)=J(\theta_r;r^*).
$$

因此全局 prospective regret 是

$$
\boxed{
\operatorname{Reg}_{G}(r)
=J(\theta_{r^*};r^*)-J(\theta_r;r^*)
}.
$$

理想问题是 `min_{r in R} Reg_G(r)`。这个定义回答“哪个 reward 会诱导更正确的 policy”，
而 BT-MLE likelihood 回答“哪个 reward 更能解释 preference observations”。当 reward class
misspecified 时，两者没有理由选择同一个近似。

若不固定 selection map，候选 reward 的不同最优 policy 可能在 `r*` 下具有不同效用，
`Reg_G(r)` 就是 set-valued 而不是单个标量。后文的 local Taylor 结论使用趋近 `theta_0`
的局部解分支；它不自动等同于这里任意远处的 global selection。

### 1.2 为什么不能直接训练全局目标

全局定义有两个不可直接实施的部分：

1. `theta_r` 是完整 policy optimization 的解映射，reward learning 与 policy learning 形成
   双层问题；
2. 评价需要目标 reward `r*`，训练时却只观察随机 pairwise preferences。

因此 `Reg_G` 是正确的效用定义，不是本项目直接反向传播的 loss。ProRM 通过局部 Taylor
近似解决第一个障碍；ProRM+ 再通过 repeated-label identification 解决第二个障碍。

### 1.3 研究所针对的维度错配

有限离散化后，把完整 reward 写成 `r in R^m`，reference policy 的可更新 tangent 维数为
`d`，受限 reward class 的有效维数为 `p`。目标设定是

$$
m\gg d\gg p.
$$

第一层压缩来自 policy：大量 reward directions 不改变当前 tangent 中的 update。第二层压缩
来自 reward class：只能在受限函数类中选择代表。这个关系是方法动机，不是实验证明；Phase 1
是否在有限 train candidates 上表现出线性不可表示性，必须由 projection diagnostic 报告。

## 2. 局部二次问题与理想 ProRM loss

### 2.1 Score、reward moment 与 Fisher

定义 reference-policy sequence score

$$
s_0(x,y)
=\nabla_\theta\log\pi_\theta(y\mid x)|_{\theta_0}\in\mathbb R^d,
$$

reward moment operator

$$
A_0r
:=\mathbb E_{x\sim\rho,y\sim\pi_0}[s_0(x,y)r(x,y)],
$$

以及 Fisher

$$
F_0
:=\mathbb E_{x\sim\rho,y\sim\pi_0}
[s_0(x,y)s_0(x,y)^\top].
$$

这些对象必须使用下游实际更新的同一坐标。Phase 1 只更新 fixed-A LoRA-B，所以候选 score、
Fisher、natural direction 与实际写回 policy 的 displacement 全部使用同一 LoRA-B layout、
scale、shape 与 hash。

### 2.2 Local policy optimization

令 `delta=theta-theta_0`。在 reference policy 附近，对期望 reward 作一阶展开、对理论中的
policy-to-reference KL 作二阶展开：

$$
J(\theta_0+\delta;r)
\approx
C(r)+\delta^\top A_0r
-\frac\beta2\delta^\top F_0\delta.
$$

记 `g_r=A_0r`。在 identifiable tangent space 上取最小范数解：

$$
\delta_r
=\beta^{-1}F_0^\dagger g_r.
$$

若 `a` 位于 `Null(F_0)`，则 `a^T s_0=0` 几乎处处。只要 reward 二阶可积，`a^Tg_r=0`，
所以 `g_r` 位于 `Range(F_0)`，伪逆表达良定义。

### 2.3 ProRM 的精确局部 regret

用目标 reward 的局部 objective

$$
G^*(\delta)=g_*^\top\delta-\frac\beta2\delta^\top F_0\delta
$$

评价 learned reward 诱导的 `delta_phi`。理想 ProRM loss 定义为

$$
\mathcal L_{\mathrm{ProRM}}(\phi)
:=G^*(\delta^*)-G^*(\delta_\phi).
$$

对二次函数配方：

$$
G^*(\delta^*)-G^*(\delta_\phi)
=\frac\beta2(\delta_\phi-\delta^*)^\top
F_0(\delta_\phi-\delta^*).
$$

再使用
`delta_phi-delta*=beta^{-1}F_0^dagger(A_0r_phi-A_0r*)`，得到

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|A_0(r_\phi-r^*)\right\|_{F_0^\dagger}^{2}
=\frac1{2\beta}
(g_{r_\phi}-g_*)^\top F_0^\dagger(g_{r_\phi}-g_*)
}.
$$

这里 `||u||^2_{F_0^dagger}:=u^T F_0^dagger u`。这是 population、无阻尼、局部二阶问题中的
精确 identity；它不是任意大 policy update 的全局保证。

#### 2.3.1 Global-to-local remainder

上述 equality 对定义出来的 quadratic local problem 是精确的；相对于原始 nonlinear
policy problem，它是 leading-order expansion。reward expectation 本身通常还有 Hessian：

$$
\mathbb E_{\pi_{\theta_0+\delta}}[r]
=C(r)+g_r^\top\delta+\frac12\delta^\top H_r\delta+O(\|\delta\|^3),
$$

不能把 `H_r` 默认为零。这里的
`Reg_{G,local}` 明确比较随 `beta -> infinity` 趋近 `theta_0` 的 local-maximizer
branches；它不是第 1 节任意 global selection 的无条件展开。先固定
`r*` 与 `r_phi`。若限制在 identifiable tangent space，并假设：

- `theta_0` 位于该坐标域内部，且 `F_0` 在该空间的最小特征值至少为某个 `kappa>0`；
- 对 `r in {r*,r_phi}`，reward expectation 与 KL 在 `theta_0` 的同一邻域三阶连续，
  相应导数与 `g_r` 有共同有限界；
- 对充分大的 `beta`，相应 stationary branch 存在，downstream solver 选择其中留在
  `O(1/beta)` 邻域内的 local maximizer；

则在 `beta -> infinity` 时

$$
\delta_r
=\frac1\beta F_0^{-1}g_r+O(\beta^{-2}),
$$

并且目标 nonlinear utility regret 满足绝对误差展开

$$
\boxed{
\operatorname{Reg}_{G,\mathrm{local}}(r_\phi)
=
\frac1{2\beta}
\|g_{r_\phi}-g_*\|_{F_0^{-1}}^2
+O(\beta^{-2})
}.
$$

常数依赖于 Fisher conditioning、reward/KL derivative bounds 与 moment bounds。若 leading
term 本身为零，只能使用 absolute `O(beta^-2)` statement，不能声称相对误差趋零。若 optimizer
跳到远处另一个 mode，上述 local branch 结论不适用。

上述 statement 首先是对固定 `r*`,`r_phi` 的逐点展开。若 `r_phi=r_phi(beta)`，或要对整个
reward class 作统一 remainder claim，则上面的邻域、`kappa`、三阶导数界、moment 界与
`O(1/beta)` branch bound 都必须在该序列或函数类上一致成立。

pilot calibration candidate 中 `beta` 与 `K_cal^{-1/2}` 同阶，所以 leading regret 为
`O(sqrt(K_cal))`，remainder 为 `O(K_cal)`，非退化情形的相对误差为
`O(sqrt(K_cal))`。正式实验会冻结一个 global `beta_0`；prespecified global-beta multiples
`c in {0.5, 2.0}` 只作正式尺度 sensitivity，并统一部署
`beta=c*beta_0`。seed-conditional `K_cal` 曲线仅用于 pilot train-only 尺度诊断；
它不能为 formal seed 重新选择下游问题或步长。

### 2.4 Projection geometry 与 reward 等价类

在有限表示中，令 `A_theta` 的列为每个 `(x,y)` 的 score，
`D_0=diag(rho(x)pi_0(y|x))`，`B_0=A_theta D_0^(1/2)`。则

$$
F_0=B_0B_0^\top,
$$

且

$$
\boxed{
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}
\left\|
P_{\operatorname{row}(B_0)}
D_0^{1/2}(r_\phi-r^*)
\right\|_2^2
}.
$$

因此：

- prompt 内统一加常数不会改变 update；
- 位于 score 零空间的 reward error 不会被惩罚；
- pointwise MSE、BT-MLE NLL 与 policy regret 是不同几何中的投影；
- 在 misspecified reward class 中，BT-MLE optimum 不保证等于 ProRM optimum；
- 若 `A_0r_1=A_0r_2`，两个 rewards 在当前局部 policy problem 中属于同一等价类。

### 2.5 Estimand hierarchy：fixed-beta 为主，fixed-K 为次

令

$$
u_r=F_0^\dagger g_r,\qquad
\|u\|_{F_0}^2=u^\top F_0u.
$$

本项目的**主 estimand**固定同一个 `beta`，不为每个 reward model 单独重标步长：

$$
\delta_{\beta,r}=\frac{u_r}{\beta},\qquad
\mathcal R_\beta(r_\phi)
=\frac{1}{2\beta}\|u_{r_\phi}-u_*\|_{F_0}^2.
$$

它就是上一节的 ProRM loss，同时惩罚 policy direction 的角度误差和自然梯度范数的校准
误差。乘上全局常数不会改变 reward-class minimizer，但固定 `beta` 是下游决策问题的一部分，
不能在方法之间分别调节后仍称为同一个 ProRM estimand。

另一个合理但不同的决策问题是固定局部 KL 半径 `K`：

$$
\max_\delta g_r^\top\delta
\quad\text{s.t.}\quad
\frac12\delta^\top F_0\delta\le K.
$$

当 `\|u_r\|_{F_0}>0` 时，其正向边界解为

$$
\delta_{K,r}
=\frac{\sqrt{2K}}{\|u_r\|_{F_0}}u_r.
$$

用目标 reward 评价该约束解得到**次级 estimand**

$$
\boxed{
\mathcal R_K^{\mathrm{con}}(r_\phi)
=g_*^\top(\delta_{K,*}-\delta_{K,r_\phi})
=\sqrt{2K}\,\|u_*\|_{F_0}
\left(1-\cos_{F_0}(u_{r_\phi},u_*)\right)
}.
$$

因此 fixed-K constrained regret 与 Fisher cosine 只检验角度；它们会主动消除 learned
direction 的范数误差。若任一 Fisher norm 为零，归一化解和 cosine 均未定义，必须 fail
closed。fixed-K 指标对 trust-region rollout 很有用，但不能替代 fixed-beta ProRM 主
estimand。

三边四响应 [closed-form example](closed_form_example.md) 给出 BT-MLE 与 population ProRM 的解析排序
反转。它只证明理想 population objectives 可以选出不同 reward；它不使用 randomized `h`，
因此不能单独证明 ProRM+ identification。后者必须在 natural `Q_0` 下由下一节的 identity 建立。

## 3. Natural-pair representation

### 3.1 固定 pair law 与方向

不引入人为 edge reweighting。定义

$$
Q_0(dx,dy,dy')
=\rho(dx)\pi_0(dy\mid x)\pi_0(dy'\mid x).
$$

给定 `e=(x,y,y')`，定义

$$
z_0(e)=s_0(x,y)-s_0(x,y'),
\qquad
t_r(e)=\Delta r(e)=r(x,y)-r(x,y').
$$

方向始终为 `left-right`：`a=1` 表示 left/`y` 获胜。交换 edge 时必须同时翻转 `z_0`、
`Delta r`、`h`，并把 `left_wins` 变为 `N-left_wins`。

### 3.2 两个精确 pair identities

以下 identity 需要标准 score regularity：对 `rho`-几乎处处的 `x`，`pi_theta(.|x)` 在
`theta_0` 邻域有不随 `theta` 改变的支持，且允许把参数微分移入归一化积分/求和。另假设

$$
\mathbb E\|s_0\|_2^2<\infty,\qquad
\mathbb E[r^2]<\infty.
$$

于是 score identity 成立，且由 Cauchy--Schwarz 有
$\mathbb E\|s_0r\|_2<\infty$。使用 repeated-label signal 时还需
$\mathbb E\|z_0h\|_2<\infty$；第 4.3 节的 bounded-probability、有限二阶矩设计与
$\mathbb E\|z_0\|_2^2<\infty$ 足以保证这一点。

Score identity 给出

$$
\mathbb E_{y\sim\pi_0(\cdot|x)}[s_0(x,y)\mid x]=0.
$$

因为 `y,y'` 条件独立，展开可得

$$
\mathbb E[z_0t_r\mid x]
=2\mathbb E[s_0r\mid x],
$$

所以

$$
\boxed{A_0r=\frac12\mathbb E_{e\sim Q_0}[z_0(e)t_r(e)]}.
$$

同理，

$$
\boxed{F_0=\frac12\mathbb E_{e\sim Q_0}[z_0(e)z_0(e)^\top]}.
$$

Population 中 node 与 pair 表示相同。工程上固定为：

- 全部 on-policy candidate nodes 估计 Fisher，降低方差；
- canonical labeled edge 估计 reward-error moment；
- 不用 edge endpoint 频率替代原 on-policy node Fisher；
- 两个 finite-sample estimators 不要求数值相等。

Phase 1 中，`P` 个 prompts、每 prompt `M=4` 个 candidates、policy dimension `d`、reward
feature dimension `H` 对应：

```text
S:          (P*M, d)   # all on-policy nodes
Z:          (P, d)     # canonical candidate 0 - candidate 1
left/right: (P, H)     # frozen reward features
Delta r,h:  (P,)
F_hat = S.T @ S / (P*M)
m_hat = Z.T @ (Delta r-h) / (2*P)
```

## 4. Repeated labels identify the target margin

### 4.1 单标签不可能逐 edge 无偏恢复 logit

单位温度 BTL 模型为

$$
a\mid e\sim\operatorname{Bernoulli}(p^*(e)),
\qquad
p^*(e)=\sigma(\Delta r^*(e)).
$$

对仅依赖一个 Bernoulli label 的任意统计量 `H(a)`，

$$
\mathbb E[H(a)\mid e]
=(1-p^*)H(0)+p^*H(1),
$$

它关于 `p*` 必为仿射函数，不可能在区间上等于非线性的 `logit(p*)`。这个命题只排除
“逐 edge、distribution-free、单标签”的无偏 target；它不否定跨 edge 参数共享的 MLE。

### 4.2 Randomized U-statistic

利用级数

$$
\operatorname{logit}(p)
=\sum_{k=1}^{\infty}\frac{p^k-(1-p)^k}{k}.
$$

对同一 edge 获取条件 iid labels `a_1,...,a_N`。记 `S_N=sum_j a_j`，定义

$$
U^+_{k,N}=\frac{\binom{S_N}{k}}{\binom Nk},
\qquad
U^-_{k,N}=\frac{\binom{N-S_N}{k}}{\binom Nk}.
$$

令 `N` 独立于 edge 与 labels，生存概率 `q_k=P(N>=k)>0`。定义

$$
h
=\sum_{k=1}^{N}
\frac{U^+_{k,N}-U^-_{k,N}}{kq_k}.
$$

条件于 `N>=k`，两个 U-statistics 分别无偏估计 `p^k` 与 `(1-p)^k`；`1/q_k` 校正第
`k` 项被计算的生存概率。对 `p in (0,1)`，正、负两部分的期望级数分别为
`sum_k p^k/k` 与 `sum_k (1-p)^k/k`，均有限；因此可分别用 Tonelli theorem 交换期望与
求和，得到

$$
\boxed{
\mathbb E[h\mid e]
=\operatorname{logit}(p^*(e))
=\Delta r^*(e)
}.
$$

实现只使用 `(S_N,N)` 的组合计数，不枚举 label 子集。

这里以及后文“严格无偏”均指具有所声明、真正无界 support 的理想随机变量 `N`。有限精度
PRNG 的单次 inverse-CDF 实现只有有限个可表示状态，因此是该 geometric law 的数值实现，
不是 bit-level 的无限支持分布。这个数值限定不允许人为 hard cap、clip、重采样或丢弃；
后几种操作会引入可观测的设计性截断，必须继续 fail closed。

### 4.3 主实验的随机截断常数

主实验固定

$$
P(N=n)=(1-\gamma)\gamma^{n-1},
\qquad
q_k=\gamma^{k-1},
\qquad
\gamma=0.9.
$$

因此 `E[N]=1/(1-gamma)=10`。oracle transform 保证
`p* in [0.25,0.75]`，并且 `gamma > max(p*,1-p*)`，即 `0.9>0.75`；这满足本实验采用的
有限二阶矩充分条件。

**引理 4.1（geometric truncation 的矩条件）.**
固定一条 edge 及其 `p in (0,1)`，令 `q_k=gamma^(k-1)`。若

$$
\gamma>\max(p,1-p),
$$

则 $\mathbb E[h^2\mid e]<\infty$。此外，只要

$$
\frac{p}{\gamma^3}>1
\quad\text{或}\quad
\frac{1-p}{\gamma^3}>1,
$$

就有 $\mathbb E[|h|^4\mid e]=\infty$。

证明：记

$$
T_k^+
=\frac{\mathbf 1\{N\ge k\}U^+_{k,N}}{kq_k}.
$$

由 $0\le U^+_{k,N}\le1$ 和 U-statistic 无偏性，

$$
\mathbb E[(T_k^+)^2\mid e]
\le \frac{p^k}{k^2q_k}.
$$

所以当 $p<\gamma$ 时，

$$
\sum_{k\ge1}\|T_k^+\|_{L_2}
\le
\sqrt{\gamma}\sum_{k\ge1}
\frac{(p/\gamma)^{k/2}}{k}
<\infty.
$$

对 $T_k^-$ 用 $1-p<\gamma$ 得到同样结论；Minkowski inequality 因而给出
$h=\sum_k(T_k^+-T_k^-)\in L_2$。另一方面，在事件
$\{N=n,S_N=n\}$ 上，

$$
h\ge \frac1{n\gamma^{n-1}},
\qquad
\Pr(N=n,S_N=n\mid e)=(1-\gamma)\gamma^{n-1}p^n.
$$

因此四阶矩至少包含常数倍的

$$
\sum_{n\ge1}\frac{(p/\gamma^3)^n}{n^4},
$$

当 $p/\gamma^3>1$ 时发散；all-losses 事件给出 $1-p$ 的情形。证毕。

无偏与低方差不是同一件事。对当前 estimator 作高精度条件枚举可得：

| `p` | `sd(h|p)`, `gamma=0.9` |
|---:|---:|
| `0.50` | `0.840942` |
| `0.40/0.60` | `0.851791` |
| `0.25/0.75` | `0.934657` |

Phase 1 的 train-only node-centered oracle RMS 约为 `0.24`；对应全六 pair-margin RMS 约为
`0.392`。所以单 edge 单份 `h` 的噪声标准差分别约为这两个尺度的 `3.5–3.9` 倍和
`2.1–2.4` 倍。这不破坏 population identity，但会显著增加 finite-sample moment 方差。

若对同一 edge 独立生成 `R` 份合法 randomized estimates `h_1,...,h_R`，则

$$
\bar h_R=\frac1R\sum_{j=1}^R h_j,\qquad
\mathbb E[\bar h_R|e]=\Delta r^*(e),\qquad
\operatorname{Var}(\bar h_R|e)=\frac1R\operatorname{Var}(h|e).
$$

因此下一 noisy-label 主臂固定 `R=4`：条件标准差减半，仍严格无偏。`gamma=0.9` 时
每份 `E[N]=10`，所以 canonical-edge arm 每 prompt 平均需要 40 个 Bernoulli labels；
all-six-pairs arm 平均需要 240 个，且 geometric tail 无上界。这是精确无偏方法的主要工程
成本，不可用 hard cap 换取固定预算。BT-MLE 必须使用四份 replicate 的全部 Bernoulli
labels。replicate boundaries 必须被 schema 保存；把 labels 直接拼接后按一个更大的 `N`
重算 `h` 不是同一个 estimator。

不得硬截断、clip 大 `N`、按 `N` 重采样或静默丢弃。memory guard 只能使 run fail closed，
不能改变 estimator。

有限二阶矩不意味着 sub-Gaussian 或有限四阶矩。在概率区间端点，这个 estimator 的
all-wins/all-losses 尾项计算中，二阶矩的锁定比例为

$$
\frac{\max(p^*,1-p^*)}{\gamma}
=\frac{0.75}{0.9}
<1,
$$

而四阶矩的对应比例为

$$
\frac{\max(p^*,1-p^*)}{\gamma^3}
=\frac{0.75}{0.9^3}
>1,
$$

所以该锁定设计在端点具有无限四阶矩。`R=4` 独立平均把条件方差除以 4，但不改变尾指数，
也不会凭空得到有限四阶矩。因此理论和论文只能声称严格无偏与有限二阶矩，不能声称
sub-Gaussian concentration 或用经验四阶矩作为稳定性保证。

在任何正式结果产生前，工程协议预先固定一份**纯描述性** tail record。它分别对
`N`、`|h_j|` 与 `|\bar h_4|` 报告 empirical `p50/p90/p95/p99/max`。对样本量 `n`，
nearest-rank 定义为先升序排列，再取一基序号 `ceil(q*n)`；maximum 为第 `n` 个次序统计量，
不做插值。record 只含标量、样本量与源 tensor SHA256，不序列化 labels、`h` 向量或
`mean_h` 向量，并明确 `descriptive_only=true`、clipping/selection/gating 全部为 false。
这些统计量用于诚实呈现 realized heavy tail；它们不得改变样本、训练、beta、seed
资格或正式判定。

随机 `N` 只决定构造 `h` 的成本：

- ProRM+ 中每个 edge 对 moment 恰好贡献一次；
- 不得按 `N` 再给 ProRM+ edge 加权；
- BT-MLE 使用全部原始 Bernoulli labels，等价于按 `N` 累计 likelihood。

### 4.4 固定重复数只能得到截断 target

若每个 edge 固定收集 `L` 个 labels，`S=sum_j a_j`，则

$$
h_L
=\sum_{k=1}^{L}\frac1k
\left[
\frac{\binom Sk}{\binom Lk}
-\frac{\binom{L-S}k}{\binom Lk}
\right].
$$

其期望只等于 logit 级数的前 `L` 项。若 `p* in [epsilon,1-epsilon]`，

$$
\left|
\mathbb E[h_L\mid e]-\Delta r^*(e)
\right|
\le
\frac{2(1-\epsilon)^{L+1}}{\epsilon(L+1)}.
$$

所以固定 `L` 的人类数据实验必须称为 **candidate-restricted truncated ProRM+ robustness**，
不能援引精确无偏 identity。

### 4.5 BTL temperature 与 cardinal scale

Repeated labels 识别的是 preference log-odds。若真实 observation model 为

$$
p^*(e)=\sigma(\Delta u^*(e)/\tau),
$$

则 `h` 无偏识别 `Delta u*/tau`，而不是另有绝对单位的 `Delta u*`。全局常数 `tau` 可以
通过把 downstream penalty 同时改写为 `beta/tau` 吸收；因此 synthetic Phase 1 固定 unit
temperature 并让 transformed oracle reward 与 `beta` 使用同一尺度。

若 annotator/edge temperature 异质，`h` 识别的是 edge-dependent rescaled margin，通常不再
对应一个全局 scalar reward potential。仅有 pairwise choices 时，cardinal utility scale 与
downstream `beta` 不能分别识别。真实人类实验必须显式建模 annotator temperature/mixture，
或把结论限制为所识别 log-odds reward 下的 prospective policy utility，不能静默把它称为
绝对 human utility。

## 5. Observable ProRM+ Fisher–GMM objective

### 5.1 Moment identification

定义预测 margin `Delta r_phi(e)` 以及

$$
m_\phi
:=\frac12\mathbb E_{e,h}
[z_0(e)(\Delta r_\phi(e)-h(e))].
$$

利用 repeated-label identity 与 natural-pair identity：

$$
\begin{aligned}
m_\phi
&=\frac12\mathbb E_{Q_0}
[z_0(\Delta r_\phi-\Delta r^*)]\\
&=A_0(r_\phi-r^*).
\end{aligned}
$$

因此既不需要恢复完整 `r*`，也不需要分别估计 `A_0r_phi` 与 `A_0r*`；直接估计二者之差。

### 5.2 Population equivalence theorem

ProRM loss 可写为

$$
\mathcal L_{\mathrm{ProRM}}(\phi)
=\frac1{2\beta}m_\phi^\top F_0^\dagger m_\phi.
$$

对 `m in Range(F_0)`，Fenchel identity 为

$$
\frac12m^\top F_0^\dagger m
=\max_v\left(v^\top m-\frac12v^\top F_0v\right).
$$

所以 ProRM+ 的 population objective 是

$$
\boxed{
\min_\phi\max_v
\frac1\beta
\left[
v^\top m_\phi
-\frac12v^\top F_0v
\right]
}.
$$

展开两个 expectations：

$$
\min_\phi\max_v
\left\{
\frac1{2\beta}
\mathbb E_{e,h}[(z_0(e)^\top v)(\Delta r_\phi(e)-h(e))]
-\frac1{2\beta}
\mathbb E_{x,y}[(s_0(x,y)^\top v)^2]
\right\}.
$$

在 natural `Q_0`、条件 iid repeated labels、第 3.2 节的 score regularity 与乘积矩条件、
$\mathbb E\|z_0h\|_2<\infty$、局部二阶近似和无阻尼 Fisher 条件下：

$$
\boxed{
\max_v\mathcal J_{\mathrm{ProRM+}}(\phi,v)
=\mathcal L_{\mathrm{ProRM}}(\phi)
=\widetilde{\operatorname{Reg}}(r_\phi)
}.
$$

这是本项目的核心 identity。最小范数 dual witness 为 `v=F_0^dagger m_phi`。

### 5.3 不要混淆 dual witness 与 policy direction

- `u_r=(F+lambda*I)^-1 g_r` 是未除以 `beta` 的 damped natural direction；
- `delta_r=u_r/beta` 才是局部 policy displacement；
- `v=(F+lambda*I)^-1 m` 是 reward-error dual witness；
- population、`lambda=0` 时 `v=beta*(delta_phi-delta*)`。

## 6. Finite-sample ridge ProRM+

令 `S in R^(n_F x d)` 为 node scores，`Z in R^(n_E x d)` 为 edge score differences。固定

$$
\widehat F_0=\frac1{n_F}S^\top S,
\qquad
\widehat m_\phi
=\frac1{2n_E}Z^\top(\Delta r_\phi-h).
$$

由于通常 `d>n_F`，经验 Fisher 必然秩亏。实际训练的 ProRM+ objective 必须显式写成

$$
\boxed{
\min_\phi\max_v
\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right]
},
$$

其中

$$
\lambda
=c\,\operatorname{mean}(\operatorname{diag}\widehat F_0),
\qquad
c>0,\qquad
\operatorname{mean}(\operatorname{diag}\widehat F_0)>0.
$$

最后一个条件排除所有 empirical scores 都为零的退化 tangent；若不满足，不能把 `lambda`
写成严格正数，也不能继续内层 solve，实验必须 fail closed。于是受控路径中 `lambda>0`。

内层唯一解与报告值为

$$
v_\phi^*=(\widehat F_0+\lambda I)^{-1}\widehat m_\phi,
$$

$$
\boxed{
\widehat L_{\mathrm{ProRM+},\lambda}(\phi)
=\frac1{2\beta}\widehat m_\phi^\top
(\widehat F_0+\lambda I)^{-1}\widehat m_\phi
}.
$$

必须区分三个层级：

| 层级 | Fisher | 可以声称什么 |
|---|---|---|
| Population theorem | `F_0^dagger`, `lambda=0` | ProRM+ inner optimum 与 local ProRM regret 精确相等 |
| Finite-sample experiment | `F_hat+lambda*I`, `lambda>0` | Ridge empirical surrogate |
| Sensitivity | `c in {1e-4,1e-3,1e-2}` | 检查结论是否依赖阻尼尺度 |

不得把 empirical ridge objective 称为 population identity 的“精确实现”。

还有一个必须写清楚的对应关系：把
$H_\lambda=\widehat F_0+\lambda I$ 当作下游局部 optimizer 的 metric 时，对应的局部
objective 必须是

$$
g^\top\delta
-\frac{\beta}{2}\delta^\top\widehat F_0\delta
-\frac{\beta\lambda}{2}\|\delta\|_2^2
=g^\top\delta-\frac{\beta}{2}\delta^\top H_\lambda\delta.
$$

此时 $\widehat m^\top H_\lambda^{-1}\widehat m/(2\beta)$ 才是该
**ridge-regularized local optimizer** 的精确 quadratic target。若有限 policy endpoint
仍只报告
$\mathbb E[r^*]-\beta D_{\rm KL}(\pi\Vert\pi_0)$、不把
$\beta\lambda\|\delta\|_2^2/2$ 计入 utility，那么 damped oracle direction 只是一个共同的
算法正控，不是该无 ridge utility 的解析最优解。下一实验因此：

1. 对所有 learner 与 oracle arm 使用完全相同的 $H_\lambda$；
2. 把 oracle comparator 称为 `oracle-step reference`，不称 global optimum；
3. 保留 $\lambda$ sensitivity；
4. 增加一个 $d<n_F$、经验 Fisher 可识别的低维 tangent positive control，直接检查
   $\lambda\to0$ 时的 population mechanism。

### 6.1 Moment 无偏不等于 loss 无偏

对一个不依赖当前样本的固定 `phi`，`\hat m_phi` 是 population moment `m_phi` 的无偏估计。
但即使先把 weighting matrix `W` 当作固定量，

$$
\mathbb E[\widehat m_\phi^\top W\widehat m_\phi]
=m_\phi^\top Wm_\phi
+\operatorname{tr}\!\left(W\operatorname{Var}(\widehat m_\phi)\right).
$$

所以 empirical quadratic 不是 population ProRM loss 的无偏 estimator；估计
`F_hat+lambda I` 的逆、再让训练后的 `phi_hat` 依赖同一批数据，只会进一步增加有限样本差异。
正确表述是：ProRM+ 有一个 unbiased identifying moment，并以 regularized empirical GMM
训练；不能写成“ProRM+ loss 本身无偏”。

这也是必须保留独立 held-out prompts、exact-margin positive control、label-replicate curve
和 sample-size sweep 的原因。用两个独立 moment batches 构造 cross-product 可以消掉固定
`phi,W` 下的对角方差项，但该 estimator 可为负，且无法自动解决 learned `phi`、estimated
Fisher 与 reward-class overfitting；因此它只适合作为诊断，不替换主训练 objective。

### 6.2 Ridge 的坐标依赖性

`lambda*I` 不具任意 tangent reparameterization invariance。
`lambda=c*mean(diag(F_hat))` 只消除统一全局尺度，不能抵消各坐标的非等比例变化。因此
fixed-A seed/state、LoRA alpha、B layout、shape、scale、flatten order 与 hash 都是 empirical
objective 的科学定义，而不只是 provenance。训练与 rollout 必须复用同一坐标。

### 6.3 Train 与 held-out 不是同一个 estimator

| 用途 | Moment/Fisher | Damping |
|---|---|---|
| theorem | population `A_0(r_phi-r*)`, `F_0` | pseudoinverse，`lambda=0` |
| train | canonical-edge `m_hat`, train node `F_hat` | 由 train Fisher 解析 |
| held-out | prompt covariance `g_hat_error`, held-out node Fisher | 每个 split 独立解析 |

Held-out covariance 直接估计 `g_error`，所以没有 pair identity 中的 `1/2`；这不是常数冲突。
更重要的是，held-out metric 会用 held-out moment、held-out Fisher 和 held-out damping
**重新求解** predicted/target directions。它保持 reward head 冻结，但不是把实际部署的
train direction 搬到 held-out split 上打分。这个区别在第 8 节写成显式公式。

## 7. PCG、reported value 与 envelope gradient

### 7.1 Matrix-free dual solve

每个 outer optimizer step 先求解

$$
(\widehat F_0+\lambda I)v=\widehat m_\phi.
$$

PCG 只调用

$$
u\longmapsto\frac1{n_F}S^\top(Su)+\lambda u,
$$

无需形成 `d x d` Fisher。artifact 中的 `S`、`Z`、`h` 仍以 FP32 保存；进入固定 policy
geometry 后，`S/Z/h`、`m_hat`、absolute damping、Fisher matvec、全部 Krylov state、dot/norm
和 true residual 统一提升为 config-locked FP64。Qwen/Skywork forward、冻结 reward feature、
reward head、autograd 与 AdamW 仍为 FP32。

这里的系统是 rank 至多 `n_F` 的 empirical Fisher 加 isotropic damping；unpreconditioned CG
保留其“低秩 + 重复 `lambda` 特征值”结构。坐标级 Jacobi scaling 会把这个重复特征值打散，
因此 controlled path 明确不使用 Jacobi。main train Fisher 有 `1536*4=6144` 个 node，精确
算术中的 Krylov 上界至多是 `rank(S)+1<=6145`；main ceiling 因而取 `8192`，smoke 的
`48*4+1=193` 下界由其 `2048` ceiling 覆盖。rank 上界只说明 ceiling 的规模充分性，不保证
有限精度收敛。

relative tolerance 固定为 `1e-5`。recursive residual 只用于 CG recurrence；每 20 次迭代及
recursive residual 首次过门时显式计算 `r_true=rhs-Ax`。只有 `||r_true||<=rtol||rhs||`
才允许 `converged`；若 recursive residual 假性过门，则从 `r_true` 显式 restart，禁止替换
residual 后沿用旧 Krylov direction。达到 ceiling 时同样以 true residual 作最终判定。
未收敛是 fail-closed condition，不是 warning。

### 7.2 三个 scalar 必须分开

解得 `v` 后，报告的 ridge quadratic 是

$$
\widehat L_{\mathrm{reported}}
=\frac1{2\beta}\widehat m_\phi^\top v.
$$

其精确 envelope gradient 为

$$
\nabla_\phi\widehat L_{\mathrm{reported}}
=\frac1{2\beta n_E}
\sum_i(z_i^\top v)\nabla_\phi\Delta r_{\phi,i}.
$$

若在 autograd 中 detach `v`，必须使用 mean-reduced surrogate

$$
\widehat L_{\mathrm{env}}
=\frac1{n_E}\sum_i
\frac{z_i^\top\operatorname{stopgrad}(v)}{2\beta}
(\Delta r_{\phi,i}-h_i).
$$

同一 full batch、精确 solve 下：

$$
\widehat L_{\mathrm{env}}
=\frac1\beta v^\top\widehat m_\phi
=2\widehat L_{\mathrm{reported}}.
$$

它的数值不是论文 objective；它只产生正确 gradient。完整 saddle diagnostic 为

$$
\widehat L_{\mathrm{saddle}}(\phi,v)
=\frac1\beta
\left[
v^\top\widehat m_\phi
-\frac12v^\top(\widehat F_0+\lambda I)v
\right].
$$

最优 `v` 时，saddle value 等于 reported value；PCG 近似时两者可能不同。仓库分别记录
reported quadratic、saddle diagnostic 和 training surrogate，禁止混用。

实现中 `v` 与 `z_i^T v/(2 beta)` 在 FP64 policy geometry 内计算；这些 scalar envelope
weights 只在进入 FP32 reward-head autograd 前转换一次。该 precision boundary 不改变上述
梯度，只避免把整个 reward learner 与 AdamW 无必要地改成 FP64。

### 7.3 外层更新顺序

每次 optimizer update 固定执行：

```text
all training-edge margins
  -> one full moment m_hat
  -> warm-started FP64 PCG with true-residual acceptance
  -> detach v
  -> microbatch accumulation of one full-data envelope gradient
  -> exactly one optimizer step
  -> recompute margins, moment and dual
```

Microbatch 只改变内存占用，不创建 batch-local moment/Fisher。一个 dual direction 不得跨多个
outer steps stale reuse。

Phase 2 不再用相同的任意固定步数定义两个 estimator。BT-MLE 与 ProRM+ 的曲率和条件数不同；
把两者都停在第 720 步会把 objective 差异与 optimizer error 混在一起。每个 objective 的主
head 必须分别通过冻结的、只读 train data 的一阶条件：

$$
\rho_\ell
=
\frac{\|\nabla_w L_\ell(w_{\rm final})\|_2}
{\max(\|\nabla_w L_\ell(w_0)\|_2,\epsilon)}
\le \tau.
$$

梯度在 optimizer update 后对完整训练集重算，不使用 clipping 后的值；ProRM+ 的 gate 必须
用 cold-start FP64 PCG 和 reported quadratic 的精确 envelope gradient。连续若干次 scheduled
check 通过后才接受第一个合格 iterate，未在最大步数内通过则整个 seed fail closed。validation、
test 与 rollout 不参与停止或 checkpoint 选择。

第 720 步仍保存为两方法共享算力的 secondary checkpoint，但不会覆盖已收敛的主 head。
zero initialization 与 AdamW 路径只定义可复现的 algorithmic tie-break；在没有 rank 证据或
显式投影时，不得声称它给出欧氏 minimum-norm 解。pilot 只用 train-only trajectory 冻结
`tau`、检查间隔、连续通过次数与最大步数，pilot seeds 永久排除在正式统计之外。

## 8. Held-out geometry and downstream evaluation

### 8.1 Prompt-centered held-out moment

每个 held-out prompt 有 `M` 个 candidates。为有限样本中精确消除 prompt reward gauge，使用

$$
\widehat g_r
=\frac1{P(M-1)}
\sum_{i=1}^{P}\sum_{j=1}^{M}
(s_{ij}-\bar s_i)(r_{ij}-\bar r_i).
$$

Covariance 的无偏分母是 `P(M-1)`；held-out node Fisher 仍用 `PM`。每个 split 从自身 Fisher
独立解析 absolute damping，并继承与训练完全相同的 `pcg_dtype/tolerance/ceiling`。主要
geometry metrics 是：

1. reward moment error 的 held-out ridge local-regret proxy；
2. predicted 与 target damped natural directions 的 undamped-Fisher squared error；
3. 两个 directions 的 Fisher cosine。

若任一 direction 的 Fisher norm 为零，cosine 未定义，正式结果必须 fail closed。

设 `H` 表示 validation 或 test split。冻结训练后的 reward head 后，held-out evaluator
重新计算

$$
u_{\phi}^{H}
=(\widehat F_H+\lambda_HI)^{-1}\widehat g_{\phi}^{H},
\qquad
u_*^{H}
=(\widehat F_H+\lambda_HI)^{-1}\widehat g_*^{H}.
$$

held-out local regret、direction error 和 cosine 都由这两个 **held-out re-solved**
directions 构造。它们估计“同一 reward function 在新 prompt/candidate geometry 中会诱导
什么局部方向”，不是实际写入 policy 的参数向量。实际部署方向只由 train quantities
定义：

$$
d_{\phi}^{\mathrm{deploy}}
=\beta^{-1}(\widehat F_{\mathrm{train}}+\lambda_{\mathrm{train}}I)^{-1}
\widehat g_{\phi}^{\mathrm{train}}.
$$

后者在 rollout 前冻结；test moment/Fisher 不得用于重新求解或修改它。因此 held-out
re-solve 指标与 deployed-direction rollout 的 ordering 不同并不构成数值矛盾，它表示
跨 split geometry 与 finite-update transfer 没有保持同一 ordering。

### 8.2 Prediction metrics 只是描述指标

Held-out preference diagnostics 报告：

- BTL negative log-likelihood；
- pairwise ordering accuracy；
- predicted BTL probability 与 operational-oracle BTL probability 的 mean absolute error。

这些指标回答 preference fit，不是 ProRM+ 的 primary success gate。它们的作用是显示
preference geometry 与 policy geometry 是否出现预期分离，不能替代 downstream evidence。

### 8.3 Matched measured-KL policy optimization

分别由 BT-MLE 与 ProRM+ 的 **train reward moments** 构造上述 deployed FP64 natural
direction，并以 FP64 train
Fisher 计算 quadratic 初值；只有真正写入 FP32 LoRA-B 参数时才按参数 dtype 转换。Fisher
quadratic

$$
\alpha_{\mathrm{init}}
=\sqrt{\frac{2\kappa}{d^\top Fd}}
$$

只提供 line-search 初值。每个方法必须独立测量固定 reference histories 上
reference policy 到 updated policy 的
sequence-level KL，并匹配到

$$
\kappa=0.01\quad\text{with relative tolerance }0.05.
$$

即 Phase 1 接受量的参数顺序是
$D_{\mathrm{KL}}(\pi_0\Vert\pi_{\alpha d})$，不是理论全局目标中的
$D_{\mathrm{KL}}(\pi_{\alpha d}\Vert\pi_0)$。两者共享 `theta_0` 处的局部 Fisher，但有限
步长下只能把该 line search 解释为一个 code-locked operational KL budget。

最终接受依据是 measured KL，不是二阶预测。随后用 common-random candidate indices 比较
zero-B、BT-MLE update 与 ProRM+ update 的 transformed-oracle reward improvement。

这个 learner-specific line search 把每个 direction 归一化到相同实测半径，因而对应
fixed-K transfer test；它不是 fixed-beta ProRM 主 estimand 的直接 rollout。固定 β 与
固定 K 回答不同问题，正式结果必须分别报告。

### 8.4 下一实验：pilot-calibrated global beta

下一实验使用新的 design identity。关键区分是：pilot 可以产生
**seed-specific calibration candidates**，但正式 estimand 使用一个对所有 formal seeds 与
policy arms 相同的 **global beta**。

对 excluded pilot seed `s`，只用 train split 的 operational-oracle rewards 与 train Fisher
构造

$$
u_{*,s}^{\rm tr}
=(\widehat F_{{\rm tr},s}+\lambda I)^{-1}\widehat g_{*,s}^{\rm tr},
\qquad
\widetilde\beta_s
=
\sqrt{
\frac{(u_{*,s}^{\rm tr})^\top
\widehat F_{{\rm tr},s}u_{*,s}^{\rm tr}}
{2K_{\rm cal}}
}.
$$

`widetilde beta_s` 使该 pilot seed 的 oracle local displacement 达到解析的 train-Fisher
二次 KL `K_cal`；它只是尺度候选，不是正式 seed `s` 的专用 beta。calibration pilot 先设

$$
\beta_{\rm base}
=
\max_{s\in\mathcal S_{\rm pilot}}\widetilde\beta_s,
\qquad
\beta^{(k)}=2^k\beta_{\rm base}.
$$

若 excluded pilot seeds 的 worst-arm mean/tail KL safety 不通过，只能运行 target-free
KL grid，并取预声明序列 `{beta_base, 2 beta_base, 4 beta_base, ...}` 中最小通过值。首个 freeze
绑定 calibration aggregate；之后每一步必须绑定紧邻的上一个 non-length safety failed
freeze aggregate 及其 exact doubled-beta recommendation，不能直接跳到更大的 grid
point。horizon parent 与 beta retry parent 分别绑定。随后把最终值与全部
numerical/horizon gates 写入新 confirmatory identity。即

$$
k_*=\min\{k\ge0:\beta^{(k)}\text{ 通过全部冻结 gate}\},
\qquad
\boxed{\beta_0=2^{k_*}\beta_{\rm base}}.
$$

若该集合为空，实验停止且不定义正式 `beta_0`；formal outcomes 产生后不得再改变。

正式部署为

$$
d_\ell^{deploy}
=
\frac{u_{\ell,\mathrm{train}}}{\beta_0},
\qquad
\ell\in\{\mathrm{BT},\mathrm{ProRM+},\mathrm{oracle}\}.
$$

因此固定 `beta_0` 保留 natural direction 的角度与范数校准误差，且不同 formal seeds 确实
对应同一个 downstream decision problem。learner-specific 或 formal-seed-specific line
search、norm normalization 与 beta calibration 都被禁止；seed-conditional `K_cal`
规则仅作 pilot train-only 尺度诊断。正式 sensitivity 只允许预注册的
`c in {0.5, 2.0}`，并直接使用 `beta=c*beta_0`；formal-seed curvature 对 beta
没有任何选择权。

主 finite-policy utility 使用 updated-policy trajectories 上的
$D_{\mathrm{KL}}(\pi_{d_\ell}\Vert\pi_0)$：

$$
J_\ell^*
=
\mathbb E_{x,y\sim\pi_{d_\ell}}[r^*(x,y)]
-\beta_0
D_{\rm KL}(\pi_{d_\ell}\Vert\pi_0).
$$

fixed-history $D_{\mathrm{KL}}(\pi_0\Vert\pi_{d_\ell})$、fixed-K normalization、
constrained regret、Fisher cosine 与 learner-specific matched-KL rollout 只是 secondary
diagnostics。test Fisher/moment 仍只用于 held-out re-solve，不参与 deployed train direction。
任何 formal mean/tail KL safety violation 都 fail closed，不触发缩放。

Pilot 是 target-free engineering stage：只发布 train convergence/rank/controls、
`widetilde beta_s`、response length/EOS/max-length 与 on-policy KL。它不得调用 held-out
evaluator、开启 final oracle scoring、计算 reward/utility/regret/learner ordering，或序列化
prompt/response text、token IDs、head/direction vectors。oracle finite-step utility control
第一次只在 fresh formal campaign 中评估。

Phase 2 还要求 raw-prompt 语义完全一致。Qwen2.5 用自己的 tokenizer/chat template 在
`truncation=False` 下渲染完整 raw prompt。对 pinned MultiPref 5,323 个 unique prompts 的
确定性 local audit 得到 88 个超过 1024 policy tokens（`1.65%`），所以先建立 5,235 个
length-eligible prompts 的 pool，再做 seeded shuffle/split；旧顺序会让三个 pilot seeds
分别包含 39、34、36 个超限 prompt 并 fail closed。artifact 绑定
unique/eligible/excluded/selected ID-list hashes 以及每个 selected prompt 的 token/hash
evidence，任何 mismatch 都硬失败。这将 Phase 2 的 prompt law 明确定义为
MultiPref 条件于 Qwen2.5-rendered length `<=1024`，而不是通过 silent truncation 改变同一
raw prompt 的语义。

Skywork Qwen3 对同一 raw prompt 加 assistant response 使用自己的 pinned tokenizer/chat
template 独立重渲染，绝不复用 Qwen2.5 token IDs。这样模型家族可以不同，但两者看到的语义
内容不能因 template 混用而不同。上述长度统计是固定输入 precheck，不是 pilot outcome。

这项新实验不重写已完成 Phase 1。Phase 1 的 learner-specific matched-KL rollout 仍是合法的
fixed-K transfer test，其 `not_passed` 状态保持不变。

## 9. Assumptions, failure modes and protections

| Assumption | Violation | Current protection |
|---|---|---|
| Local reward/KL Taylor model is adequate | ProRM identity cannot be extrapolated to a large update | One update; measured-KL budget `0.01` |
| Theoretical and measured KL orientations are interchangeable only locally | Finite-step fixed-K matching could differ from the policy-to-reference regularized objective | Record explicit argument order; interpret Phase 1 matching as an operational secondary estimand |
| Candidates are sampled from `pi_0` | Score identity and Fisher distribution are wrong | Same FP32 instance; exact tokens; no post-generation filtering |
| Tangent coordinates match | Objective weights unusable policy directions | Fixed-A, zero-B, identical layout/scale/hash |
| Edge law is natural `Q_0` | Pair moment no longer identifies `A_0r` without correction | Canonical `0-1` endpoints are independent base samples |
| Repeated labels are conditionally iid BTL | `h` no longer identifies one target margin | Controlled Phase 1 oracle; human data only robustness |
| `N` is independent with correct survival law | Randomized estimator becomes biased | Named RNG streams; no clipping or silent resampling |
| Rewards and scores have sufficient moments | Fisher/moment variance can diverge | Bounded oracle transform; probability floor |
| Train and evaluation targets are isolated | Target leakage invalidates comparison | Train tensor schema cannot contain true rewards |
| FP32 Krylov reaches a residual floor or recursive/true residual diverge | A numerically invalid direction could be reported as converged | FP64 policy geometry; explicit true-residual gate; false crossing restarts |
| PCG and measured-KL search converge | Direction or step size is undefined | Fail closed; failed seed cannot be discarded |
| Measured KL is locally bracketable on the positive ray | Bisection cannot locate target | Accept only measured KL within tolerance |

If policy optimization moves materially away from `pi_0`, candidates, scores, Fisher and repeated-label
moments must be regenerated around the new reference. The one-reference theorem does not justify reusing
old geometry indefinitely.

## 10. Mathematical objects and compatibility code paths

Public terminology is ProRM/ProRM+. The repository and Python namespace remain `Smart-Reward-Model` and
`smart_reward`; several internal identifiers retain historical names for artifact compatibility.

| Object | Current implementation |
|---|---|
| randomized `h` | `src/smart_reward/annotations.py` |
| `m_hat`, reported quadratic, envelope weights | `src/smart_reward/objective.py` |
| matrix-free Fisher and PCG | `src/smart_reward/linear.py`, `pcg.py` |
| fixed-A LoRA score/layout | `src/smart_reward/hf.py`, `scores.py` |
| BT-MLE / ProRM+ base trainers | `src/smart_reward/training.py` |
| Phase-2 objective-specific convergence and positive controls | `src/smart_reward/phase2_training.py`, `phase2_controls.py` |
| held-out policy geometry | `src/smart_reward/metrics.py` |
| fixed-β/fixed-K estimands | `src/smart_reward/metrics.py`, `estimand_audit.py` |
| natural directions and both measured-KL orientations | `src/smart_reward/rollout.py`, `policy_update.py` |
| common-β calibration and finite-policy utility | `src/smart_reward/common_beta.py` |
| real-model orchestration | `src/smart_reward/phase1.py`, `phase1_rollout.py`, `phase2_rollout.py`, `phase2_hf.py` |

The public CLI is `prorm`; the historical `smart-reward` executable remains a compatibility alias during
migration.

## 11. Foundations and contribution boundary

The project composes established tools:

- Fisher/natural policy-gradient geometry builds on
  [Kakade, 2001](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html)
  and trust-region policy optimization
  [Schulman et al., 2015](https://arxiv.org/abs/1502.05477);
- pairwise logistic preferences use the
  [Bradley–Terry model](https://academic.oup.com/biomet/article-abstract/39/3-4/324/326091);
- unbiased estimates of powers use classical
  [Hoeffding U-statistics](https://www.jstor.org/stable/2235637);
- randomized truncation follows the general debiasing ideas of
  [McLeish, 2011](https://arxiv.org/abs/1005.2228) and
  [Rhee & Glynn, 2015](https://web.stanford.edu/~glynn/papers/2015/RheeG15.pdf);
- the moment-estimation language builds on
  [Hansen, 1982](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/).

These ingredients are not individually claimed as new. The proposed combination is:

1. define reward-model quality prospectively through downstream policy regret;
2. reduce the global bilevel objective to a local Fisher-inverse ProRM target;
3. identify its reward-error moment from natural pairs using randomized repeated labels;
4. train the resulting ProRM+ objective with a matrix-free Fisher–GMM dual;
5. test the mechanism under fixed tangent coordinates, leakage isolation, held-out geometry, and a
   globally shared fixed-`beta` policy update; keep matched measured-KL as a secondary transfer diagnostic.

### 11.1 AuxDPO as an experimental precedent

The paper *Why DPO is a Misspecified Estimator and How to Fix It* motivates a
useful evidence pattern: pair a clean analytic misspecification construction
with matched data/compute, capacity stress, many seeds, base/oracle controls,
and separate ID/OOD evaluation. Phase 2 adopts that pattern and strengthens the
primary campaign to the exact preregistered ordered list of 30 paired formal
seeds `20260901` through `20260930`; stopping after observing outcomes is not
permitted.

The method itself is not imported. AuxDPO's auxiliary null-space
parameterization and IPO/DPOP comparisons concern direct preference
optimization. ProRM+ concerns reward-model estimation for downstream policy
regret, so repeated-label BT-MLE remains the primary baseline. Capacity,
sample-size, all-six-pair and OOD/human experiments test robustness or external
validity; they cannot replace the fixed-global-beta controlled mechanism test.

## 12. Paper claim boundary

1. Exact equality holds for the population, local, undamped pseudoinverse target. Its connection to the
   nonlinear problem is the local-branch expansion of Section 2.3.1, not an arbitrary-global-optimizer
   theorem. Finite-sample ridge is a regularized surrogate and requires damping sensitivity.
2. Phase 1 uses a restricted reward class and an operational oracle. It does not establish human utility.
3. A capacity bottleneck does not prove misspecification. The train-only projection residual is descriptive.
4. If real annotators violate homogeneous conditionally iid BTL, `h` identifies a different object.
5. Fixed-`L` human data supports only truncated ProRM+ robustness.
6. The closed-form example proves a population ProRM/BT-MLE ordering reversal, not the ProRM+ repeated-label
   theorem or an empirical effect.
7. Preference NLL, accuracy and probability MAE are diagnostics. A Phase 2
   positive mechanism claim requires the intersection of held-out
   fixed-global-beta geometry, ProRM+ versus BT-MLE finite utility, ProRM+
   versus zero-B improvement, and oracle-step versus zero-B positive control.
8. The pinned HPC4 five-seed Phase 1 campaign has completed with preregistered status `not_passed`.
   Consequently, the repository makes no empirical claim that ProRM+ outperforms BT-MLE under this locked
   setting. This result does not alter the population identity above; see
   [phase1_results.md](phase1_results.md).
9. Phase 2 is a one-reference, one-step LoRA-B policy test. It is not PPO,
   multi-step RLHF, or evidence that the local expansion remains accurate for
   arbitrary policy shifts.
10. The pilot is target-free and non-confirmatory. Its convergence, beta,
    length, and KL diagnostics cannot be interpreted as learner efficacy.
