# Fisher 修正 + TRPO 正式实验协议

## 冻结问题

主问题是在同一实际 forward-KL 条件下，Pro-RM 的单步策略更新是否比 MLE-RM
获得更高的 fresh-test oracle reward improvement。唯一主条件为
`kappa = 1e-3`；`3e-4` 与 `3e-3` 是预注册敏感性条件。

主 estimand 为三个 seed 上

```text
fresh_rollout_reward_improvement(Pro-RM, kappa=1e-3)
-
fresh_rollout_reward_improvement(MLE-RM, kappa=1e-3)
```

的逐 seed 值、均值与 sample standard deviation。三个 seed 只支持描述性证据。

## 数据隔离

`train=3072` 和 `validation=512` 保持不变。旧 test 已参与事后诊断，不能再作为
确认性 test。新 test 是同一 pinned MultiPref revision、Qwen chat template、512-token
过滤和 `prompt_split_seed=20261000` 下，确定性 shuffle 后区间 `[4096, 4608)` 的
512 个合格 prompt。选择过程不读取 reward、类别或模型结果。

旧 artifact 的 train/validation 候选、policy score、reward feature 和 raw oracle score
可以按组件复用；复用必须逐项验证 prompt/candidate 顺序、LoRA-A、tangent layout、
模型 revisions 和文件 SHA-256。fresh test 必须重新生成。旧 receipt 保持逐字节不变，
新 artifact 在 evidence 和 materialize receipt inputs 中记录来源组件哈希。

## Fisher 选择

经验 Fisher 保持 raw second moment：

```text
F_hat = S.T S / N
F_lambda = F_hat + lambda_relative * mean(diag(F_hat)) I
```

候选固定为 `{0.1, 1.0, 10.0}`。每个 seed 在 train prompts 内按 prompt ID 做相同的
平衡五折划分。每个候选在四折求 oracle direction，并按四折 raw Fisher 缩放到
`0.5 * delta.T F_hat delta = 1e-3`；剩余一折计算有限候选 oracle reward improvement。

先对每个 seed 的五折取均值。候选必须在每个 seed 上均为正。跨 seed 均值最好的候选
定义 best；保留均值不低于 `best_mean - SE(best)` 的候选，并在其中选择最大的
`lambda_relative`。三 seed 和三种 reward source 共用该唯一结果。没有候选合格时
fail closed。

## 单步 TRPO

对于 reward source `m`：

```text
g_m = A r_m
d_m = (F_hat + lambda * mean(diag(F_hat)) I)^-1 g_m
alpha = sqrt(2 * kappa / (d_m.T F_hat d_m))
delta_B = alpha * d_m
```

damping 只决定方向，trust-region 缩放使用未加 damping 的 raw Fisher。固定 beta
不参与 adapter 生成，也不作为本轮主 estimand。

## 真实 KL 校准

九个初始 adapter 分别在 validation prompts 上从更新后策略采样，每个 prompt 四个
response；估计 `KL(updated || pi0)` 的统计单位是 prompt。接受条件同时为：

- 点估计落在 `[0.8*kappa, 1.2*kappa]`；
- prompt-clustered normal 95% CI 上端不超过 `1.5*kappa`。

最多四次确定性尝试。下一步 multiplier 使用局部二次关系
`scale *= sqrt(kappa / observed_KL)`，单次变化最多四倍；所有尝试共享确定性随机流。
不读取 test 做缩放。四次均失败则该 policy component fail closed。

## 阶段与恢复

```text
materialize(fresh test + validated train/validation reuse)
-> fisher-crossfit[seed]
-> fisher-select
-> reward[seed] (MLE validated reuse; Pro refit; three directions)
-> adapters[seed, method, kappa]
-> kl-calibration[seed, method, kappa]
-> kl-calibration-aggregate[seed]
-> rollout[seed, ten policies]
-> rollout-aggregate[seed]
-> three-seed-aggregate
-> integrity audit
-> immutable archive
-> local key report
```

方括号内每个组件有独立 checkpoint/receipt。失败只重算该组件及其科学下游。
HPC4 GPU 阶段排除已知故障节点 `gpu19`，除非独立 GPU gate 后另行修改。
