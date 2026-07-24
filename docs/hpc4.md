# HKUST HPC4 运行规范

本文只描述仓库当前 Slurm/Apptainer 接口。邮件确认 ITSO 账号已关联 Slurm account
`sigroup`，登录地址为 `hpc4.ust.hk`，并列出 CPU 分区 `amd/intel` 与 GPU 分区
`gpu-a30`、`gpu-l20`、`gpu-rtx5880`、`gpu-rtx4090d`。邮件没有证明当前 QoS、wall-time、
显存、驱动、节点健康或计算节点外网；这些都必须在首次登录和 GPU smoke job 中实测。

正式方法名是 ProRM+，正式对照是 repeated-label BT-MLE。仓库目录
`Smart-Reward-Model`、Python package `smart_reward` 和 project/scratch 路径作为兼容基础设施
保留。公开 CLI 为 `prorm`，公开 environment keys 统一为 `PRORM_*`；旧 `SRM_*` keys 只作为
迁移期兼容 alias 被脚本接受。

## 0. 从登录到结论的七个门

| Gate | 执行动作 | 通过标准 | 失败后 |
|---|---|---|---|
| 1. Account | 私密完成 password+Duo/2FA，再运行 `preflight.sh` | account、quota、partition、project/scratch、Apptainer 可用 | 不申请 GPU |
| 2. Driver + candidate | host probe、committed definition、SIF SHA256 | driver 已观察；candidate image 身份完整 | 不进入 CUDA smoke |
| 3. CUDA | `submit_gpu_smoke.sh` | 单 GPU、版本、Qwen3 class、`pip check` 全通过 | 修 image/partition |
| 4. HF staging | 在 CPU 计算节点严格串行执行 smoke stage、验收、main stage | 两个 stage job 成功并产生 config-specific inventory | 不提交 controlled job |
| 5. Model smoke | `submit_controlled.sh configs/smoke.yaml ...` | artifact、BT-MLE/ProRM+、KL、rollout、memory evidence 完整 | 定位实现或容量问题 |
| 6. Main | 同一环境提交 `configs/main.yaml` | 五个 paired seeds 主链完整 | 保留失败证据后诊断 |
| 7. Aggregate | CPU 节点运行 `submit_aggregate.sh` | 身份一致、原子发布并写出预注册判据 | 不手改结果 |

这些 gate 必须顺序执行。后一个 gate 成功不能补偿前一个 gate 的失败；尤其不能在登录节点
直接运行模型命令，也不能因为单个 smoke 成功就宣称实验结论成立。

Gate 2 产生的只是 **candidate image**。只有 Gate 3 在目标 HPC4 partition 上以 exit code 0
完成后，该 image SHA 才能称为 **HPC4-validated**。host probe 报告不是 CUDA 容器验证的替代。

### Completed formal campaign record

锁定的 FP64 Phase 1 campaign 已于 2026-07-24 完成。producer Git commit 是
`738401d15ff97c941f3066bab1973aab37a60bc6`，main config hash 是
`ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6`。

| Gate | Accepted evidence |
|---|---|
| Main | Seeds `20260722–20260726` 分别由 jobs `1642737, 1642741, 1643282, 1642736, 1643747` 产生；全部 `COMPLETED (0:0)` |
| Aggregate | Job `1645205`, `amd/cpu13`, `COMPLETED (0:0)`, `00:00:33` |
| Aggregate SHA256 | `97c2caf7790caf30d3a1108a71b766a9efb9f432312d22a72552cbf911eb15c3` |
| Manifest SHA256 | `c1fd5a9399be0fe39ed97d1e3409c9d26919bac2ff173ae078db198d17f07c38` |
| Scientific status | `not_passed` |

完整指标、逐 seed 结果和结论见 [phase1_results.md](phase1_results.md)。`not_passed` 是有效科研
结果，不是 Gate 7 的工程失败。

## 1. 不可违反的原则

- 登录节点只做 Git、quota/partition 检查、文件传输和 `sbatch`；禁止加载模型、提取 score
  或训练。登录节点的 user namespace 已确认禁用，因此也禁止在登录节点直接执行
  `apptainer exec` 或 `stage_hf_assets.py`。
- 正式实验只使用 SHA256 校验过的 Apptainer `.sif`，不在计算 job 中 `pip install`。
- model、tokenizer 和 dataset 必须提前缓存 config 指定的 commit revision；正式 job 强制
  Hugging Face offline。首次下载只能通过 `scripts/hpc4/submit_hf_stage.sh` 提交到
  `amd` 或 `intel` CPU 计算分区。smoke/main 共用一个 cache，首次 staging 必须严格串行：
  smoke 报告通过后才可提交 main，禁止并发写 cache。
- 同一个 paired comparison 固定同一 GPU partition；smoke 后记录实际 GPU 型号，不混用
  不同型号的 seed。
- 第一阶段是一节点、一 GPU。未获得明确 GPU-hour 预算前，array concurrency 默认 1，
  不提交无界 sweep 或多节点作业。
- credential 只通过受控 secret 注入；不得写入仓库、YAML、manifest、artifact 或 Slurm log。
- ITSO password、Duo/2FA response、SSH private key 和 recovery code 只由用户在受信任的 SSH
  客户端中处理；绝不发送给 Codex，也不要求 Codex 代填。
- formal array 排队和执行期间不得切换、pull、修改或复用其 checkout 做开发。submit 绑定
  当时的 clean `HEAD`；compute 从该 commit 建立私有 detached scratch clone 并在每个 phase
  边界复验，登录 checkout 后续变化不会进入 job。
- 提交与 compute 都拒绝任何导出的 `APPTAINER*`/`SINGULARITY*` ambient control variable；
  所有 bind 与容器内环境只能来自受审脚本的显式 allowlist。

## 2. 存储分层

按邮件给出的目录与配额规划；首次登录仍必须用 `squota`、`df` 和实际权限验证：

| 逻辑路径 | 用途 | 约束 |
|---|---|---|
| `$HOME` | Git checkout、小配置 | 不放模型、dataset、artifact 或 checkpoint |
| `$PRORM_PROJECT_ROOT` | validated image、pinned HF snapshots、可复用 artifact、正式结果 | 必须是绝对组共享根；建议 `/project/sigroup/smart-reward-model` |
| `$PRORM_SCRATCH_ROOT` | 每 job staging、临时 cache/checkpoint | 必须是绝对个人 scratch 根；建议 `/scratch/$USER/smart-reward-model` |

脚本为每个 job 建立：

```text
$PRORM_SCRATCH_ROOT/jobs/$SLURM_JOB_ID/
```

正式持久化布局：

```text
$PRORM_PROJECT_ROOT/
├── images/                         # validated .sif
├── hf-cache/
│   ├── hub/                        # Qwen/Skywork/MultiPref raw snapshots
│   ├── datasets/                   # processed Datasets/Arrow cache
│   └── inventories/                # config-specific content inventories
├── system-reports/                 # build/staging/host/GPU evidence
├── slurm-logs/                     # scheduler stdout/stderr
├── artifacts/<config-sha>/<image-sha>/<inventory-sha>/<git-commit>/seed-N/
│                                    # candidates, repeated labels, S/features/oracle
└── runs/<run-name>/<config-sha>/
    ├── seed-N/job-J/               # manifest/comparison/rollout + artifact symlink
    └── aggregate/                  # SUCCESS + aggregate/manifest/hash evidence
```

Qwen 和 Skywork base weights 只保存在 `hf-cache/hub`。训练得到的 bias-free linear RM head
直接序列化在 `comparison.json`；当前 controlled experiment 不保存另一份完整 Qwen checkpoint，
也不导出 production LoRA adapter。policy LoRA-B update 在 rollout 时从 head 与固定 geometry
重构。

不要并发提交相同 config/image/Git/inventory/seed 身份；project artifact 以该身份为唯一主副本。
每个成功 `project_run` 必须有相对 symlink `artifact` 指向该 content-addressed artifact；这样
scratch 删除后，`comparison.json` 和 `rollout.json` 中的相对 POSIX path 仍可解析。

## 3. 首次登录与 preflight

校外先连接 HKUST VPN。首次 SSH 会要求 ITSO password 与 Duo/2FA；这些值只在 SSH 客户端
中输入，不复制到对话、命令历史、文件或环境变量。验证服务器 host key 后完成登录，再取得
公开 Git checkout。仓库内命令路径都相对 Git 根目录；只有 project/scratch 两个跨节点存储
锚点是绝对路径：

```bash
ssh YOUR_ITSO@hpc4.ust.hk
git clone https://github.com/youth-flow/Smart-Reward-Model.git
cd Smart-Reward-Model
test "$(git remote get-url origin)" = \
  "https://github.com/youth-flow/Smart-Reward-Model.git"
git rev-parse --verify HEAD

# .env.hpc4 is ignored by Git. Never edit the tracked example for a formal run.
test -e .env.hpc4 || cp scripts/hpc4/env.example .env.hpc4
source .env.hpc4
mkdir -p \
  "${PRORM_PROJECT_ROOT}"/{images,hf-cache,system-reports,slurm-logs,artifacts,runs} \
  "${PRORM_SCRATCH_ROOT}/jobs"
bash scripts/hpc4/preflight.sh
```

已经 clone 的用户跳过 `git clone`，但仍必须执行 remote、commit 与 clean-worktree 检查。公开
asset staging 不需要 Hugging Face token。SSH key 可在首次 password+Duo 登录后按 ITSO 官方
流程配置；private key 始终留在用户设备，不能提交到 Git。

`preflight.sh` 使用 account `sigroup`，检查：

- `squota` 与 `squota -A sigroup`；
- scheduler/partition 可见性和当前 queue；
- `PRORM_PROJECT_ROOT`、`PRORM_SCRATCH_ROOT` 存在且可写；
- filesystem capacity；
- module 信息与 `apptainer --version`。

任一步非零即停止。不要为了“通过”而在脚本后追加 `|| true`；只有 `module avail` 的展示性
输出被脚本有意允许失败。

## 4. 固定镜像与离线输入

host driver gate 已在 `gpu-l20` job `1640437` 完成：实际设备为 NVIDIA L20（46,068 MiB），
driver `570.211.01`，`nvidia-smi` 报告最高 CUDA 12.8。该报告固定了
[`containers/prorm-hpc4.def`](../containers/prorm-hpc4.def) 的 candidate base：
`pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime`，并用 OCI digest
`sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3`
锁定。CUDA 12.6 是 PyTorch 2.7.1 的 stable build，并被实测 host driver 支持。

image-build commit `b057bc9e134f1844248d655ed0f6c340af03099f` 产生的镜像随后通过 HPC4 GPU
environment smoke：job `1640778`，validated SIF SHA256 为
`d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb`。持久化报告路径为
`$PRORM_PROJECT_ROOT/system-reports/gpu-smoke-1640778.txt`。只有这个精确 SHA256 具有当前
validated identity；其他构建或传输结果仍是 candidate，必须重新通过 GPU smoke。

首次环境固化必须完成以下闭环：

1. 提交 `submit_host_gpu_probe.sh`，在最小 GPU allocation 中记录 host driver、
   `nvidia-smi` 和实际 GPU 型号；
2. 选择与该 driver 兼容的 CUDA/PyTorch base，并把完整 `.def` 纳入 Git；**已完成**；
3. 由 `.github/workflows/build-hpc4-image.yml` 使用 Apptainer 1.5.2 构建 `.sif`，运行
   definition test 与 `pip check`，再把 raw SIF 作为 ORAS artifact 发布到 GHCR；
4. 保存 definition Git commit、base/ORAS/SIF 三层 digest、build log、sorted `pip freeze`
   与 image SHA256；
5. GPU smoke 以 exit code 0 验证 candidate image 后，才把该 SHA 称为 validated image；
   **已由 job `1640778` 完成**；
6. 通过 `submit_hf_stage.sh` 在 CPU 计算节点先按 `configs/smoke.yaml` 预缓存全部 HF
   assets；确认 Slurm 与持久化报告均通过后，再按 `configs/main.yaml` staging，保存两份
   config-specific `repo_id/revision/file SHA256` inventory；
7. staging job 在 `TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1` 下验证 snapshot、
   config、tokenizer 与 dataset resolution；这一步不虚称已实例化模型权重。

先提交不依赖容器的 host probe：

```bash
bash scripts/hpc4/submit_host_gpu_probe.sh gpu-l20
squeue -u "$USER"
```

该 job 的正式报告为
`$PRORM_PROJECT_ROOT/system-reports/host-gpu-probe-1640437.txt`，SHA256 为
`96d3e4e2deaa9eb0948385d6d3a9ea2e81150736b50779c76ba84aec4430ce85`。

HPC4 登录节点不能本地构建 definition：其 Apptainer 没有 SUID builder，subuid/subgid
为空，而且**登录节点 user namespace 已禁用**。唯一正式构建路径是 GitHub Actions root
runner 构建 raw SIF，以
`oras://ghcr.io/youth-flow/smart-reward-model-hpc4:git-<commit>` 发布，并在 workflow 中
fail-closed 验证匿名 manifest pull。HPC4 随后按 immutable manifest digest 拉取相同 SIF
bytes，而不是把 Docker image 重新转换一次。相同限制也意味着登录节点不能直接
`apptainer exec` 完成 HF 下载；该工作必须由下一节的 CPU Slurm staging job 执行。

镜像构建身份与当前运行源码身份必须分离。获取镜像时固定使用已验证的 image-build commit，
不能用当前 `git rev-parse HEAD` 代替；当前 source `HEAD` 可以包含之后审查通过的 staging
或 control-plane 修改：

```bash
source .env.hpc4
image_build_commit=b057bc9e134f1844248d655ed0f6c340af03099f
bash scripts/hpc4/fetch_candidate_image.sh "${image_build_commit}"
```

fetcher 先对 GHCR tag 的 OCI manifest bytes 做 SHA256，取得 immutable manifest digest，
要求 manifest 恰好含一个 SIF layer，再按 digest 拉取。下载后的文件 SHA256 必须等于该
layer digest；已有不同 `images/prorm.sif` 时拒绝覆盖。base OCI digest、ORAS manifest
digest 与 SIF SHA256 是三个不同身份，绝不能互换。后续 staging/controlled manifest 绑定
提交时的 source Git SHA，同时独立绑定 validated SIF SHA256；source SHA 不要求等于上述
image-build commit。

image/cache 的公开配置值使用相对于 project root 的路径；submit 脚本会 canonicalize 为
Apptainer 所需的绝对路径，并拒绝 path escape：

```bash
export PRORM_IMAGE=images/prorm.sif
export PRORM_HF_CACHE=hf-cache
export PRORM_IMAGE_SHA256=d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb

test -f "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}"
mkdir -p "${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"
printf '%s  %s\n' \
  "${PRORM_IMAGE_SHA256}" "${PRORM_PROJECT_ROOT}/${PRORM_IMAGE}" \
  | sha256sum --check
```

`PRORM_PROJECT_ROOT` 与 `PRORM_SCRATCH_ROOT` 必须是绝对路径；`PRORM_IMAGE` 和
`PRORM_HF_CACHE` 可以是相对 project root 的资产路径。`PRORM_IMAGE_SHA256` 必须是 64 位
小写 hex；submit 和 compute 脚本都会再次校验。cache 至少要能离线解析 `configs/*.yaml`
中的以下固定资产：

- `allenai/multipref` dataset revision；
- Qwen policy/reward-feature model 与 tokenizer revision；
- Skywork oracle model 与 tokenizer revision。

formal MultiPref 路径先用 `huggingface_hub` 在本地 cache 精确解析 pinned snapshot，再按文件名
排序并直接读取其中的 `data/train-*.parquet` shards，调用的是本地 Parquet builder，而不是
`load_dataset("allenai/multipref", ...)`。这是硬性离线约束：Datasets 3.6 即使收到 offline
flags，repo-ID loader 仍可能发起 Hub metadata 查询。staging 的 offline-resolution evidence
与 controlled materialization 都走同一 parquet 路径；缺 shard、revision 不符或任何网络
依赖都会硬失败。

首次 staging 必须对 smoke 与 main 两份 config 分别执行。两者当前引用相同 repo revision
并共用同一个 cache，但各自的 config hash 与 inventory identity 不可互相替代。登录节点
只负责执行 `sbatch` 封装脚本；禁止在登录节点直接调用 `apptainer exec` 或
`stage_hf_assets.py`。选择一个允许计算节点访问公共 Hugging Face 的 CPU 分区，当前脚本
只接受 `amd` 或 `intel`。默认 staging walltime 为 `04:00:00`；只有管理员强制的 partition
上限更低时，才改为获批的更低值。

先只提交 smoke stage：

```bash
set -euo pipefail
export PRORM_HF_STAGE_WALLTIME=04:00:00

smoke_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/smoke.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
smoke_stage_job="${smoke_stage_job%%;*}"
test -n "${smoke_stage_job}"
squeue -j "${smoke_stage_job}"
```

`submit_hf_stage.sh` 要求 clean Git checkout、已校验的 validated SIF SHA 和 project/scratch
路径。每个 job 在 `$PRORM_SCRATCH_ROOT/jobs/$SLURM_JOB_ID` 建立 submitted commit 的 detached
source，在 CPU 计算节点中运行容器，并把不可覆盖的报告持久化到：

```text
$PRORM_PROJECT_ROOT/system-reports/hf-stage-<job-id>.log
```

smoke stage 离开队列后，必须先确认它为 Slurm `COMPLETED`、`ExitCode=0:0`，且持久化
报告包含独立一行 `status=passed`：

```bash
sacct -j "${smoke_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
smoke_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${smoke_stage_job}.log"
tail -n 20 "${smoke_stage_report}"
grep -Fx 'status=passed' "${smoke_stage_report}"
```

只有上述三项全部通过，才提交 main stage：

```bash
main_stage_job="$(
  bash scripts/hpc4/submit_hf_stage.sh \
    configs/main.yaml amd "${PRORM_HF_STAGE_WALLTIME}"
)"
main_stage_job="${main_stage_job%%;*}"
test -n "${main_stage_job}"
squeue -j "${main_stage_job}"
```

main stage 离开队列后执行同样的验收；只有 `COMPLETED`、`ExitCode=0:0` 和
`status=passed` 全部成立，才接受两份 inventories：

```bash
sacct -j "${main_stage_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition
main_stage_report="${PRORM_PROJECT_ROOT}/system-reports/hf-stage-${main_stage_job}.log"
tail -n 20 "${main_stage_report}"
grep -Fx 'status=passed' "${main_stage_report}"
sha256sum "${PRORM_PROJECT_ROOT}/${PRORM_HF_CACHE}"/inventories/*.json
```

该工具的 JSON inventory 只含 cache-relative POSIX path、每个 snapshot 文件 SHA256、
package versions 和 offline-resolution evidence。它会 local-only resolve model repository、
`AutoConfig`、tokenizer 与 MultiPref dataset；不会加载 Qwen/Skywork weight tensors。真正的
weight/class/显存验证属于 controlled model smoke。

不要在登录节点手工执行 `--verify-only`。`submit_controlled.sh` 选定 config-specific
inventory 后，compute job 会在 materialization 前自动运行无网络 `--verify-only`，重算内容
并要求 bytes 与已存在 inventory 完全一致；验证失败会硬停止，不会覆盖正式 inventory。

`submit_controlled.sh` 根据 config hash 固定选择
`$PRORM_HF_CACHE/inventories/<config-hash>.json`，校验其 SHA256，并把绝对 inventory path 与
`PRORM_HF_INVENTORY_SHA256` 送入 job。compute 在 materialization 前再次 `--verify-only`；
manifest、artifact producer、comparison/rollout environment identity 和 aggregate 都绑定
同一 `hf_inventory_sha256`。因此 cache 内容改变不能沿用旧正式身份。

预缓存是独立 staging 操作，不得在正式 allocation 中使用 `--allow-download`。当前容器 job
设置：

```text
HF_HOME=$PRORM_HF_CACHE
HF_HUB_CACHE=$PRORM_HF_CACHE/hub
HF_DATASETS_CACHE=$PRORM_HF_CACHE/datasets
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
```

本地 Windows/CPU environment 不能冒充 HPC4 lock。candidate 的 exact version lock 位于
`containers/requirements-hpc4.lock`；关键锚点是 `torch==2.7.1`（digest-locked base，
pip 不重装）、`transformers==4.52.3`、`peft==0.15.2`、`accelerate==1.7.0`、
`datasets==3.6.0`、`huggingface-hub==0.31.4`、`tokenizers==0.21.1`、
`numpy==1.26.4`、`pyarrow==17.0.0` 与 `fsspec==2025.3.0`。Skywork pinned config 的
`Qwen3ForSequenceClassification` 和 `transformers_version=4.52.3` 决定了该硬锚点。
完整实际安装集由 build 与 GPU smoke 各自保存 sorted `pip freeze --all`；最终运行身份是
SIF SHA256，不宣称 `apt` mirror 能在任意未来时刻重建 bit-identical 文件。

Gate 5（controlled model smoke）前至少要能定位以下证据：

| 证据 | 持久化位置 |
|---|---|
| Apptainer definition + Git commit | repository |
| Candidate/validated SIF + SHA256 | `$PRORM_PROJECT_ROOT/images/` |
| Definition commit + build log | repository + `$PRORM_PROJECT_ROOT/system-reports/` |
| `pip check` + sorted `pip freeze` | `system-reports/` |
| HF repo/revision/cache inventory | `$PRORM_PROJECT_ROOT/hf-cache/inventories/` |
| Networked stage + offline verify logs | `$PRORM_PROJECT_ROOT/system-reports/` |
| Host driver、partition、GPU model | host-probe and GPU-smoke reports |

## 5. GPU smoke：先验证环境，不跑模型

提交十分钟单 GPU smoke：

```bash
bash scripts/hpc4/submit_gpu_smoke.sh gpu-l20
squeue -u "$USER"
```

参数必须是脚本 allowlist 中的一个分区：
`gpu-a30|gpu-l20|gpu-rtx5880|gpu-rtx4090d`。如果 `gpu-l20` 的 account/QoS 不允许，
根据 preflight 输出选择邮件列出的其他分区；不要盲目反复提交。

`gpu_smoke.sbatch` 在 `apptainer exec --cleanenv --nv` 内验证：

- image SHA256；
- `nvidia-smi`；
- 容器内 `git --version`；
- Python 3.11、PyTorch 2.7.1、compiled CUDA 12.6 与全部 exact Python lock；
- Transformers 恰为 4.52.3，且 Qwen3 sequence-classification auto mapping 正确；
- Qwen2/Qwen3、GenerationConfig、PEFT、Datasets 与 Safetensors contract；
- scheduler visibility 变量被显式传入容器，且 `torch.cuda.is_available()` 为真、GPU count
  **恰好为 1**；
- device name 为 L20、compute capability 为 `(8,9)`；
- 一个实际 FP32 CUDA matrix forward/backward 与 synchronize；
- `python -m pip check` 无依赖冲突；
- 排序后的完整 `python -m pip freeze`。

任一检查失败时 job 非零。无论成功失败，trap 都把报告同步到：

```text
$PRORM_PROJECT_ROOT/system-reports/gpu-smoke-<job-id>.txt
```

正式 GPU smoke job `1640778` 已满足 Slurm `COMPLETED`、exit code 0 与全部报告检查，验证的
SIF SHA256 是
`d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb`。这只关闭 Gate 3：
GPU smoke 验证 runtime/class，不加载模型权重，也不产生 HF cache inventory。Gate 4 的两个
CPU staging job 完成后，Gate 5 才首次实例化 Qwen/Skywork。任何不同 SHA 的镜像都必须重新
执行本节，不得继承 job `1640778` 的 validated 身份。

## 6. Controlled job 提交

`submit_controlled.sh` 的前置条件是：精确 validated SIF、smoke/main 两个 CPU staging job
均成功、当前 config 对应的 inventory 已存在且 SHA256 可验证。缺少任一条件都必须返回
Gate 4；不得让 controlled GPU allocation 临时下载或重建 cache。

提交时的 clean source `HEAD` 是本次运行的源码身份，可以晚于 image-build commit
`b057bc9e134f1844248d655ed0f6c340af03099f`。run manifest 分别保存 source Git SHA 与
validated SIF SHA256，禁止把二者相等作为前置条件，也禁止用 source `HEAD` 重新推导镜像
身份。

接口要求三个必选位置参数和一个可选的 seed-index selection：

```text
submit_controlled.sh <config.yaml> <gpu-partition> <walltime> [<index>|<start>-<end>]
```

`walltime` 只能是 `HH:MM:SS` 或 `D-HH:MM:SS`。先用 smoke config，时间根据本集群实测
填写。可选第四参数是 config seed list 的 zero-based 单个 index 或闭区间；省略时提交全部
indices。它只能拆分调度，不能改变、重排或漏掉最终五个 formal seeds。formal config 必须是
repo 内已跟踪文件。提交前确认使用已审查并推送的 commit：

```bash
git fetch origin main
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

export PRORM_SMOKE_WALLTIME=REPLACE_WITH_APPROVED_PILOT_WALLTIME
bash scripts/hpc4/submit_controlled.sh \
  configs/smoke.yaml gpu-l20 "${PRORM_SMOKE_WALLTIME}"
```

submit 脚本在申请 allocation 前 fail-fast：验证 clean worktree、tracked config、exact `HEAD`、
image SHA、config-specific inventory path/SHA，并导出 `PRORM_GIT_COMMIT`、
`PRORM_HF_INVENTORY` 与 `PRORM_HF_INVENTORY_SHA256`。compute 再次要求当前 `HEAD` 等于提交
commit；排队期间切换 checkout 不会静默改变实验，而会使 job 硬失败。

检查 Slurm log 与 project run output，确认下列步骤全部完成：

1. 严格解析 config、selected seed、config SHA256 与 submission Git identity；
2. 对 config-specific cache 做无网络 `--verify-only`，要求 inventory bytes/SHA 不变，并写
   `hf-inventory-verification.json` 作为本次 job evidence；
3. 用 `env-report --seed` 写包含 inventory digest 的 seed-specific `run-manifest.json`；
4. 离线加载 pinned MultiPref/Qwen/Skywork，创建或完整校验 Phase-1 artifact，并用
   `artifact-materialization.json` 明确记录本 job 是 `materialized` 还是 `reused`；
5. `controlled-compare --run-manifest` 先绑定 manifest SHA、formal environment 与 artifact
   producer，再在 config 声明的 damping（main 为三档）运行 fixed-step paired
   BT-MLE/ProRM+；
6. 从主 damping head 构造两个 policy direction，分别匹配 measured KL；
7. 用 common-random test rollouts 和冻结 oracle transform 写 rollout evidence；
8. trap 将小型最终证据同步到 project，创建指向 authoritative artifact 的相对 symlink，并
   原子写入互斥的 `SUCCESS` 或 `FAILED` completion marker。

smoke 保留 main 的 rank-4/最后四层 LoRA tangent、16-candidate KL probe 和 oracle batch 上限，
只缩 prompts 与 outer steps。它必须验证 candidate/token schema、LoRA-A hash/layout、PCG
convergence、KL tolerance 和输出 hash；`memory-materialize.json`、`memory-compare.json`、
`memory-rollout.json` 分别记录 PyTorch allocated/reserved 峰值。smoke 数字不能进入论文主表。

以下数值是**历史 pre-fix 环境基线**，来自 controlled smoke job `1641441`、semantic config
`756270b082c059250f0d04a93dd08198cd07d8f6202f9dcaa7b5549b37247967`、source commit
`c48db19`、validated SIF SHA256
`d6fc044b4fa303747908783ea057d5b8946f613bfec6a6ca301e3a02fd7719cb`。它证明环境可运行，
但不能替代当前 FP64 campaign 的实际资源记录：

| 资源 | Historical smoke |
|---|---:|
| Wall-clock | `00:03:15` |
| Peak GPU allocated/reserved | `9,376,474,112 / 24,404,557,824 B` |
| Host MaxRSS / allocated CPUs | `3,906,312 KiB / 16` |
| Scratch peak | `21,926,294 B` |
| Project persistent bytes | `18,467,909 B` artifact + `1,574,291 B` run directory |

接受的五 seed FP64 main campaign 给出最终容量事实：

| 资源 | Formal main 实测 |
|---|---:|
| Requested walltime / seed | `12:00:00` |
| Elapsed / seed | mean `02:59:02`; range `02:49:53–03:11:52` |
| Total GPU time / peak concurrency | `14:55:11`; peak `2` |
| Allocation / seed | `1 × NVIDIA L20`, `16 CPUs`, `120 GiB` host memory |
| Peak GPU allocated / reserved | `10,112,134,656 / 24,404,557,824 B` |
| Host MaxRSS range | `5,066,212–8,745,552 KiB` |
| Content-addressed artifact / seed | `589,174,876–589,667,887 B` |
| Small run evidence / seed | `9,406,293–9,863,826 B` |

storage 实测统一采用 GNU
`du --bytes --summarize --no-dereference -- <directory>`：统计 apparent bytes 和目录/相对 symlink
本身，不跟随 run 内的 artifact symlink 重复计算 artifact target。

上述数字是本次 locked identity 的实际记录，不保证另一模型、prompt budget 或集群状态具有相同
耗时。array concurrency 只改变 campaign makespan，不减少总 GPU time。如果 site-default
CPU/RAM 不足，必须先把资源请求作为脚本变更提交并重跑 smoke；不得在 main 中临时使用未经
记录的资源设置。

smoke 通过后，在**同一 partition/GPU 型号**提交 main 五 seed array：

```bash
export PRORM_ARRAY_CONCURRENCY=2
export PRORM_MAIN_WALLTIME=12:00:00
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}" 0-3
```

2026-07-23 查询到 `l20_qos MaxSubmitPU=4, MaxJobsPU=2`；五元素 array 会在提交阶段被
`QOSMaxSubmitJobPerUserLimit` 拒绝。上面的 `0-3` 先占四个 submit slots；其中至少一个 task
终止、当前 submitted count 降至 3 后，再提交最后一个 index：

```bash
bash scripts/hpc4/submit_controlled.sh \
  configs/main.yaml gpu-l20 "${PRORM_MAIN_WALLTIME}" 4
```

submit 脚本在 committed identity 中读取完整 seed 数，并形成
`--array=<selected-start>-<selected-end>%$PRORM_ARRAY_CONCURRENCY`。selection 必须是范围内的
单 index 或连续闭区间，`PRORM_ARRAY_CONCURRENCY` 必须是正整数。当前用户决定拉满已确认
QoS，因此 accepted FP64 campaign 取 2；`MaxJobsPU=2` 仍是所有 split arrays 的全局并发上限。

### 6.1 Phase 2 outcome-blind pilot

Phase 2 使用两个被同时锁定的 config：

- `configs/common_beta_pilot_base.yaml` 定义 real-model materialization，决定 HF inventory
  与 content-addressed artifact identity；
- `configs/common_beta_pilot.yaml` 是固定 seeds
  `20260801,20260802,20260803`、`stage=pilot`、
  `formal_eligibility=false` 的 overlay，决定 pilot design identity。

提交前的本地 deterministic input audit 必须使用 pinned Qwen2.5 tokenizer/template 完整渲染
全部 5,323 个 unique MultiPref prompts。当前锁定输入有 88 个超过 1024 policy tokens
（`1.65%`），eligible pool 为 5,235；旧 shuffle-first 顺序会让三个 pilot tasks 分别包含
39、34、36 个超限 prompt。当前 contract 是先建立 `<=1024` eligible pool，再 seeded
shuffle/split，绝不截断；artifact 保存 unique/eligible/excluded/selected counts 与 ID-list
hashes。这个检查是 reproducible preflight，不是 Phase 2 结果。

先在 CPU 计算节点为 **base config** 建立/验证 inventory；不能用 overlay hash 代替 base
inventory identity：

```bash
git fetch origin main
test -z "$(git status --porcelain --untracked-files=normal)"
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"

bash scripts/hpc4/submit_hf_stage.sh \
  configs/common_beta_pilot_base.yaml amd 02:00:00
```

staging job 必须 `COMPLETED (0:0)`，对应
`$PRORM_HF_CACHE/inventories/<base-config-hash>.json` 存在且 verify-only 通过。随后提交 pilot；
HPC4 已观察到 `l20_qos MaxJobsPU=2`，所以脚本默认 array concurrency 2：

```bash
export PRORM_PHASE2_ARRAY_CONCURRENCY=2
bash scripts/hpc4/submit_phase2_pilot.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  gpu-l20 1-00:00:00
```

当前可执行的 Phase 2 pilot GPU design 固定为 `gpu-l20`（NVIDIA L20）；专用 pilot 提交
入口会拒绝任何其他 GPU partition。已实现的 confirmatory wrapper 同样只接受
`gpu-l20`，避免把跨硬件 producer 差异混入方法效应。CPU pilot aggregation 与正式
campaign finalization 仍只允许 `amd|intel`，不受此 GPU 锁定影响。

可选第五参数仍是一个 zero-based index 或连续闭区间。该专用入口在 allocation 前验证：

- clean exact Git commit（上面的 operator precheck 另行保证它已推送到 `origin/main`）；
- overlay/base/`identities.json` 的 committed bytes、file SHA256 与 semantic hash；
- overlay 对 base config hash 的绑定、3-seed pilot 身份和 `formal_eligibility=false`；
- validated SIF、base-config inventory、绝对 project/scratch roots；
- account `sigroup`、单节点、单 task、单 GPU 与锁定的 `gpu-l20` partition。

每个 array task 的权威目录是

```text
$PRORM_PROJECT_ROOT/runs/phase2-pilot/<phase2-design-sha>/
└── seed-<seed>/job-<array-job-id>_<task-id>/
    ├── phase2-pilot-diagnostics.json
    ├── phase2-pilot-diagnostics.diagnostics.jsonl
    ├── phase2-output-verification.json
    ├── run-manifest.json
    ├── artifact -> <content-addressed base artifact>
    └── SUCCESS | FAILED
```

base artifact 单独发布到

```text
$PRORM_PROJECT_ROOT/artifacts/<base-config-hash>/<image-sha>/<inventory-sha>/<git-commit>/seed-<seed>/
```

pilot 输出的 schema 是 `common-beta-pilot-diagnostics/v2` 与
`common-beta-pilot-diagnostic-row/v2`。它们只允许 train-only convergence/rank/control、
train-only beta candidates、response length/EOS/max-length 和 on-policy KL evidence。作业内
verifier 会拒绝 held-out、reward、utility、regret、oracle outcome、head/direction vectors、
prompt/response text 或 token IDs；pilot 不调用 held-out evaluator，也不开启 final
oracle-scoring session。`SUCCESS` 只表示这条 outcome-blind 工程链和信息边界通过，不表示
ProRM+ 胜过 BT-MLE，也不能进入正式 aggregate。

三个声明 seed 都产生完整 `SUCCESS` 目录后，只能通过 CPU Slurm job 做严格聚合。先将三个
路径分别替换为与当前 committed design SHA 对应的唯一成功目录；输出父目录必须预先存在，
目标文件必须尚不存在：

```bash
design_sha=REPLACE_WITH_COMMITTED_DESIGN_SHA256
pilot_root="${PRORM_PROJECT_ROOT}/runs/phase2-pilot/${design_sha}"
aggregate_dir="${pilot_root}/aggregate"
mkdir -p "${aggregate_dir}"

run_seed_1=REPLACE_WITH_20260801_SUCCESS_DIR
run_seed_2=REPLACE_WITH_20260802_SUCCESS_DIR
run_seed_3=REPLACE_WITH_20260803_SUCCESS_DIR

aggregate_job="$(
  bash scripts/hpc4/submit_phase2_pilot_aggregate.sh \
    configs/common_beta_pilot.yaml \
    configs/common_beta_pilot_base.yaml \
    "${aggregate_dir}/calibration.json" \
    amd 01:00:00 \
    "${run_seed_1}" "${run_seed_2}" "${run_seed_3}"
)"
aggregate_job="${aggregate_job%%;*}"
test -n "${aggregate_job}"
squeue -j "${aggregate_job}"
```

任务离开队列后，要求 `COMPLETED`、`ExitCode=0:0`，并验证原子发布的 JSON：

```bash
sacct -j "${aggregate_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition,MaxRSS
calibration_aggregate="${aggregate_dir}/calibration.json"
test -f "${calibration_aggregate}"
aggregate_log="${PRORM_PROJECT_ROOT}/slurm-logs/prorm-phase2-pilot-aggregate-${aggregate_job}.out"
calibration_sha="$(sha256sum -- "${calibration_aggregate}" | awk '{print $1}')"
grep -Fx \
  "Phase-2 pilot aggregate published: ${calibration_aggregate} sha256=${calibration_sha}" \
  "${aggregate_log}"
```

aggregate job 不发布 `.sha256` sidecar；其权威 digest 是计算后与 Slurm publication log
逐字匹配的 `calibration_sha`。calibration aggregate 只接受或拒绝当前 horizon；仅在当前
horizon 通过时，它才给新的 freeze identity 提供
`beta_base=max_s(beta_tilde_s)`。它本身永远不接受 confirmatory selection。

把该值、`calibration_sha` 和所有冻结字段写入一个新的 committed overlay identity：
`design.stage=pilot`、`design.pilot_phase=freeze`。随后完整重跑固定 pilot seeds
`20260801,20260802,20260803`。aggregate identity 必须在 **GPU submit** 和 **CPU
aggregate submit** 两处都传入。

第一次 freeze 的模板是：

```bash
calibration_aggregate=REPLACE_WITH_ACCEPTED_CALIBRATION_JSON

bash scripts/hpc4/submit_phase2_pilot.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  gpu-l20 1-00:00:00 \
  --beta-source-aggregate "${calibration_aggregate}" \
  --horizon-parent-aggregate "${calibration_aggregate}"

# 三个 freeze SUCCESS 目录产生后：
bash scripts/hpc4/submit_phase2_pilot_aggregate.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  REPLACE_WITH_NEW_FREEZE_AGGREGATE_JSON \
  amd 01:00:00 \
  REPLACE_WITH_20260801_FREEZE_SUCCESS_DIR \
  REPLACE_WITH_20260802_FREEZE_SUCCESS_DIR \
  REPLACE_WITH_20260803_FREEZE_SUCCESS_DIR \
  --beta-source-aggregate "${calibration_aggregate}" \
  --horizon-parent-aggregate "${calibration_aggregate}"
```

如果 freeze 仅因非长度 KL safety 失败，下一个身份只能使用精确
`next_global_beta=2*previous_beta`，且其 `--beta-source-aggregate` 必须是**紧邻的上一份失败
freeze aggregate**；`--horizon-parent-aggregate` 仍是接受当前 horizon 的 calibration
aggregate。禁止跳过 beta 网格点。提交模板是：

```bash
previous_failed_freeze=REPLACE_WITH_IMMEDIATELY_PRECEDING_FREEZE_AGGREGATE
accepted_horizon_calibration=REPLACE_WITH_ACCEPTED_CALIBRATION_AGGREGATE

bash scripts/hpc4/submit_phase2_pilot.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  gpu-l20 1-00:00:00 \
  --beta-source-aggregate "${previous_failed_freeze}" \
  --horizon-parent-aggregate "${accepted_horizon_calibration}"

# 三个 retry SUCCESS 目录产生后，CPU aggregate 使用完全相同的两个父证据：
bash scripts/hpc4/submit_phase2_pilot_aggregate.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  REPLACE_WITH_NEW_RETRY_AGGREGATE_JSON \
  amd 01:00:00 \
  REPLACE_WITH_20260801_RETRY_SUCCESS_DIR \
  REPLACE_WITH_20260802_RETRY_SUCCESS_DIR \
  REPLACE_WITH_20260803_RETRY_SUCCESS_DIR \
  --beta-source-aggregate "${previous_failed_freeze}" \
  --horizon-parent-aggregate "${accepted_horizon_calibration}"
```

若长度 gate 失败，则不能只调 beta：必须在 `[256,512,1024]` 中进入下一 horizon，
建立新的 `pilot_phase=calibration` identity，并从三个 seeds 重新开始。此时 GPU 与 CPU
提交都只绑定紧邻的长度失败 aggregate：

```bash
previous_length_failure=REPLACE_WITH_IMMEDIATELY_PRECEDING_LENGTH_FAILURE

bash scripts/hpc4/submit_phase2_pilot.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  gpu-l20 1-00:00:00 \
  --horizon-parent-aggregate "${previous_length_failure}"

# 三个新-horizon calibration SUCCESS 目录产生后：
bash scripts/hpc4/submit_phase2_pilot_aggregate.sh \
  configs/common_beta_pilot.yaml \
  configs/common_beta_pilot_base.yaml \
  REPLACE_WITH_NEW_HORIZON_CALIBRATION_AGGREGATE_JSON \
  amd 01:00:00 \
  REPLACE_WITH_20260801_HORIZON_SUCCESS_DIR \
  REPLACE_WITH_20260802_HORIZON_SUCCESS_DIR \
  REPLACE_WITH_20260803_HORIZON_SUCCESS_DIR \
  --horizon-parent-aggregate "${previous_length_failure}"
```

每个箭头都要求先提交新的 overlay/design identity、更新 `configs/identities.json`，并使用
新的 no-overwrite output。只有最终 `selection_accepted=true` 的 freeze aggregate 才能生成
confirmatory identity。

正式实验只有在 pilot 结束后，才建立新的 confirmatory config：冻结一个对所有 formal
seeds/arms 相同的 global `beta_0`、response horizon、optimization/KL/length gates，并使用
顺序锁定的 30 个 paired seeds `20260901`–`20260930`。正式 beta sensitivity 只运行固定的
`beta in {0.5*beta_0, 2.0*beta_0}`；seed-specific curvature calibration 仅是 pilot
train-only 诊断，禁止进入 confirmatory jobs。本文件不硬编码 Phase 2 job ID 或实时状态；
以 immutable run evidence、`squeue` 与 `sacct` 为准。

### 6.2 Historical numerical incident

旧 main semantic hash
`7b3f12ba1c2b6f73deb8ca2fe0da303424f108876fb4a0ad12e68df312cbf7b2`、source
`f16edb12e9d09ddb797d9f3eb2f7cf8aa4c6d29e` 的 seed `20260722` / job `1641489`
在主阻尼 zero-head 初始 ProRM+ solve 达到 2048 iterations 后，true relative residual
`2.717e-5` 仍高于 `1e-5`，因此硬失败。

对应 `runs/controlled-main/7b3f12…f7b2/seed-20260722/job-1641489/` 的 `FAILED`、Slurm log、
manifest 与 artifact 保留；comparison、rollout、`SUCCESS` 不存在。该 run 不是 accepted seed，
也没有产生可解释为 `passed/not_passed` 的 aggregate。当前修订不放宽 tolerance 或 damping：
固定 policy geometry 改为 FP64，main ceiling 改为 8192，并使用新的 config/Git/content-addressed
路径完整重跑五 seeds。
该并发上限是**每次 array submission** 的上限；按上述时机提交第二个 array 后，campaign
全局最多同时运行两个 GPU task，仍受 `MaxJobsPU=2` 约束，总 GPU-hours 不变。
最终验收必须覆盖 indices `0,1,2,3,4` 各一次，拆分提交不能成为换 seed 或丢弃失败 task 的
手段。

## 7. 作业内的可复现与恢复语义

`controlled.sbatch` 不依赖 shell 当前的 Python。它在 job scratch 中建立 submitted commit 的
私有 detached clone，所有 Python 命令均从该不可被登录 checkout 切换影响的源码进入同一
clean Apptainer 环境；job staging 与 HF cache 被显式 bind，`PYTHONPATH` 指向 detached
commit 的 `src`。manifest 的 `selected_seed` 必须等于 array task 对应 seed；CUDA comparison
还要求 clean Git、submission `PRORM_GIT_COMMIT==HEAD`、image SHA、HF inventory SHA、
Slurm account `sigroup`、partition、CUDA 可见且恰好一个有名称的 GPU。comparison 保存
manifest bytes SHA 与解析后的 formal environment identity；rollout 再将当前进程与该 identity
逐字段匹配并写回结果。

Artifact 路径是：

```text
$PRORM_PROJECT_ROOT/artifacts/<config-sha>/<image-sha>/<inventory-sha>/<git-commit>/seed-<seed>
```

如果该目录已有 `metadata.json`，job 会复制到本次 scratch 并由 loader 重新验证 config
hash、seed、tensor SHA256、schema、shape/dtype/finiteness、split 与 producer
Git/image/inventory digest；验证失败不会静默重建或继续。若不存在，则在 job staging 完整
materialize，并通过 project 内同 filesystem staging、锁与原子 rename 安装，避免并发发布
半成品。

每次提交使用新的 `job-$SLURM_JOB_ID` run 目录，不覆盖旧结果。退出 trap 只同步已经存在的
`run-manifest.json`、`hf-inventory-verification.json`、`artifact-materialization.json`、
`comparison.json`、`rollout.json`、`updated_rollouts.jsonl` 和本 job 实际产生的
`memory-*.json`；复用 artifact 的 job 不伪造 `memory-materialize.json`。`SUCCESS`/`FAILED`
marker 记录
workload/final exit code。部分/失败 job 仍
保留 manifest 与 log 供诊断，但不得纳入 aggregate。大体积 artifact 不在每个 run 目录重复
复制。成功 run 的
`$project_run/artifact` 是指向上面 content-addressed project artifact 的**相对 symlink**；
验收必须执行 `test -e "$project_run/artifact/metadata.json"`。

失败 marker、manifest、log 与部分产物不得删除、覆盖或手改成成功；任何 solver/config 修订
必须进入新 Git/config identity，并在新 content-addressed 路径从头运行。

不要通过修改 project artifact 文件来“续跑”。materialization 是不可变数据层；reward
head comparison 固定步数且成本较小，失败后用同 config/seed 新 job 重跑。只有未来长时
scale-up 才需要另行设计原子 checkpoint/resume contract。

## 8. 监控与验收

常用只读命令：

```bash
squeue -u "$USER"
job_id=REPLACE_WITH_JOB_ID
sacct -j "${job_id}" \
  --format=JobID,State,Elapsed,ExitCode,AllocTRES,AllocCPUS,ReqMem,MaxRSS
tail -n 200 \
  "${PRORM_PROJECT_ROOT}/slurm-logs/prorm-controlled-ARRAY_JOB_TASK.out"
```

每个 seed 只有同时满足以下条件才算有效：

- Slurm state `COMPLETED`、`ExitCode=0:0`、run 有 `SUCCESS` 且没有 `FAILED` marker；
- manifest 的 config hash、`selected_seed`、Git commit、image hash、HF inventory hash、
  account、partition 和唯一 GPU model 正确；
- `hf-inventory-verification.json.inventory_sha256` 等于 manifest 的 inventory hash；
- comparison 的 manifest SHA/environment identity 与同目录 `run-manifest.json` 完全一致；
- artifact identity/integrity 检查通过；
- run-local `artifact` relative symlink 可解析到 content-addressed project artifact；
- config 明确记录 `pcg_dtype=float64`；训练、held-out 与 rollout-direction solve 运行时均
  fail closed。对 schema 已序列化的 final/rollout evidence 检查 `converged`、显式 true
  relative residual `<=1e-5`，以及存在时的 reason `converged|zero_rhs`；iteration ceiling
  不能被误当成成功；
- 主阻尼 comparison 存在且 PCG converged；每档 sensitivity 要么有完整结果，要么有显式
  failure record，任何 sensitivity seed 都不能被静默删除；
- BT-MLE/ProRM+ measured KL 各自在 `0.01 ± 5%`，rollout 文件完整；
- 该 seed 没有读取 test 进行训练/选择，也没有手工改动产物。

OOM、offline snapshot missing、主阻尼训练 PCG、rollout direction PCG、measured-KL、hash
mismatch 或 dirty/错误 commit 失败，都会使该 seed 的主链无效。sensitivity PCG failure 不
删除整个 seed：作业保留 failure record，继续产出可诊断 aggregate，但最终预注册状态必须是
`not_passed`。任何情况下都不得编辑 JSON 把失败改成成功。

## 9. 五 seed 聚合

Slurm array 不自动做跨 seed 聚合。先逐一用上一节的 `sacct`、log、manifest 和 symlink 条件
验收。每个 seed 必须**显式选择一个** exit-code-0 的 `job-J`；不要用 `latest`、`find | head`
或通配符偷偷选择重复/失败 run。登录节点不得直接执行 `apptainer exec` 或 host
`aggregate-results`；它只调用 `submit_aggregate.sh`，真正的验证与聚合在 `amd` 或 `intel`
CPU Slurm job 内使用 validated SIF 和 submitted commit 完成。

先把五个预注册 seed 与各自**唯一一个已验收成功**的 controlled job 写成明确映射。每个所选
job 必须是 Slurm `COMPLETED`、`ExitCode=0:0`，且对应
`runs/.../seed-<seed>/job-<job-id>/SUCCESS` schema 与 job ID 均正确、`FAILED` 不存在。下列五
个值必须来自逐 seed 验收记录，不得复用同一个 job ID：

| Seed | Accepted controlled job |
|---:|---:|
| `20260722` | `1642737` |
| `20260723` | `1642741` |
| `20260724` | `1643282` |
| `20260725` | `1642736` |
| `20260726` | `1643747` |

下面保留通用重跑模板；只有新 campaign 逐 seed 验收后，才把占位符替换为该 campaign 的 job
IDs。不得把上表历史 IDs 作为未来重跑的默认输入。提交一个单节点、单 task、1 CPU、8 GiB、无
GPU 的 formal aggregation job：

```bash
set -euo pipefail
job_20260722=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260723=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260724=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260725=REPLACE_WITH_ACCEPTED_JOB_ID
job_20260726=REPLACE_WITH_ACCEPTED_JOB_ID

aggregate_job="$(
  bash scripts/hpc4/submit_aggregate.sh \
    configs/main.yaml amd 01:00:00 \
    "20260722=${job_20260722}" \
    "20260723=${job_20260723}" \
    "20260724=${job_20260724}" \
    "20260725=${job_20260725}" \
    "20260726=${job_20260726}"
)"
aggregate_job="${aggregate_job%%;*}"
test -n "${aggregate_job}"
squeue -j "${aggregate_job}"
```

接口为：

```text
submit_aggregate.sh <main-config.yaml> <amd|intel> <walltime> [--source-commit <full_commit>] <seed=controlled_job_id>...
```

submit 层要求 clean/tracked submitted commit、validated image、main config-specific inventory，
并用 committed `configs/identities.json` 约束 config hash 和 seed 数。compute 层建立 detached
exact-commit source，要求输入映射与 `configs/main.yaml` 的五个 seeds 精确相等且 job ID
互不重复；随后复核每个 `SUCCESS` marker、manifest、artifact symlink、
comparison/rollout/updated-JSONL hashes 以及共同 Git/image/inventory/account/partition/GPU
identity。任一输入失败时不发布 final aggregate。

默认情况下，clean submission `HEAD` 同时是 aggregation control-plane commit 与
source/producer commit。只有 wrapper-only 后继修复需要验证既有 seed 时，才在 walltime 后显式
传入 `--source-commit <full-producer-commit>`。该 source 必须是 control-plane `HEAD` 的祖先，
且当前 worktree config bytes 必须与 source commit 中的 config blob 完全一致；否则提交硬失败。
compute job 从 source commit 的 detached checkout 执行原始 aggregation Python/config，并用同一
commit 严格校验五个 run manifest 和 artifact producer。最终
`aggregation-manifest.json` 的 `control_plane.git_commit` 与
`aggregation_source.git_commit` 分别记录两种身份，不能把 wrapper commit 冒充为实验来源。

本次 campaign 的 aggregate 恢复记录验证了该语义。Jobs `1645152`（`amd/cpu26`）与
`1645166`（`intel/cpu07`）均在容器创建阶段以 exit `127` 失败；系统 Apptainer 配置试图
auto-bind 不存在的 `/opt/knem-1.1.4.90mlnx3`。它们没有创建 final/staging aggregate，也没有
运行统计 CLI。后继 control-plane commit
`535f5abf7a858e35b33d5ac461d8c8e58add41fa` 将容器参数固定为
`--no-mount home,cwd,bind-paths`，再显式绑定 job/project roots；job `1645205` 以
`--source-commit 738401d15ff97c941f3066bab1973aab37a60bc6` 从原 producer checkout 执行原始
aggregation code/config 并成功发布。这个恢复修复了集群 mount 漂移，没有改变任何 seed、
metric、bootstrap 或 success criterion。

aggregation job 离开队列后先验收 Slurm 状态和 log：

```bash
sacct -j "${aggregate_job}" \
  --format=JobID,State,Elapsed,ExitCode,Partition,MaxRSS
tail -n 40 \
  "${PRORM_PROJECT_ROOT}/slurm-logs/prorm-aggregate-${aggregate_job}.out"
```

aggregation job 只有同时满足 Slurm `COMPLETED`、`ExitCode=0:0`，且下列 final
directory/文件校验全部通过，才可被接受。成功路径会把 staging directory 原子、
no-overwrite 发布为：

```text
$PRORM_PROJECT_ROOT/runs/controlled-main/<main-config-hash>/aggregate/
├── SUCCESS
├── aggregate.json
├── aggregate.json.sha256
└── aggregation-manifest.json
```

当前 locked `configs/main.yaml` 在 committed `configs/identities.json` 中的 semantic hash 为
`ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6`。执行最终文件验收：

```bash
main_config_hash=ae5d628ee47ff74a1fa2b89478c40b4fdd289935d8cf58dcbcf98b42f69a0df6
aggregate_dir="${PRORM_PROJECT_ROOT}/runs/controlled-main/${main_config_hash}/aggregate"
test -d "${aggregate_dir}"
test -f "${aggregate_dir}/SUCCESS"
test -f "${aggregate_dir}/aggregation-manifest.json"
(
  cd "${aggregate_dir}"
  sha256sum --check aggregate.json.sha256
)
grep -Fx 'status=SUCCESS' "${aggregate_dir}/SUCCESS"
grep -E '^pre_registered_evidence_status=(passed|not_passed)$' \
  "${aggregate_dir}/SUCCESS"
```

这里必须区分两层状态：

- `SUCCESS` 中的 `status=SUCCESS` 只表示 CPU aggregation job 完整验证五个来源、成功运行聚合器
  并原子发布证据。
- 科研结论只由 `aggregate.json` 的 `pre_registered_evidence.status` 决定，并在 `SUCCESS`
  中镜像为 `pre_registered_evidence_status=passed` 或 `not_passed`。只有 `passed` 支持预注册
  主张；`not_passed` 是有效、应保留的否定结果，不是工程失败，也不得因此换 seed、阻尼或重跑。

聚合器会拒绝缺 seed、重复/额外 seed 或非 main damping identity；在同一 aggregate 中加入
prompt-level `test_rollout_improvement`，聚合所有声明 damping 的 local-regret evidence、记录
PCG failure/non-reversal 并执行预注册 criteria。该状态不是 p-value 或“显著”标签。若 final
`aggregate/` 已存在，提交会拒绝覆盖，必须把它作为既有证据审计。

最终目录必须连同五个 manifest、comparison、rollout、updated JSONL、artifact symlink、
controlled/aggregation Slurm logs、image/build/staging/smoke reports 与两份 HF inventory 一起
长期保存。

### 9.1 Scratch 清理

scratch 在验收前是故障恢复证据，不自动删除。只有 aggregate 已原子写出并校验、五个 project
run 文件齐全、每个 `artifact/metadata.json` symlink 可解析后，才按**确切 job ID**清理成功
job；失败 job 先保留诊断。先打印并人工复核目标，再执行删除：

```bash
completed_job_ids=(
  REPLACE_20260722 REPLACE_20260723 REPLACE_20260724
  REPLACE_20260725 REPLACE_20260726
)
targets=()
for job_id in "${completed_job_ids[@]}"; do
  target="$(realpath -e -- "${PRORM_SCRATCH_ROOT}/jobs/${job_id}")"
  case "${target}" in
    "${PRORM_SCRATCH_ROOT}"/jobs/*) ;;
    *) echo "refusing scratch escape: ${target}" >&2; exit 2 ;;
  esac
  printf 'review cleanup target: %s\n' "${target}"
  targets+=("${target}")
done

# Run only after reviewing every printed absolute path.
rm -r -- "${targets[@]}"
```

该命令不使用 glob、不删除 `$PRORM_SCRATCH_ROOT/jobs` 本身，也不触碰 project 主副本。删除后
用 `test -f`/`test -e` 再确认 aggregate、run JSON 和 artifact symlink 仍可访问。

## 10. 故障处理顺序

1. **preflight 失败**：先解决 account、partition、quota、权限或 Apptainer；不提交 GPU job。
2. **image/hash 失败**：停止；重新传输或确认 digest，绝不绕过校验。
3. **CUDA smoke 失败**：记录 report，换兼容 image/module；不进入模型 smoke。
   Transformers `<4.52.3`、`>=5` 或缺少 Qwen3 classification class 都属于 image 硬失败。
4. **offline snapshot missing**：在 staging 环境补齐 config 的精确 revision，再重提 job；
   不给正式 job 开网。
5. **OOM**：先从 smoke 记录峰值并验证是否为实现问题；任何改变 microbatch/config 的 run
   都产生新 config hash，不能混入原主实验。
6. **主阻尼 PCG、rollout direction PCG 或 KL 不收敛**：该 seed 主链硬失败；诊断数值几何，
   不得放宽 tolerance/damping，也不得把“只增加 cap”当作充分修复。dtype、cap 或 solver
   semantics 改变都必须形成新 Git/config identity，并完整重跑五 seeds。
7. **Sensitivity PCG 不收敛**：保留 comparison 中的 failure record，继续完成可执行的主链和
   aggregate；结果必须为 `not_passed`，不得丢弃该 damping/seed 后重算均值。
8. **artifact/hash mismatch**：隔离损坏目录，保留证据；不要覆盖。确认目标后用新目录重建。
9. **manifest/environment mismatch**：不得手改 manifest 或 comparison；在相同 commit、image、
   account、partition、GPU model 下重跑受影响 seed。若环境必须改变，则全部五 seed 重跑。

任何需要改变预注册 config 的修复，都必须重新运行全部五个 paired seeds。

## 11. Phase 2 confirmatory：exact-30、no-retry 正式运行手册

本节是正式 Phase 2 的权威操作入口。上文出现的 calibration/freeze “retry”只表示
**outcome-blind pilot 设计产生新 identity**，不适用于 confirmatory seed。正式 campaign
固定 seeds `20260901`–`20260930`，每个 seed 只有 `attempt-1`，不允许 retry、requeue、
replacement seed 或手工补跑。

本仓库文档不记录尚未产生的 confirmatory config SHA、Git commit、image/inventory SHA、
accepted-freeze SHA、Slurm job ID 或当前运行状态。以下 `REPLACE_WITH_*` 必须在正式提交时
由已接受的 freeze、已提交 config identities 和 HPC4 实际环境解析，不能从示例猜测。

### 11.1 提交前身份与环境

先确认：

- 已接受的 strict freeze aggregate 存在，并与 confirmatory overlay 中绑定的 SHA256 一致；
- confirmatory overlay 和 base 都是 `configs/*.yaml` 下的 tracked file，且
  `configs/identities.json` 中二者均声明 `seed_count=30`；
- 当前 checkout 是将要执行的 exact commit，worktree clean；正式提交前先将该 commit 推送
  到协作远程；
- SIF、SIF SHA256、base-specific HF inventory 和离线 cache 已通过既有 preflight/staging；
- `PRORM_PROJECT_ROOT` 与 `PRORM_SCRATCH_ROOT` 是两个互不包含的 canonical writable
  absolute directory；
- shell 中不存在导出的 `APPTAINER*`、`SINGULARITY*` 或 `SBATCH_*` 覆盖变量。

```bash
set -euo pipefail

cd /ABS/PATH/TO/Smart-Reward-Model
git status --short
git rev-parse HEAD

export PRORM_PROJECT_ROOT=/ABS/PATH/TO/PROJECT_DATA
export PRORM_SCRATCH_ROOT=/ABS/PATH/TO/SCRATCH
export PRORM_IMAGE=/ABS/PATH/TO/LOCKED_IMAGE.sif
export PRORM_IMAGE_SHA256=REPLACE_WITH_64_HEX_IMAGE_SHA256
export PRORM_HF_CACHE=/ABS/PATH/TO/OFFLINE_HF_CACHE
export PRORM_PHASE2_ARRAY_CONCURRENCY=2
```

`PRORM_PHASE2_ARRAY_CONCURRENCY` 只能是 `1` 或 `2`，默认是 `2`；它只限制同时占用的
L20 数量，不改变 exact-30 seed 集或 attempt policy。

### 11.2 唯一正式提交与 held-array 恢复

只提交完整数组，不按 seed 拆分：

```bash
overlay=configs/REPLACE_WITH_CONFIRMATORY_OVERLAY.yaml
base=configs/REPLACE_WITH_CONFIRMATORY_BASE.yaml
accepted_freeze=/ABS/PATH/TO/accepted-freeze-aggregate.json
gpu_walltime=REPLACE_WITH_HH_MM_SS_OR_D_HH_MM_SS

submission="$(
  bash scripts/hpc4/submit_phase2_confirmatory.sh \
    "${overlay}" \
    "${base}" \
    "${accepted_freeze}" \
    gpu-l20 \
    "${gpu_walltime}"
)"
printf '%s\n' "${submission}"
array_job_id="${submission%%;*}"
test -n "${array_job_id}"
```

可选第六参数只允许字面值 `0-29`；省略即可。`0`、子区间或第二个 array 都会被拒绝。
wrapper 的顺序是：

1. 校验 clean Git、双 config identity、accepted freeze、SIF 和 inventory；
2. 用 `--hold --array=0-29%1|2 --no-requeue` 提交唯一数组；
3. 在 registry lock 下原子提交 exact-30 submission record；
4. fsync registry 后才执行 `scontrol release`。

如果进程在第 3 步之后、第 4 步之前中断，**原样重跑上面的同一命令**。wrapper 会从
immutable registry 找到同一个 held array，复核其 Slurm record 并 release；它不会再次调用
`sbatch`。不要手工提交新数组，也不要自行执行 `scontrol release` 绕过复核。

正式 design root 由已提交 overlay 的 semantic identity 决定：

```bash
design_sha=REPLACE_WITH_COMMITTED_CONFIRMATORY_DESIGN_SHA256
design_root="${PRORM_PROJECT_ROOT}/runs/phase2-confirmatory/${design_sha}"
registry="${design_root}/campaign-registry"

test -f "${registry}/submissions/array-${array_job_id}.json"
test -z "$(find "${registry}/recoveries" -mindepth 1 -maxdepth 1 -print -quit)"
```

### 11.3 监控

只读监控不会改变 campaign：

```bash
squeue -j "${array_job_id}" \
  -o '%.18i %.9P %.24j %.8T %.10M %.6D %R'

sacct -X -j "${array_job_id}" \
  --format=Cluster,JobIDRaw,JobID,State,ExitCode,Elapsed,MaxRSS

tail -n 100 \
  "${PRORM_PROJECT_ROOT}/slurm-logs/prorm-phase2-confirmatory-${array_job_id}_"*.out
```

对 task `t`，seed 固定为 `20260901+t`，canonical path 固定为：

```text
<design-root>/seed-<seed>/attempt-1/job-<array-job-id>_<array-task-id>/
```

完整 success 必须有 `SUCCESS` 与 `phase2-success-terminal.json`；compute failure 在 GPU
publisher 完成原子 rename 后表现为 `FAILURE_PENDING`。隐藏的
`.job-<array-job-id>_<task>.in-progress-<slurm-job-id>` 不是 terminal input。

### 11.4 失败分流：先看 canonical path

对每个 Slurm terminal non-success task，严格按下表处理：

| 观测 | 唯一动作 |
|---|---|
| canonical job directory 含 `SUCCESS` | 已是 terminal success；不再处理 |
| canonical job directory 含 `FAILURE_PENDING` | compute terminalizer |
| canonical job directory 含 `FAILED` 或 `SCHEDULER_FAILED` | 已是 terminal failure；幂等复核即可 |
| canonical job directory不存在，且 `sacct` root row 是 terminal non-success | scheduler terminalizer |
| Slurm 尚非 terminal，或 evidence 不唯一 | 等待/调查；不得猜测分类或创建 terminal |

GPU publisher 在隐藏 staging 中完成文件同步、hash、receipt、ledger、marker 验证与 fsync，
并在 registry lock 下把完整目录原子 rename 到 canonical path。因此，rename 前 hard kill
不会留下 canonical job，由 scheduler terminalizer 负责；一旦 canonical
`FAILURE_PENDING` 存在，scheduler terminalizer 不再拥有该 slot。

classification 是明确的 operator/scientific judgment，不得用默认值掩盖未知原因。
`failure_stage`、`failure_class`、`failure_type` 必须是小写安全 token；
`failure_class` 只能是
`infrastructure|scientific|safety|identity|numerical|software`。先将受控的简短故障描述
做 SHA256；原始描述保留在受控日志/incident record，不把任意日志内容拼入 JSON。

#### A. canonical `FAILURE_PENDING`：compute terminalizer

```bash
seed=REPLACE_WITH_SEED
array_task_id=REPLACE_WITH_TASK_0_TO_29
failed_job_dir="${design_root}/seed-${seed}/attempt-1/job-${array_job_id}_${array_task_id}"
test -f "${failed_job_dir}/FAILURE_PENDING"

message='REPLACE_WITH_CONTROLLED_FAILURE_SUMMARY'
message_sha256="$(printf '%s' "${message}" | sha256sum | awk '{print $1}')"
classification=/ABS/PATH/TO/compute-failure-classification.json
```

`classification` 文件必须恰好包含：

```json
{
  "failure_stage": "replace_with_safe_stage",
  "failure_class": "replace_with_allowed_class",
  "failure_type": "replace_with_safe_type",
  "failure_message_sha256": "REPLACE_WITH_64_HEX_MESSAGE_SHA256"
}
```

然后：

```bash
bash scripts/hpc4/terminalize_phase2_compute_failure.sh \
  "${overlay}" \
  "${failed_job_dir}" \
  "${classification}"

test -f "${failed_job_dir}/FAILED"
test -f "${failed_job_dir}/phase2-failure-terminal.json"
```

该命令只接受已有 `FAILURE_PENDING`、attempt ledger 与 canonical evidence 的目录，原子生成
v2 failure terminal 并把 marker 变为 `FAILED`；重复调用只会确认同一终态。

#### B. canonical job 不存在：scheduler terminalizer

先保存**恰好一行**未经改写的 root record：

```bash
seed=REPLACE_WITH_SEED
array_task_id=REPLACE_WITH_TASK_0_TO_29
sacct_raw=/ABS/PATH/TO/sacct-${array_job_id}_${array_task_id}.txt

sacct -X -n -P \
  -j "${array_job_id}_${array_task_id}" \
  -o Cluster,JobIDRaw,JobID,State,ExitCode \
  > "${sacct_raw}"

test "$(wc -l < "${sacct_raw}")" -eq 1
cat "${sacct_raw}"
```

这一行必须是
`Cluster|JobIDRaw|JobID|State|ExitCode`，属于目标 array task，cluster 是 `hpc4`，
且 state 为 terminal non-success。不要传 step rows、表头、筛选后的摘要或多行输出。

在没有 run manifest/artifact 的 hard-kill 情况下，scheduler classification 的完整 schema
如下；`final_outcome_reveal_started` 必须按真实执行边界显式填写 `true` 或 `false`：

```json
{
  "failure_stage": "scheduler_reconciliation",
  "failure_class": "infrastructure",
  "failure_type": "replace_with_safe_type",
  "failure_message_sha256": "REPLACE_WITH_64_HEX_MESSAGE_SHA256",
  "final_outcome_reveal_started": false,
  "evidence_availability": {
    "schema_version": "phase2-seed-failure-evidence-availability/v1",
    "run_manifest": {
      "status": "unavailable",
      "reason": "not_published_before_hard_termination"
    },
    "artifact_metadata": {
      "status": "unavailable",
      "reason": "not_produced_before_failure"
    },
    "environment_identity": {
      "status": "unavailable",
      "reason": "not_recoverable_from_scheduler_evidence"
    }
  }
}
```

若某项 evidence 实际可用，必须按 schema 填 `available` 的真实 digest/value，不能套用上面的
`unavailable`。执行 reconciliation 前，从原 submission record/正式身份设置：

```bash
export PRORM_GIT_COMMIT=REPLACE_WITH_SUBMITTED_GIT_COMMIT
export PRORM_CLUSTER_NAME=hpc4
export PRORM_PHASE2_ACCEPTED_FREEZE_AGGREGATE_SHA256=REPLACE_WITH_ACCEPTED_FREEZE_SHA256

scheduler_classification=/ABS/PATH/TO/scheduler-failure-classification.json
bash scripts/hpc4/terminalize_phase2_scheduler_failure.sh \
  "${overlay}" \
  "${seed}" \
  "${array_job_id}" \
  "${array_task_id}" \
  1 \
  "${scheduler_classification}" \
  "${sacct_raw}"
```

attempt index 必须是字面值 `1`。该命令要求 canonical job 不存在，验证 held-array
submission、可用时的 execution registry 和 raw `sacct` SHA256，然后在 registry lock 下
原子发布 scheduler failure terminal。它不会授权 retry。

### 11.5 Registry resolver 与 CPU campaign finalizer

只有 30 个 slot 都 terminal 后才提交 finalizer。通常不要直接调用 resolver：
`submit_phase2_campaign_finalize.sh` 会在提交前和 CPU compute 内各运行一次 resolver，验证：

- 只有一个 exact ordered `0..29` attempt-1 submission；
- task-to-seed 映射严格为 `20260901..20260930`；
- Git、design/base、SIF、inventory、accepted-freeze 与 producer hashes 一致；
- `recoveries/` 为空，每个 seed 只有唯一 terminal owner；
- terminal manifest、marker、attempt ledger 与 nested success evidence 的 hashes 一致。

只读审计需要时，resolver 接口为：

```bash
python3 scripts/hpc4/resolve_phase2_campaign_registry.py \
  "${design_root}" \
  "${design_sha}" \
  REPLACE_WITH_BASE_CONFIG_HASH \
  REPLACE_WITH_SUBMITTED_GIT_COMMIT \
  REPLACE_WITH_IMAGE_SHA256 \
  REPLACE_WITH_HF_INVENTORY_SHA256
```

正式 publication 被锁定为一次性的 `campaign-final/`：

```bash
campaign_dir="${design_root}/campaign-final"

finalizer_job="$(
  bash scripts/hpc4/submit_phase2_campaign_finalize.sh \
    "${overlay}" \
    "${base}" \
    "${campaign_dir}/phase2-campaign-terminal.json" \
    "${campaign_dir}/phase2-primary-aggregate.json" \
    amd \
    REPLACE_WITH_WALLTIME
)"
printf '%s\n' "${finalizer_job}"
```

finalizer 完成后：

```bash
sacct -X -j "${finalizer_job%%;*}" \
  --format=JobID,State,ExitCode,Elapsed,MaxRSS
test -f "${campaign_dir}/phase2-campaign-terminal.json"
```

若 30 个 terminal 全部成功，才允许存在
`phase2-primary-aggregate.json`，并由其报告 `passed|not_passed` 与 primary intervals。
若任一 seed terminal failure，campaign 状态必须是
`not_passed_due_to_seed_failure`，`phase2-primary-aggregate.json` 必须不存在，也不得计算、
展示或用替代 seed 补出 primary CI。这是预注册 estimand 的终态，不是待重跑的工程状态。
