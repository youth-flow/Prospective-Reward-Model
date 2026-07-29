#!/usr/bin/env bash
#SBATCH --job-name=prorm-gpu-smoke
#SBATCH --account=sigroup
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

set -euo pipefail
umask 027

: "${PRORM_REPO_ROOT:?}"
: "${PRORM_IMAGE:?}"
: "${PRORM_IMAGE_SHA256:?}"
: "${PRORM_GIT_COMMIT:?}"
: "${PRORM_REPORT_ROOT:?}"
: "${SLURM_JOB_ID:?}"

[[ "${SLURM_JOB_PARTITION}" = "gpu-l20" ]] || {
  echo "GPU smoke requires gpu-l20" >&2
  exit 2
}
printf '%s  %s\n' "${PRORM_IMAGE_SHA256}" "${PRORM_IMAGE}" \
  | sha256sum --check --status
[[ "$(git -C "${PRORM_REPO_ROOT}" rev-parse HEAD)" = "${PRORM_GIT_COMMIT}" ]]
[[ -z "$(git -C "${PRORM_REPO_ROOT}" status --porcelain)" ]]

report="${PRORM_REPORT_ROOT}/gpu-smoke-${SLURM_JOB_ID}.txt"
[[ ! -e "${report}" ]] || {
  echo "refusing existing GPU smoke report: ${report}" >&2
  exit 2
}
exec > >(tee "${report}") 2>&1

visibility=()
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; do
  if [[ -v "${variable}" ]]; then
    visibility+=(--env "${variable}=${!variable}")
  fi
done

echo "job_id=${SLURM_JOB_ID}"
echo "partition=${SLURM_JOB_PARTITION}"
echo "git_commit=${PRORM_GIT_COMMIT}"
echo "image_sha256=${PRORM_IMAGE_SHA256}"
nvidia-smi

apptainer exec --cleanenv --nv \
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT}" \
  --pwd "${PRORM_REPO_ROOT}" \
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" \
  "${visibility[@]}" \
  "${PRORM_IMAGE}" \
  python - <<'PY'
import sys
from importlib.metadata import version

import torch
from datasets import DownloadConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load, save
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

expected = {
    "accelerate": "1.7.0",
    "datasets": "3.6.0",
    "huggingface-hub": "0.31.4",
    "numpy": "1.26.4",
    "peft": "0.15.2",
    "pyarrow": "17.0.0",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.1",
    "transformers": "4.52.3",
}
observed = {name: version(name) for name in expected}
if observed != expected:
    raise SystemExit(f"package lock mismatch: {observed!r}")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"expected Python 3.11, got {sys.version}")
if version("torch").split("+", 1)[0] != "2.7.1" or torch.version.cuda != "12.6":
    raise SystemExit("torch/CUDA runtime does not match the frozen image contract")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("exactly one allocated CUDA GPU must be visible")
if "L20" not in torch.cuda.get_device_name(0):
    raise SystemExit(f"expected NVIDIA L20, got {torch.cuda.get_device_name(0)!r}")
if torch.cuda.get_device_capability(0) != (8, 9):
    raise SystemExit("L20 compute capability must be (8, 9)")
required = (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DownloadConfig,
    LoraConfig,
    get_peft_model,
)
if any(item is None for item in required):
    raise SystemExit("a required Qwen/Skywork/PEFT/Datasets symbol is unavailable")
roundtrip = load(save({"x": torch.arange(8, dtype=torch.float32)}))
if not torch.equal(roundtrip["x"], torch.arange(8, dtype=torch.float32)):
    raise SystemExit("safetensors round trip changed data")
a = torch.randn(256, 256, device="cuda", requires_grad=True)
b = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (a @ b).square().mean()
loss.backward()
torch.cuda.synchronize()
if not torch.isfinite(loss) or a.grad is None or b.grad is None:
    raise SystemExit("CUDA FP32 forward/backward failed")
print("gpu", torch.cuda.get_device_name(0))
print("cuda_fp32_loss", loss.item())
PY

apptainer exec --cleanenv \
  --bind "${PRORM_REPO_ROOT}:${PRORM_REPO_ROOT}" \
  --pwd "${PRORM_REPO_ROOT}" \
  --env "PYTHONPATH=${PRORM_REPO_ROOT}/src" \
  "${PRORM_IMAGE}" \
  python -m smart_reward.cli config-check configs/smoke.yaml

apptainer exec --cleanenv "${PRORM_IMAGE}" python -m pip check
echo "status=passed"
