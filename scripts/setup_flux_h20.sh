#!/usr/bin/env bash
# H20 上准备 FLUX.1-dev 正式对照实验（在【H20 机器】上运行，不是本机 Windows）
#
# 目标：把 scripts/train_flux.sh 需要的 4 个模型文件和 sd-scripts 准备好。
#   flux1-dev.safetensors / ae.safetensors  -> black-forest-labs/FLUX.1-dev (gated)
#   clip_l.safetensors / t5xxl_fp16.safetensors -> comfyanonymous/flux_text_encoders
#
# ⚠️ 官方 BFL 仓库是 gated：需先在网页接受 FLUX.1 [dev] 非商业许可，
#    并用 `huggingface-cli login` 或 HF_TOKEN 环境变量提供 token。
#    若暂时拿不到 token，可用 MODE=fp8 走非 gated 的一体化拆分方案，
#    但 fp8 会损失质量，只适合先跑通链路，不建议作为正式对照结果。
#
# 用法：
#   HF_TOKEN=hf_xxx bash scripts/setup_flux_h20.sh
#   MODE=fp8 bash scripts/setup_flux_h20.sh          # 非 gated 降级方案
#   SKIP_MODELS=1 bash scripts/setup_flux_h20.sh     # 模型已手动放好，只装 sd-scripts
set -euo pipefail

ROOT="${H20_ROOT:-$(pwd)}"
SD_SCRIPTS="${SD_SCRIPTS:-$ROOT/sd-scripts}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
MODE="${MODE:-official}"
PY="${PY:-python3}"

mkdir -p "$MODEL_DIR"

echo "=========================================================="
echo " FLUX.1-dev setup for H20 | ROOT=$ROOT | MODE=$MODE"
echo "=========================================================="

# ---- 1. 环境体检 ----
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo "[warn] nvidia-smi not found - are you on the H20 host?"
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null \
  || echo "[warn] torch not importable yet - install sd-scripts requirements first"

FREE_MB="$(df -Pm "$ROOT" | awk 'NR==2{print $4}')"
echo "[disk] $ROOT free ${FREE_MB}MiB"
if [[ "$FREE_MB" -lt 60000 ]]; then
  echo "[warn] 建议至少 60 GB 空闲（4 个模型约 34 GB + 训练输出 / 缓存）"
fi

# ---- 2. sd-scripts ----
if [[ ! -f "$SD_SCRIPTS/flux_train_network.py" ]]; then
  echo "[sd-scripts] cloning kohya-ss/sd-scripts ..."
  git clone https://github.com/kohya-ss/sd-scripts.git "$SD_SCRIPTS"
  (cd "$SD_SCRIPTS" && "$PY" -m pip install -r requirements.txt)
else
  echo "[sd-scripts] already present: $SD_SCRIPTS"
fi

# ---- 3. 模型 ----
if [[ "${SKIP_MODELS:-0}" == "1" ]]; then
  echo "[models] SKIP_MODELS=1 - assuming files are already in $MODEL_DIR"
else
  "$PY" -c "import huggingface_hub" 2>/dev/null || "$PY" -m pip install -U huggingface_hub

  if [[ "$MODE" == "official" ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
      echo "[ABORT] MODE=official 需要 HF_TOKEN（BFL 仓库 gated）。"
      echo "        先到 https://huggingface.co/black-forest-labs/FLUX.1-dev 接受许可，"
      echo "        然后 HF_TOKEN=hf_xxx bash scripts/setup_flux_h20.sh"
      echo "        或改用 MODE=fp8 走非 gated 降级方案。"
      exit 1
    fi
    echo "[models] downloading official fp16 weights (about 34 GB) ..."
    HF_TOKEN="$HF_TOKEN" "$PY" - "$MODEL_DIR" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
out = sys.argv[1]
jobs = [
    ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
    ("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
    ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
]
for repo, name in jobs:
    print("  ", repo, name, flush=True)
    p = hf_hub_download(repo_id=repo, filename=name, local_dir=out,
                        token=os.environ.get("HF_TOKEN"))
    print("   ->", p, flush=True)
PY
  elif [[ "$MODE" == "fp8" ]]; then
    echo "[models] non-gated fp8 bundle (about 22.4 GB) ..."
    "$PY" - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
out = sys.argv[1]
for repo, name in [
    ("Comfy-Org/flux1-dev", "flux1-dev-fp8.safetensors"),
    ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
    ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn.safetensors"),
]:
    print("  ", repo, name, flush=True)
    print("   ->", hf_hub_download(repo_id=repo, filename=name, local_dir=out), flush=True)
PY
    echo "[models] splitting bundle into DiT + AE ..."
    "$PY" "$ROOT/scripts/split_flux_checkpoint.py" \
      --src "$MODEL_DIR/flux1-dev-fp8.safetensors" \
      --out-dir "$MODEL_DIR" --parts dit,ae
    echo "[models] fp8 模式需覆盖环境变量："
    echo "  FLUX_DEV=$MODEL_DIR/flux1-dev-fp8-dit.safetensors"
    echo "  T5XXL=$MODEL_DIR/t5xxl_fp8_e4m3fn.safetensors"
    echo "  并在 train_flux.sh 参数里加 --fp8_base（质量低于 fp16 正式版）"
  else
    echo "[ABORT] unknown MODE=$MODE (official | fp8)"
    exit 1
  fi
fi

echo ""
echo "=========================================================="
echo " 下一步："
echo "   SD_SCRIPTS=$SD_SCRIPTS MODEL_DIR=$MODEL_DIR \\"
echo "     DRY_RUN=1 bash scripts/train_flux.sh"
echo "   SD_SCRIPTS=$SD_SCRIPTS MODEL_DIR=$MODEL_DIR \\"
echo "     BENCH=1 bash scripts/train_flux.sh      # 200 步记录 s/it"
echo "   SD_SCRIPTS=$SD_SCRIPTS MODEL_DIR=$MODEL_DIR \\"
echo "     bash scripts/train_flux.sh              # 正式 2000 步"
echo ""
echo " 注意：train_flux.sh 默认读 $ROOT/models 下的"
echo " flux1-dev.safetensors / clip_l.safetensors / t5xxl_fp16.safetensors / ae.safetensors"
echo " fp8 模式请用 FLUX_DEV / T5XXL / AE 单独覆盖。"
echo " S0 数据集仍是正式训练的唯一硬门。"
echo "=========================================================="
