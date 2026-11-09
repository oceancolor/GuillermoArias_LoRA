#!/usr/bin/env bash
# FLUX.2 [klein] 4B Base —— 本机 8GB 路径（路径 D）准备脚本
#
# 只做「环境预检 + 下载 + 摆放文件」，**不自动安装 Python 依赖**。
# 原因：本机是 sm_120 / Blackwell + Python 3.13，ai-toolkit 官方安装命令里的
# torch==2.6.0+cu124 不支持 Blackwell，且 scipy==1.12.0 在 py3.13 需从源码编译。
# 依赖安装必须人工确认，详见 FLUX_LoRA训练方案.md §1.1。
#
# 用法：
#   DRY_RUN=1        bash scripts/setup_flux_klein_local.sh   # 只预检、只打印
#   SKIP_DOWNLOAD=1  bash scripts/setup_flux_klein_local.sh   # 只检查已下载文件
#                    bash scripts/setup_flux_klein_local.sh   # 预检 + 下载
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-E:/lora_local}"
MODEL_DIR="${MODEL_DIR:-$LOCAL_ROOT/models/klein4b}"
TOOLKIT_DIR="${TOOLKIT_DIR:-$LOCAL_ROOT/ai-toolkit}"
ZIP="${ZIP:-$LOCAL_ROOT/cache/ai-toolkit-main.zip}"
PY="${PY:-E:/lora_local/venv/Scripts/python.exe}"

BASE_URL="https://hf-mirror.com"
TRANSFORMER_URL="$BASE_URL/black-forest-labs/FLUX.2-klein-base-4B/resolve/main/flux-2-klein-base-4b.safetensors"
AE_URL="$BASE_URL/ai-toolkit/flux2_vae/resolve/main/ae.safetensors"
TOOLKIT_URL="https://gh-proxy.com/https://github.com/ostris/ai-toolkit/archive/refs/heads/main.zip"

# 本机 ComfyUI 里已有的 FLUX.2 VAE（约 321 MiB），与 ai-toolkit 的 ae.safetensors 体积一致
COMFY_VAE="${COMFY_VAE:-E:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI/models/vae/flux2-vae.safetensors}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "=========================================================="
echo " FLUX.2 klein 4B Base | LOCAL 8GB PATH D PREFLIGHT"
echo " models : $MODEL_DIR"
echo " toolkit: $TOOLKIT_DIR"
echo "=========================================================="

# ---- 1. 预检 ----
WARN=0

FREE_VRAM="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
if [[ -n "$FREE_VRAM" ]]; then
  echo "[preflight] VRAM free = ${FREE_VRAM} MiB"
  if (( FREE_VRAM < 7000 )); then
    echo "[WARN] 空闲显存不足 7000 MiB；请先退出 ComfyUI / 浏览器硬件加速。8GB 路径对 shared memory 极其敏感。"
    WARN=1
  fi
fi

FREE_E="$(df -m /e 2>/dev/null | awk 'NR==2{print $4}' || true)"
if [[ -n "$FREE_E" ]]; then
  echo "[preflight] E: free = ${FREE_E} MiB (约 $(( FREE_E / 1024 )) GB)"
  if (( FREE_E < 30000 )); then
    echo "[WARN] E 盘建议至少留 30 GB（权重约 8.1 GB + Qwen3-4B 约 8 GB + 缓存 + checkpoint）。"
    WARN=1
  fi
fi

if [[ -f "$PY" ]]; then
  PYVER="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo unknown)"
  echo "[preflight] existing venv Python = $PYVER"
  if [[ "$PYVER" == "3.13" ]]; then
    echo "[WARN] ai-toolkit 的 scipy==1.12.0 在 Python 3.13 无预编译 wheel，会把 Fortran 编译器当依赖。"
    echo "       处置：把 requirements.txt 里的 pin 放宽为 scipy>=1.14，或另建 Python 3.12 venv。"
    WARN=1
  fi
  echo "[WARN-REMINDER] 不要执行官方的 pip install torch==2.6.0+cu124 —— sm_120 / Blackwell 需要 cu128+。"
  echo "                本机 ComfyUI 已有 torch 2.12.1+cu130，安装依赖时必须保留它。"
fi

echo "[reminder] 页面文件 E:\\pagefile.sys 32–48GB 需重启 Windows 才生效；"
echo "           本路径稳态下 CPU 内存压力远小于 sd-scripts FLUX.1，但启动阶段仍建议先重启。"

if [[ "$WARN" == "1" ]]; then
  echo "[preflight] 有警告项，但均不阻断下载；正式训练前须逐项处理。"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] 仅预检，不下载、不写文件。"
  echo "[dry-run] 下载清单："
  echo "  $TRANSFORMER_URL  (7.75 GB)"
  echo "  $AE_URL           (336 MB)"
  echo "  $TOOLKIT_URL      (36 MB)"
  exit 0
fi

mkdir -p "$MODEL_DIR" "$(dirname "$ZIP")"

# ---- 2. 权重 ----
fetch() {
  local url="$1"
  local dest="$2"
  if [[ -s "$dest" ]]; then
    echo "[skip] 已存在 $dest ($(du -h "$dest" | cut -f1))"
    return 0
  fi
  if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
    echo "[ABORT] SKIP_DOWNLOAD=1 但文件缺失：$dest"
    return 1
  fi
  echo "[download] $url"
  curl -L --fail --retry 3 --retry-delay 3 -C - -o "$dest" "$url"
}

fetch "$TRANSFORMER_URL" "$MODEL_DIR/flux-2-klein-base-4b.safetensors"

if [[ ! -s "$MODEL_DIR/ae.safetensors" ]]; then
  if [[ -s "$COMFY_VAE" ]]; then
    echo "[reuse] 复用本机已有 FLUX.2 VAE：$COMFY_VAE"
    cp "$COMFY_VAE" "$MODEL_DIR/ae.safetensors"
  else
    fetch "$AE_URL" "$MODEL_DIR/ae.safetensors"
  fi
fi

# ---- 3. ai-toolkit 源码 ----
if [[ -d "$TOOLKIT_DIR" && -f "$TOOLKIT_DIR/run.py" ]]; then
  echo "[skip] ai-toolkit 已存在：$TOOLKIT_DIR"
elif [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  echo "[ABORT] SKIP_DOWNLOAD=1 但 ai-toolkit 源码缺失：$TOOLKIT_DIR"
  exit 1
else
  fetch "$TOOLKIT_URL" "$ZIP"
  echo "[unzip] $ZIP -> $LOCAL_ROOT"
  "$PY" - "$ZIP" "$LOCAL_ROOT" <<'PY'
import sys, zipfile
from pathlib import Path
z, root = Path(sys.argv[1]), Path(sys.argv[2])
with zipfile.ZipFile(z) as f:
    f.extractall(root)
top = root / "ai-toolkit-main"
dest = root / "ai-toolkit"
if top.exists() and not dest.exists():
    top.rename(dest)
print("[unzip] OK ->", dest)
PY
fi

# ---- 4. 后续步骤 ----
echo ""
echo "=========================================================="
echo " 文件就绪。后续人工步骤（本脚本刻意不自动执行）："
echo "  1) 在 $TOOLKIT_DIR 建 venv，先装与 sm_120 匹配的 torch（保留 cu128+），"
echo "     再放宽 scipy pin / 显式装 torchaudio，最后装 requirements.txt。"
echo "  2) 首次运行设 HF_ENDPOINT=https://hf-mirror.com，让 toolkit 自动拉取 Qwen3-4B（约 8 GB）。"
echo "  3) 完成 S0 数据集门禁后："
echo "     HF_ENDPOINT=https://hf-mirror.com venv/Scripts/python.exe run.py \\"
echo "       \"C:/Users/Administrator/WorkBuddy/LoRA训练/configs/ai_toolkit_klein_4b_local.yaml\""
echo "  4) 训练中看任务管理器 GPU 页的「专用 / 共享 GPU 内存」："
echo "     共享内存持续上涨 = Windows WDDM 静默溢出，速度会掉 10–50 倍且不报错。"
echo "=========================================================="
