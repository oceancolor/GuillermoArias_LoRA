#!/usr/bin/env bash
# One-time local training environment setup (native Windows / Git Bash)
#
# Design: reuse ComfyUI's torch 2.12.1+cu130 through a .pth file instead of
# copying a ~5 GB venv or rebuilding PyTorch. New packages land in a fresh venv
# and take precedence; the original ComfyUI venv is never modified.
#
# Usage:  bash scripts/setup_local_env.sh
# Rerunnable: every step checks whether it is already done.
# Dependencies are installed into the NEW venv only. The explicit package list
# deliberately avoids sd-scripts' requirements.txt, which may pin an older
# torch and clobber the CUDA 13 / sm_120 build.
set -euo pipefail

cd "$(dirname "$0")/.."

COMFY_ROOT="${COMFY_ROOT:-E:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI}"
BASE_PY="${BASE_PY:-E:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/standalone-env/python.exe}"
COMFY_VENV="$COMFY_ROOT/.venv"
LOCAL_ROOT="${LOCAL_ROOT:-E:/lora_local}"
VENV="$LOCAL_ROOT/venv"
SD_SCRIPTS="$LOCAL_ROOT/sd-scripts"
PY_INDEX="${PY_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
SD_REPO="${SD_REPO:-https://github.com/kohya-ss/sd-scripts.git}"
BASE_MODEL="${BASE_MODEL:-$COMFY_ROOT/models/checkpoints/sd_xl_base_1.0.safetensors}"
PKGS=(
  accelerate diffusers omegaconf voluptuous toml ftfy tqdm imagesize rich arrow
  albumentations opencv-python-headless PyYAML psutil
)

export HF_HOME="$LOCAL_ROOT/cache/hf"
export TEMP="$LOCAL_ROOT/cache/tmp"
export TMP="$TEMP"
export PIP_CACHE_DIR="$LOCAL_ROOT/cache/pip"
mkdir -p "$HF_HOME" "$TEMP" "$PIP_CACHE_DIR" "$HF_HOME/accelerate"

echo "=== 0. Preflight ==="
for p in "$BASE_PY" "$COMFY_VENV/Lib/site-packages" "$BASE_MODEL"; do
  if [[ ! -e "$p" ]]; then echo "[ABORT] missing: $p" >&2; exit 1; fi
  echo "  OK  $p"
done
E_FREE_MB="$(df -Pm /e 2>/dev/null | awk 'NR==2{print $4}' || true)"
echo "  E: free = ${E_FREE_MB:-unknown} MiB"
if [[ -n "$E_FREE_MB" && "$E_FREE_MB" -lt 30000 ]]; then
  echo "[ABORT] E: needs at least 30000 MiB free" >&2
  exit 1
fi

echo "=== 1. Fresh venv ==="
if [[ ! -f "$VENV/Scripts/python.exe" ]]; then
  "$BASE_PY" -m venv "$VENV"
  echo "  created $VENV"
else
  echo "  already exists, skipping"
fi
PY="$VENV/Scripts/python.exe"

echo "=== 2. Link ComfyUI site-packages (read-only reuse) ==="
PTH="$VENV/Lib/site-packages/_comfy_base.pth"
printf '%s\n' "$COMFY_VENV/Lib/site-packages" > "$PTH"
"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| cap', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')"

echo "=== 3. sd-scripts (source archive, multi-channel) ==="
# git:// over HTTPS to GitHub is unreliable here (observed: OpenSSL unexpected
# eof during clone). Download a source zip instead and try three channels.
if [[ ! -f "$SD_SCRIPTS/sdxl_train_network.py" ]]; then
  ZIP="$LOCAL_ROOT/sd-scripts-main.zip"
  SD_URLS=(
    "https://gh-proxy.com/https://github.com/kohya-ss/sd-scripts/archive/refs/heads/main.zip"
    "https://codeload.github.com/kohya-ss/sd-scripts/zip/refs/heads/main"
    "https://github.com/kohya-ss/sd-scripts/archive/refs/heads/main.zip"
  )
  GOT=0
  for U in "${SD_URLS[@]}"; do
    echo "  trying $U"
    if curl -fL --retry 1 --connect-timeout 15 --max-time 300 -o "$ZIP" "$U"; then
      GOT=1
      break
    fi
    echo "  channel failed, trying next"
  done
  if [[ "$GOT" != "1" ]]; then
    echo "[ABORT] could not download sd-scripts from any channel" >&2
    exit 1
  fi
  "$PY" -c "
import pathlib, shutil, sys, zipfile
root, zipf = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with zipfile.ZipFile(zipf) as z:
    bad = z.testzip()
    if bad: raise SystemExit('corrupt archive member: ' + bad)
    tmp = root / '_sd_extract'
    z.extractall(tmp)
src = next(p for p in (root / '_sd_extract').iterdir()
           if p.is_dir() and (p / 'sdxl_train_network.py').exists())
dst = root / 'sd-scripts'
if dst.exists(): shutil.rmtree(dst)
shutil.move(str(src), str(dst))
shutil.rmtree(root / '_sd_extract')
print('  extracted to', dst)
" "$LOCAL_ROOT" "$ZIP" || { echo "[ABORT] extraction failed" >&2; exit 1; }
else
  echo "  already present, skipping"
fi

echo "=== 4. Training dependencies (explicit set, never touch torch) ==="
"$PY" -m pip install --disable-pip-version-check --timeout 40 --retries 3 \
  -i "$PY_INDEX" "${PKGS[@]}" || {
  echo "[warn] package install failed; rerun this script after fixing the error" >&2
  exit 1
}

echo "=== 5. Non-interactive accelerate config ==="
ACC_CFG="$HF_HOME/accelerate/default_config.yaml"
if [[ ! -f "$ACC_CFG" ]]; then
  cat > "$ACC_CFG" <<'YAML'
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML
  echo "  wrote $ACC_CFG"
else
  echo "  already exists, skipping"
fi

echo "=== 5b. Pre-cache SDXL tokenizers via hf-mirror ==="
# sdxl_train_network.py loads TWO CLIP tokenizers on startup. The default HF
# endpoint is unreachable through this host's proxy (502 Bad Gateway), so seed
# the cache now; training then runs without any network dependency.
HF_ENDPOINT="https://hf-mirror.com" HF_HOME="$HF_HOME" "$PY" -c "
from transformers import CLIPTokenizer
for m in ['openai/clip-vit-large-patch14',
          'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k']:
    t = CLIPTokenizer.from_pretrained(m)
    print('  OK', m, 'vocab', len(t))
" || { echo "[ABORT] tokenizer pre-cache failed" >&2; exit 1; }

echo "=== 6. Self-check ==="
"$PY" -c "
import sys
mods = ['torch','accelerate','diffusers','transformers','safetensors','einops',
        'numpy','PIL','scipy','toml','voluptuous','omegaconf','yaml','cv2']
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        print('  MISSING', m, type(e).__name__, e); sys.exit(1)
import torch
assert torch.cuda.is_available(), 'CUDA not available in the new venv'
assert torch.__version__.startswith('2.12'), 'torch was downgraded: ' + torch.__version__
print('  torch', torch.__version__, '| device', torch.cuda.get_device_name(0),
      '| cap', torch.cuda.get_device_capability(0))
print('  all modules OK')
" || { echo "[ABORT] self-check failed" >&2; exit 1; }

if "$PY" "$SD_SCRIPTS/sdxl_train_network.py" --help > "$LOCAL_ROOT/train_network_help.txt" 2>&1; then
  echo "  sdxl_train_network.py --help OK"
else
  echo "  [warn] sdxl_train_network.py --help failed; last lines:"
  tail -15 "$LOCAL_ROOT/train_network_help.txt"
fi

echo "=== DONE ==="
echo " venv       : $PY"
echo " sd-scripts : $SD_SCRIPTS"
echo " HF cache   : $HF_HOME"
echo " next       : bash scripts/make_bench_dataset.py --out $LOCAL_ROOT/bench/10_mdtr"
echo "              BENCH=1 DATASET_CONFIG=configs/dataset_local_bench.toml bash scripts/train_local_sdxl.sh"
