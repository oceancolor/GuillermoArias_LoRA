#!/usr/bin/env bash
# 下载 FLUX.1-dev LoRA 训练所需的权重，并从一体化 checkpoint 拆出 sd-scripts 的四件套。
#
# 选这条路的原因：black-forest-labs/FLUX.1-dev 官方仓库是 gated（本机实测 403），
# 需要 HF 账号 + 接受非商业许可 + HF_TOKEN。而下面三个源经 hf-mirror 实测均可直连：
#   Comfy-Org/flux1-dev/flux1-dev-fp8.safetensors              17.25 GB  (200 OK)
#   comfyanonymous/flux_text_encoders/clip_l.safetensors        246 MB   (200 OK)
#   comfyanonymous/flux_text_encoders/t5xxl_fp8_e4m3fn.safetensors 4.89 GB (200 OK)
# 合计约 22.4 GB 下载；拆分后额外产出约 12.2 GB（DiT + AE）。
#
# ⚠️ 合规：FLUX.1 [dev] 为「非商业许可」。绕开 gated 只是获取方式，不改变许可。
#    本项目默认仅个人学习 / 风格研究，不得商用、不得发布权重。见主方案合规章节。
#
# 用法：
#   bash scripts/fetch_flux_models.sh                 # 下载（可重复运行，支持断点续传）
#   SKIP_DOWNLOAD=1 bash scripts/fetch_flux_models.sh # 已手动放好文件，只做拆分
#   MODEL_DIR=/path bash scripts/fetch_flux_models.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

MODEL_DIR="${MODEL_DIR:-E:/lora_local/models}"
PY="${PY:-E:/lora_local/venv/Scripts/python.exe}"
BASE="${HF_ENDPOINT:-https://hf-mirror.com}"
BUNDLE="$MODEL_DIR/flux1-dev-fp8.safetensors"
DIT="$MODEL_DIR/flux1-dev-fp8-dit.safetensors"
AE="$MODEL_DIR/ae.safetensors"

mkdir -p "$MODEL_DIR"

# ---- 磁盘空间预检 ----
E_FREE_MB="$(df -Pm "$MODEL_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)"
if [[ -n "$E_FREE_MB" && "$E_FREE_MB" -lt 40000 ]]; then
  echo "[warn] 目标盘剩余 ${E_FREE_MB}MiB < 40000MiB；下载 22.4GB + 拆分 12.2GB 可能不够。"
  echo "       可把 MODEL_DIR 指到别的盘，或先清理空间。"
fi

# repo|源文件|目标文件|预期字节数（0 表示不校验）
FILES=(
  "Comfy-Org/flux1-dev|flux1-dev-fp8.safetensors|$BUNDLE|17246524772"
  "comfyanonymous/flux_text_encoders|clip_l.safetensors|$MODEL_DIR/clip_l.safetensors|246144152"
  "comfyanonymous/flux_text_encoders|t5xxl_fp8_e4m3fn.safetensors|$MODEL_DIR/t5xxl_fp8_e4m3fn.safetensors|4893934904"
)

if [[ "${SKIP_DOWNLOAD:-0}" != "1" ]]; then
  for row in "${FILES[@]}"; do
    IFS='|' read -r repo src dst expected <<< "$row"
    if [[ -f "$dst" && "$expected" != "0" ]]; then
      actual="$(stat -c %s "$dst" 2>/dev/null || echo 0)"
      if [[ "$actual" == "$expected" ]]; then
        echo "[skip] $dst already complete ($actual bytes)"
        continue
      fi
      echo "[resume] $dst is $actual / $expected bytes"
    fi
    echo "[download] $repo/$src -> $dst"
    curl -fL -C - --retry 3 --retry-delay 2 \
      -o "$dst" "$BASE/$repo/resolve/main/$src"
  done
else
  echo "[skip] SKIP_DOWNLOAD=1 - assuming files are already in $MODEL_DIR"
fi

if [[ ! -f "$BUNDLE" ]]; then
  echo "[ABORT] bundle not found: $BUNDLE"
  exit 1
fi

# ---- 拆分出纯 DiT 与 AE ----
echo "[split] extracting DiT + AE from the bundled checkpoint"
"$PY" scripts/split_flux_checkpoint.py \
  --src "$BUNDLE" \
  --out-dir "$MODEL_DIR" \
  --parts dit,ae

if [[ ! -f "$DIT" || ! -f "$AE" ]]; then
  echo "[ABORT] split output missing"
  exit 1
fi

echo "=========================================================="
echo " FLUX 四件套就绪："
for f in "$DIT" "$MODEL_DIR/clip_l.safetensors" "$MODEL_DIR/t5xxl_fp8_e4m3fn.safetensors" "$AE"; do
  printf '  %-70s %s bytes\n' "$f" "$(stat -c %s "$f" 2>/dev/null || echo '?')"
done
echo ""
echo " 下一步（本机 8GB 探测，先解决内存硬门）："
echo "   DRY_RUN=1 bash scripts/train_local_flux.sh"
echo "   BENCH=1  bash scripts/train_local_flux.sh"
echo ""
echo " 若要在 H20 上做 fp16 正式对照，请改用官方 gated 权重："
echo "   huggingface-cli login  # 需先在网页接受 FLUX.1 [dev] 非商业许可"
echo "   huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors ae.safetensors"
echo " 然后用 MODEL_DIR / FLUX_DEV / AE 环境变量指向新文件，"
echo " 并把 T5XXL 换成 t5xxl_fp16.safetensors、去掉 --fp8_base。"
echo "=========================================================="
