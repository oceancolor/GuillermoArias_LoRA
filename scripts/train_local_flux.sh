#!/usr/bin/env bash
# FLUX.1-dev LoRA —— 本机 RTX 5060 Ti 8GB 可行性探测（不是正式训练）
#
# 官方 8GB 配方：fp8 底模 + fp8 T5-XXL + blocks_to_swap 28 +
# gradient_checkpointing + 全量缓存 + Adafactor + batch1 + 768px。
# 参考 sd-scripts/docs/flux_train_network.md §6.1 (VRAM Usage Optimization)。
#
# ⚠️ 显存不是本机的主要瓶颈，CPU 内存 / 提交上限才是。
#    FLUX.1-dev fp8 底模约 12 GB、T5-XXL fp8 约 5 GB，blocks_to_swap
#    还要把大量 transformer block 常驻 CPU 内存。16 GB 物理内存 +
#    17 GB 页面文件（commit limit 约 35 GB）在这台机器上不够用，
#    SDXL（底模仅 6.5 GB）实测就已经撞到内存墙。
#    下方预检会在不满足条件时直接阻断，避免白跑几小时。
#
# 用法：
#   DRY_RUN=1 bash scripts/train_local_flux.sh       # 只打印命令
#   BENCH=1 bash scripts/train_local_flux.sh         # 200 步 s/it 探测
#   bash scripts/train_local_flux.sh                 # 正式跑（必须先过 S0）
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SD_SCRIPTS="${SD_SCRIPTS:-E:/lora_local/sd-scripts}"
VENV_PY="${VENV_PY:-E:/lora_local/venv/Scripts/python.exe}"
MODEL_DIR="${MODEL_DIR:-E:/lora_local/models}"
# 注意：必须是【拆分后的纯 DiT】文件，不能是 Comfy-Org 的一体化 checkpoint。
# 用 scripts/fetch_flux_models.sh 一键下载 + 拆分即可得到下面四个文件。
FLUX_DEV="${FLUX_DEV:-$MODEL_DIR/flux1-dev-fp8-dit.safetensors}"
CLIP_L="${CLIP_L:-$MODEL_DIR/clip_l.safetensors}"
T5XXL="${T5XXL:-$MODEL_DIR/t5xxl_fp8_e4m3fn.safetensors}"
AE="${AE:-$MODEL_DIR/ae.safetensors}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/configs/dataset_local_flux.toml}"
LOCAL_ROOT="${LOCAL_ROOT:-E:/lora_local}"
LOG_DIR="${LOG_DIR:-$LOCAL_ROOT/logs}"
CACHE_DIR="${CACHE_DIR:-$LOCAL_ROOT/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/output/flux_local}"
NAME="${NAME:-mdtr_style_flux_local_v1}"

STEPS="${STEPS:-1200}"
BATCH="${BATCH:-1}"
LR="${LR:-1e-4}"
DIM="${DIM:-8}"
ALPHA="${ALPHA:-8}"
SEED="${SEED:-42}"
BLOCKS_TO_SWAP="${BLOCKS_TO_SWAP:-28}"
T5_TOKENS="${T5_TOKENS:-256}"
BENCH="${BENCH:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CACHE_DIR" "$CACHE_DIR/tmp"

# ---- 1. 模型文件预检 ----
MISSING=0
for f in "$FLUX_DEV" "$CLIP_L" "$T5XXL" "$AE" "$DATASET_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "[ABORT] missing required file: $f"
    MISSING=1
  fi
done
if [[ "$MISSING" == "1" ]]; then
  echo "        本机没有可直接训练的 FLUX.1-dev 组件。"
  echo "        下载清单见 FLUX_LoRA训练方案.md §3（合计约 17.4 GB）。"
  echo "        注意：已有的 flux-2-klein-4b-nvfp4.safetensors 属于 FLUX.2，"
  echo "        sd-scripts 当前版本没有 FLUX.2 trainer，不能用于本脚本。"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then exit 1; fi
fi

# ---- 2. 内存 / 提交上限预检（FLUX 的真正硬门）----
if [[ -f "$VENV_PY" ]]; then
  MEM="$("$VENV_PY" -c '
import ctypes
class M(ctypes.Structure):
    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),
    ("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),
    ("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),
    ("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),
    ("ullAvailExtendedVirtual",ctypes.c_ulonglong)]
m=M(); m.dwLength=ctypes.sizeof(M)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
g=1024**3
print(f"{m.ullTotalPhys/g:.1f} {m.ullAvailPhys/g:.1f} {m.ullTotalPageFile/g:.1f} {m.ullAvailPageFile/g:.1f}")
' 2>/dev/null || true)"
  if [[ -n "$MEM" ]]; then
    read -r TOTAL_PHYS FREE_PHYS COMMIT_LIMIT FREE_COMMIT <<< "$MEM"
    echo "[preflight] RAM total=${TOTAL_PHYS}GB free=${FREE_PHYS}GB | commit_limit=${COMMIT_LIMIT}GB free=${FREE_COMMIT}GB"
    # commit limit = RAM + pagefile. FLUX fp8 + block swap 需要 >= 48 GB。
    if awk -v c="$COMMIT_LIMIT" 'BEGIN{exit !(c < 48)}'; then
      echo "[BLOCK] commit limit ${COMMIT_LIMIT}GB < 48GB —— 本机 FLUX 会复现 SDXL 的内存耗尽。"
      echo "        处置（任选其一，见 FLUX_LoRA训练方案.md §2）："
      echo "        a) 把页面文件迁到 E: 并增大到 32–48 GB（E: 有约 78 GB 空闲）"
      echo "        b) 关闭占内存程序释放 2–3 GB，并把页面文件一并增大"
      echo "        c) 加装物理内存到 32 GB"
      echo "        若只想验证参数：DRY_RUN=1 bash scripts/train_local_flux.sh"
      if [[ "${DRY_RUN:-0}" != "1" && "${FORCE:-0}" != "1" ]]; then exit 1; fi
      if [[ "${FORCE:-0}" == "1" ]]; then
        echo "        FORCE=1 已设置，继续（大概率在模型加载阶段失败）。"
      else
        echo "        DRY_RUN=1，仅继续打印参数，不执行训练。"
      fi
    fi
  fi
fi

# ---- 3. 显存预检 ----
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
TOTAL_VRAM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
if [[ -n "$USED" && -n "$TOTAL_VRAM" ]]; then
  FREE_VRAM=$((TOTAL_VRAM - USED))
  echo "[preflight] VRAM free=${FREE_VRAM}MiB / ${TOTAL_VRAM}MiB"
  if (( FREE_VRAM < 7000 )); then
    echo "[BLOCK] 显存空闲不足 7000MiB；先退出 ComfyUI / 浏览器硬件加速。"
    if [[ "${DRY_RUN:-0}" != "1" && "${FORCE:-0}" != "1" ]]; then exit 1; fi
  fi
fi

# ---- 4. 缓存全部重定向到 E: ----
export HF_HOME="$CACHE_DIR/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TEMP="$CACHE_DIR/tmp"
export TMP="$CACHE_DIR/tmp"

if [[ "$BENCH" == "1" ]]; then
  echo "[BENCH] 200 步 s/it 探测（合成图集不含风格信号，绝不用于正式训练）"
  STEPS=200
  SAVE_EVERY=200
  # 只有用户显式指定 DATASET_CONFIG 时才沿用；否则切到 768px 合成图集配置。
  if [[ "$DATASET_CONFIG" == "$ROOT/configs/dataset_local_flux.toml" ]]; then
    DATASET_CONFIG="$ROOT/configs/dataset_local_flux_bench.toml"
  fi
else
  SAVE_EVERY="${SAVE_EVERY:-300}"
fi

# ---- 5. 8GB 专用参数 ----
ARGS=(
  --pretrained_model_name_or_path="$FLUX_DEV"
  --clip_l="$CLIP_L"
  --t5xxl="$T5XXL"
  --ae="$AE"
  --dataset_config="$DATASET_CONFIG"
  --output_dir="$OUTPUT_DIR"
  --output_name="$NAME"
  --logging_dir="$LOG_DIR"

  --network_module=networks.lora_flux
  --network_dim="$DIM"
  --network_alpha="$ALPHA"
  --network_train_unet_only

  --learning_rate="$LR"
  --lr_scheduler=constant_with_warmup
  --lr_warmup_steps=$(( STEPS / 20 ))
  --optimizer_type=adafactor
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
  --max_grad_norm=1.0

  --mixed_precision=bf16
  --save_precision=bf16
  --save_model_as=safetensors
  --train_batch_size="$BATCH"
  --gradient_accumulation_steps=1
  --max_train_steps="$STEPS"
  --save_every_n_steps="$SAVE_EVERY"
  --seed="$SEED"

  # 省显存四件套：fp8 底座 + block swap + 重算 + 全量缓存
  --fp8_base
  --blocks_to_swap="$BLOCKS_TO_SWAP"
  --gradient_checkpointing
  --cache_latents
  --cache_latents_to_disk
  --cache_text_encoder_outputs
  --cache_text_encoder_outputs_to_disk
  --t5xxl_max_token_length="$T5_TOKENS"
  --sdpa
  --max_data_loader_n_workers=2
  --persistent_data_loader_workers

  # FLUX flow-matching 三件套（缺一不可）
  --timestep_sampling=flux_shift
  --model_prediction_type=raw
  --guidance_scale=1.0
)
# 禁止同时加 --cpu_offload_checkpointing：flux_train_network.py 会断言冲突。
# 禁止 --highvram / --lowram：前者是 H20 专用，后者在 8GB 卡上必 OOM。

echo "=========================================================="
echo " FLUX.1-dev LoRA | LOCAL 8GB PROBE | steps=$STEPS batch=$BATCH dim=$DIM"
echo " fp8_base=on blocks_to_swap=$BLOCKS_TO_SWAP t5_tokens=$T5_TOKENS"
echo " dataset : $DATASET_CONFIG"
echo " output  : $OUTPUT_DIR"
echo " NOTE    : 探测性质；正式训练仍须先通过 S0 数据门禁"
echo "=========================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] full launch command (NOT executed):"
  printf ' %q' "$VENV_PY" -m accelerate.commands.launch --num_cpu_threads_per_process=4 \
    "$SD_SCRIPTS/flux_train_network.py" "${ARGS[@]}"
  echo ""
  exit 0
fi

"$VENV_PY" -m accelerate.commands.launch --num_cpu_threads_per_process=4 \
  "$SD_SCRIPTS/flux_train_network.py" "${ARGS[@]}"
