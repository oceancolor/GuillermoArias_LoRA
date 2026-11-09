#!/usr/bin/env bash
# FLUX.1-dev 风格 LoRA —— H20 96GB 对照实验（显存富余，不做任何量化/offload）
# 用法：
#   BENCH=1 bash scripts/train_flux.sh        # 先测 200 步，记录 s/it
#   bash scripts/train_flux.sh                # 正式 2000 步
#   DRY_RUN=1 bash scripts/train_flux.sh      # 只打印命令，检查参数
#
# 前置：H20 上需准备 4 个模型文件（见 FLUX_LoRA训练方案.md §3），
# 以及 sd-scripts（git clone kohya-ss/sd-scripts）。本机路径可用
# SD_SCRIPTS=E:/lora_local/sd-scripts 覆盖。
#
# 合规：FLUX.1-dev 为【非商业许可】；商用请改用 FLUX.1-schnell / FLUX.2 klein。
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SD_SCRIPTS="${SD_SCRIPTS:-$ROOT/sd-scripts}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
FLUX_DEV="${FLUX_DEV:-$MODEL_DIR/flux1-dev.safetensors}"
CLIP_L="${CLIP_L:-$MODEL_DIR/clip_l.safetensors}"
T5XXL="${T5XXL:-$MODEL_DIR/t5xxl_fp16.safetensors}"
AE="${AE:-$MODEL_DIR/ae.safetensors}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/configs/dataset_flux.toml}"
BENCH="${BENCH:-0}"

if [[ "$BENCH" == "1" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/flux_bench}"
  NAME="${NAME:-flux_bench}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/flux_mdtr}"
  NAME="${NAME:-mdtr_style_flux_v1}"
fi

STEPS="${STEPS:-2000}"
BATCH="${BATCH:-2}"
LR="${LR:-1e-4}"
DIM="${DIM:-16}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR"

# ---- 模型文件预检：FLUX 训练器启动很慢，缺文件时提前失败更省机时 ----
MISSING=0
for f in "$FLUX_DEV" "$CLIP_L" "$T5XXL" "$AE" "$DATASET_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "[ABORT] missing required file: $f"
    MISSING=1
  fi
done
if [[ "$MISSING" == "1" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "        See FLUX_LoRA训练方案.md §3 for the download manifest."
  exit 1
fi
if [[ "$MISSING" == "1" ]]; then
  echo "[dry-run] continuing despite missing files (parameter check only)"
fi
if [[ ! -f "$SD_SCRIPTS/flux_train_network.py" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "[ABORT] sd-scripts not found at $SD_SCRIPTS (override with SD_SCRIPTS=...)"
  exit 1
fi

if [[ "$BENCH" == "1" ]]; then
  echo "[BENCH] 200 步基准测试：记录 s/it 与显存峰值"
  STEPS=200
  SAVE_EVERY=200
else
  SAVE_EVERY="${SAVE_EVERY:-250}"
fi

# flow-matching 三件套为硬要求：
#   --timestep_sampling flux_shift / --model_prediction_type raw / --guidance_scale 1.0
# 旧版 sd-scripts 若无 flux_shift，改用：--timestep_sampling shift --discrete_flow_shift 3.1582
ARGS=(
  --pretrained_model_name_or_path="$FLUX_DEV"
  --clip_l="$CLIP_L"
  --t5xxl="$T5XXL"
  --ae="$AE"
  --dataset_config="$DATASET_CONFIG"
  --output_dir="$OUTPUT_DIR"
  --output_name="$NAME"

  --network_module=networks.lora_flux
  --network_dim="$DIM"
  --network_alpha="$DIM"
  --network_train_unet_only

  --learning_rate="$LR"
  --lr_scheduler=constant_with_warmup
  --lr_warmup_steps=$(( STEPS / 20 ))
  --optimizer_type=adamw8bit
  --max_grad_norm=1.0

  --mixed_precision=bf16
  --save_precision=bf16
  --save_model_as=safetensors
  --train_batch_size="$BATCH"
  --gradient_accumulation_steps=1
  --max_train_steps="$STEPS"
  --save_every_n_steps="$SAVE_EVERY"
  --seed="$SEED"

  --resolution=1024
  # --cache_latents 必须显式给出：sd-scripts 不会因 *_to_disk 自动开启它。
  # 缺了它，H20 上每个 step 都要重跑一次 VAE，机时被白白烧掉。
  --cache_latents
  --cache_latents_to_disk
  --cache_text_encoder_outputs
  --cache_text_encoder_outputs_to_disk
  --sdpa
  --highvram
  --persistent_data_loader_workers
  --max_data_loader_n_workers=4

  --timestep_sampling=flux_shift
  --model_prediction_type=raw
  --guidance_scale=1.0
)

# 仅当出现 OOM 时才加下面这一行（会额外增加前向重算开销，H20 上属于纯亏损）：
# ARGS+=( --gradient_checkpointing )

echo "=========================================================="
echo " FLUX.1-dev style LoRA | H20 | steps=$STEPS batch=$BATCH dim=$DIM lr=$LR"
echo " output: $OUTPUT_DIR"
echo "=========================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] full launch command (NOT executed):"
  printf ' %q' accelerate launch --num_cpu_threads_per_process=8 \
    "$SD_SCRIPTS/flux_train_network.py" "${ARGS[@]}"
  echo ""
  exit 0
fi

accelerate launch --num_cpu_threads_per_process=8 \
  "$SD_SCRIPTS/flux_train_network.py" "${ARGS[@]}"
