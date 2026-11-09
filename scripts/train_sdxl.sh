#!/usr/bin/env bash
# SDXL 1.0 风格 LoRA —— H20 96GB 优化版
# 用法：
#   bash scripts/train_sdxl.sh                # 正式训练 3000 步
#   BENCH=1 bash scripts/train_sdxl.sh        # 先跑 200 步 benchmark 测 s/it（强制第一步）
# 可用环境变量覆盖：BASE_MODEL / STEPS / BATCH / LR / DIM / SEED / OUTPUT_DIR / NAME / TRAIN_SCRIPT
#
# 注意：SDXL 必须用 sdxl_train_network.py。普通 train_network.py 是 SD1/SD2
# 训练器（is_sdxl=False、加载 v1 tokenizer），用它喂 SDXL 底模是错的。
# 首次运行需能访问 HF Hub 下载 CLIP tokenizer；若不可达，先设
#   export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SD_SCRIPTS="${SD_SCRIPTS:-$ROOT/sd-scripts}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-sdxl_train_network.py}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/sd_xl_base_1.0.safetensors}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/configs/dataset_sdxl.toml}"
VAL_PROMPTS="${VAL_PROMPTS:-$ROOT/configs/val_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/sdxl_mdtr}"
LOG_DIR="${LOG_DIR:-$ROOT/output/logs_sdxl_mdtr}"
NAME="${NAME:-mdtr_style_sdxl_v1}"

STEPS="${STEPS:-3000}"
BATCH="${BATCH:-6}"
LR="${LR:-1e-4}"
DIM="${DIM:-64}"
ALPHA="${ALPHA:-32}"
SEED="${SEED:-42}"
BENCH="${BENCH:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [[ "$BENCH" == "1" ]]; then
  echo "[BENCH] 200 步基准测试：记录 s/it 与显存峰值，用于重排机时预算"
  STEPS=200
  SAVE_EVERY=200
else
  SAVE_EVERY="${SAVE_EVERY:-250}"
fi

# H20 关键点：
#   - 不启用 --gradient_checkpointing（显存充足，重算白白损耗本就偏弱的算力）
#   - 不启用 CPU offload / blocks_to_swap / fp8_base
#   - 预缓存 latent 与文本编码输出（H20 算力弱，缓存是「用显存换算力」的核心）
#   - bf16、SDPA、大 batch、持久化 dataloader
ARGS=(
  --pretrained_model_name_or_path="$BASE_MODEL"
  --dataset_config="$DATASET_CONFIG"
  --output_dir="$OUTPUT_DIR"
  --output_name="$NAME"
  --logging_dir="$LOG_DIR"
  --log_with=tensorboard

  --network_module=networks.lora
  --network_dim="$DIM"
  --network_alpha="$ALPHA"
  --network_train_unet_only

  --learning_rate="$LR"
  --text_encoder_lr=0
  --lr_scheduler=cosine_with_restarts
  --lr_warmup_steps=$(( STEPS / 20 ))
  --optimizer_type=AdamW8bit
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
  --cache_latents
  --cache_latents_to_disk
  --cache_text_encoder_outputs
  --cache_text_encoder_outputs_to_disk
  --sdpa
  --persistent_data_loader_workers
  --max_data_loader_n_workers=8

  --sample_sampler=euler_a
  --sample_prompts="$VAL_PROMPTS"
)

if [[ "$BENCH" != "1" ]]; then
  ARGS+=( --sample_every_n_steps="$SAVE_EVERY" )
fi

# 可选消融：取消注释以启用
# ARGS+=( --min_snr_gamma=5 )
# ARGS+=( --noise_offset=0.0357 )
# ARGS+=( --network_args "loraplus_lr_ratio=16" )

echo "=========================================================="
echo " SDXL style LoRA | H20 | steps=$STEPS batch=$BATCH dim=$DIM lr=$LR"
echo " script : $TRAIN_SCRIPT"
echo " model  : $BASE_MODEL"
echo " dataset: $DATASET_CONFIG"
echo " output : $OUTPUT_DIR"
echo "=========================================================="

accelerate launch --num_cpu_threads_per_process=8 \
  "$SD_SCRIPTS/$TRAIN_SCRIPT" "${ARGS[@]}"
