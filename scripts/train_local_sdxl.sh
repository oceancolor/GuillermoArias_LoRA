#!/usr/bin/env bash
# SDXL 1.0 style LoRA - LOCAL Windows host (RTX 5060 Ti 8 GB / 16 GB RAM)
# Run with Git Bash. See the local-training assessment report (v1.1) in the project root.
#
# Usage:
#   BENCH=1 bash scripts/train_local_sdxl.sh               # 200-step benchmark, DO THIS FIRST
#   RUN_TAG=g2_main STEPS=7200 SAVE_STATE=1 bash scripts/train_local_sdxl.sh
#   RUN_TAG=g2_main STEPS=7200 RESUME=<state_dir> bash scripts/train_local_sdxl.sh
#
# Overridable env vars:
#   SD_SCRIPTS, TRAIN_SCRIPT, BASE_MODEL, VENV_PY, LOCAL_ROOT, DATASET_CONFIG,
#   VAL_PROMPTS, OUTPUT_DIR, LOG_DIR, CACHE_DIR, RUN_TAG, NAME, STEPS, BATCH,
#   LR, DIM, ALPHA, SEED, SAVE_EVERY, SAVE_STATE, RESUME, BENCH, SAMPLE,
#   MIN_FREE_MIB, WARN_FREE_MIB, FORCE, DRY_RUN, DISABLE_MMAP, LOWRAM, FULL_BF16,
#   NO_HALF_VAE, DL_WORKERS, MAX_GRAD_NORM, CUDA_ALLOC_CONF
#
# IMPORTANT before running: close ComfyUI and every other GPU-holding GUI app.
# For unattended long runs also: pause Windows Update, disable sleep/hibernate,
# and confirm E: has >= 30 GB free. All caches are redirected to E:.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SD_SCRIPTS="${SD_SCRIPTS:-E:/lora_local/sd-scripts}"
# SDXL MUST use sdxl_train_network.py. The plain train_network.py is the
# SD1/SD2 trainer (its Trainer sets is_sdxl = False and loads a v1 tokenizer).
TRAIN_SCRIPT="${TRAIN_SCRIPT:-sdxl_train_network.py}"
BASE_MODEL="${BASE_MODEL:-E:/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI/models/checkpoints/sd_xl_base_1.0.safetensors}"
VENV_PY="${VENV_PY:-E:/lora_local/venv/Scripts/python.exe}"
DATASET_CONFIG="${DATASET_CONFIG:-$ROOT/configs/dataset_local_sdxl.toml}"
VAL_PROMPTS="${VAL_PROMPTS:-$ROOT/configs/val_prompts.txt}"
LOCAL_ROOT="${LOCAL_ROOT:-E:/lora_local}"
LOG_DIR="${LOG_DIR:-$LOCAL_ROOT/logs}"
CACHE_DIR="${CACHE_DIR:-$LOCAL_ROOT/cache}"

# Each scheduled experiment writes to its own directory, so ablations never
# overwrite each other and a crashed group can be resumed independently.
RUN_TAG="${RUN_TAG:-}"
if [[ -n "$RUN_TAG" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/output/$RUN_TAG}"
  NAME="${NAME:-$RUN_TAG}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-$LOCAL_ROOT/output/sdxl_mdtr_local}"
  NAME="${NAME:-mdtr_style_sdxl_local_v1}"
fi

STEPS="${STEPS:-2400}"
BATCH="${BATCH:-1}"
LR="${LR:-1e-4}"
DIM="${DIM:-16}"
ALPHA="${ALPHA:-8}"
SEED="${SEED:-42}"
BENCH="${BENCH:-0}"
SAMPLE="${SAMPLE:-0}"
SAVE_STATE="${SAVE_STATE:-0}"
RESUME="${RESUME:-}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6400}"
WARN_FREE_MIB="${WARN_FREE_MIB:-7000}"
# sd-scripts warns that a non-zero max_grad_norm enables clip_grad_norm, which
# allocates an extra all-gradient-norm workspace. On an 8 GB WDDM card the very
# first optimizer.step() can OOM there even with >6 GiB reported free.
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
# Windows uses spawn for DataLoader workers: each worker re-imports torch and
# duplicates the parent's address space. On a 16 GB / 34.77 GB commit-limit host
# that is enough to push the first CUDA allocation over the edge. 0 is the
# safest setting (and latent/TE caches are already on disk, so workers add
# almost nothing); raise it only on hosts with more RAM.
DL_WORKERS="${DL_WORKERS:-0}"
# SDXL's fp16 VAE is NaN-prone, but under --full_bf16 the VAE dtype becomes
# bf16, which has fp32's exponent range and does not need --no_half_vae.
# Leaving the flag OFF avoids an extra fp32 VAE copy on the CPU.
NO_HALF_VAE="${NO_HALF_VAE:-0}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CACHE_DIR"

# ---- 1. VRAM preflight: the local card only has 8151 MiB ----
# Measured 2026-11-09 after ComfyUI exited: used=1199, free=6694. That 1.2 GB
# is the Windows desktop compositor / browsers / chat clients; under WDDM those
# allocations are evictable when CUDA requests memory, so 6400 is a realistic
# pass line. 7000+ is still the comfortable target and produces a warning.
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
TOTAL="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]' || true)"
if [[ -n "$USED" && -n "$TOTAL" ]]; then
  FREE=$((TOTAL - USED))
  echo "[preflight] VRAM total=${TOTAL}MiB used=${USED}MiB free=${FREE}MiB"
  if (( FREE < MIN_FREE_MIB )); then
    echo "[ABORT] free VRAM ${FREE}MiB < ${MIN_FREE_MIB}MiB."
    echo "        Close ComfyUI, browsers with HW acceleration and games, then retry."
    if [[ "${FORCE:-0}" != "1" ]]; then exit 1; fi
    echo "        FORCE=1 set, continuing anyway (OOM is likely)."
  elif (( FREE < WARN_FREE_MIB )); then
    echo "[warn] free VRAM ${FREE}MiB < ${WARN_FREE_MIB}MiB (comfortable target)."
    echo "        WDDM can evict desktop allocations, so training will be attempted."
    echo "        If the first 200 steps OOM: close WeChat / Steam / browser HW accel,"
    echo "        or use resolution 896, or raise MIN_FREE_MIB to abort early."
  fi
else
  echo "[warn] nvidia-smi unavailable - skipping VRAM preflight"
fi

# ---- 2. Long-run readiness checks (warnings only, never change OS settings) ----
if [[ "$BENCH" != "1" ]]; then
  E_FREE_MB="$(df -Pm /e 2>/dev/null | awk 'NR==2{print $4}' || true)"
  if [[ -n "$E_FREE_MB" && "$E_FREE_MB" -lt 30000 ]]; then
    echo "[warn] E: free ${E_FREE_MB}MiB < 30000MiB - checkpoints may fill the disk"
  fi
  echo "[long-run] Make sure Windows Update is paused and sleep/hibernate are off."
  if [[ "$SAVE_STATE" != "1" ]]; then
    echo "[long-run] SAVE_STATE=0 - set SAVE_STATE=1 to allow RESUME after interruption."
  fi
fi

# ---- 3. Redirect every cache off the nearly-full C: drive ----
export HF_HOME="$CACHE_DIR/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_DIR/hf/hub"
# sd-scripts downloads CLIP tokenizers from the HF hub on first run. The
# default endpoint is unreachable through this host's proxy (502); hf-mirror
# works. Files are cached in HF_HOME, so later runs are effectively offline.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TEMP="$CACHE_DIR/tmp"
export TMP="$CACHE_DIR/tmp"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
# expandable_segments is not supported on Windows (torch warns and ignores it).
# max_split_size_mb IS supported and reduces allocator fragmentation on 8 GB
# WDDM hosts; opt in with CUDA_ALLOC_CONF=max_split_size_mb:128.
if [[ -n "${CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"
  echo "[alloc] PYTORCH_CUDA_ALLOC_CONF=$CUDA_ALLOC_CONF"
fi
mkdir -p "$HF_HOME" "$TEMP"

# ---- 4. Launcher ----
# Fall back to PATH accelerate only if the dedicated venv is missing (e.g. the
# user runs this script before setup_local_env.sh).
if [[ -n "$VENV_PY" && -f "$VENV_PY" ]]; then
  LAUNCH=("$VENV_PY" -m accelerate.commands.launch --num_cpu_threads_per_process=4)
else
  echo "[warn] training venv not found at ${VENV_PY:-<unset>}; falling back to PATH accelerate"
  echo "        Run: bash scripts/setup_local_env.sh"
  LAUNCH=(accelerate launch --num_cpu_threads_per_process=4)
fi

if [[ "$BENCH" == "1" ]]; then
  echo "[BENCH] 200 steps - record s/it and peak VRAM, then recompute the schedule"
  STEPS=200
  SAVE_EVERY=200
else
  # 600 keeps a 19,400-step schedule at ~30 checkpoints instead of ~65.
  SAVE_EVERY="${SAVE_EVERY:-600}"
fi

# ---- 5. 8 GB specific arguments ----
# gradient_checkpointing + full latent/text-encoder caching + Adafactor + batch 1
# are mandatory at 8 GB. sdxl_train_network.py DOES support
# --cache_text_encoder_outputs(_to_disk); the plain train_network.py is the
# SD1/SD2 trainer and must NOT be used for SDXL. SDPA (not xformers /
# flash-attn) is the only reliable attention backend on native Windows.
ARGS=(
  --pretrained_model_name_or_path="$BASE_MODEL"
  --dataset_config="$DATASET_CONFIG"
  --output_dir="$OUTPUT_DIR"
  --output_name="$NAME"
  --logging_dir="$LOG_DIR"

  --network_module=networks.lora
  --network_dim="$DIM"
  --network_alpha="$ALPHA"
  --network_train_unet_only

  --learning_rate="$LR"
  --text_encoder_lr=0
  --lr_scheduler=constant_with_warmup
  --lr_warmup_steps=100
  --optimizer_type=adafactor
  --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False"
  --max_grad_norm="$MAX_GRAD_NORM"

  --mixed_precision=bf16
  --save_precision=bf16
  --save_model_as=safetensors
  --train_batch_size="$BATCH"
  --gradient_accumulation_steps=1
  --max_train_steps="$STEPS"
  --save_every_n_steps="$SAVE_EVERY"
  --seed="$SEED"

  --cache_latents
  --cache_latents_to_disk
  --cache_text_encoder_outputs
  --cache_text_encoder_outputs_to_disk
  --gradient_checkpointing
  --sdpa
  --max_data_loader_n_workers="$DL_WORKERS"
)
if (( DL_WORKERS > 0 )); then
  ARGS+=(--persistent_data_loader_workers)
fi
# Only needed for fp16 training; under full_bf16 the VAE is bf16 already.
if [[ "$NO_HALF_VAE" == "1" ]]; then
  ARGS+=(--no_half_vae)
fi

# --disable_mmap_load_safetensors is NOT a free workaround on a 16 GB host.
# In sd-scripts it switches to safetensors.torch.load(open(path,"rb").read()),
# which holds ~7 GB of raw bytes PLUS ~7 GB of tensors at the same time.
# Measured 2026-11-09: that path leaves only 1.02 GB of free physical RAM and
# 5.98 GB of pagefile at its peak; the process then dies with 0xC0000005 while
# injecting text encoder 2. mmap'ed loading takes 1.9s vs 31.2s and leaves
# 8.41 GB of free RAM. An isolated diag of the TE2 conversion/injection passes
# in BOTH modes (517 keys, 0 missing, 0 unexpected), so the crash was memory
# pressure, not a transformers/ABI or key mismatch problem.
# Set DISABLE_MMAP=1 only if mmap genuinely crashes on your host.
if [[ "${DISABLE_MMAP:-0}" == "1" ]]; then
  ARGS+=(--disable_mmap_load_safetensors)
fi

# --lowram loads weights straight into VRAM. Measured 2026-11-09 on this 8 GB
# card: it OOMs while loading U-Net (7.85 GiB allocated, 0 free). Keep it OFF
# by default; only opt in on hosts with more VRAM but little system RAM.
if [[ "${LOWRAM:-0}" == "1" ]]; then
  ARGS+=(--lowram)
fi

# --full_bf16 is the CPU-RAM fix for this 16 GB host. sd-scripts builds the
# U-Net on the meta device in fp32; when dtype=None, accelerate's
# set_module_tensor_to_device() calls value.to(torch.float32) for EVERY weight
# of the fp16 SDXL checkpoint, materialising a second ~10 GB fp32 copy. The
# 2026-11-09 mmap benchmark died there with:
#   RuntimeError: DefaultCPUAllocator: not enough memory:
#   you tried to allocate 6553600 bytes.
# With --full_bf16 the U-Net is loaded in bf16 directly, so no conversion copy
# is made. The only hard requirement is --mixed_precision=bf16 (already set).
# NOTE: the earlier note claiming --no_half_vae was also required was wrong;
# source (sdxl_train_util.match_mixed_precision) asserts only the precision.
# Under full_bf16 the VAE dtype is bf16, so the fp32 VAE flag is unnecessary
# and is now opt-in via NO_HALF_VAE=1.
if [[ "${FULL_BF16:-0}" == "1" ]]; then
  ARGS+=(--full_bf16)
fi

# Optimizer state is small with Adafactor (no second-moment matrix), so saving
# it is cheap insurance for a 10+ hour unattended run.
if [[ "$SAVE_STATE" == "1" && "$BENCH" != "1" ]]; then
  ARGS+=(--save_state)
fi
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume="$RESUME")
  echo "[resume] restoring training state from $RESUME"
fi

# Periodic sample generation competes for the same 8 GB and is OFF by default.
# Evaluate checkpoints later in ComfyUI (already installed locally).
if [[ "$SAMPLE" == "1" && "$BENCH" != "1" ]]; then
  ARGS+=(
    --sample_sampler=euler_a
    --sample_prompts="$VAL_PROMPTS"
    --sample_every_n_steps="$SAVE_EVERY"
  )
fi

SHOW_TAG="${RUN_TAG:-(none)}"
SHOW_STATE="$SAVE_STATE"
[[ "$BENCH" == "1" ]] && SHOW_STATE="off(bench)"
echo "=========================================================="
echo " SDXL style LoRA | LOCAL 8GB | tag=$SHOW_TAG steps=$STEPS batch=$BATCH dim=$DIM lr=$LR"
echo " script  : $TRAIN_SCRIPT"
echo " base    : $BASE_MODEL"
echo " dataset : $DATASET_CONFIG"
echo " output  : $OUTPUT_DIR"
echo " caches  : $CACHE_DIR"
echo " save    : every $SAVE_EVERY steps | save_state=$SHOW_STATE | sample=$SAMPLE"
echo " NOTE    : steps are NOT the H20 3000; they follow views/image, see report sec.3"
echo "=========================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] full launch command (NOT executed):"
  printf ' %q' "${LAUNCH[@]}" "$SD_SCRIPTS/$TRAIN_SCRIPT" "${ARGS[@]}"
  echo ""
  exit 0
fi

"${LAUNCH[@]}" "$SD_SCRIPTS/$TRAIN_SCRIPT" "${ARGS[@]}"
