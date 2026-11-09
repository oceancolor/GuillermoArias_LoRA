#!/usr/bin/env bash
# Unattended 10h+ local SDXL experiment schedule (RTX 5060 Ti 8GB / 16GB RAM)
# Run with Git Bash. See the local-training assessment report (v1.1) sec.4.1 / sec.6.
#
# Prerequisite: run the 200-step benchmark FIRST and pass its measured s/it.
#   BENCH=1 bash scripts/train_local_sdxl.sh
#   S_IT=2.0 DRY_RUN=1 bash scripts/run_local_ablations.sh     # preview plan
#   S_IT=2.0 bash scripts/run_local_ablations.sh               # run the window
#
# 2026-11-10: the 768-bucket benchmark completed at 0.73 s/it (report
# appendix C). Do NOT feed that number into S_IT for the real 1024 run -
# rerun BENCH=1 with configs/dataset_local_bench.toml (1024) first.
#
# Overridable env vars:
#   S_IT (required), PLAN (auto|full|trim|light|minimal), MAX_HOURS, DRY_RUN,
#   FORCE_REDO, INTERVAL, SAVE_STATE, LOCAL_ROOT, DONE_DIR, LOG_DIR,
#   DATASET_SINGLE, SINGLE_DIR
#
# Group-level resume: a completed group leaves <tag>.done; rerunning the script
# skips those groups. A crashed group leaves <tag>.failed and is retried next time.
# GPU temperature / VRAM are sampled every INTERVAL seconds into <tag>_gpu.csv.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

LOCAL_ROOT="${LOCAL_ROOT:-E:/lora_local}"
DONE_DIR="${DONE_DIR:-$LOCAL_ROOT/schedule}"
LOG_DIR="${LOG_DIR:-$LOCAL_ROOT/logs}"
S_IT="${S_IT:-}"
PLAN="${PLAN:-auto}"
MAX_HOURS="${MAX_HOURS:-10.5}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_REDO="${FORCE_REDO:-0}"
INTERVAL="${INTERVAL:-60}"
SAVE_STATE="${SAVE_STATE:-1}"
DATASET_SINGLE="${DATASET_SINGLE:-$ROOT/configs/dataset_local_single_palette.toml}"
SINGLE_DIR="${SINGLE_DIR:-$ROOT/dataset/train/10_mdtr_single}"
DEFAULT_CFG="$ROOT/configs/dataset_local_sdxl.toml"

if [[ -z "$S_IT" ]]; then
  echo "[ABORT] S_IT is required. Run 'BENCH=1 bash scripts/train_local_sdxl.sh' first," >&2
  echo "        then pass the measured seconds/iteration, e.g. S_IT=2.0 ..." >&2
  exit 2
fi
if ! awk -v s="$S_IT" 'BEGIN{exit !(s+0>0)}'; then
  echo "[ABORT] S_IT must be a positive number, got: $S_IT" >&2
  exit 2
fi

# tag|steps|dim|alpha|lr|seed|dataset_config_override|bench
# G0/G1/G2 are mandatory; ablations are cut from the tail when time is short.
# NOTE: never rename this array to GROUPS - that is a bash built-in holding the
# current user's group IDs, and assigning to it is silently ignored.
RUN_GROUPS=(
  "g0_bench|200|16|8|1e-4|42||1"
  "g1_pilot|1200|16|8|1e-4|42||0"
  "g2_main_v80|7200|16|8|1e-4|42||0"
  "g3_dim32|3600|32|16|1e-4|42||0"
  "g4_palette_single|3600|16|8|1e-4|42|$DATASET_SINGLE|0"
  "g5_lr2x_seed43|3600|16|8|2e-4|43||0"
)

# Pick the largest plan that fits MAX_HOURS (0.5h reserved for evaluation images).
# full > trim > light > minimal; G0/G1/G2 are never cut.
PLAN_INFO="$(awk -v s="$S_IT" -v m="$MAX_HOURS" -v p="$PLAN" 'BEGIN{
  name[1]="full";    st[1]=19400; keep[1]=6
  name[2]="trim";    st[2]=15800; keep[2]=5
  name[3]="light";   st[3]=12200; keep[3]=4
  name[4]="minimal"; st[4]=8600;  keep[4]=3
  sel=0
  if (p!="auto") { for(i=1;i<=4;i++) if(name[i]==p) sel=i }
  else { for(i=1;i<=4;i++){ h=st[i]*s/3600+0.5; if(h<=m){sel=i;break} } }
  if (sel==0 && p=="auto") sel=4
  if (sel==0) { print "BADPLAN 0 0"; exit }
  printf "%s %d %.2f\n", name[sel], keep[sel], st[sel]*s/3600+0.5
}')"
read -r PLAN_CHOSEN KEEP EST_HOURS <<<"$PLAN_INFO"
if [[ "$PLAN_CHOSEN" == "BADPLAN" ]]; then
  echo "[ABORT] unknown PLAN '$PLAN' (use auto|full|trim|light|minimal)" >&2
  exit 2
fi

# G4 needs an optional single-palette subset; skip cleanly when it is absent.
shopt -s nullglob
SINGLE_IMGS=("$SINGLE_DIR"/*.png "$SINGLE_DIR"/*.jpg "$SINGLE_DIR"/*.jpeg "$SINGLE_DIR"/*.webp)
shopt -u nullglob
G4_OK=0
if [[ -f "$DATASET_SINGLE" && ${#SINGLE_IMGS[@]} -gt 0 ]]; then G4_OK=1; fi

echo "=========================================================="
echo " LOCAL LONG-RUN SCHEDULE"
echo " measured s/it = $S_IT | window = ${MAX_HOURS}h | plan = $PLAN_CHOSEN ($KEEP groups)"
echo " estimated wall time (incl. 0.5h eval) = ${EST_HOURS}h"
if awk -v h="$EST_HOURS" -v m="$MAX_HOURS" 'BEGIN{exit !(h>m)}'; then
  echo " [warn] estimate exceeds the window - consider PLAN=light/minimal"
fi
if [[ "$G4_OK" != "1" ]]; then
  echo " [info] G4 skipped: build dataset/train/10_mdtr_single/ to enable it"
fi
echo "=========================================================="
printf "%-22s %7s %5s %5s %8s %6s  %s\n" "GROUP" "STEPS" "DIM" "ALPHA" "LR" "SEED" "STATUS"
for line in "${RUN_GROUPS[@]:0:$KEEP}"; do
  IFS='|' read -r TAG ST DIM ALPHA LR SEED CFG ISBENCH <<<"$line"
  STATUS="run"
  [[ -f "$DONE_DIR/$TAG.done" && "$FORCE_REDO" != "1" ]] && STATUS="skip(done)"
  if [[ -n "$CFG" && "$G4_OK" != "1" ]]; then STATUS="skip(no subset)"; fi
  # Keep skip(...) states visible in dry-run so the preview matches a real rerun.
  if [[ "$DRY_RUN" == "1" && "$STATUS" == "run" ]]; then STATUS="dry-run"; fi
  printf "%-22s %7s %5s %5s %8s %6s  %s\n" "$TAG" "$ST" "$DIM" "$ALPHA" "$LR" "$SEED" "$STATUS"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] no training started."
  exit 0
fi

mkdir -p "$DONE_DIR" "$LOG_DIR"
MON_PID=""
trap 'if [[ -n "${MON_PID:-}" ]]; then kill "$MON_PID" 2>/dev/null || true; fi' EXIT INT TERM

start_monitor() {
  # Split the locals: referring to $tag inside the same `local` statement is an
  # unbound variable under `set -u` on some bash builds.
  local tag="$1"
  local log="$LOG_DIR/${tag}_gpu.csv"
  echo "timestamp,temp_c,mem_used_mib,util_gpu_pct,clock_sm_mhz" > "$log"
  (
    while true; do
      line="$(nvidia-smi --query-gpu=temperature.gpu,memory.used,utilization.gpu,clocks.sm \
              --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
      if [[ -n "$line" ]]; then
        printf "%s,%s\n" "$(date +%FT%T)" "$line" >> "$log"
      fi
      sleep "$INTERVAL"
    done
  ) &
  MON_PID=$!
}

FAILED=0
for line in "${RUN_GROUPS[@]:0:$KEEP}"; do
  IFS='|' read -r TAG ST DIM ALPHA LR SEED CFG ISBENCH <<<"$line"
  DONE="$DONE_DIR/$TAG.done"
  if [[ -f "$DONE" && "$FORCE_REDO" != "1" ]]; then
    echo "[skip] $TAG already done"
    continue
  fi
  if [[ -n "$CFG" && "$G4_OK" != "1" ]]; then
    echo "[skip] $TAG - single-palette subset missing"
    continue
  fi
  # Clearing a stale marker must never abort the whole window under set -e.
  rm -f "$DONE_DIR/$TAG.failed" 2>/dev/null || true
  echo ""
  echo "=== [$TAG] steps=$ST dim=$DIM alpha=$ALPHA lr=$LR seed=$SEED bench=$ISBENCH ==="
  start_monitor "$TAG"
  if env \
      RUN_TAG="$TAG" STEPS="$ST" DIM="$DIM" ALPHA="$ALPHA" LR="$LR" SEED="$SEED" \
      BENCH="$ISBENCH" SAVE_STATE="$SAVE_STATE" SAVE_EVERY=600 \
      DATASET_CONFIG="${CFG:-$DEFAULT_CFG}" \
      LOCAL_ROOT="$LOCAL_ROOT" LOG_DIR="$LOG_DIR" \
      bash "$ROOT/scripts/train_local_sdxl.sh" 2>&1 | tee "$LOG_DIR/${TAG}.log"; then
    touch "$DONE"
    echo "[done] $TAG"
  else
    touch "$DONE_DIR/$TAG.failed"
    FAILED=$((FAILED+1))
    echo "[FAILED] $TAG - see $LOG_DIR/${TAG}.log ; continuing with the next group"
  fi
  if [[ -n "$MON_PID" ]]; then
    kill "$MON_PID" 2>/dev/null || true
    MON_PID=""
  fi
done

echo ""
echo "=== SCHEDULE FINISHED ==="
echo " plan=$PLAN_CHOSEN est=${EST_HOURS}h failed_groups=$FAILED"
echo " checkpoints : $LOCAL_ROOT/output/<tag>/"
echo " gpu logs    : $LOG_DIR/<tag>_gpu.csv"
echo " next        : copy weights into ComfyUI models/loras, generate a fixed-prompt"
echo "               grid with configs/val_prompts.txt, then blind-score each group."
if (( FAILED > 0 )); then exit 1; fi
