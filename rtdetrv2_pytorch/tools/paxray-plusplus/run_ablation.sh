#!/usr/bin/env bash
# Unattended PAX-Ray++ OT-assignment ablation: train base, train assign, compare on test.
#
# Sequential on ONE free GPU (the box shares 4x A100; 1-3 were busy at launch).
# Each stage is skipped if it already finished, so the script can be re-run safely.
#
#   nohup tools/paxray-plusplus/run_ablation.sh > output/paxray_ablation.log 2>&1 &
#
# Run from rtdetrv2_pytorch/.
set -u

PY=../.venv/bin/python
CFG_DIR=configs/rtdetrv2
BASE_CFG=$CFG_DIR/rtdetrv2_r50vd_paxray_base.yml
ASSIGN_CFG=$CFG_DIR/rtdetrv2_r50vd_paxray_assign.yml
TEST_JSON=/mnt/data/abi/datasets/PAX-RayPlusPlus/annotations/test.json
OUT=output
mkdir -p $OUT

log() { echo "[$(date '+%F %T')] $*"; }

# Pick a GPU that is actually free: <5 GB used and <20% utilization. Waits rather
# than trampling someone else's job; gives up after ~2 h so the script cannot hang
# forever unnoticed.
pick_gpu() {
  for attempt in $(seq 1 240); do
    while IFS=, read -r idx used util; do
      idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' '); util=$(echo "$util" | tr -d ' ')
      if [ "$used" -lt 5000 ] && [ "$util" -lt 20 ]; then
        echo "$idx"; return 0
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
             --format=csv,noheader,nounits)
    log "no free GPU (attempt $attempt/240); sleeping 30s" >&2
    sleep 30
  done
  return 1
}

# train <name> <config>; retries once at half batch size if CUDA OOMs.
train() {
  local name=$1 cfg=$2
  local outdir="$OUT/rtdetrv2_r50vd_paxray_$name"
  local logf="$OUT/paxray_${name}_train.log"

  if [ -f "$outdir/best.pth" ] && grep -q "test-eval. wrote per-class metrics" "$logf" 2>/dev/null; then
    log "SKIP $name: $outdir/best.pth exists and test-eval already ran"
    return 0
  fi

  local gpu
  gpu=$(pick_gpu) || { log "FATAL: no free GPU for $name after 2 h"; return 1; }
  log "START $name on GPU $gpu -> $logf"

  CUDA_VISIBLE_DEVICES=$gpu $PY tools/scripts/train.py -c "$cfg" > "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] && grep -qi "out of memory" "$logf"; then
    log "$name OOMed at the configured batch size; retrying at total_batch_size=16"
    gpu=$(pick_gpu) || { log "FATAL: no free GPU for $name retry"; return 1; }
    CUDA_VISIBLE_DEVICES=$gpu $PY tools/scripts/train.py -c "$cfg" \
      -u train_dataloader.total_batch_size=16 optimizer.lr=0.0001 \
      > "$logf" 2>&1
    rc=$?
  fi
  [ $rc -eq 0 ] && log "DONE $name (rc=0)" || log "FAILED $name (rc=$rc) -- see $logf"
  return $rc
}

log "=== PAX-Ray++ OT-assignment ablation ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | sed 's/^/    /'

train base   "$BASE_CFG"   ; BASE_RC=$?
train assign "$ASSIGN_CFG" ; ASSIGN_RC=$?

# ---- comparison on the TEST split ----------------------------------------------
# Both configs in one per_tooth_metrics run so they are read at identical thresholds.
# The sweep matters: score_thr 0.5 is miscalibrated for Sinkhorn assignment scores
# (it cost the ortho assign run a large slice of recall), so the low thresholds are
# where an honest base-vs-assign comparison lives.
if [ $BASE_RC -eq 0 ] && [ $ASSIGN_RC -eq 0 ]; then
  gpu=$(pick_gpu) || gpu=0
  log "START test comparison on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu $PY tools/scripts/per_tooth_metrics.py \
    -c "$BASE_CFG" "$ASSIGN_CFG" \
    --split "$TEST_JSON" \
    --thresholds 0.02 0.05 0.1 0.2 0.3 0.4 0.5 \
    --iou 0.5 --per-class \
    --save-csv    "$OUT/paxray_test_per_class.csv" \
    --save-agg-csv "$OUT/paxray_test_aggregate.csv" \
    > "$OUT/paxray_test_comparison.log" 2>&1
  log "comparison rc=$? -> $OUT/paxray_test_comparison.log"

  $PY tools/paxray-plusplus/log_comparison_wandb.py \
    --agg-csv "$OUT/paxray_test_aggregate.csv" \
    --per-class-csv "$OUT/paxray_test_per_class.csv" \
    >> "$OUT/paxray_test_comparison.log" 2>&1
  log "wandb comparison upload rc=$?"
else
  log "SKIPPING comparison: base_rc=$BASE_RC assign_rc=$ASSIGN_RC"
fi

log "=== finished (base_rc=$BASE_RC assign_rc=$ASSIGN_RC) ==="
