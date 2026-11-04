#!/usr/bin/env bash
# Test overview of the 4 IOP-trained models: dump predictions (one model per GPU, each
# in its own detached screen), then -- once all four are done -- compute the tables.
# Everything survives the launching shell. Run from anywhere:
#   bash rtdetrv2_pytorch/tools/scripts/test_overview/run_all.sh
set -u
cd "$(dirname "$0")/../../.."          # rtdetrv2_pytorch/
PY=../.venv/bin/python
OUT=output/test_overview
LOGS=$OUT/logs
mkdir -p $LOGS $OUT/preds $OUT/tables
rm -f $OUT/done_*

launch() {   # gpu model config ckpt
  local gpu=$1 model=$2 cfg=$3 ckpt=$4
  screen -dmS test_overview_$model bash -c "
    cd $(pwd)
    CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PY tools/scripts/test_overview/dump_predictions.py \
        --model $model -c $cfg -r $ckpt > $LOGS/dump_$model.log 2>&1
    echo \$? > $OUT/done_$model"
}

launch 0 rtdetr_base configs/rtdetrv2/rtdetrv2_r50vd_iop_base_bs32.yml output/rtdetrv2_r50vd_iop_base_bs32/best.pth
launch 1 rtdetr_ot   configs/rtdetrv2/rtdetrv2_r50vd_iop_assign_p20.yml output/rtdetrv2_r50vd_iop_assign_p20/best.pth
launch 2 yolo_base   configs/yolo11/yolo11l_iop_base_bs24.yml           output/yolo11l_iop_base_bs24/best.pth
launch 3 yolo_ot     configs/yolo11/yolo11l_iop_assign_p20_nms_no_orient_bs24.yml output/yolo11l_iop_assign_p20_nms_no_orient_bs24/best.pth

# analysis: wait for all four dumps, then wandb references + tables
screen -dmS test_overview_analysis bash -c "
  cd $(pwd)
  while [ \$(ls $OUT/done_* 2>/dev/null | wc -l) -lt 4 ]; do sleep 20; done
  cat $OUT/done_* > $LOGS/dump_exit_codes.txt
  PYTHONUNBUFFERED=1 $PY tools/scripts/test_overview/pull_wandb_refs.py > $LOGS/wandb_refs.log 2>&1
  PYTHONUNBUFFERED=1 $PY tools/scripts/test_overview/analyze.py > $LOGS/analyze.log 2>&1
  echo \$? > $OUT/done_analysis"
screen -ls | grep test_overview_
