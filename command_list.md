cd rtdetrv2_pytorch
screen -dmS r50_base bash -c 'CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tools/scripts/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_ortho_base.yml 2>&1 | tee output/r50_base_train.log'

=================================================

cd rtdetrv2_pytorch
screen -dmS r50_base bash -c 'CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tools/scripts/train.py -c configs/rtdetrv2/rtdetrv2_r50vd_ortho_assign.yml 2>&1 | tee output/r50_assign_train.log'