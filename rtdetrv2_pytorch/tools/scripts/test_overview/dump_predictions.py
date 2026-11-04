"""Dump raw per-image predictions of ONE trained model on every test-overview split.

Builds the solver from the model's own config + checkpoint exactly like the
in-training test eval (det_solver._test_eval): EMA weights, the config's own
postprocessor, the config's own val transforms (incl. OrientationChannel when the
config has it). Only the dataset (and batch size / workers) is swapped per split:

  iop_val            IOP-Compass val (1000 imgs), mirror_buccal=True  -> threshold choice
  iop_test           IOP-Compass test (500 imgs), mirror_buccal=True  -> sanity check vs wandb
  iop_test_nomirror  IOP-Compass test, mirror_buccal=False (buccal views in the native
                     IOP convention, x0 = 1 - VIEW_X0) -> in-domain positive control
                     for the mirror probe
  ortho_test         OrthoTracked_301_montages test (155 imgs), as stored
  ortho_test_bflip   same, LEFT/RIGHT images + boxes mirrored (OrthoBuccalFlipDetection)

Records are stored like det_engine.evaluate builds them for PerToothMetrics:
predictions at original-image scale from the postprocessor, GT boxes rescaled from
model-input to original scale; plus image metadata (file, view, patient).

One model per process (two YOLO configs in one process break the registry).
Run from rtdetrv2_pytorch/:
  CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tools/scripts/test_overview/dump_predictions.py \
      --model rtdetr_base -c configs/rtdetrv2/rtdetrv2_r50vd_iop_base_bs32.yml \
      -r output/rtdetrv2_r50vd_iop_base_bs32/best.pth
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import torch

import buccal_flip_dataset  # noqa: F401  (registers OrthoBuccalFlipDetection)
from src.core import YAMLConfig
from src.data.ortho_views import IOP_VIEW_TO_NAME, iop_view_token
from src.solver import TASKS

IOP_ROOT = '/mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/IOP-Compass_1024'
IOP_SPLITS = '/mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/splits'
ORTHO_ROOT = '/mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_301_montages'

SPLITS = {
    'iop_val': dict(type='IOPCompassDetection', img_folder=IOP_ROOT,
                    split_file=f'{IOP_SPLITS}/val.txt', mirror_buccal=True),
    'iop_test': dict(type='IOPCompassDetection', img_folder=IOP_ROOT,
                     split_file=f'{IOP_SPLITS}/test.txt', mirror_buccal=True),
    'iop_test_nomirror': dict(type='IOPCompassDetection', img_folder=IOP_ROOT,
                              split_file=f'{IOP_SPLITS}/test.txt', mirror_buccal=False),
    'ortho_test': dict(type='OrthoDetection', img_folder=ORTHO_ROOT,
                       split_file=f'{ORTHO_ROOT}/test.txt'),
    'ortho_test_bflip': dict(type='OrthoBuccalFlipDetection', img_folder=ORTHO_ROOT,
                             split_file=f'{ORTHO_ROOT}/test.txt'),
}


def image_meta(img_info):
    """(view, patient) for an IOP or OrthoTracked COCO image record."""
    fn = img_info['file_name']
    token = iop_view_token(fn)
    if token is not None:
        return IOP_VIEW_TO_NAME[token], Path(fn).parts[0]            # Patient_<n>
    return Path(fn).stem.upper(), Path(fn).parts[0]                    # <patient>/<date>/VIEW.png


@torch.no_grad()
def run_split(cfg, model, postprocessor, device, split, bs, workers):
    vcfg = copy.deepcopy(cfg.yaml_cfg['val_dataloader'])
    transforms = vcfg['dataset']['transforms']
    vcfg['dataset'] = dict(SPLITS[split], transforms=transforms)
    vcfg['total_batch_size'] = bs
    vcfg['num_workers'] = workers
    vcfg['shuffle'] = False
    vcfg['drop_last'] = False
    name = f'overview_{split}_dataloader'
    cfg.yaml_cfg[name] = vcfg
    loader = cfg.build_dataloader(name)
    ds = loader.dataset
    print(f'[{split}] {len(ds)} images, dataset={type(ds).__name__}, transforms={transforms["ops"]}')

    records = []
    t0 = time.time()
    for samples, targets in loader:
        samples = samples.to(device)
        outputs = model(samples)
        orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(device)
        results = postprocessor(outputs, orig_sizes)
        ih, iw = samples.shape[-2], samples.shape[-1]
        for t, r in zip(targets, results):
            ow, oh = t['orig_size'].tolist()
            gtb = t['boxes'].float()
            if gtb.numel():
                gtb = gtb * torch.tensor([ow / iw, oh / ih, ow / iw, oh / ih])
            img_id = int(t['image_id'].flatten()[0])
            info = ds.coco.loadImgs(img_id)[0]
            view, patient = image_meta(info)
            records.append(dict(
                image_id=img_id, file_name=info['file_name'], view=view, patient=patient,
                x0=info.get('x0'), mirrored=bool(info.get('mirrored', False)),
                orig_size=(ow, oh), input_size=(iw, ih), in_channels=int(samples.shape[1]),
                pred_labels=r['labels'].cpu().numpy().astype(int),
                pred_scores=r['scores'].float().cpu().numpy().astype(float),
                pred_boxes=r['boxes'].float().cpu().numpy().astype(float),
                gt_labels=t['labels'].cpu().numpy().astype(int),
                gt_boxes=gtb.cpu().numpy().astype(float),
            ))
    dt = time.time() - t0
    print(f'[{split}] done in {dt:.0f}s ({len(records) / max(dt, 1e-6):.1f} img/s); '
          f'mean #preds/img {np.mean([len(r["pred_labels"]) for r in records]):.1f}, '
          f'input channels {records[0]["in_channels"] if records else "?"}')
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='short model key used in file names')
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('-r', '--resume', required=True)
    ap.add_argument('--splits', nargs='+', default=list(SPLITS))
    ap.add_argument('--out', default='output/test_overview/preds')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path('output/test_overview/solver_tmp') / args.model
    cfg = YAMLConfig(args.config, resume=args.resume, use_wandb=False, tuning=None,
                     output_dir=str(scratch), summary_dir=str(scratch / 'summary'))
    cfg.yaml_cfg['use_wandb'] = False
    cfg.yaml_cfg['val_dataloader']['total_batch_size'] = args.bs
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    solver.eval()   # builds model/EMA/postprocessor, loads the checkpoint (incl. EMA)
    model = (solver.ema.module if solver.ema else solver.model).eval()
    postprocessor = solver.postprocessor.eval()
    print(f'model={args.model} config={args.config} ckpt={args.resume} '
          f'ema={solver.ema is not None} model={type(model).__name__} '
          f'postprocessor={type(postprocessor).__name__}')

    for split in args.splits:
        records = run_split(cfg, model, postprocessor, solver.device, split, args.bs, args.workers)
        path = out / f'{args.model}__{split}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dict(model=args.model, config=args.config, ckpt=args.resume,
                             split=split, dataset=SPLITS[split], records=records), f)
        print(f'[{split}] wrote {path}')


if __name__ == '__main__':
    main()
