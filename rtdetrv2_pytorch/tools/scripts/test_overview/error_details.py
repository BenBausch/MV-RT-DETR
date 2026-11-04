"""Where do the misclassifications come from? (companion of analyze.py)

For every misclassified GT tooth (compute_metrics definition: no same-class prediction
overlaps it, a different-class prediction localizes it at IoU >= 0.5; label = the
highest-IoU such prediction) at the fixed 0.5 threshold, writes:

  confusion_pairs.csv   (model, split, view, gt_fdi, pred_fdi, type, shift, count)
                        shift = pred tooth number - GT tooth number for same-quadrant
                        confusions (negative = labelled as a more mesial tooth)
  neighbour_shift.csv   same-quadrant neighbour swaps per (model, split, view group):
                        mesial (-1) vs distal (+1) counts
  swap_concentration.csv  per (model, split, view, swap type): number of images with
                        >= 1 such swap, and the share taken by the worst image
                        (are the rare swap types a few broken images or diffuse?)
  gt_teeth_per_image.csv  mean number of GT teeth / image and FDI coverage per view

Only aggregate counts are written (no image names). Run from rtdetrv2_pytorch/:
  ../.venv/bin/python tools/scripts/test_overview/error_details.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from analyze import GROUPS, IOU_THR, MODELS, SWAP_TYPES, swap_type, write_csv
from src.data.ortho_views import _FDI_ORDER

SPLITS = ['iop_test', 'ortho_test', 'ortho_test_bflip', 'iop_test_nomirror']
THR = 0.5


def miscls_of_image(rec, thr):
    keep = rec['pred_scores'] >= thr
    pl, pb = rec['pred_labels'][keep], rec['pred_boxes'][keep]
    gl, gb = rec['gt_labels'], rec['gt_boxes']
    out = []
    if not (len(pl) and len(gl)):
        return out
    iou = box_iou(torch.tensor(pb, dtype=torch.float32), torch.tensor(gb, dtype=torch.float32)).numpy()
    for gi in range(len(gl)):
        same = pl == gl[gi]
        if same.any() and iou[same, gi].max() > 0.0:
            continue
        other = np.nonzero(~same)[0]
        if not other.size:
            continue
        j = other[np.argmax(iou[other, gi])]
        if iou[j, gi] >= IOU_THR:
            out.append((int(gl[gi]), int(pl[j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='output/test_overview/preds')
    ap.add_argument('--out', default='output/test_overview/tables')
    args = ap.parse_args()
    preds, out = Path(args.preds), Path(args.out)

    pairs, shifts, conc, teeth = [], [], [], []
    for m in MODELS:
        for s in SPLITS:
            p = preds / f'{m}__{s}.pkl'
            if not p.exists():
                continue
            with open(p, 'rb') as f:
                R = pickle.load(f)['records']
            cnt = Counter()
            per_img = defaultdict(Counter)                  # (view, type) -> Counter(img idx)
            shift_cnt = defaultdict(Counter)
            for i, r in enumerate(R):
                for g, pr in miscls_of_image(r, THR):
                    t = swap_type(g, pr)
                    gq, gn = int(_FDI_ORDER[g][0]), int(_FDI_ORDER[g][1])
                    pq, pn = int(_FDI_ORDER[pr][0]), int(_FDI_ORDER[pr][1])
                    sh = pn - gn if gq == pq else ''
                    cnt[(r['view'], _FDI_ORDER[g], _FDI_ORDER[pr], t, sh)] += 1
                    per_img[(r['view'], t)][i] += 1
                    if t == 'neighbour':
                        for gname, views in GROUPS.items():
                            if r['view'] in views:
                                shift_cnt[gname]['mesial' if sh < 0 else 'distal'] += 1
            for (v, g, pr, t, sh), n in sorted(cnt.items()):
                pairs.append(dict(model=m, split=s, view=v, gt_fdi=g, pred_fdi=pr, type=t, shift=sh, count=n))
            for gname in GROUPS:
                c = shift_cnt[gname]
                shifts.append(dict(model=m, split=s, view_group=gname, mesial=c['mesial'], distal=c['distal']))
            for (v, t), c in sorted(per_img.items()):
                tot = sum(c.values())
                conc.append(dict(model=m, split=s, view=v, type=t, n_teeth=tot, n_images=len(c),
                                 n_images_in_view=sum(r['view'] == v for r in R),
                                 worst_image_count=max(c.values()),
                                 worst_image_share=max(c.values()) / tot,
                                 top2_share=sum(sorted(c.values())[-2:]) / tot))
            if m == MODELS[0]:
                for v in sorted({r['view'] for r in R}):
                    rv = [r for r in R if r['view'] == v]
                    n = np.array([len(r['gt_labels']) for r in rv])
                    present = Counter(_FDI_ORDER[c] for r in rv for c in r['gt_labels'])
                    teeth.append(dict(split=s, view=v, n_imgs=len(rv), mean_teeth=n.mean(), min_teeth=n.min(),
                                      max_teeth=n.max(), n_distinct_fdi=len(present),
                                      fdi_present=' '.join(f'{k}:{present[k]}' for k in _FDI_ORDER if present[k])))
            print(f'done {m} {s}')
    write_csv(out / 'confusion_pairs.csv', pairs)
    write_csv(out / 'neighbour_shift.csv', shifts)
    write_csv(out / 'swap_concentration.csv', conc)
    write_csv(out / 'gt_teeth_per_image.csv', teeth)


if __name__ == '__main__':
    main()
