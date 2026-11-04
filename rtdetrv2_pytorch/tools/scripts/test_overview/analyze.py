"""Metrics tables for the IOP-trained models' test overview (from dumped predictions).

Reads output/test_overview/preds/<model>__<split>.pkl (see dump_predictions.py) and
writes CSVs to output/test_overview/tables/:

  thresholds.csv       best-micro-F1 score threshold per model, chosen on IOP val
  totals.csv           per (model, split, thr_kind)             -- whole split
  per_view.csv         per (model, split, thr_kind, view group) -- CENTER/UP/DOWN/LEFT/RIGHT,
                       BUCCAL (L+R), NONBUCCAL (C+U+D), ALL; with patient-bootstrap 95% CI
  per_class.csv        per (model, split, thr_kind, FDI) + a pooled 'wisdom' row (18/28/38/48)
  threshold_sweep.csv  micro P/R/F1 + macro F1 per (model, split, view group, thr 0.05..0.70)
  swaps.csv            misclassified teeth by confusion type, per (model, split, thr_kind, view group)
  contrasts.csv        the mirror-convention contrasts with patient-bootstrap 95% CIs
  summary.md           the key tables in markdown

thr_kind: 'fixed' = 0.5 (training/test protocol), 'val_best' = per-model best micro-F1
threshold on IOP val. IoU threshold 0.5 throughout.

Per-image matching reuses src.misc.per_tooth_eval.match_image (the core of
compute_metrics); group metrics are sums of per-image per-class counts, which is
exactly what compute_metrics does (asserted against compute_metrics for every split).

Run from rtdetrv2_pytorch/:  ../.venv/bin/python tools/scripts/test_overview/analyze.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from torchvision.ops import box_iou
import torch

from src.data.ortho_views import _FDI_ORDER
from src.misc.per_tooth_eval import compute_metrics, match_image

MODELS = ['rtdetr_base', 'rtdetr_ot', 'yolo_base', 'yolo_ot']
MODEL_NAMES = {'rtdetr_base': 'RT-DETR base', 'rtdetr_ot': 'RT-DETR OT',
               'yolo_base': 'YOLO11-L base', 'yolo_ot': 'YOLO11-L OT'}
SPLITS = ['iop_val', 'iop_test', 'iop_test_nomirror', 'ortho_test', 'ortho_test_bflip']
VIEWS = ['CENTER', 'UP', 'DOWN', 'LEFT', 'RIGHT']
GROUPS = {v: (v,) for v in VIEWS}
GROUPS.update(BUCCAL=('LEFT', 'RIGHT'), NONBUCCAL=('CENTER', 'UP', 'DOWN'), ALL=tuple(VIEWS))
GROUP_ORDER = VIEWS + ['BUCCAL', 'NONBUCCAL', 'ALL']
FIXED_THR = 0.5
IOU_THR = 0.5
SWEEP = [round(0.05 * i, 2) for i in range(1, 15)]            # 0.05 .. 0.70
NC = len(_FDI_ORDER)
WISDOM = [_FDI_ORDER.index(c) for c in ('18', '28', '38', '48')]
N_BOOT = 2000
FIELDS = ('tp', 'fp', 'fn', 'correct', 'misloc', 'miscls', 'missed', 'dup')
SWAP_TYPES = ('lr_exact', 'lr_other', 'upper_lower', 'diagonal', 'neighbour', 'same_quad_other')


def swap_type(gt_c, pred_c):
    """Confusion type of a GT tooth labelled as another class."""
    gq, gn = int(_FDI_ORDER[gt_c][0]), int(_FDI_ORDER[gt_c][1])
    pq, pn = int(_FDI_ORDER[pred_c][0]), int(_FDI_ORDER[pred_c][1])
    if gq == pq:
        return 'neighbour' if abs(gn - pn) == 1 else 'same_quad_other'
    pair = {gq, pq}
    if pair in ({1, 2}, {3, 4}):                 # across the midline: left/right swap
        return 'lr_exact' if gn == pn else 'lr_other'
    if pair in ({1, 4}, {2, 3}):
        return 'upper_lower'
    return 'diagonal'


def image_counts(rec, thr):
    """Per-class counts for one image (arrays [NC]) + swap-type counts of its
    misclassified teeth, using the same matching as compute_metrics."""
    tp, fp, fn, kept, correct, misloc, miscls, missed, _ = match_image(
        rec['pred_labels'], rec['pred_scores'], rec['pred_boxes'],
        rec['gt_labels'], rec['gt_boxes'], thr, IOU_THR)
    c = {k: np.zeros(NC, dtype=np.int64) for k in FIELDS}
    for name, d in (('tp', tp), ('fp', fp), ('fn', fn), ('correct', correct),
                    ('misloc', misloc), ('miscls', miscls), ('missed', missed)):
        for k, v in d.items():
            c[name][int(k)] += v
    if len(kept):
        counts = np.bincount(kept, minlength=NC)
        c['dup'] = np.maximum(counts - 1, 0)
    swaps = dict.fromkeys(SWAP_TYPES, 0)
    # label of each misclassified tooth = the highest-IoU other-class kept prediction
    keep = rec['pred_scores'] >= thr
    pl, pb = rec['pred_labels'][keep], rec['pred_boxes'][keep]
    gl, gb = rec['gt_labels'], rec['gt_boxes']
    if len(pl) and len(gl):
        iou = box_iou(torch.tensor(pb, dtype=torch.float32),
                      torch.tensor(gb, dtype=torch.float32)).numpy()
        for gi in range(len(gl)):
            same = pl == gl[gi]
            if same.any() and iou[same, gi].max() > 0.0:
                continue                                   # correct or mislocalized
            other = np.nonzero(~same)[0]
            if not other.size:
                continue
            j = other[np.argmax(iou[other, gi])]
            if iou[j, gi] >= IOU_THR:
                swaps[swap_type(int(gl[gi]), int(pl[j]))] += 1
    n_teeth = len(gl)
    return c, swaps, n_teeth


class SplitCounts:
    """Per-image count arrays for one (model, split, thr)."""

    def __init__(self, records, thr):
        self.records = records
        self.thr = thr
        per = [image_counts(r, thr) for r in records]
        self.c = {k: np.stack([p[0][k] for p in per]) for k in FIELDS}          # [I, NC]
        self.swaps = {k: np.array([p[1][k] for p in per]) for k in SWAP_TYPES}  # [I]
        self.n_teeth = np.array([p[2] for p in per])
        self.views = np.array([r['view'] for r in records])
        self.patients = np.array([r['patient'] for r in records])
        self.has_dup = self.c['dup'].sum(1) > 0

    def mask(self, group):
        return np.isin(self.views, GROUPS[group])


def summarize(c, mask):
    """Metrics of the images in mask, from summed per-class counts (== compute_metrics)."""
    s = {k: c[k][mask].sum(0) for k in FIELDS}
    tp, fp, fn = s['tp'].sum(), s['fp'].sum(), s['fn'].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    active = (s['tp'] + s['fp'] + s['fn']) > 0
    pc_p = np.divide(s['tp'], s['tp'] + s['fp'], out=np.zeros(NC), where=(s['tp'] + s['fp']) > 0)
    pc_r = np.divide(s['tp'], s['tp'] + s['fn'], out=np.zeros(NC), where=(s['tp'] + s['fn']) > 0)
    pc_f1 = np.divide(2 * pc_p * pc_r, pc_p + pc_r, out=np.zeros(NC), where=(pc_p + pc_r) > 0)
    sup = s['tp'] + s['fn']
    return dict(micro_p=p, micro_r=r, micro_f1=f1,
                macro_f1=float(pc_f1[active].mean()) if active.any() else 0.0,
                weighted_f1=float(np.average(pc_f1[active], weights=sup[active])) if sup[active].sum() else 0.0,
                tp=int(tp), fp=int(fp), fn=int(fn),
                correct=int(s['correct'].sum()), misloc=int(s['misloc'].sum()),
                miscls=int(s['miscls'].sum()), missed=int(s['missed'].sum()),
                total_dups=int(s['dup'].sum())), s, pc_p, pc_r, pc_f1


def micro_f1_from(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def boot_group_f1(sc, group, rng_seed):
    """Patient-bootstrap distribution of micro F1 for the images of a view group."""
    m = sc.mask(group)
    pats = np.unique(sc.patients[m])
    idx = {p: i for i, p in enumerate(pats)}
    pid = np.array([idx[p] for p in sc.patients[m]])
    per_pat = np.zeros((len(pats), 3))
    for k, key in enumerate(('tp', 'fp', 'fn')):
        np.add.at(per_pat[:, k], pid, sc.c[key][m].sum(1))
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(pats), size=(N_BOOT, len(pats)))
    tot = per_pat[draws].sum(1)                                # [B, 3]
    return micro_f1_from(tot[:, 0], tot[:, 1], tot[:, 2]), pats, per_pat


def boot_paired_delta(sc_a, sc_b, group, rng_seed):
    """Paired (same patients) bootstrap of F1_b - F1_a for a view group."""
    ma, mb = sc_a.mask(group), sc_b.mask(group)
    pats = np.unique(sc_a.patients[ma])
    idx = {p: i for i, p in enumerate(pats)}
    out = []
    for sc, m in ((sc_a, ma), (sc_b, mb)):
        pid = np.array([idx[p] for p in sc.patients[m]])
        per_pat = np.zeros((len(pats), 3))
        for k, key in enumerate(('tp', 'fp', 'fn')):
            np.add.at(per_pat[:, k], pid, sc.c[key][m].sum(1))
        out.append(per_pat)
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(pats), size=(N_BOOT, len(pats)))
    fa = micro_f1_from(*out[0][draws].sum(1).T)
    fb = micro_f1_from(*out[1][draws].sum(1).T)
    return fb - fa


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')


def fmt(x, nd=3):
    return f'{x:.{nd}f}' if isinstance(x, (float, np.floating)) else str(x)


def md_table(rows, cols, header=None):
    header = header or cols
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r.get(c, '')) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='output/test_overview/preds')
    ap.add_argument('--out', default='output/test_overview/tables')
    args = ap.parse_args()
    preds, out = Path(args.preds), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = {}
    for m in MODELS:
        for s in SPLITS:
            p = preds / f'{m}__{s}.pkl'
            if p.exists():
                with open(p, 'rb') as f:
                    recs[(m, s)] = pickle.load(f)['records']
            else:
                print(f'WARN missing {p}')
    models = [m for m in MODELS if any((m, s) in recs for s in SPLITS)]

    # ---- threshold sweep (all splits, all view groups) ----------------------------
    counts = {}                                 # (model, split, thr) -> SplitCounts
    sweep_rows = []
    for (m, s), R in recs.items():
        for thr in sorted(set(SWEEP) | {FIXED_THR}):
            sc = SplitCounts(R, thr)
            counts[(m, s, thr)] = sc
            for g in GROUP_ORDER:
                mk = sc.mask(g)
                if not mk.any():
                    continue
                met = summarize(sc.c, mk)[0]
                sweep_rows.append(dict(model=m, split=s, view_group=g, thr=thr, n_imgs=int(mk.sum()),
                                       **{k: met[k] for k in ('micro_p', 'micro_r', 'micro_f1', 'macro_f1',
                                                              'tp', 'fp', 'fn', 'miscls', 'missed')}))
        print(f'swept {m} {s}')
    write_csv(out / 'threshold_sweep.csv', sweep_rows)

    # ---- threshold choice on IOP val -------------------------------------------------
    thr_rows, thr_best = [], {}
    for m in models:
        cand = [r for r in sweep_rows if r['model'] == m and r['split'] == 'iop_val' and r['view_group'] == 'ALL']
        if not cand:
            thr_best[m] = FIXED_THR
            continue
        best = max(cand, key=lambda r: (r['micro_f1'], -abs(r['thr'] - FIXED_THR)))
        thr_best[m] = best['thr']
        at05 = next(r for r in cand if r['thr'] == FIXED_THR)
        thr_rows.append(dict(model=m, val_best_thr=best['thr'], val_micro_f1_at_best=best['micro_f1'],
                             val_micro_f1_at_fixed_0p5=at05['micro_f1']))
    write_csv(out / 'thresholds.csv', thr_rows)
    thr_kinds = {'fixed': lambda m: FIXED_THR, 'val_best': lambda m: thr_best[m]}

    # ---- totals / per-view / per-class / swaps ---------------------------------------
    totals, per_view, per_class, swaps_rows = [], [], [], []
    boot = {}                                           # (m, s, kind, group) -> draws
    for (m, s), R in recs.items():
        for kind, thr_of in thr_kinds.items():
            thr = thr_of(m)
            sc = counts[(m, s, thr)]
            cm = compute_metrics(R, thr, IOU_THR)       # canonical totals
            met, summed, pc_p, pc_r, pc_f1 = summarize(sc.c, np.ones(len(R), bool))
            assert abs(met['micro_f1'] - cm['micro_f1']) < 1e-9 and met['tp'] == cm['tp'] \
                and met['miscls'] == cm['miscls'] and abs(met['macro_f1'] - cm['macro_f1']) < 1e-9, \
                (m, s, kind, met, {k: cm[k] for k in ('micro_f1', 'tp', 'miscls', 'macro_f1')})
            n_gt = met['correct'] + met['misloc'] + met['miscls'] + met['missed']
            base = dict(model=m, split=s, thr_kind=kind, thr=thr)
            totals.append(dict(**base, n_imgs=len(R), n_teeth=n_gt,
                               micro_p=cm['micro_p'], micro_r=cm['micro_r'], micro_f1=cm['micro_f1'],
                               macro_f1=cm['macro_f1'], weighted_f1=cm['weighted_f1'],
                               tp=cm['tp'], fp=cm['fp'], fn=cm['fn'],
                               correct=cm['correct'], misloc=cm['misloc'], miscls=cm['miscls'],
                               missed=cm['missed'], imgs_with_dup=cm['imgs_with_dup'],
                               dup_img_pct=100.0 * cm['imgs_with_dup'] / len(R), total_dups=cm['total_dups'],
                               mean_iou_assoc=cm['mean_iou_assoc']))
            for c in range(NC):
                sup = int(summed['tp'][c] + summed['fn'][c])
                per_class.append(dict(**base, fdi=_FDI_ORDER[c], class_id=c, support=sup,
                                      tp=int(summed['tp'][c]), fp=int(summed['fp'][c]), fn=int(summed['fn'][c]),
                                      precision=pc_p[c], recall=pc_r[c], f1=pc_f1[c], dup=int(summed['dup'][c]),
                                      correct=int(summed['correct'][c]), misloc=int(summed['misloc'][c]),
                                      miscls=int(summed['miscls'][c]), missed=int(summed['missed'][c])))
            w = WISDOM
            wtp, wfp, wfn = (int(summed[k][w].sum()) for k in ('tp', 'fp', 'fn'))
            wp = wtp / (wtp + wfp) if wtp + wfp else 0.0
            wr = wtp / (wtp + wfn) if wtp + wfn else 0.0
            per_class.append(dict(**base, fdi='wisdom(18/28/38/48)', class_id=-1, support=wtp + wfn,
                                  tp=wtp, fp=wfp, fn=wfn, precision=wp, recall=wr,
                                  f1=2 * wp * wr / (wp + wr) if wp + wr else 0.0,
                                  dup=int(summed['dup'][w].sum()), correct=int(summed['correct'][w].sum()),
                                  misloc=int(summed['misloc'][w].sum()), miscls=int(summed['miscls'][w].sum()),
                                  missed=int(summed['missed'][w].sum()),
                                  macro_f1_wisdom=float(np.mean(pc_f1[w]))))

            for gi, g in enumerate(GROUP_ORDER):
                mk = sc.mask(g)
                if not mk.any():
                    continue
                gm = summarize(sc.c, mk)[0]
                n_gt = gm['correct'] + gm['misloc'] + gm['miscls'] + gm['missed']
                if s != 'iop_val':
                    draws, pats, _ = boot_group_f1(sc, g, rng_seed=1000 * gi + 7)
                    boot[(m, s, kind, g)] = draws
                    lo, hi = ci(draws)
                    n_pat = len(pats)
                else:
                    lo = hi = float('nan')
                    n_pat = len(np.unique(sc.patients[mk]))
                per_view.append(dict(**base, view_group=g, n_imgs=int(mk.sum()), n_patients=n_pat, n_teeth=n_gt,
                                     micro_p=gm['micro_p'], micro_r=gm['micro_r'], micro_f1=gm['micro_f1'],
                                     micro_f1_ci_lo=lo, micro_f1_ci_hi=hi,
                                     macro_f1=gm['macro_f1'], tp=gm['tp'], fp=gm['fp'], fn=gm['fn'],
                                     correct=gm['correct'], misloc=gm['misloc'], miscls=gm['miscls'],
                                     missed=gm['missed'],
                                     correct_pct=100 * gm['correct'] / n_gt if n_gt else 0.0,
                                     misloc_pct=100 * gm['misloc'] / n_gt if n_gt else 0.0,
                                     miscls_pct=100 * gm['miscls'] / n_gt if n_gt else 0.0,
                                     missed_pct=100 * gm['missed'] / n_gt if n_gt else 0.0,
                                     imgs_with_dup=int(sc.has_dup[mk].sum()),
                                     dup_img_pct=100.0 * sc.has_dup[mk].mean(), total_dups=gm['total_dups']))
                sw = {k: int(sc.swaps[k][mk].sum()) for k in SWAP_TYPES}
                n_sw = sum(sw.values())
                assert n_sw == gm['miscls'], (m, s, kind, g, n_sw, gm['miscls'])
                swaps_rows.append(dict(**base, view_group=g, n_teeth=n_gt, miscls=gm['miscls'],
                                       miscls_per100=100 * gm['miscls'] / n_gt if n_gt else 0.0, **sw,
                                       lr_swap=sw['lr_exact'] + sw['lr_other'],
                                       lr_swap_per100=100 * (sw['lr_exact'] + sw['lr_other']) / n_gt if n_gt else 0.0,
                                       lr_share_of_miscls=(sw['lr_exact'] + sw['lr_other']) / n_sw if n_sw else 0.0,
                                       neighbour_per100=100 * sw['neighbour'] / n_gt if n_gt else 0.0))
        print(f'tabulated {m} {s}')
    write_csv(out / 'totals.csv', totals)
    write_csv(out / 'per_view.csv', per_view)
    write_csv(out / 'per_class.csv', per_class)
    write_csv(out / 'swaps.csv', swaps_rows)

    # ---- mirror-convention contrasts (patient bootstrap) -------------------------------
    def pv(m, s, kind, g):
        return next((r for r in per_view if r['model'] == m and r['split'] == s
                     and r['thr_kind'] == kind and r['view_group'] == g), None)

    contrasts = []
    for m in models:
        for kind, thr_of in thr_kinds.items():
            thr = thr_of(m)
            have = lambda s: (m, s, thr) in counts
            # (1) domain drop per view group, and difference-in-differences buccal vs non-buccal
            if have('iop_test') and have('ortho_test'):
                for g in GROUP_ORDER:
                    a, b = pv(m, 'iop_test', kind, g), pv(m, 'ortho_test', kind, g)
                    d = boot[(m, 'iop_test', kind, g)] - boot[(m, 'ortho_test', kind, g)]
                    lo, hi = ci(d)
                    contrasts.append(dict(model=m, thr_kind=kind, thr=thr, contrast='drop_iop_to_ortho',
                                          view_group=g, value=a['micro_f1'] - b['micro_f1'], ci_lo=lo, ci_hi=hi,
                                          f1_a=a['micro_f1'], f1_b=b['micro_f1'],
                                          err_ratio=(1 - b['micro_f1']) / (1 - a['micro_f1']) if a['micro_f1'] < 1 else float('nan')))
                did = (boot[(m, 'iop_test', kind, 'BUCCAL')] - boot[(m, 'ortho_test', kind, 'BUCCAL')]) \
                    - (boot[(m, 'iop_test', kind, 'NONBUCCAL')] - boot[(m, 'ortho_test', kind, 'NONBUCCAL')])
                val = (pv(m, 'iop_test', kind, 'BUCCAL')['micro_f1'] - pv(m, 'ortho_test', kind, 'BUCCAL')['micro_f1']) \
                    - (pv(m, 'iop_test', kind, 'NONBUCCAL')['micro_f1'] - pv(m, 'ortho_test', kind, 'NONBUCCAL')['micro_f1'])
                lo, hi = ci(did)
                contrasts.append(dict(model=m, thr_kind=kind, thr=thr, contrast='did_buccal_minus_nonbuccal_drop',
                                      view_group='BUCCAL-NONBUCCAL', value=val, ci_lo=lo, ci_hi=hi))
            # (2) flip probe on OrthoTracked (paired over patients) and the IOP positive control
            for s_a, s_b, name in (('ortho_test', 'ortho_test_bflip', 'flip_probe_ortho'),
                                   ('iop_test', 'iop_test_nomirror', 'nomirror_control_iop')):
                if not (have(s_a) and have(s_b)):
                    continue
                for g in GROUP_ORDER:
                    a, b = pv(m, s_a, kind, g), pv(m, s_b, kind, g)
                    d = boot_paired_delta(counts[(m, s_a, thr)], counts[(m, s_b, thr)], g,
                                          rng_seed=17 + GROUP_ORDER.index(g))
                    lo, hi = ci(d)
                    contrasts.append(dict(model=m, thr_kind=kind, thr=thr, contrast=name, view_group=g,
                                          value=b['micro_f1'] - a['micro_f1'], ci_lo=lo, ci_hi=hi,
                                          f1_a=a['micro_f1'], f1_b=b['micro_f1']))
    write_csv(out / 'contrasts.csv', contrasts)

    # ---- sanity: views that are NOT flipped must be bit-identical between variants -------
    sanity_lines = []
    for m in models:
        for s_a, s_b in (('ortho_test', 'ortho_test_bflip'), ('iop_test', 'iop_test_nomirror')):
            if (m, s_a) not in recs or (m, s_b) not in recs:
                continue
            for v in ('CENTER', 'UP', 'DOWN'):
                ra = [r for r in recs[(m, s_a)] if r['view'] == v]
                rb = [r for r in recs[(m, s_b)] if r['view'] == v]
                same = len(ra) == len(rb) and all(
                    np.array_equal(x['pred_labels'], y['pred_labels'])
                    and np.allclose(x['pred_scores'], y['pred_scores'], atol=1e-5)
                    and np.allclose(x['pred_boxes'], y['pred_boxes'], atol=1e-2)
                    and np.array_equal(x['gt_labels'], y['gt_labels']) for x, y in zip(ra, rb))
                sanity_lines.append(f'{m:12s} {s_a} vs {s_b} {v:6s}: predictions identical = {same}')
    (out / 'sanity_unflipped_views.txt').write_text('\n'.join(sanity_lines) + '\n')
    print('\n'.join(sanity_lines))

    # ---- summary.md --------------------------------------------------------------------
    md = ['# Test overview -- auto-generated tables', '',
          f'Thresholds chosen on IOP val: ' + ', '.join(f'{m}={thr_best[m]}' for m in models), '']
    for kind in thr_kinds:
        md += [f'## Totals ({kind})', '',
               md_table([r for r in totals if r['thr_kind'] == kind and r['split'] != 'iop_val'] +
                        [r for r in totals if r['thr_kind'] == kind and r['split'] == 'iop_val'],
                        ['model', 'split', 'thr', 'n_imgs', 'n_teeth', 'micro_p', 'micro_r', 'micro_f1',
                         'macro_f1', 'miscls', 'missed', 'dup_img_pct']), '']
        for s in SPLITS[1:]:
            md += [f'### Per view, {s} ({kind})', '',
                   md_table([r for r in per_view if r['thr_kind'] == kind and r['split'] == s],
                            ['model', 'view_group', 'n_imgs', 'n_teeth', 'micro_f1', 'micro_f1_ci_lo',
                             'micro_f1_ci_hi', 'macro_f1', 'correct_pct', 'misloc_pct', 'miscls_pct',
                             'missed_pct', 'dup_img_pct']), '']
        md += [f'## Contrasts ({kind})', '',
               md_table([r for r in contrasts if r['thr_kind'] == kind],
                        ['model', 'contrast', 'view_group', 'value', 'ci_lo', 'ci_hi', 'f1_a', 'f1_b',
                         'err_ratio']), '']
        md += [f'## Swaps ({kind})', '',
               md_table([r for r in swaps_rows if r['thr_kind'] == kind and r['split'] != 'iop_val'
                         and r['view_group'] in ('BUCCAL', 'NONBUCCAL', 'ALL')],
                        ['model', 'split', 'view_group', 'n_teeth', 'miscls', 'miscls_per100', 'lr_exact',
                         'lr_other', 'upper_lower', 'diagonal', 'neighbour', 'same_quad_other',
                         'lr_swap_per100', 'lr_share_of_miscls']), '']
        md += [f'## Wisdom teeth ({kind})', '',
               md_table([r for r in per_class if r['thr_kind'] == kind and (r['class_id'] in WISDOM or r['class_id'] == -1)
                         and r['split'] in ('iop_val', 'iop_test', 'ortho_test')],
                        ['model', 'split', 'fdi', 'support', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1']), '']
    md += ['## Sanity: un-flipped views identical', '', '```', *sanity_lines, '```', '']
    (out / 'summary.md').write_text('\n'.join(md))
    print(f'wrote {out / "summary.md"}')


if __name__ == '__main__':
    main()
