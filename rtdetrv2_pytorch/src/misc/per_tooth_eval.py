"""Per-tooth (FDI) detection metrics for the RT-DETR eval loop.

COCO mAP is a poor headline for the ortho task: each FDI tooth class occurs at most
once per image, so the metric that matters is "did we localize+label each present
tooth", plus "did we ever predict the same tooth class twice". This module provides
the pure metric core (precision/recall/F1 at micro/macro/weighted aggregation, the
duplicate-class rate, and a recall-loss decomposition) plus a `PerToothMetrics`
accumulator that mirrors `UltralyticsDetMetrics` (update-per-image, compute-once) and
also builds the top-1 class-confusion matrix.

Boxes are xyxy in the SAME coordinate space for predictions and ground truth.

Aggregation schemas:
  - micro    = pool TP/FP/FN across classes, then compute once
  - macro    = per-class metric, unweighted mean
  - weighted = per-class metric, mean weighted by support (GT instances per class)
"""

from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import box_iou


METRIC_KEY = {"f1": "f1", "precision": "p", "recall": "r"}


def match_image(
    pred_labels, pred_scores, pred_boxes, gt_labels, gt_boxes, score_thr, iou_thr
):
    """Greedy class-aware IoU matching for one image. Returns (tp, fp, fn) dicts
    keyed by class id, the list of kept predicted class ids (for dup counting), and
    the recall-loss buckets (correct / mislocalized / misclassified / missed) per class.

    Recall-loss decomposition, per present (GT) tooth -- mutually exclusive and
    exhaustive, so correct == TP and misloc + misclassified + missed == FN:
      correct      : a same-class prediction overlaps it with IoU >= iou_thr
      mislocalized : no correct, but a same-class prediction overlaps it (IoU > 0)
      misclassified: no same-class overlap, but a DIFFERENT-class prediction
                     localizes it (IoU >= iou_thr) -- a real wrong-label error
      missed       : nothing localizes the tooth (no box at IoU >= iou_thr, no
                     same-class overlap) -- a detection / threshold failure
    """
    keep = pred_scores >= score_thr
    pl, ps, pb = pred_labels[keep], pred_scores[keep], pred_boxes[keep]
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    correct, misloc = defaultdict(int), defaultdict(int)
    misclassified, missed = defaultdict(int), defaultdict(int)

    matched_gt = np.zeros(len(gt_labels), dtype=bool)
    iou = None
    if len(pl) and len(gt_labels):
        iou = box_iou(
            torch.tensor(pb, dtype=torch.float32),
            torch.tensor(gt_boxes, dtype=torch.float32),
        ).numpy()
        order = np.argsort(-ps)  # match high-confidence predictions first
        matched_pred = np.zeros(len(pl), dtype=bool)
        for pi in order:
            cand = [
                gi
                for gi in range(len(gt_labels))
                if not matched_gt[gi]
                and gt_labels[gi] == pl[pi]
                and iou[pi, gi] >= iou_thr
            ]
            if cand:
                gi = max(cand, key=lambda g: iou[pi, g])
                matched_gt[gi] = True
                matched_pred[pi] = True
                tp[pl[pi]] += 1
        for pi in range(len(pl)):
            if not matched_pred[pi]:
                fp[pl[pi]] += 1
    else:
        for c in pl:
            fp[c] += 1
    for gi in range(len(gt_labels)):
        cg = gt_labels[gi]
        if not matched_gt[gi]:
            fn[cg] += 1
        # best overlap of same-class vs different-class predictions with this tooth
        if iou is not None and len(pl):
            same = np.nonzero(pl == cg)[0]
            other = np.nonzero(pl != cg)[0]
            best_same = float(iou[same, gi].max()) if same.size else 0.0
            best_other = float(iou[other, gi].max()) if other.size else 0.0
        else:
            best_same = best_other = 0.0
        if best_same >= iou_thr:
            correct[cg] += 1
        elif best_same > 0.0:
            misloc[cg] += 1
        elif best_other >= iou_thr:
            misclassified[cg] += 1
        else:
            missed[cg] += 1
    return tp, fp, fn, pl, correct, misloc, misclassified, missed


def compute_metrics(records, score_thr, iou_thr):
    TP, FP, FN, DUP = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    CORRECT, MISLOC = defaultdict(int), defaultdict(int)
    MISCLS, MISSED = defaultdict(int), defaultdict(int)
    imgs_with_dup, total_dups, n_imgs = 0, 0, len(records)
    for rec in records:
        tp, fp, fn, kept, correct, misloc, miscls, missed = match_image(
            rec["pred_labels"],
            rec["pred_scores"],
            rec["pred_boxes"],
            rec["gt_labels"],
            rec["gt_boxes"],
            score_thr,
            iou_thr,
        )
        for c, v in tp.items():
            TP[c] += v
        for c, v in fp.items():
            FP[c] += v
        for c, v in fn.items():
            FN[c] += v
        for c, v in correct.items():
            CORRECT[c] += v
        for c, v in misloc.items():
            MISLOC[c] += v
        for c, v in miscls.items():
            MISCLS[c] += v
        for c, v in missed.items():
            MISSED[c] += v
        counts = np.bincount(kept) if len(kept) else np.array([])
        dup = int((counts[counts > 1] - 1).sum()) if counts.size else 0
        total_dups += dup
        imgs_with_dup += int(dup > 0)
        for c in np.nonzero(counts > 1)[0]:  # surplus predictions per class
            DUP[int(c)] += int(counts[c] - 1)

    classes = sorted(set(TP) | set(FP) | set(FN))
    per_class, f1s, ps, rs, support = {}, [], [], [], []
    for c in classes:
        tp, fp, fn = TP[c], FP[c], FN[c]
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        sup = tp + fn  # GT instances of this class (support)
        per_class[c] = dict(
            tp=tp, fp=fp, fn=fn, p=p, r=r, f1=f1, support=sup, dup=DUP[c],
            correct=CORRECT[c], misloc=MISLOC[c], miscls=MISCLS[c], missed=MISSED[c],
        )
        f1s.append(f1)
        ps.append(p)
        rs.append(r)
        support.append(sup)
    tTP, tFP, tFN = sum(TP.values()), sum(FP.values()), sum(FN.values())
    micro_p = tTP / (tTP + tFP) if (tTP + tFP) else 0.0
    micro_r = tTP / (tTP + tFN) if (tTP + tFN) else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    )
    # support-weighted: per-class metric weighted by GT instances of that class
    w = np.array(support, dtype=float)
    wsum = w.sum()

    def wavg(vals):
        return float(np.average(vals, weights=w)) if wsum else 0.0

    return dict(
        per_class=per_class,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        macro_p=float(np.mean(ps)) if ps else 0.0,
        macro_r=float(np.mean(rs)) if rs else 0.0,
        weighted_f1=wavg(f1s),
        weighted_p=wavg(ps),
        weighted_r=wavg(rs),
        micro_p=micro_p,
        micro_r=micro_r,
        micro_f1=micro_f1,
        tp=tTP,
        fp=tFP,
        fn=tFN,
        correct=sum(CORRECT.values()),
        misloc=sum(MISLOC.values()),
        miscls=sum(MISCLS.values()),
        missed=sum(MISSED.values()),
        imgs_with_dup=imgs_with_dup,
        total_dups=total_dups,
        n_imgs=n_imgs,
    )


class PerToothMetrics:
    """In-loop accumulator mirroring UltralyticsDetMetrics: feed one image at a time
    via `update(...)`, then call `compute()` once. Predictions and GT are xyxy in the
    SAME (original-image) coordinate space.

    Also accumulates a top-1 class-confusion matrix (for each GT box, the
    highest-confidence prediction overlapping it at IoU >= confusion_iou,
    class-agnostic; NOT score-thresholded -- matches the previous Ultralytics output).
    """

    def __init__(self, names, score_thr=0.5, iou_thr=0.5, confusion_iou=0.5):
        self.names = {int(k): v for k, v in (names or {}).items()}
        self.score_thr = float(score_thr)
        self.iou_thr = float(iou_thr)
        self.confusion_iou = float(confusion_iou)
        self.records = []
        self.nc = (max(self.names) + 1) if self.names else 0
        self.confusion = np.zeros((self.nc, self.nc), dtype=np.int64) if self.nc else None

    @torch.no_grad()
    def update(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls):
        pb = pred_boxes.detach().cpu().numpy().astype(float)
        ps = pred_scores.detach().cpu().numpy().astype(float)
        pl = pred_cls.detach().cpu().numpy().astype(int)
        gb = gt_boxes.detach().cpu().numpy().astype(float)
        gl = gt_cls.detach().cpu().numpy().astype(int)
        self.records.append(dict(
            pred_labels=pl, pred_scores=ps, pred_boxes=pb, gt_labels=gl, gt_boxes=gb,
        ))

        # top-1 class-confusion matrix (class-agnostic overlap, highest-conf pred per GT)
        if self.confusion is not None and len(gl) and len(pl):
            iou = box_iou(
                torch.as_tensor(gb, dtype=torch.float32),
                torch.as_tensor(pb, dtype=torch.float32),
            ).numpy()
            for i in range(iou.shape[0]):
                cand = np.nonzero(iou[i] >= self.confusion_iou)[0]
                if cand.size:
                    j = int(cand[np.argmax(ps[cand])])
                    gi, pj = int(gl[i]), int(pl[j])
                    if 0 <= gi < self.nc and 0 <= pj < self.nc:
                        self.confusion[gi, pj] += 1

    def compute(self):
        m = compute_metrics(self.records, self.score_thr, self.iou_thr)
        m["confusion"] = self.confusion
        m["names"] = self.names
        return m

    def print_table(self, metrics=None):
        m = metrics or self.compute()
        if not m:
            print("(no per-tooth metrics)")
            return m
        print(f"Per-tooth metrics (score_thr={self.score_thr}, IoU={self.iou_thr}):")
        print(f"  micro  P/R/F1: {m['micro_p']:.3f} / {m['micro_r']:.3f} / {m['micro_f1']:.3f}")
        print(f"  macro     F1 : {m['macro_f1']:.3f}    weighted F1: {m['weighted_f1']:.3f}")
        sup = m["correct"] + m["misloc"] + m["miscls"] + m["missed"]
        if sup:
            print(f"  recall-loss (of {sup} present teeth): "
                  f"correct {m['correct']} ({100*m['correct']/sup:.1f}%), "
                  f"misloc {m['misloc']} ({100*m['misloc']/sup:.1f}%), "
                  f"misclass {m['miscls']} ({100*m['miscls']/sup:.1f}%), "
                  f"missed {m['missed']} ({100*m['missed']/sup:.1f}%)")
        dup_pct = 100.0 * m["imgs_with_dup"] / m["n_imgs"] if m["n_imgs"] else 0.0
        print(f"  duplicate-class: {m['imgs_with_dup']}/{m['n_imgs']} imgs ({dup_pct:.1f}%), "
              f"{m['total_dups']} surplus boxes")
        return m
