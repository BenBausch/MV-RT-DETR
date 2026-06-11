"""Ultralytics-equivalent detection metrics for the RT-DETR eval loop.

Reuses Ultralytics' own `box_iou` and `ap_per_class`, and replicates its
`match_predictions` (greedy IoU matching across the 10 thresholds 0.50:0.95),
so the numbers line up with `ultralytics` validation rather than pycocotools.

Reports, per class and overall ("all"):
  - P, R         : precision / recall at the max-F1 confidence point
  - mAP50        : AP at IoU 0.50
  - mAP50-95     : AP averaged over IoU 0.50:0.95
  - Instances    : number of GT boxes
  - mean matched IoU@0.5 : average IoU of true-positive matches (extra, vs Ultralytics)

Predictions and ground truth must be xyxy boxes in the SAME coordinate space.
"""
import numpy as np
import torch

from ultralytics.utils.metrics import box_iou, ap_per_class


class UltralyticsDetMetrics:
    def __init__(self, names, confusion_iou=0.5):
        # names: {class_id: name}
        self.names = {int(k): v for k, v in (names or {}).items()}
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.niou = self.iouv.numel()
        self.tp, self.conf, self.pred_cls, self.target_cls = [], [], [], []
        self.matched_iou = []
        self.seen = 0

        # top-1 class-confusion matrix: for each GT box, the highest-confidence
        # prediction overlapping it at IoU >= confusion_iou (class-agnostic match)
        # contributes confusion[gt_class, pred_class] += 1.
        self.confusion_iou = confusion_iou
        self.nc = (max(self.names) + 1) if self.names else 0
        self.confusion = np.zeros((self.nc, self.nc), dtype=np.int64) if self.nc else None

    def _match(self, pred_classes, true_classes, iou):
        """Replicates ultralytics BaseValidator.match_predictions (non-scipy)."""
        correct = np.zeros((pred_classes.shape[0], self.niou)).astype(bool)
        correct_class = true_classes[:, None] == pred_classes  # [nl, npr]
        iou = (iou * correct_class).cpu().numpy()
        for i, threshold in enumerate(self.iouv.tolist()):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool)

    @torch.no_grad()
    def update(self, pred_boxes, pred_scores, pred_cls, gt_boxes, gt_cls):
        self.seen += 1
        gt_cls = gt_cls.detach().cpu()
        self.target_cls.append(gt_cls)

        npr = int(pred_boxes.shape[0])
        if npr == 0:
            self.tp.append(torch.zeros((0, self.niou), dtype=torch.bool))
            self.conf.append(torch.zeros(0))
            self.pred_cls.append(torch.zeros(0))
            return

        pred_boxes = pred_boxes.detach().cpu().float()
        pred_scores = pred_scores.detach().cpu().float()
        pred_cls = pred_cls.detach().cpu()

        if gt_boxes.shape[0]:
            gt_boxes = gt_boxes.detach().cpu().float()
            iou = box_iou(gt_boxes, pred_boxes)  # [nl, npr], class-agnostic
            correct = self._match(pred_cls, gt_cls, iou)
            iou_cc = (iou * (gt_cls[:, None] == pred_cls)).numpy()
            m0 = correct[:, 0].numpy()
            if m0.any():
                self.matched_iou.append(iou_cc.max(0)[m0])

            # top-1 confusion: per GT, highest-confidence overlapping prediction
            if self.confusion is not None:
                iou_np, sc = iou.numpy(), pred_scores.numpy()
                gc, pc = gt_cls.numpy(), pred_cls.numpy()
                for i in range(iou_np.shape[0]):
                    cand = np.nonzero(iou_np[i] >= self.confusion_iou)[0]
                    if cand.size:
                        j = int(cand[np.argmax(sc[cand])])
                        gi, pj = int(gc[i]), int(pc[j])
                        if 0 <= gi < self.nc and 0 <= pj < self.nc:
                            self.confusion[gi, pj] += 1
        else:
            correct = torch.zeros((npr, self.niou), dtype=torch.bool)

        self.tp.append(correct)
        self.conf.append(pred_scores)
        self.pred_cls.append(pred_cls)

    def compute(self):
        if not self.tp:
            return {}
        tp = torch.cat(self.tp).numpy()
        conf = torch.cat(self.conf).numpy()
        pred_cls = torch.cat(self.pred_cls).numpy()
        target_cls = torch.cat(self.target_cls).numpy().astype(int)

        nc = (max(self.names) + 1) if self.names else (int(target_cls.max()) + 1 if target_cls.size else 1)
        nt = np.bincount(target_cls, minlength=nc)
        mean_iou = float(np.concatenate(self.matched_iou).mean()) if self.matched_iou else 0.0

        confusion = self.confusion.copy() if self.confusion is not None else None

        if tp.shape[0] == 0:  # no predictions at all
            return dict(mp=0.0, mr=0.0, map50=0.0, map=0.0, mean_iou=mean_iou,
                        seen=self.seen, instances=int(nt.sum()), per_class={},
                        confusion=confusion, names=self.names)

        res = ap_per_class(tp, conf, pred_cls, target_cls, names=self.names)
        p, r, ap, unique = res[2], res[3], res[5], res[6]
        ap50, ap5095 = ap[:, 0], ap.mean(1)

        per_class = {}
        for i, c in enumerate(unique):
            per_class[int(c)] = dict(p=float(p[i]), r=float(r[i]),
                                     ap50=float(ap50[i]), ap=float(ap5095[i]),
                                     n=int(nt[c]) if c < len(nt) else 0)
        return dict(
            mp=float(p.mean()) if len(p) else 0.0,
            mr=float(r.mean()) if len(r) else 0.0,
            map50=float(ap50.mean()) if len(ap50) else 0.0,
            map=float(ap5095.mean()) if len(ap5095) else 0.0,
            mean_iou=mean_iou,
            seen=self.seen,
            instances=int(nt.sum()),
            per_class=per_class,
            confusion=confusion,
            names=self.names,
        )

    def print_table(self, metrics=None):
        m = metrics or self.compute()
        if not m:
            print("(no ultralytics metrics)")
            return m
        cols = f"{'Class':>16}{'Images':>8}{'Instances':>11}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>10}"
        print("Ultralytics-equivalent metrics:")
        print(cols)
        print(f"{'all':>16}{m['seen']:>8}{m['instances']:>11}"
              f"{m['mp']:>9.3f}{m['mr']:>9.3f}{m['map50']:>9.3f}{m['map']:>10.3f}")
        for cid, d in sorted(m['per_class'].items()):
            name = self.names.get(cid, str(cid))
            print(f"{name:>16}{m['seen']:>8}{d['n']:>11}"
                  f"{d['p']:>9.3f}{d['r']:>9.3f}{d['ap50']:>9.3f}{d['ap']:>10.3f}")
        print(f"  mean matched IoU@0.5: {m['mean_iou']:.4f}")

        conf = m.get('confusion')
        if conf is not None and conf.sum() > 0:
            off = [(self.names.get(i, str(i)), self.names.get(j, str(j)), int(conf[i, j]))
                   for i in range(len(conf)) for j in range(len(conf))
                   if i != j and conf[i, j] > 0]
            off.sort(key=lambda x: -x[2])
            total = int(conf.sum())
            correct = int(conf.trace())
            print(f"  top-1 matched: {correct}/{total} correct class; "
                  f"top confusions (gt->pred): "
                  + ", ".join(f"{g}->{p}:{c}" for g, p, c in off[:8]))
        return m


def write_confusion(confusion, names, out_dir):
    """Write the top-1 class-confusion matrix as two CSVs in out_dir:
      confusion_matrix.csv : full NxN counts (rows=gt, cols=pred)
      confusion_pairs.csv  : gt_class,pred_class,count,correct (nonzero, count-sorted)
    """
    import os
    if confusion is None:
        return
    confusion = np.asarray(confusion)
    nc = confusion.shape[0]
    labels = [str(names.get(i, i)) for i in range(nc)]
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'confusion_matrix.csv'), 'w') as f:
        f.write('gt\\pred,' + ','.join(labels) + '\n')
        for i in range(nc):
            f.write(labels[i] + ',' + ','.join(str(int(v)) for v in confusion[i]) + '\n')

    pairs = [(labels[i], labels[j], int(confusion[i, j]), int(i == j))
             for i in range(nc) for j in range(nc) if confusion[i, j] > 0]
    pairs.sort(key=lambda x: -x[2])
    with open(os.path.join(out_dir, 'confusion_pairs.csv'), 'w') as f:
        f.write('gt_class,pred_class,count,correct\n')
        for g, p, c, ok in pairs:
            f.write(f'{g},{p},{c},{ok}\n')
