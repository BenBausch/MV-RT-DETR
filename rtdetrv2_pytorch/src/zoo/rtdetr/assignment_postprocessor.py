"""Post-processor that turns the OT assignment into unique-per-class detections.

Unlike `RTDETRPostProcessor` (top-k over independent per-query scores, which can
emit the same class many times), this reads `assignment_log` from the model and
produces *at most one box per class* via an exact assignment, exactly enforcing
the "each class occurs at most once" constraint at inference time.

Copyright(c) 2026. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision

from scipy.optimize import linear_sum_assignment

from ...core import register


__all__ = ['RTDETRAssignmentPostProcessor']


@register()
class RTDETRAssignmentPostProcessor(nn.Module):
    __share__ = ['num_classes', 'remap_mscoco_category']

    def __init__(self,
                 num_classes=32,
                 score_threshold=0.0,
                 use_hungarian=True,
                 remap_mscoco_category=False) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.score_threshold = score_threshold
        self.use_hungarian = use_hungarian
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'num_classes={self.num_classes}, use_hungarian={self.use_hungarian}, ' \
               f'score_threshold={self.score_threshold}'

    @torch.no_grad()
    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        assert 'assignment_log' in outputs, \
            'RTDETRAssignmentPostProcessor requires a model with an assignment head'
        log_P = outputs['assignment_log']          # [B, N+1, C+1]
        boxes = outputs['pred_boxes']              # [B, N, 4] cxcywh normalized
        B, N, C = log_P.shape[0], log_P.shape[1] - 1, log_P.shape[2] - 1

        # assignment probabilities over real (query, class) pairs
        P = log_P[:, :N, :C].exp()                 # [B, N, C]

        bbox_xyxy = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_xyxy = bbox_xyxy * orig_target_sizes.repeat(1, 2).unsqueeze(1)

        results = []
        all_labels, all_boxes, all_scores = [], [], []
        for b in range(B):
            Pb = P[b]                              # [N, C]
            if self.use_hungarian:
                # minimize negative log-prob -> one query per class (cost over [N, C])
                cost = (-log_P[b, :N, :C]).cpu().numpy()
                q_idx, c_idx = linear_sum_assignment(cost)
                q_idx = torch.as_tensor(q_idx, device=Pb.device)
                c_idx = torch.as_tensor(c_idx, device=Pb.device)
            else:
                # each class grabs its most likely query (may reuse a query; rare
                # after Sinkhorn). Cheaper, no scipy dependency.
                c_idx = torch.arange(C, device=Pb.device)
                q_idx = Pb.argmax(dim=0)

            scores = Pb[q_idx, c_idx]
            sel_boxes = bbox_xyxy[b][q_idx]

            keep = scores > self.score_threshold
            lab, box, sco = c_idx[keep], sel_boxes[keep], scores[keep]

            if self.deploy_mode:
                all_labels.append(lab); all_boxes.append(box); all_scores.append(sco)
            else:
                results.append(dict(labels=lab, boxes=box, scores=sco))

        if self.deploy_mode:
            return all_labels, all_boxes, all_scores

        if self.remap_mscoco_category:
            from ...data.dataset import mscoco_label2category
            for r in results:
                r['labels'] = torch.tensor(
                    [mscoco_label2category[int(x.item())] for x in r['labels'].flatten()]
                ).to(r['boxes'].device).reshape(r['labels'].shape)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
