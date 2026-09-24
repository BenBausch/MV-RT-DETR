"""Loss for the YOLO11 detector: ultralytics' own training loss + the OT head loss.

  loss_box / loss_cls / loss_dfl   ultralytics `v8DetectionLoss` (task-aligned
                                   assigner, CIoU, BCE, DFL) on all dense anchors,
                                   i.e. exactly how YOLO11 is trained upstream
  loss_assignment / loss_proto_div the OT head's matching NLL + prototype penalty,
                                   identical to the RT-DETR runs: the Hungarian
                                   matcher pairs GT with the top-k candidates the
                                   head saw, and `assignment_nll` supervises it

Loss gains live in `weight_dict` (ultralytics defaults box 7.5 / cls 0.5 / dfl 1.5),
so the ultralytics loss itself runs with unit gains.

Copyright(c) 2026. All Rights Reserved.
"""

import torch
import torch.nn as nn

from types import SimpleNamespace

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner

from ...core import register
from ..rtdetr.rtdetrv2_criterion import assignment_nll


__all__ = ['YOLOCriterion']


class _TALDetectionLoss(v8DetectionLoss):
    """`v8DetectionLoss` built from explicit head settings instead of a live
    DetectionModel, so the criterion is configured on its own like every other
    criterion here. Mirrors `v8DetectionLoss.__init__` field for field."""

    def __init__(self, num_classes, reg_max, strides, tal_topk, device):
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = SimpleNamespace(box=1.0, cls=1.0, dfl=1.0)
        self.stride = torch.tensor(strides, dtype=torch.float32)
        self.nc = num_classes
        self.no = num_classes + reg_max * 4
        self.reg_max = reg_max
        self.device = device
        self.use_dfl = reg_max > 1
        self.class_weights = None
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=num_classes, alpha=0.5, beta=6.0, stride=list(strides))
        self.bbox_loss = BboxLoss(reg_max).to(device)
        self.proj = torch.arange(reg_max, dtype=torch.float, device=device)


@register()
class YOLOCriterion(nn.Module):
    __inject__ = ['matcher', ]
    __share__ = ['num_classes', ]

    def __init__(self,
                 matcher=None,
                 weight_dict=None,
                 losses=('yolo', ),
                 num_classes=80,
                 reg_max=16,
                 strides=(8, 16, 32),
                 tal_topk=10):
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict or {'loss_box': 7.5, 'loss_cls': 0.5, 'loss_dfl': 1.5}
        self.losses = list(losses)
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = list(strides)
        self.tal_topk = tal_topk
        # ultralytics' loss is a plain object holding device-bound tensors, so it is
        # bound to the device of the first batch it sees (see `_tal_loss`).
        self._tal = None
        assert 'assignment' not in self.losses or matcher is not None, \
            'the assignment loss needs a matcher to pair GT with the candidates'

    def _tal_loss(self, device):
        if self._tal is None or self._tal.device != device:
            self._tal = _TALDetectionLoss(
                self.num_classes, self.reg_max, self.strides, self.tal_topk, device)
        return self._tal

    @staticmethod
    def _ultralytics_batch(targets, device):
        """Framework targets (normalized cxcywh boxes) -> ultralytics' flat batch."""
        return {
            'batch_idx': torch.cat([torch.full((len(t['labels']), ), i, device=device)
                                    for i, t in enumerate(targets)]).float(),
            'cls': torch.cat([t['labels'] for t in targets]).float(),
            'bboxes': torch.cat([t['boxes'] for t in targets]).float(),
        }

    def forward(self, outputs, targets, **kwargs):
        losses = {}

        if 'yolo' in self.losses:
            raw = outputs['yolo_raw']
            device = raw['scores'].device
            # the model runs under autocast; compute the loss in fp32 like the
            # rest of the criterion (the engine disables autocast around it)
            raw = {'boxes': raw['boxes'].float(), 'scores': raw['scores'].float(),
                   'feats': raw['feats']}
            _, box_cls_dfl, _ = self._tal_loss(device).get_assigned_targets_and_loss(
                raw, self._ultralytics_batch(targets, device))
            losses.update(loss_box=box_cls_dfl[0], loss_cls=box_cls_dfl[1],
                          loss_dfl=box_cls_dfl[2])

        if 'assignment' in self.losses:
            cand = {'pred_logits': outputs['pred_logits'].float(),
                    'pred_boxes': outputs['pred_boxes'].float()}
            indices = self.matcher(cand, targets)['indices']
            losses['loss_assignment'] = assignment_nll(outputs['assignment_log'], targets, indices)

        if 'proto_div' in self.losses:
            losses['loss_proto_div'] = outputs['proto_div']

        return {k: v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
