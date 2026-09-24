"""Ultralytics YOLO11 as a detector in this framework, with the optional OT head.

Wraps an ultralytics `DetectionModel` (COCO-pretrained weights) so it trains with
the same datasets, transforms, solver, EMA and per-tooth evaluation as RT-DETR --
only the detector changes.

YOLO is a dense detector: every anchor point (21,504 at 1024^2 with strides
8/16/32) predicts a box and per-class logits, and duplicates are normally removed
by NMS. The `OTAssignmentHead` instead needs a small candidate set with a box,
logits and an embedding per candidate -- what RT-DETR's decoder queries provide.
Here the candidates are the top-`num_candidates` anchors by max class logit (300,
as RT-DETR's queries), and a candidate's embedding is the anchor's penultimate
feature in the Detect head's class and box branches (the features its logits and
box are predicted from), i.e. the analogue of the decoder hidden state.

YOLO's one-to-many training makes ~17 near-identical anchors per object confident,
so plain top-k hands the head many duplicates per tooth; Sinkhorn then splits the
tooth's mass across them and the dustbin drops it. `candidate_nms_iou` runs
class-agnostic NMS before the top-k (training and inference) so the head sees about
one candidate per object, as with RT-DETR's one-to-one queries.

Output dict:
  yolo_raw      ultralytics' raw head output {boxes, scores, feats}, for the TAL loss
  pred_logits   [B, N, C] candidate class logits
  pred_boxes    [B, N, 4] candidate boxes, cxcywh normalized to the input size
  pred_embed    [B, N, D] candidate embeddings (D = class + box branch widths)
With an assignment head N = num_candidates; without one N = all anchors, so the
standard NMS post-processor sees every prediction (as ultralytics' own val does).

Copyright(c) 2026. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils.tal import make_anchors

from ...core import register


__all__ = ['YOLO11']


@register()
class YOLO11(nn.Module):
    __inject__ = ['assignment_head', ]
    __share__ = ['num_classes', ]

    def __init__(self,
                 num_classes=80,
                 cfg='yolo11l.yaml',
                 weights='yolo11l.pt',
                 in_channels=3,
                 num_candidates=300,
                 candidate_nms_iou=None,
                 assignment_head: nn.Module=None):
        super().__init__()
        self.det = DetectionModel(cfg, ch=in_channels, nc=num_classes, verbose=False)
        if weights:
            # ultralytics' own finetune transfer: loads every shape-matching tensor,
            # skipping the COCO 80-class logit convs (re-initialized by bias_init).
            self.det.load(load_checkpoint(weights)[0])
            if in_channels > 3:
                # load() copied the RGB stem into the first 3 input channels; start
                # the extra (orientation) channels at zero like PResNet does, so the
                # pretrained stem's response is unchanged at init.
                with torch.no_grad():
                    self.det.model[0].conv.weight[:, 3:] = 0

        self.head = self.det.model[-1]
        assert type(self.head) is Detect and not self.head.end2end, \
            f'{cfg}: expected a plain (one-to-many) YOLO Detect head, got {type(self.head).__name__}'

        self.num_classes = num_classes
        self.num_candidates = num_candidates
        # IoU of the class-agnostic NMS that de-duplicates the OT head's candidates
        # (None -> plain top-k by max logit)
        self.candidate_nms_iou = candidate_nms_iou
        # optional learnable OT class-assignment head (None -> vanilla YOLO + NMS)
        self.assignment_head = assignment_head

    @property
    def embed_dim(self):
        """Width of `pred_embed`: class-branch + box-branch penultimate channels."""
        return self.head.cv3[0][-1].in_channels + self.head.cv2[0][-1].in_channels

    def _neck(self, x):
        """Backbone + neck: ultralytics' layer loop up to (not including) Detect."""
        y = []
        for m in self.det.model[:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.det.save else None)
        return [x if j == -1 else y[j] for j in self.head.f]

    @torch.no_grad()
    def _select_candidates(self, scores, boxes, k):
        """[B, k] indices of the anchors handed to the OT head.

        scores: [B, A] max class logit per anchor; boxes: [B, A, 4] xyxy in pixels.
        """
        if self.candidate_nms_iou is None:
            return scores.topk(k, dim=1).indices
        # NMS over the strongest 10k anchors only (the rest is background); keep the
        # survivors in score order, topped up with the best suppressed anchors if
        # fewer than k survive so every image yields exactly k candidates
        pre = scores.topk(min(10 * k, scores.shape[1]), dim=1).indices    # score-sorted
        idx = []
        for b in range(scores.shape[0]):
            keep = torchvision.ops.nms(boxes[b, pre[b]].float(), scores[b, pre[b]].float(),
                                       self.candidate_nms_iou)
            if keep.numel() < k:
                suppressed = torch.ones(pre.shape[1], dtype=torch.bool, device=pre.device)
                suppressed[keep] = False
                keep = torch.cat([keep, suppressed.nonzero(as_tuple=True)[0]])
            idx.append(pre[b, keep[:k]])
        return torch.stack(idx)

    def forward(self, x, targets=None):
        H, W = x.shape[-2:]
        feats = self._neck(x)
        head = self.head

        # Detect's per-level branches, split before their final 1x1 conv so the
        # penultimate features can serve as candidate embeddings.
        dist, logits, embeds = [], [], []
        for i in range(head.nl):
            hb = head.cv2[i][:-1](feats[i])
            hc = head.cv3[i][:-1](feats[i])
            dist.append(head.cv2[i][-1](hb).flatten(2))
            logits.append(head.cv3[i][-1](hc).flatten(2))
            embeds.append(torch.cat([hc, hb], dim=1).flatten(2))
        dist = torch.cat(dist, -1)                         # [B, 4*reg_max, A]
        logits = torch.cat(logits, -1)                     # [B, C, A]
        embeds = torch.cat(embeds, -1)                     # [B, D, A]

        anchors, strides = make_anchors(feats, head.stride, 0.5)
        boxes = head.decode_bboxes(head.dfl(dist), anchors.t().unsqueeze(0)) * strides.t()
        boxes = boxes / boxes.new_tensor([W, H, W, H]).view(1, 4, 1)   # cxcywh, normalized

        out = {'yolo_raw': {'boxes': dist, 'scores': logits, 'feats': feats}}
        logits, boxes = logits.transpose(1, 2), boxes.transpose(1, 2)   # [B, A, *]

        if self.assignment_head is None:
            out.update(pred_logits=logits, pred_boxes=boxes)
            return out

        k = min(self.num_candidates, logits.shape[1])
        pixel_xyxy = torchvision.ops.box_convert(boxes, 'cxcywh', 'xyxy') * boxes.new_tensor([W, H, W, H])
        idx = self._select_candidates(logits.max(-1).values, pixel_xyxy, k)   # [B, k]
        out.update(
            pred_logits=logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1])),
            pred_boxes=boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4)),
            pred_embed=embeds.gather(2, idx.unsqueeze(1).expand(-1, embeds.shape[1], -1)).transpose(1, 2),
        )
        return self.assignment_head(out)

    def deploy(self, ):
        self.eval()
        return self
