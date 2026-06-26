"""Learnable optimal-transport assignment head for RT-DETR.

Each decoder query carries a class distribution, a box, and a hidden embedding,
but RT-DETR predicts every query independently -- nothing stops the same class
(e.g. an FDI tooth number) from being predicted on several boxes, and nothing
decides which of two competing boxes is the better fit for a class.

This head turns the per-query predictions into a *global* assignment between
queries and classes under the constraint that each class is used at most once.
It is fully learnable and differentiable:

  1. build a token per query  f_i = proj([emb_i || box_i || logits_i])
  2. a small (default 3-layer) Transformer encoder lets every query attend to
     every other query -> context-aware features g_i
  3. score g_i against a learnable prototype per class -> cost matrix S [N, C]
  4. augment S with a learnable dustbin row/column (absent class / dropped box)
  5. Sinkhorn normalization -> doubly-normalized soft assignment P, whose column
     constraint *is* the "each class at most once" rule
  6. the caller reads off P (argmax / Hungarian) to retain at most one box per class

Reference for the matching layer: SuperGlue (Sarlin et al., CVPR 2020).

Copyright(c) 2026. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


__all__ = ['OTAssignmentHead']


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """Perform Sinkhorn normalization in the log domain for numerical stability."""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, bin_score, iters: int):
    """Differentiable partial optimal transport with a learnable dustbin.

    Args:
        scores: [B, M, N] affinities (M queries, N classes).
        bin_score: scalar parameter filling the dustbin row/column.
        iters: number of Sinkhorn iterations.

    Returns:
        log_P: [B, M+1, N+1] log assignment. Row M is the dustbin (absent class),
            column N is the dustbin (dropped/background query). Each real row sums
            to 1 and each real column sums to 1 over the augmented matrix.
    """
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = bin_score.view(1, 1, 1).expand(b, m, 1)
    bins1 = bin_score.view(1, 1, 1).expand(b, 1, n)
    alpha = bin_score.view(1, 1, 1).expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)  # [B, M+1, N+1]

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu = log_mu[None].expand(b, -1)
    log_nu = log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # undo the marginal normalization (multiply probabilities by M+N)
    return Z


@register()
class OTAssignmentHead(nn.Module):
    """Sinkhorn-based class assignment over the decoder queries.

    Adds `assignment_log` ([B, N+1, C+1] log assignment) to the model output dict
    without modifying `pred_logits` / `pred_boxes`, so the detector keeps working
    as before and the assignment is an extra, supervised, head.
    """

    __share__ = ['num_classes']

    def __init__(self,
                 num_classes=32,
                 hidden_dim=256,
                 dim=256,
                 num_layers=3,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 sinkhorn_iters=50,
                 use_box_pos=True,
                 use_pos_proto=True,
                 num_prototypes=1,
                 proto_repulsion=0.0,
                 proto_margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.sinkhorn_iters = sinkhorn_iters
        # multiple appearance prototypes per class let one tooth type cover its
        # diverse looks; their per-class similarities are reduced with a soft-max
        # (logsumexp) so a query matches its best-fitting mode.
        self.num_prototypes = num_prototypes
        self.proto_repulsion = proto_repulsion
        self.proto_margin = proto_margin

        in_dim = hidden_dim + 4 + num_classes
        self.input_proj = nn.Linear(in_dim, dim)
        self.input_norm = nn.LayerNorm(dim)

        # explicit box positional encoding -- 4 coords would otherwise be drowned
        # out next to the high-dimensional embedding, and position is the dominant
        # cue for tooth identity.
        self.use_box_pos = use_box_pos
        if use_box_pos:
            self.box_pos = nn.Sequential(
                nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers, enable_nested_tensor=False)

        # num_prototypes learnable prototype vectors per class
        self.class_proto = nn.Parameter(torch.empty(num_classes * num_prototypes, dim))
        nn.init.normal_(self.class_proto, std=dim ** -0.5)

        # explicit per-class spatial prior: a learnable location prototype scored
        # directly against the box positional embedding, so the cost has a direct
        # geometry-vs-class term (tooth identity is dominated by anatomical
        # position). Requires the box positional encoding to be enabled.
        self.use_pos_proto = use_pos_proto and use_box_pos
        if self.use_pos_proto:
            self.pos_proto = nn.Parameter(torch.empty(num_classes * num_prototypes, dim))
            nn.init.normal_(self.pos_proto, std=dim ** -0.5)
            self.beta = nn.Parameter(torch.tensor(1.0))

        # logits-as-prior bias weight, and the single learnable dustbin score
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.bin_score = nn.Parameter(torch.tensor(1.0))
        # learnable Sinkhorn temperature (log-space), initialised to sqrt(dim)
        self.log_temperature = nn.Parameter(torch.tensor(dim ** 0.5).log())

    def _proto_repulsion(self, proto):
        """Angular anti-collapse penalty over the per-class prototype group.

        Normalizes each prototype, then hinge-penalizes any intra-class prototype
        pair whose cosine similarity exceeds ``1 - proto_margin``, so the p
        prototypes of a class are pushed apart only while they are too similar.
        """
        p = self.num_prototypes
        if p == 1 or self.proto_repulsion == 0:
            return proto.new_zeros(())
        pn = F.normalize(proto.view(self.num_classes, p, -1), dim=-1)
        gram = torch.matmul(pn, pn.transpose(1, 2))            # [K, p, p] cosine sims
        off = gram - torch.eye(p, device=proto.device)         # zero the diagonal
        pen = (off - (1.0 - self.proto_margin)).clamp(min=0.0)
        return (pen ** 2).sum() / (self.num_classes * p * (p - 1))

    def forward(self, outputs):
        emb = outputs['pred_embed']       # [B, N, hidden_dim]
        boxes = outputs['pred_boxes']     # [B, N, 4]  (cxcywh, normalized)
        logits = outputs['pred_logits']   # [B, N, C]

        tokens = torch.cat([emb, boxes, logits], dim=-1)
        f = self.input_norm(self.input_proj(tokens))
        pe = self.box_pos(boxes) if self.use_box_pos else None
        if pe is not None:
            f = f + pe

        g = self.encoder(f)               # [B, N, dim]
        B, N = g.shape[:2]
        K, P = self.num_classes, self.num_prototypes

        # affinity of each query to each class prototype, plus RT-DETR's own
        # logits as a unary prior so the head *corrects* rather than relearns.
        # the P prototypes per class are reduced with a soft-max (logsumexp) so a
        # query matches its best-fitting appearance mode (logsumexp over a single
        # prototype is the identity, recovering the one-prototype-per-class head).
        temperature = self.log_temperature.exp()
        sim = torch.matmul(g, self.class_proto.t()).view(B, N, K, P)
        scores = torch.logsumexp(sim / temperature, dim=-1) \
            + self.alpha * logits          # [B, N, C]

        # explicit geometry-vs-class term: how well each box's position matches the
        # learned canonical location of every tooth class.
        if self.use_pos_proto:
            psim = torch.matmul(pe, self.pos_proto.t()).view(B, N, K, P)
            scores = scores + self.beta * torch.logsumexp(psim / temperature, dim=-1)

        # Sinkhorn runs in fp32 regardless of AMP autocast for stability.
        log_P = log_optimal_transport(
            scores.float(), self.bin_score.float(), self.sinkhorn_iters)

        outputs['assignment_log'] = log_P          # [B, N+1, C+1]

        # anti-collapse penalty over the appearance (and spatial) prototypes; the
        # criterion reads this scalar as `loss_proto_div`.
        proto_div = self._proto_repulsion(self.class_proto)
        if self.use_pos_proto:
            proto_div = proto_div + self._proto_repulsion(self.pos_proto)
        outputs['proto_div'] = proto_div
        return outputs
