"""Layer-block diagram of the OTAssignmentHead with visualtorch.

The head is instantiated from the given config (so all dims -- hidden_dim, dim,
num_layers, dim_feedforward, num_classes, num_prototypes, num_queries -- are the
ones used in training) and re-wired into a tracing-friendly module: every real
submodule (input_proj, input_norm, box_pos MLP, encoder self-attention/FFN) and
parameter (prototypes, alpha/beta, dustbin, temperature) is reused as-is, and the
functional steps of `OTAssignmentHead.forward` (concat, residual adds, prototype
similarity, score fusion, Sinkhorn) are wrapped in small blocks so they appear as
boxes. The diagram module is asserted numerically equal to the real head.

Tensors are traced as [B, 1, N, D] instead of [B, N, D]: visualtorch discards the
sequence axis of 2D (per-sample) shapes, while a 3D shape is drawn as a block whose
height is the feature width D and whose depth is the number of queries N.

  python tools/visualization/visualize_assignment_head.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_ortho_assign.yml \
    -o output/viz_assign_head
  # -> assignment_head_flow.png  (layered volumetric blocks)
  #    assignment_head_graph.png (node/edge view)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import visualtorch
from PIL import ImageFont

from src.core import YAMLConfig, create
from src.zoo.rtdetr.assignment_head import log_optimal_transport


class TokenConcat(nn.Module):
    """[emb || box || logits] -> [.., N, hidden_dim + 4 + C]."""

    def forward(self, emb, boxes, logits):
        return torch.cat([emb, boxes, logits], dim=-1)


class BoxPosAdd(nn.Module):
    """f + box positional encoding."""

    def forward(self, f, pe):
        return f + pe


class ResidualAdd(nn.Module):
    """Pre-norm residual connection x + sublayer(x)."""

    def forward(self, x, h):
        return x + h


class SelfAttention(nn.Module):
    """Multi-head self-attention over the N queries (wraps the layer's MHA).

    The MHA is deliberately not registered as a child and its `forward` is called
    directly, so the tracer draws it as one block and it can run on the [B, 1, N, D]
    layout used for the diagram.
    """

    def __init__(self, mha):
        super().__init__()
        self.__dict__['mha'] = mha

    def forward(self, x):
        h = x.squeeze(1)
        return self.mha.forward(h, h, h, need_weights=False)[0].unsqueeze(1)


class PreNormAttention(nn.Module):
    """x + SelfAttn(LN(x)) -- first half of a norm_first TransformerEncoderLayer."""

    def __init__(self, layer):
        super().__init__()
        self.norm1 = layer.norm1
        self.self_attn = SelfAttention(layer.self_attn)
        self.add = ResidualAdd()

    def forward(self, x):
        return self.add(x, self.self_attn(self.norm1(x)))


class PreNormFFN(nn.Module):
    """x + W2 GELU(W1 LN(x)) -- second half of a norm_first TransformerEncoderLayer."""

    def __init__(self, layer):
        super().__init__()
        self.norm2 = layer.norm2
        self.linear1 = layer.linear1
        self.act = nn.GELU()
        self.linear2 = layer.linear2
        self.add = ResidualAdd()

    def forward(self, x):
        return self.add(x, self.linear2(self.act(self.linear1(self.norm2(x)))))


class AppearancePrototypeSim(nn.Module):
    """logsumexp_p <g_i, class_proto_{c,p}> / T -> [.., N, C]."""

    def __init__(self, head):
        super().__init__()
        self.class_proto = head.class_proto
        self.log_temperature = head.log_temperature
        self.K, self.P = head.num_classes, head.num_prototypes

    def forward(self, g):
        sim = torch.matmul(g, self.class_proto.t()).view(*g.shape[:-1], self.K, self.P)
        return torch.logsumexp(sim / self.log_temperature.exp(), dim=-1)


class SpatialPrototypeSim(nn.Module):
    """logsumexp_p <pe_i, pos_proto_{c,p}> / T -> [.., N, C]."""

    def __init__(self, head):
        super().__init__()
        self.pos_proto = head.pos_proto
        self.log_temperature = head.log_temperature
        self.K, self.P = head.num_classes, head.num_prototypes

    def forward(self, pe):
        psim = torch.matmul(pe, self.pos_proto.t()).view(*pe.shape[:-1], self.K, self.P)
        return torch.logsumexp(psim / self.log_temperature.exp(), dim=-1)


class ScoreFusion(nn.Module):
    """S = appearance + alpha * logits (+ beta * spatial) -> [.., N, C]."""

    def __init__(self, head):
        super().__init__()
        self.alpha = head.alpha
        self.beta = head.beta if head.use_pos_proto else None

    def forward(self, app, logits, spatial=None):
        scores = app + self.alpha * logits
        if spatial is not None:
            scores = scores + self.beta * spatial
        return scores


class SinkhornDustbin(nn.Module):
    """Dustbin-augmented log-Sinkhorn -> assignment_log [.., N+1, C+1]."""

    def __init__(self, head):
        super().__init__()
        self.bin_score = head.bin_score
        self.iters = head.sinkhorn_iters

    def forward(self, scores):
        return log_optimal_transport(scores.squeeze(1), self.bin_score, self.iters).unsqueeze(1)


class AssignmentHeadDiagram(nn.Module):
    """Same computation as OTAssignmentHead.forward, split into traceable blocks."""

    def __init__(self, head):
        super().__init__()
        self.use_box_pos = head.use_box_pos
        self.use_pos_proto = head.use_pos_proto
        self.concat = TokenConcat()
        self.input_proj = head.input_proj
        self.input_norm = head.input_norm
        if self.use_box_pos:
            self.box_pos = head.box_pos
            self.pos_add = BoxPosAdd()
        blocks = []
        for layer in head.encoder.layers:
            blocks += [PreNormAttention(layer), PreNormFFN(layer)]
        self.encoder = nn.Sequential(*blocks)
        self.app_sim = AppearancePrototypeSim(head)
        if self.use_pos_proto:
            self.pos_sim = SpatialPrototypeSim(head)
        self.fusion = ScoreFusion(head)
        self.sinkhorn = SinkhornDustbin(head)

    def forward(self, emb, boxes, logits):
        f = self.input_norm(self.input_proj(self.concat(emb, boxes, logits)))
        pe = None
        if self.use_box_pos:
            pe = self.box_pos(boxes)
            f = self.pos_add(f, pe)
        g = self.encoder(f)
        app = self.app_sim(g)
        spatial = self.pos_sim(pe) if self.use_pos_proto else None
        return self.sinkhorn(self.fusion(app, logits, spatial))


COLOR_MAP = {
    visualtorch.Input: {'fill': '#BBBBBB'},
    TokenConcat: {'fill': '#DDDDDD'},
    BoxPosAdd: {'fill': '#DDDDDD'},
    ResidualAdd: {'fill': '#DDDDDD'},
    nn.Linear: {'fill': '#009E73'},
    nn.LayerNorm: {'fill': '#56B4E9'},
    nn.GELU: {'fill': '#F0E442'},
    SelfAttention: {'fill': '#D55E00'},
    AppearancePrototypeSim: {'fill': '#CC79A7'},
    SpatialPrototypeSim: {'fill': '#882255'},
    ScoreFusion: {'fill': '#E69F00'},
    SinkhornDustbin: {'fill': '#0072B2'},
}


def build_head(cfg_path):
    cfg = YAMLConfig(cfg_path)
    head = create('OTAssignmentHead', cfg.global_cfg).eval()
    num_queries = cfg.global_cfg['RTDETRTransformerv2']['num_queries']
    return head, num_queries


def input_shapes(head, num_queries, batch=1):
    hidden_dim = head.input_proj.in_features - 4 - head.num_classes
    return ((batch, 1, num_queries, hidden_dim),
            (batch, 1, num_queries, 4),
            (batch, 1, num_queries, head.num_classes))


def check_equivalence(head, diagram, num_queries):
    """Guard: the diagram module must compute exactly what the real head does."""
    emb, boxes, logits = (torch.randn(s) for s in input_shapes(head, num_queries, batch=2))
    boxes = boxes.sigmoid()
    with torch.no_grad():
        ref = head({'pred_embed': emb.squeeze(1), 'pred_boxes': boxes.squeeze(1),
                    'pred_logits': logits.squeeze(1)})['assignment_log']
        out = diagram(emb, boxes, logits).squeeze(1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/rtdetrv2/rtdetrv2_r50vd_ortho_assign.yml')
    parser.add_argument('-o', '--out', default='output/viz_assign_head')
    parser.add_argument('--font-size', type=int, default=18)
    args = parser.parse_args()

    head, num_queries = build_head(args.config)
    diagram = AssignmentHeadDiagram(head).eval()
    check_equivalence(head, diagram, num_queries)

    shapes = input_shapes(head, num_queries)
    font = ImageFont.load_default(size=args.font_size)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = dict(font=font, color_map=COLOR_MAP, legend=True, show_dimension=True, padding=40)

    flow_path = out_dir / 'assignment_head_flow.png'
    visualtorch.render(
        diagram, shapes, style='flow', to_file=str(flow_path),
        scale_xy=0.3, min_xy=24, min_z=28, spacing=60, shade_step=20,
        level_gap=30, connector_width=2, **common)

    graph_path = out_dir / 'assignment_head_graph.png'
    visualtorch.render(
        diagram, shapes, style='graph', to_file=str(graph_path),
        show_neurons=False, show_arrows=True, layer_spacing=170, node_size=60, **common)

    n_params = sum(p.numel() for p in head.parameters())
    print(f'OTAssignmentHead: {n_params / 1e6:.2f}M params, N={num_queries} queries, '
          f'C={head.num_classes} classes, P={head.num_prototypes} prototypes/class')
    print(f'wrote {flow_path}\nwrote {graph_path}')


if __name__ == '__main__':
    main()
