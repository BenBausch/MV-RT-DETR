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

The grouped flow figure uses visualtorch's (private, pinned 1.4.1) flow layout to get
every block's position, then draws nested, titled frames around logical stages
(input projection, box PE, encoder -> layer -> sublayer, OT assignment) and an
"N x D" dimension label below each block.

  python tools/visualization/visualize_assignment_head.py \
    -c configs/rtdetrv2/rtdetrv2_r50vd_ortho_assign.yml \
    -o output/viz_assign_head
  # -> assignment_head_flow.png          (layered volumetric blocks)
  #    assignment_head_flow_grouped.png  (same, with nested group frames + per-block dims)
  #    assignment_head_graph.png         (node/edge view)
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import visualtorch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont
from visualtorch.flow import _draw_diagram_frame, _draw_legend, _prepare_flow_render

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
    """Same computation as OTAssignmentHead.forward, split into traceable blocks.

    `draw_layers` truncates the encoder to its first k layers, so the (identical)
    repeated layers can be drawn once and annotated "xN" instead.
    """

    def __init__(self, head, draw_layers=None):
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
        for layer in list(head.encoder.layers)[:draw_layers]:
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


def group_tree(head):
    """Nested stages of the head: (title, style, member module prefixes, children).

    A node belongs to a group if its qualified module name equals or is nested under
    one of the prefixes; a parent's members are the union of its children's.
    The encoder is drawn as a single (pre-norm) layer annotated with the repeat count.
    """
    n = len(head.encoder.layers)
    return [
        ('Decoder outputs', 'input', ['<input>'], []),
        ('Input projection', 'proj', ['concat', 'input_proj', 'input_norm', 'pos_add'], []),
        ('Box PE + spatial prior', 'box', ['box_pos', 'pos_sim'], []),
        (f'Transformer encoder layer  × {n}', 'layer', [], [
            ('Self-attention', 'attn', ['encoder.0'], []),
            ('Feed-forward', 'ffn', ['encoder.1'], []),
        ]),
        ('OT assignment', 'assign', [], [
            (f'Prototypes (P={head.num_prototypes})', 'score', ['app_sim', 'fusion'], []),
            ('Sinkhorn', 'sinkhorn', ['sinkhorn'], []),
        ]),
    ]


# (frame fill, frame edge / title pill) per group style
GROUP_STYLE = {
    'input': ('#F4F4F4', '#6E6E6E'),
    'proj': ('#EAF3FB', '#3A78B5'),
    'box': ('#F7EBF2', '#8E3B67'),
    'encoder': ('#F5F2FB', '#5B4A9B'),
    'layer': ('#EAE5F6', '#5B4A9B'),
    'attn': ('#FCEBE0', '#B5541A'),
    'ffn': ('#E4F4ED', '#1E7F5C'),
    'assign': ('#FDF7EA', '#9A6A12'),
    'score': ('#FBEBCB', '#9A6A12'),
    'sinkhorn': ('#E1ECF7', '#1F5F99'),
}


def load_font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal'))
    return ImageFont.truetype(path, size)


def render_grouped_flow(diagram, head, shapes, out_path, font_size=18, margin=130, pad_leaf=14, pad_step=36):
    font = load_font(font_size)
    dim_font = load_font(font_size - 2)
    title_font = load_font(font_size, bold=True)
    flow_kw = dict(min_z=28, min_xy=24, max_z=400, max_xy=2000, scale_z=0.1, scale_xy=0.3,
                   draw_volume=True, spacing=170, shade_step=20, opacity=255, padding=10)
    transparent = (255, 255, 255, 0)

    setup = _prepare_flow_render(
        diagram, shapes, input_dtype=None, type_ignore=None, color_map=COLOR_MAP, palette='okabe_ito',
        low_dim_orientation='z', font=font, show_dimension=False, legend=False, level_gap=30,
        show_input=True, outline_width=1, one_dim_orientation=None, legend_position='bottom-left',
        background_fill=transparent, **flow_kw)
    blocks = _draw_diagram_frame(
        setup, reveal_up_to=len(setup.column_layout.boxes_by_column) - 1, draw_funnel_flag=True,
        connector_fill=None, connector_width=2, padding=flow_kw['padding'], show_dimension=False,
        font_color='black', legend=False, draw_volume=True, shade_step=flow_kw['shade_step'],
        opacity=255, spacing=flow_kw['spacing'], background_fill=transparent,
        legend_position='bottom-left')

    # per-node qualified module name, block extent (incl. 3D extrusion) and dims label
    names = {id(m): name for name, m in diagram.named_modules()}
    label_h = dim_font.getbbox('Ag')[3] + 8
    nodes = []
    for column in setup.architecture.columns:
        for layer in column:
            box = setup.column_layout.id_to_box.get(layer.node_id)
            if box is None:
                continue
            name = '<input>' if isinstance(layer.module, visualtorch.Input) else names[id(layer.module)]
            n_tok, d = layer.output_shape[-2:]
            nodes.append(dict(
                name=name, label=f'{n_tok}×{d}',
                front=(box.x1 + margin, box.y2 + margin, box.x2 + margin),
                extent=(box.x1 + margin, box.y1 - box.de + margin, box.x2 + box.de + margin, box.y2 + label_h + margin)))

    def members(prefixes):
        return [nd for nd in nodes if any(nd['name'] == p or nd['name'].startswith(p + '.') for p in prefixes)]

    def pill_size(title):
        tx1, ty1, tx2, ty2 = title_bbox(title)
        return tx2 - tx1 + 20, ty2 - ty1 + 10

    def title_bbox(title):
        return ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), title, font=title_font)

    # resolve the tree into frames, outer groups padded enough to clear their children
    frames = []

    def resolve(group, level):
        title, style, prefixes, children = group
        nds = members(prefixes)
        height = 0
        for child in children:
            child_nds, child_height = resolve(child, level + 1)
            nds += child_nds
            height = max(height, child_height + 1)
        if nds:
            pad = pad_leaf + pad_step * height
            x1 = min(nd['extent'][0] for nd in nds) - pad
            y1 = min(nd['extent'][1] for nd in nds) - pad
            x2 = max(nd['extent'][2] for nd in nds) + pad
            y2 = max(nd['extent'][3] for nd in nds) + pad
            # widen symmetrically so the title pill never overhangs its frame
            overhang = (pill_size(title)[0] + 28) - (x2 - x1)
            if overhang > 0:
                x1, x2 = x1 - overhang / 2, x2 + overhang / 2
            frames.append((level, title, style, (x1, y1, x2, y2)))
        return nds, height

    for group in group_tree(head):
        resolve(group, 0)
    frames.sort(key=lambda f: f[0])  # parents first so children draw on top

    w, h = blocks.size
    canvas = Image.new('RGBA', (w + 2 * margin, h + 2 * margin), 'white')
    draw = ImageDraw.Draw(canvas)
    for _, _, style, rect in frames:
        fill, edge = GROUP_STYLE[style]
        if style == 'layer':  # stacked "cards" behind a repeated block
            for off in (24, 12):
                draw.rounded_rectangle(tuple(v + off for v in rect), radius=14, fill=fill, outline=edge, width=2)
        draw.rounded_rectangle(rect, radius=14, fill=fill, outline=edge, width=2)
    canvas.alpha_composite(blocks, (margin, margin))

    draw = ImageDraw.Draw(canvas)
    for nd in nodes:
        fx1, fy2, fx2 = nd['front']
        pos = ((fx1 + fx2) / 2, fy2 + 6)
        lx1, ly1, lx2, ly2 = draw.textbbox(pos, nd['label'], font=dim_font, anchor='mt')
        draw.rounded_rectangle((lx1 - 4, ly1 - 2, lx2 + 4, ly2 + 2), radius=4, fill=(255, 255, 255, 230))
        draw.text(pos, nd['label'], font=dim_font, fill='#222222', anchor='mt')
    for _, title, style, (x1, y1, _, _) in frames:
        _, edge = GROUP_STYLE[style]
        tx1, ty1, _, _ = title_bbox(title)
        pill_w, pill_h = pill_size(title)
        px, py = x1 + 14, y1 - pill_h / 2
        draw.rounded_rectangle((px, py, px + pill_w, py + pill_h), radius=pill_h / 2, fill=edge)
        draw.text((px + 10 - tx1, py + 5 - ty1), title, font=title_font, fill='white')

    # crop to content, then append the block-type legend
    bbox = Image.eval(canvas.convert('L'), lambda v: 255 - v).getbbox()
    canvas = canvas.crop((max(bbox[0] - 30, 0), max(bbox[1] - 30, 0),
                          min(bbox[2] + 30, canvas.width), min(bbox[3] + 30, canvas.height)))
    canvas = _draw_legend(canvas, setup.layer_types, setup.color_map, font, 'black', True,
                          flow_kw['shade_step'], 255, 40, 30, 'white', 'bottom-left')
    canvas.convert('RGB').save(out_path)


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
    check_equivalence(head, AssignmentHeadDiagram(head).eval(), num_queries)
    # identical encoder layers are drawn once (annotated "x N" in the grouped figure)
    diagram = AssignmentHeadDiagram(head, draw_layers=1).eval()

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

    grouped_path = out_dir / 'assignment_head_flow_grouped.png'
    render_grouped_flow(diagram, head, shapes, grouped_path, font_size=args.font_size)

    graph_path = out_dir / 'assignment_head_graph.png'
    visualtorch.render(
        diagram, shapes, style='graph', to_file=str(graph_path),
        show_neurons=False, show_arrows=True, layer_spacing=170, node_size=60, **common)

    n_params = sum(p.numel() for p in head.parameters())
    print(f'OTAssignmentHead: {n_params / 1e6:.2f}M params, N={num_queries} queries, '
          f'C={head.num_classes} classes, P={head.num_prototypes} prototypes/class')
    print(f'wrote {flow_path}\nwrote {grouped_path}\nwrote {graph_path}')


if __name__ == '__main__':
    main()
