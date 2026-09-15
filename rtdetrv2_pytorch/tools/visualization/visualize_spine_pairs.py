"""Visualize BUU-LSPINE: the AP and LA view of the same patient, side by side.

Loads the dataset through the config (so it shows exactly what `SpineDetection`
hands the model, including the synthesized S1 box) and pairs the two views by
patient id. One output image per patient: AP on the left, LA on the right.

Without ``-r`` it draws ground truth only. With ``-r <checkpoint>`` it also runs
the model and overlays predictions, using **the config's own postprocessor** --
so an assignment-head config is decoded through `RTDETRAssignmentPostProcessor`
(one box per class via Sinkhorn + Hungarian) rather than plain top-k.

Two stages:
  --stage transformed  (default) what the dataloader yields: post-transform image
                       at the model's input resolution, boxes decoded back from
                       normalized cxcywh. This is the end-to-end pipeline check.
  --stage raw          the original JPEG at full resolution with the boxes as
                       built from the CSV edge lines. Best for judging annotations.

Ground truth is drawn as thin dashed boxes, predictions as solid boxes labelled
`<level> <score>`; both are coloured by class, so a correct prediction sits right
on top of its GT box in the same colour.

Examples
--------
# ground truth only, 8 val patients
python tools/visualization/visualize_spine_pairs.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_spine_base.yml --split val --num 8

# predictions from the trained assignment-head model, over GT
python tools/visualization/visualize_spine_pairs.py \
  -c configs/rtdetrv2/rtdetrv2_r50vd_spine_assign.yml \
  -r output/rtdetrv2_r50vd_spine_assign/best.pth \
  --split val --num 8 --device cuda:3 --output-dir viz_spine_assign

Run from the rtdetrv2_pytorch/ directory. Note that --split train applies the
training augmentations, so the image will look photometrically distorted.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

from src.core import YAMLConfig, yaml_utils
from src.data.dataset.spine_dataset import patient_id_from_path, view_from_path
from src.data.transforms._transforms import ConvertBoxes

# one colour per level, in L1..L5 + S1 order
CLASS_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#00d5ff']
VIEW_ORDER = ['AP', 'LA']
CAPTION_H = 34


def load_font(size=16):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_dataset(config, split, update):
    """Build the dataset for a split straight from the config.

    train/val come from their own dataloader sections; `test` reuses the val
    section with the sibling test.txt, the same way det_solver derives its test
    loader, so the transforms match. Returns (cfg, dataset) -- the cfg is needed
    to build the model and its matching postprocessor.
    """
    cfg = YAMLConfig(config, **yaml_utils.parse_cli(update))
    if split == 'test':
        vcfg = cfg.yaml_cfg['val_dataloader']
        val_split = vcfg['dataset']['split_file']
        test_split = str(Path(val_split).with_name('test.txt'))
        if not Path(test_split).exists():
            raise SystemExit(f'no test split at {test_split}')
        vcfg['dataset']['split_file'] = test_split
    loader = cfg.train_dataloader if split == 'train' else cfg.val_dataloader
    return cfg, loader.dataset


def build_model(cfg, ckpt_path, device):
    """Load a checkpoint into the config's model, preferring EMA weights.

    The postprocessor comes from the same config, so an assignment-head run is
    decoded with RTDETRAssignmentPostProcessor (one box per class) and a plain
    run with RTDETRPostProcessor (top-k) -- no flag needed, the config decides.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['ema']['module'] if 'ema' in ckpt else ckpt['model']
    model = cfg.model
    missing, unexpected = model.load_state_dict(state, strict=False)
    src = 'ema' if 'ema' in ckpt else 'model'
    print(f'loaded {src} weights from {ckpt_path} '
          f'(epoch {ckpt.get("last_epoch", "?")}; '
          f'{len(missing)} missing, {len(unexpected)} unexpected)')
    return model.to(device).eval(), cfg.postprocessor.to(device).eval()


@torch.no_grad()
def predict(model, postprocessor, dataset, idx, device, score_thr):
    """Run the model on one dataset item. Returns boxes at ORIGINAL image scale."""
    img, target = dataset[idx]
    samples = img[None].to(device)
    orig = target['orig_size'].reshape(1, 2).to(device)      # (w, h)
    outputs = model(samples)
    res = postprocessor(outputs, orig)[0]
    keep = res['scores'] >= score_thr
    return (res['boxes'][keep].cpu().tolist(),
            res['labels'][keep].cpu().tolist(),
            res['scores'][keep].cpu().tolist())


def index_by_patient(dataset):
    """patient id -> {'AP': idx, 'LA': idx}, in dataset order."""
    pairs = {}
    for idx, img_id in enumerate(dataset.ids):
        fn = dataset.coco.loadImgs(img_id)[0]['file_name']
        pairs.setdefault(patient_id_from_path(fn), {})[view_from_path(fn)] = idx
    return pairs


def raw_panel(dataset, idx):
    """Original image + boxes as xyxy pixels, straight from the COCO index."""
    img_id = dataset.ids[idx]
    im = Image.open(dataset.img_paths[idx]).convert('RGB')
    boxes, labels = [], []
    for a in dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=img_id)):
        x, y, w, h = a['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(a['category_id'])
    return im, boxes, labels


def box_convention(dataset):
    """(fmt, normalized) of the boxes this dataset's transforms emit.

    It has to be read off the config: ConvertBoxes' normalize step divides the
    tensor, which drops the tv_tensor BoundingBoxes wrapper, so the format is no
    longer recoverable from the tensor itself. The train pipeline ends in
    normalized cxcywh, the val one leaves boxes as xyxy pixels.
    """
    fmt, normalized = 'xyxy', False
    for op in getattr(dataset._transforms, 'transforms', []):
        if isinstance(op, ConvertBoxes):
            fmt = (op.fmt or fmt).lower()
            normalized = op.normalize
    return fmt, normalized


def transformed_panel(dataset, idx, fmt, normalized):
    """What the dataloader yields, decoded back to a viewable image + xyxy pixels."""
    img, target = dataset[idx]

    # ConvertPILImage(scale=True) leaves float [0,1]; a 4th orientation channel,
    # if the config enables one, is dropped for display.
    arr = img[:3] if img.shape[0] > 3 else img
    arr = (arr.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    im = Image.fromarray(arr)
    W, H = im.size

    boxes = torch.as_tensor(target['boxes'], dtype=torch.float32).reshape(-1, 4)
    if normalized:
        boxes = boxes * torch.tensor([W, H, W, H], dtype=torch.float32)
    if boxes.numel() and fmt != 'xyxy':
        boxes = box_convert(boxes, in_fmt=fmt, out_fmt='xyxy')
    return im, boxes.tolist(), target['labels'].tolist()


def _clip(box, W, H):
    """Clamp to the frame; PIL raises on x1 < x0 (augmentation can push boxes out)."""
    x0, y0, x1, y1 = box
    return [min(max(x0, 0), W), min(max(y0, 0), H),
            min(max(x1, x0, 0), W), min(max(y1, y0, 0), H)]


def _dashed_rect(drw, box, color, width, dash=9):
    """Dashed rectangle -- PIL has no dash support, so step along each edge."""
    x0, y0, x1, y1 = box
    for x in range(int(x0), int(x1), dash * 2):
        drw.line([x, y0, min(x + dash, x1), y0], fill=color, width=width)
        drw.line([x, y1, min(x + dash, x1), y1], fill=color, width=width)
    for y in range(int(y0), int(y1), dash * 2):
        drw.line([x0, y, x0, min(y + dash, y1)], fill=color, width=width)
        drw.line([x1, y, x1, min(y + dash, y1)], fill=color, width=width)


def draw_panel(im, gt, preds, names, font, title):
    """Draw GT (dashed) and, if given, predictions (solid + score) on a copy of `im`.

    `gt` is (boxes, labels); `preds` is (boxes, labels, scores) or None. Both are
    xyxy in the panel's own pixel space, and both are coloured by class -- so a
    correct prediction lands on its GT box in the same colour.
    """
    im = im.copy()
    drw = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, W // 400)

    gt_boxes, gt_labels = gt
    for box, lab in zip(gt_boxes, gt_labels):
        color = CLASS_COLORS[lab % len(CLASS_COLORS)]
        box = _clip(box, W, H)
        if preds is None:
            drw.rectangle(box, outline=color, width=lw)
            text = names.get(lab, str(lab))
            tb = drw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            tx, ty = box[0], max(box[1] - th - 4, 0)
            drw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
            drw.text((tx + 3, ty + 2), text, fill='white', font=font)
        else:
            # GT recedes to a dashed outline so the prediction reads on top
            _dashed_rect(drw, box, color, max(lw - 1, 1))

    if preds is not None:
        pr_boxes, pr_labels, pr_scores = preds
        for box, lab, sco in zip(pr_boxes, pr_labels, pr_scores):
            color = CLASS_COLORS[lab % len(CLASS_COLORS)]
            box = _clip(box, W, H)
            drw.rectangle(box, outline=color, width=lw + 1)
            text = f'{names.get(lab, str(lab))} {sco:.2f}'
            tb = drw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            tx, ty = box[0], max(box[1] - th - 4, 0)
            drw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
            drw.text((tx + 3, ty + 2), text, fill='white', font=font)

    panel = Image.new('RGB', (W, H + CAPTION_H), 'black')
    panel.paste(im, (0, 0))
    ImageDraw.Draw(panel).text((6, H + 8), title, fill='white', font=font)
    return panel


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--config', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--stage', default='transformed', choices=['transformed', 'raw'],
                    help='dataloader output (default) or the original image')
    ap.add_argument('--num', type=int, default=8,
                    help='how many patients (0 = all); ignored with --patients')
    ap.add_argument('--patients', nargs='+', help='explicit 4-digit patient ids')
    ap.add_argument('--height', type=int, default=900, help='per-panel display height')
    ap.add_argument('--output-dir', default='viz_spine')
    ap.add_argument('--format', default='png', choices=['png', 'jpg'],
                    help='PNG (default) is lossless, so box edges and faint bone '
                         'detail stay crisp; JPG is smaller')
    ap.add_argument('-r', '--resume', help='checkpoint .pth; overlays predictions '
                                           '(uses EMA weights when present)')
    ap.add_argument('--thrh', type=float, default=0.5,
                    help='score threshold for drawn predictions')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('-u', '--update', nargs='+', help='update yaml config')
    args = ap.parse_args()

    cfg, dataset = build_dataset(args.config, args.split, args.update)
    names = dataset.category2name
    pairs = index_by_patient(dataset)
    fmt, normalized = box_convention(dataset)
    font = load_font(16)

    model = postprocessor = None
    if args.resume:
        device = torch.device(args.device)
        model, postprocessor = build_model(cfg, args.resume, device)
        print(f'postprocessor: {type(postprocessor).__name__}')

    if args.patients:
        missing = [p for p in args.patients if p not in pairs]
        if missing:
            print(f'  WARN not in the {args.split} split, skipping: {missing}')
        selected = [p for p in args.patients if p in pairs]
    else:
        selected = sorted(pairs)
        selected = selected[:args.num] if args.num else selected

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in selected:
        panels = []
        for view in VIEW_ORDER:
            idx = pairs[pid].get(view)
            if idx is None:
                print(f'  WARN patient {pid} has no {view} view in this split')
                continue
            if args.stage == 'raw':
                im, boxes, labels = raw_panel(dataset, idx)
            else:
                im, boxes, labels = transformed_panel(dataset, idx, fmt, normalized)

            preds = None
            if model is not None:
                pb, pl, ps = predict(model, postprocessor, dataset, idx,
                                     device, args.thrh)
                # predictions come back at ORIGINAL image scale; the transformed
                # panel is at model-input scale, so map them into the panel's space.
                if args.stage != 'raw':
                    ow, oh = dataset.coco.loadImgs(dataset.ids[idx])[0]['width'], \
                             dataset.coco.loadImgs(dataset.ids[idx])[0]['height']
                    sx, sy = im.size[0] / ow, im.size[1] / oh
                    pb = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in pb]
                preds = (pb, pl, ps)

            # scale to a common display height, boxes with it
            scale = args.height / im.size[1]
            im = im.resize((max(int(im.size[0] * scale), 1), args.height))
            boxes = [[v * scale for v in b] for b in boxes]
            if preds is not None:
                preds = ([[v * scale for v in b] for b in preds[0]], preds[1], preds[2])
                title = (f'{pid}  {view}   GT {len(boxes)} (dashed) | '
                         f'pred {len(preds[0])} @{args.thrh}: '
                         f'{", ".join(names.get(l, str(l)) for l in preds[1])}')
            else:
                title = (f'{pid}  {view}  {len(boxes)} boxes: '
                         f'{", ".join(names.get(l, str(l)) for l in labels)}')
            panels.append(draw_panel(im, (boxes, labels), preds, names, font, title))

        if not panels:
            continue
        canvas = Image.new('RGB', (sum(p.size[0] for p in panels),
                                   max(p.size[1] for p in panels)), 'black')
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.size[0]
        dst = out_dir / f'{pid}_{args.stage}.{args.format}'
        canvas.save(dst, **({'quality': 92} if args.format == 'jpg' else {}))
        print(f'{dst}  ({len(panels)} views)')

    print(f'\nwrote {len(selected)} patients to {out_dir}/  '
          f'(split={args.split}, stage={args.stage}; colours: '
          f'{", ".join(f"{n}={c}" for n, c in zip(names.values(), CLASS_COLORS))})')


if __name__ == '__main__':
    main()
