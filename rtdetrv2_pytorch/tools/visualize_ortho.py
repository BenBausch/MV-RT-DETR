"""Visualize RT-DETR predictions on the OrthoTracked dataset.

Draws predicted boxes (with FDI tooth names + scores) and, optionally, the
ground-truth boxes for side-by-side comparison on the same image.

Examples
--------
# sample 12 val images, overlay GT, write to ./viz
python tools/visualize_ortho.py \
  -c configs/rtdetrv2/rtdetrv2_r18vd_ortho.yml \
  -r output/rtdetrv2_r18vd_ortho/best.pth \
  --split val --num 12 --gt --device cuda:2 --output-dir viz

# a single image
python tools/visualize_ortho.py -c <cfg> -r <ckpt> --im-file /path/CENTER.png

Run from the rtdetrv2_pytorch/ directory. Pick a free GPU with --device
(e.g. cuda:2) so it does not compete with a training run.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from src.core import YAMLConfig
from src.data.ortho_views import x0_from_filename


def load_font(size=16):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def names_from_coco(ann_file):
    """category_id -> name from a COCO json."""
    with open(ann_file) as f:
        coco = json.load(f)
    return {c["id"]: c["name"] for c in coco.get("categories", [])}, coco


def index_gt(coco):
    """file_name -> list of (category_id, [x,y,w,h]) from a loaded COCO dict."""
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        fn = by_id[a["image_id"]]
        gt.setdefault(fn, []).append((a["category_id"], a["bbox"]))
    return gt


def draw_box(drw, xyxy, color, text, font, text_above=True):
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    drw.rectangle([x0, y0, x1, y1], outline=color, width=3)
    if text:
        tb = drw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y0 - th - 2 if (text_above and y0 - th - 2 > 0) else y0 + 1
        drw.rectangle([x0, ty, x0 + tw + 2, ty + th + 2], fill=color)
        drw.text((x0 + 1, ty + 1), text, fill="white", font=font)


def build_model(cfg, ckpt_path, device):
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        print(f"loaded weights from {ckpt_path} "
              f"({'ema' if 'ema' in checkpoint else 'model'})")
    else:
        print("WARNING: no checkpoint given -> random weights (layout test only)")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            return self.postprocessor(self.model(images), orig_sizes)

    return Model().to(device).eval()


@torch.no_grad()
def run(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    device = torch.device(args.device)
    model = build_model(cfg, args.resume, device)
    font = load_font(16)

    # resolve class names + (optional) GT + image list
    cat2name, gt = {}, {}
    root = Path(args.root) if args.root else None
    images = []

    if args.im_file:
        images = [Path(args.im_file)]
        if args.ann_file:
            cat2name, coco = names_from_coco(args.ann_file)
    else:
        ann = args.ann_file or (root / "coco" / f"instances_{args.split}.json")
        cat2name, coco = names_from_coco(ann)
        gt = index_gt(coco) if args.gt else {}
        img_root = root or Path(os.path.commonpath(
            [args.root] if args.root else ["."]))
        files = [im["file_name"] for im in coco["images"]]
        files = files[: args.num] if args.num else files
        images = [(Path(args.root) / f) if args.root else Path(f) for f in files]
        rel_lookup = {str((Path(args.root) / f) if args.root else Path(f)): f
                      for f in files}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # match the model's input resolution and channel count from the config
    sz = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])
    in_ch = int(cfg.yaml_cfg.get('PResNet', {}).get('in_channels', 3))
    print(f"input size {sz}, in_channels {in_ch}"
          + (" (appending orientation channel)" if in_ch == 4 else ""))
    tf = T.Compose([T.Resize((sz[0], sz[1])), T.ToTensor()])

    for img_path in images:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        orig = torch.tensor([w, h])[None].to(device)
        data = tf(im)[None].to(device)

        # orientation-encoding 4th channel (inference: no flip), matching training
        if in_ch == 4:
            _, _, H, W = data.shape
            x0 = x0_from_filename(str(img_path))
            row = 2.0 * torch.sigmoid(10.0 * (torch.linspace(0, 1, W) - x0)) - 1.0
            chan = row.view(1, 1, 1, W).expand(1, 1, H, W).to(data.device)
            data = torch.cat([data, chan], dim=1)

        labels, boxes, scores = model(data, orig)
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        keep = scores > args.thrh

        drw = ImageDraw.Draw(im)

        # ground truth in green
        rel = None
        if gt:
            rel = rel_lookup.get(str(img_path))
            for cid, (bx, by, bw, bh) in [(c, b) for c, b in gt.get(rel, [])]:
                draw_box(drw, [bx, by, bx + bw, by + bh], "lime",
                         cat2name.get(cid, str(cid)), font, text_above=False)

        # predictions in red
        npred = 0
        for lab, box, sco in zip(labels[keep], boxes[keep], scores[keep]):
            cid = int(lab.item())
            name = cat2name.get(cid, str(cid))
            draw_box(drw, box.tolist(), "red",
                     f"{name} {sco.item():.2f}", font, text_above=True)
            npred += 1

        stem = (rel or img_path.name).replace("/", "__")
        dst = out_dir / f"{Path(stem).with_suffix('')}.jpg"
        im.save(dst, quality=92)
        ngt = len(gt.get(rel, [])) if gt else 0
        print(f"{dst.name}: {npred} preds>{args.thrh}"
              + (f", {ngt} gt" if gt else ""))

    print(f"\nwrote {len(images)} images to {out_dir}/  "
          "(red = prediction, green = ground truth)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-r", "--resume", help="checkpoint .pth (uses EMA weights if present)")
    ap.add_argument("-f", "--im-file", help="single image instead of a split")
    ap.add_argument("--root", default="/mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9",
                    help="dataset root (img_folder); file_names are relative to it")
    ap.add_argument("--ann-file", help="explicit COCO json (else <root>/coco/instances_<split>.json)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--num", type=int, default=12, help="how many images from the split (0 = all)")
    ap.add_argument("--gt", action="store_true", help="overlay ground-truth boxes (green)")
    ap.add_argument("--thrh", type=float, default=0.5, help="score threshold for predictions")
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--output-dir", default="viz")
    run(ap.parse_args())
