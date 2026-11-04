#!/usr/bin/env python
"""Check that the resized IOP-Compass copy kept its annotations aligned.

For each sampled image, writes one panel with three columns:
  1. release image + release annotations (drawn at full res, shown downscaled)
  2. resized image + rescaled annotations (from resize_dataset.py)
  3. what training sees: IOPCompassDetection on the resized copy (mirror_buccal on)
     after Resize(1024x1024), with its target boxes
Contours are outlines, boxes are thin rectangles, labels are FDI codes.

Also prints a numeric check over the sampled images: every tooth's contour is
rasterised at release resolution, downscaled with nearest-neighbour, and compared
(IoU) to the rescaled contour rasterised on the resized image.

    python tools/visualization/visualize_iop_resize.py --n 12
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from src.data.dataset.iop_dataset import IOPCompassDetection  # noqa: E402
from src.data.ortho_views import _FDI_ORDER  # noqa: E402
from src.data.transforms import Compose  # noqa: E402

PANEL_H = 480


def draw_teeth(img, teeth, width):
    draw = ImageDraw.Draw(img)
    for t in teeth:
        color = t.get("color", "#00ff00")
        for c in t["contours"]:
            pts = [tuple(p) for p in c]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=color, width=width)
        draw.rectangle([t["x_min"], t["y_min"], t["x_max"], t["y_max"]], outline=color, width=max(1, width // 2))
        draw.text((t["x_min"] + 2, t["y_min"] + 1), str(t["FDI_NUM"]), fill="white")
    return img


def fit(img):
    return img.resize((round(img.width * PANEL_H / img.height), PANEL_H), Image.BILINEAR)


def raster(contours, size):
    m = Image.new("1", size, 0)
    d = ImageDraw.Draw(m)
    for c in contours:
        if len(c) >= 3:
            d.polygon([tuple(p) for p in c], fill=1)
    return np.asarray(m, dtype=bool)


def contour_iou(orig_tooth, new_tooth, orig_size, new_size):
    big = Image.fromarray(raster(orig_tooth["contours"], orig_size))
    down = np.asarray(big.resize(new_size, Image.NEAREST), dtype=bool)
    small = raster(new_tooth["contours"], new_size)
    union = (down | small).sum()
    return (down & small).sum() / union if union else 1.0


def loader_view(ds, idx):
    img, target = ds[idx]
    img = img.convert("RGB") if isinstance(img, Image.Image) else img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for box, label in zip(target["boxes"].tolist(), target["labels"].tolist()):
        cx, cy, bw, bh = box
        x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        draw.rectangle([x0, y0, x1, y1], outline="yellow", width=2)
        draw.text((x0 + 2, y0 + 1), _FDI_ORDER[label], fill="white")
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/IOP-Compass")
    ap.add_argument("--dst", default="/mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/IOP-Compass_1024")
    ap.add_argument("--split", default="/mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/splits/train.txt")
    ap.add_argument("--out", default="output/viz_iop_resize")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst, out = Path(args.src), Path(args.dst), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The pipeline's own resize + box format, so column 3 is exactly what training sees.
    transforms = Compose(ops=[
        {"type": "Resize", "size": [1024, 1024]},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ])
    ds = IOPCompassDetection(str(dst), args.split, transforms=transforms, mirror_buccal=True, num_classes=32)
    index = {name: i for i, name in enumerate(im["file_name"] for im in ds.coco.dataset["images"])}

    rels = sorted(index)
    random.seed(args.seed)
    sample = random.sample(rels, args.n)
    # make sure every view token is represented
    for token in ("Center", "Left", "Right", "Upper", "Down"):
        if not any(f"IOP_{token}_" in r for r in sample):
            sample.append(next(r for r in rels if f"IOP_{token}_" in r))

    ious, box_err, worst = [], [], []
    for rel in sample:
        a0 = json.loads((src / rel).with_suffix(".json").read_text())
        a1 = json.loads((dst / rel).with_suffix(".json").read_text())
        im0 = Image.open(src / rel).convert("RGB")
        im1 = Image.open(dst / rel).convert("RGB")
        sx, sy = a1["resize"]["scale_x"], a1["resize"]["scale_y"]

        for t0, t1 in zip(a0["teeth"], a1["teeth"]):
            ious.append(contour_iou(t0, t1, im0.size, im1.size))
            worst.append((ious[-1], rel, t1["FDI_NUM"], t1["pixel_area"]))
            pts = np.array([p for c in t1["contours"] for p in c])
            box_err.append(max(abs(pts[:, 0].min() - t1["x_min"]), abs(pts[:, 0].max() - t1["x_max"]),
                               abs(pts[:, 1].min() - t1["y_min"]), abs(pts[:, 1].max() - t1["y_max"])))

        col1 = fit(draw_teeth(im0, a0["teeth"], width=max(2, im0.width // 400)))
        col2 = fit(draw_teeth(im1, a1["teeth"], width=2))
        col3 = fit(loader_view(ds, index[rel]))
        panel = Image.new("RGB", (col1.width + col2.width + col3.width + 20, PANEL_H + 24), "black")
        x = 0
        for col, title in ((col1, f"release {im0.width}x{im0.height}"),
                           (col2, f"resized {im1.width}x{im1.height} (sx={sx:.4f}, sy={sy:.4f})"),
                           (col3, "loader: mirror_buccal + Resize 1024x1024")):
            panel.paste(col, (x, 24))
            ImageDraw.Draw(panel).text((x + 4, 6), title, fill="white")
            x += col.width + 10
        name = out / (rel.replace("/", "__").replace(".png", ".jpg"))
        panel.save(name, quality=90)
        print(f"  wrote {name}  ({len(a1['teeth'])} teeth)")

    ious = np.array(ious)
    print(f"\ncontour IoU (release mask downscaled vs rescaled contour), {len(ious)} teeth: "
          f"mean {ious.mean():.3f}, median {np.median(ious):.3f}, min {ious.min():.3f}, "
          f"<0.8: {(ious < 0.8).sum()}")
    for iou, rel, fdi, area in sorted(worst)[:3]:
        print(f"  lowest: IoU {iou:.3f}  {rel} FDI {fdi}, area {area:.0f} px at the resized scale")
    print(f"box vs rescaled-contour extent, max abs diff: {max(box_err):.3f} px")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
