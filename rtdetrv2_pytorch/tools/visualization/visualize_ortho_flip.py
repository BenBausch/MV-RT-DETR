"""Visualize the ortho flip-equivariance and the orientation-encoding channel.

Uses the ACTUAL implemented transforms (RandomHorizontalFlipFDI, OrientationChannel)
so what you see is exactly what training feeds the model.

Produces two sets of panels in --out:
  1. flip_<subject>.jpg : ground-truth boxes+FDI labels, original (left) vs
     horizontally flipped with adapted labels (right). ~5 subjects.
  2. orient_<VIEW>.jpg  : the orientation channel overlaid as a heatmap (blue=-1
     left .. red=+1 right, green line = zero crossing), original vs flipped, for
     each of the 5 views (CENTER/DOWN/LEFT/RIGHT/UP).

Run (no GPU needed):
  python tools/visualize_ortho.py \
    --root /mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9 \
    --out output/viz_ortho
"""
import argparse
import copy
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from PIL import Image as PILImage, ImageDraw, ImageFont

from src.data.dataset.coco_dataset import CocoDetection
from src.data.transforms._transforms import RandomHorizontalFlipFDI, OrientationChannel
from src.data.ortho_views import VIEW_NAMES, view_id_from_filename

ROOT_DEFAULT = "/mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9"


def load_font(size=18):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = load_font(44)

# one color per FDI quadrant (first digit of the code); makes the flip quadrant
# swap (1<->2, 3<->4) visually obvious.
QUAD_COLORS = {
    1: (255, 60, 60),     # upper right - red
    2: (255, 220, 0),     # upper left  - yellow
    3: (0, 200, 255),     # lower left  - cyan
    4: (255, 0, 220),     # lower right - magenta
}


def _quad_color(name):
    return QUAD_COLORS.get(int(name[0]) if name[:1] in "1234" else 0, (255, 255, 255))


def draw_boxes(pil, boxes, labels, names):
    pil = pil.convert("RGB").copy()
    drw = ImageDraw.Draw(pil)
    for b, l in zip(boxes.tolist(), labels.tolist()):
        x0, y0, x1, y1 = [float(v) for v in b]
        txt = names.get(int(l), str(int(l)))
        color = _quad_color(txt)
        drw.rectangle([x0, y0, x1, y1], outline=color, width=5)
        tb = drw.textbbox((0, 0), txt, font=FONT)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(0, y0 - th - 4)
        drw.rectangle([x0, ty, x0 + tw + 4, ty + th + 4], fill=color)
        drw.text((x0 + 2, ty + 2), txt, fill="black", font=FONT)
    return pil


def overlay_orientation(pil, chan):
    """chan: [H,W] in [-1,1]. Blue=left(-), red=right(+), alpha ~ |value|."""
    img = np.asarray(pil.convert("RGB"), dtype=np.float32)
    v = chan.numpy()
    H, W = v.shape
    color = np.zeros((H, W, 3), dtype=np.float32)
    color[..., 0] = np.clip(v, 0, 1) * 255      # red for +
    color[..., 2] = np.clip(-v, 0, 1) * 255     # blue for -
    a = (0.55 * np.abs(v))[..., None]
    blended = (img * (1 - a) + color * a).astype(np.uint8)
    out = PILImage.fromarray(blended)
    # zero-crossing column (where |v| is smallest) as a green line
    zcol = int(np.argmin(np.abs(v[0])))
    drw = ImageDraw.Draw(out)
    drw.line([(zcol, 0), (zcol, H - 1)], fill=(0, 255, 0), width=3)
    return out


def hcat(a, b, gap=12, label_a="original", label_b="flipped"):
    w = a.width + b.width + gap
    h = max(a.height, b.height)
    canvas = PILImage.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    drw = ImageDraw.Draw(canvas)
    drw.text((6, 6), label_a, fill=(255, 255, 0), font=FONT)
    drw.text((a.width + gap + 6, 6), label_b, fill=(255, 255, 0), font=FONT)
    return canvas


def to_tensor(pil):
    return F.pil_to_tensor(pil.convert("RGB")).float() / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--ann", default=None, help="COCO json (default <root>/coco/instances_train.json)")
    ap.add_argument("--out", default="output/viz_ortho")
    ap.add_argument("--n-subjects", type=int, default=5)
    ap.add_argument("--slope", type=float, default=10.0)
    args = ap.parse_args()

    root = Path(args.root)
    ann = args.ann or str(root / "coco" / "instances_train.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = CocoDetection(img_folder=str(root), ann_file=ann, transforms=None,
                       return_masks=False, remap_mscoco_category=False)
    names = ds.category2name  # {cat_id: FDI code}
    num_classes = len(ds.categories)

    flip = RandomHorizontalFlipFDI(p=1.0, num_classes=num_classes)   # force flip
    ori = OrientationChannel(enabled=True, slope=args.slope)

    images = ds.coco.dataset["images"]
    fname_by_id = {im["id"]: im["file_name"] for im in images}

    # ---------- Part 1: flip + adapted labels, ~N subjects ----------
    # prefer CENTER views (clear left/right anterior teeth) for legibility
    center_idx = [i for i, _id in enumerate(ds.ids)
                  if view_id_from_filename(fname_by_id[_id]).__class__ and
                  fname_by_id[_id].upper().endswith("CENTER.PNG")]
    pick = (center_idx or list(range(len(ds))))[: args.n_subjects]
    for n, idx in enumerate(pick):
        img, tgt = ds.load_item(idx)
        orig = draw_boxes(img, tgt["boxes"], tgt["labels"], names)
        fimg, ftgt, _ = flip(img.copy(), copy.deepcopy(tgt), ds)
        flp = draw_boxes(fimg, ftgt["boxes"], ftgt["labels"], names)
        subj = Path(fname_by_id[ds.ids[idx]]).parts[-3]  # patient folder
        panel = hcat(orig, flp, label_a="orig labels", label_b="flipped+remapped")
        panel.save(out / f"flip_{n:02d}_{subj}.jpg", quality=92)
        print(f"flip_{n:02d}_{subj}.jpg")

    # ---------- Part 2: orientation overlay, one image per view ----------
    first_by_view = {}
    for i, _id in enumerate(ds.ids):
        v = Path(fname_by_id[_id]).stem.upper()
        if v in VIEW_NAMES and v not in first_by_view:
            first_by_view[v] = i
    for view in VIEW_NAMES:
        if view not in first_by_view:
            print(f"(no image found for view {view})")
            continue
        idx = first_by_view[view]
        img, tgt = ds.load_item(idx)

        o_img = ori(to_tensor(img), copy.deepcopy(tgt), ds)[0]
        orig = overlay_orientation(img, o_img[3])

        fimg, ftgt, _ = flip(img.copy(), copy.deepcopy(tgt), ds)
        f_img = ori(to_tensor(fimg), ftgt, ds)[0]
        flp = overlay_orientation(fimg, f_img[3])

        panel = hcat(orig, flp, label_a=f"{view} orientation", label_b=f"{view} flipped")
        panel.save(out / f"orient_{view}.jpg", quality=92)
        print(f"orient_{view}.jpg")

    print(f"\nwrote panels to {out}/")


if __name__ == "__main__":
    main()
