"""Convert the OrthoTracked YOLO-bbox dataset to COCO detection format for RT-DETR.

The dataset stores one image per view (UP/DOWN/CENTER/LEFT/RIGHT.png) under
`<root>/<patient>/<date>/`, each with a sibling YOLO `.txt` label
(`cls cx cy w h`, all normalized). The split files `train.txt`/`val.txt`/`test.txt`
list absolute image paths.

This writes `<out>/instances_<split>.json` with:
  - file_name : path of the image RELATIVE to `--root` (so `img_folder=<root>`)
  - category_id: the original YOLO class index (0-based) -> set num_classes accordingly
  - boxes in COCO [x, y, w, h] absolute pixels

Usage:
  python tools/yolo2coco_ortho.py \
    --root /mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9 \
    --out  /mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9/coco
"""
import argparse
import json
import os
from pathlib import Path

import yaml
from PIL import Image


def load_names(root: Path):
    """Read class index -> name from data.yaml (names: {0: '11', ...})."""
    data_yaml = root / "data.yaml"
    names = {}
    if data_yaml.exists():
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        raw = cfg.get("names", {})
        if isinstance(raw, dict):
            names = {int(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):
            names = {i: str(v) for i, v in enumerate(raw)}
    return names


def read_image_list(split_file: Path):
    with open(split_file) as f:
        return [ln.strip() for ln in f if ln.strip()]


def yolo_to_coco_box(parts, img_w, img_h):
    """`cls cx cy w h` (normalized) -> (cls, [x, y, w, h] abs)."""
    cls = int(float(parts[0]))
    cx, cy, w, h = (float(v) for v in parts[1:5])
    bw, bh = w * img_w, h * img_h
    x = (cx * img_w) - bw / 2.0
    y = (cy * img_h) - bh / 2.0
    return cls, [x, y, bw, bh]


def convert_split(root: Path, split_file: Path, max_class: int):
    images, annotations = [], []
    ann_id = 1
    seen_classes = set()
    skipped = 0

    for img_id, img_path in enumerate(read_image_list(split_file), start=1):
        img_path = Path(img_path)
        if not img_path.is_absolute():
            img_path = (root / img_path).resolve()
        if not img_path.exists():
            print(f"  WARN missing image, skipping: {img_path}")
            skipped += 1
            continue

        with Image.open(img_path) as im:
            img_w, img_h = im.size

        rel = os.path.relpath(img_path, root)
        images.append({
            "id": img_id,
            "file_name": rel,
            "width": img_w,
            "height": img_h,
        })

        label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, box = yolo_to_coco_box(parts, img_w, img_h)
            if box[2] <= 0 or box[3] <= 0:
                continue
            seen_classes.add(cls)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,
                "bbox": [round(v, 2) for v in box],
                "area": round(box[2] * box[3], 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    return images, annotations, seen_classes, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for instances_*.json (default: <root>/coco)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--single-class", action="store_true",
                    help="collapse all teeth to one 'tooth' class (tooth/no-tooth "
                         "detection); writes instances_<split>_tooth.json")
    args = ap.parse_args()

    root = args.root.resolve()
    out = (args.out or (root / "coco")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    names = load_names(root)
    # Categories cover every index that appears in any split, unioned with data.yaml.
    all_seen = set()
    per_split = {}
    for split in args.splits:
        sf = root / f"{split}.txt"
        if not sf.exists():
            print(f"skip {split}: {sf} not found")
            continue
        imgs, anns, seen, skipped = convert_split(root, sf, max_class=None)
        per_split[split] = (imgs, anns)
        all_seen |= seen
        print(f"{split}: {len(imgs)} images, {len(anns)} anns, "
              f"classes {min(seen) if seen else '-'}..{max(seen) if seen else '-'}, "
              f"skipped {skipped} missing")

    if args.single_class:
        categories = [{"id": 0, "name": "tooth", "supercategory": "tooth"}]
        suffix = "_tooth"
        print("\nsingle-class mode -> num_classes to set in config: 1 (category id 0)")
    else:
        max_class = max(all_seen) if all_seen else (max(names) if names else 0)
        categories = [
            {"id": c, "name": names.get(c, str(c)), "supercategory": "tooth"}
            for c in range(max_class + 1)
        ]
        suffix = ""
        print(f"\nnum_classes to set in config: {max_class + 1} "
              f"(category ids 0..{max_class})")

    for split, (imgs, anns) in per_split.items():
        if args.single_class:
            anns = [{**a, "category_id": 0} for a in anns]
        coco = {"images": imgs, "annotations": anns, "categories": categories}
        dst = out / f"instances_{split}{suffix}.json"
        with open(dst, "w") as f:
            json.dump(coco, f)
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
