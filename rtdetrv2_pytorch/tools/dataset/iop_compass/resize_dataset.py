#!/usr/bin/env python
"""Write a downscaled copy of IOP-Compass with every annotation rescaled to match.

The release photos are ~2896x1944 PNGs; decoding one takes ~0.3 s, which makes
training data-loader bound. This writes the same tree (``Patient_<n>/IOP_<View>_<n>
.{png,json}``) with the long side resized to ``--long-side`` (aspect ratio kept), so
the split files and ``IOPCompassDetection`` work unchanged -- only ``img_folder``
changes. Photos already smaller than ``--long-side`` are copied at their own size
(never upscaled; the training Resize handles them as before).

Every geometric field of each tooth is scaled with the per-axis factors
``sx = new_w / w`` and ``sy = new_h / h`` (they differ slightly because the new size
is rounded to whole pixels): ``x_min/x_max/centroid_x`` and all contour x by ``sx``,
``y_*`` by ``sy``, ``pixel_area`` by ``sx * sy``. Coordinates are stored as floats.
The original size and factors are recorded in a top-level ``resize`` block, so
predictions can be mapped back to the release resolution.

    python tools/dataset/iop_compass/resize_dataset.py \
        --src /mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/IOP-Compass \
        --dst /mnt/data/abi/datasets/Orthodontics/Public_Datasets/IOP-Compass/IOP-Compass_1024
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from PIL import Image


def scale_tooth(tooth, sx, sy):
    t = dict(tooth)
    for k in ("x_min", "x_max", "centroid_x"):
        t[k] = round(tooth[k] * sx, 3)
    for k in ("y_min", "y_max", "centroid_y"):
        t[k] = round(tooth[k] * sy, 3)
    t["pixel_area"] = round(tooth["pixel_area"] * sx * sy, 3)
    t["contour"] = [[round(x * sx, 3), round(y * sy, 3)] for x, y in tooth["contour"]]
    t["contours"] = [[[round(x * sx, 3), round(y * sy, 3)] for x, y in c]
                     for c in tooth["contours"]]
    return t


def process(job):
    src_png, dst_png, long_side = job
    with Image.open(src_png) as im:
        w, h = im.size
        s = min(1.0, long_side / max(w, h))   # never upscale smaller release photos
        new_w, new_h = round(w * s), round(h * s)
        # BILINEAR in PIL is antialiased when downscaling, like torchvision's Resize.
        im.convert("RGB").resize((new_w, new_h), Image.BILINEAR).save(dst_png)
    sx, sy = new_w / w, new_h / h

    src_json, dst_json = src_png.with_suffix(".json"), dst_png.with_suffix(".json")
    ann = json.loads(src_json.read_text())
    ann["teeth"] = [scale_tooth(t, sx, sy) for t in ann["teeth"]]
    ann["resize"] = {"orig_width": w, "orig_height": h, "width": new_w, "height": new_h,
                     "scale_x": sx, "scale_y": sy}
    dst_json.write_text(json.dumps(ann))
    return (w, h), (new_w, new_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="release root with Patient_<n> dirs")
    ap.add_argument("--dst", required=True, help="output root (same layout)")
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    jobs = []
    for png in sorted(src.glob("Patient_*/IOP_*.png")):
        out = dst / png.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((png, out, args.long_side))
    print(f"resizing {len(jobs)} images to long side {args.long_side} -> {dst}")

    sizes = {}
    with Pool(args.workers) as pool:
        for i, (orig, new) in enumerate(pool.imap_unordered(process, jobs, chunksize=8), 1):
            sizes[(orig, new)] = sizes.get((orig, new), 0) + 1
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    print(f"done: {len(jobs)} images, {len(sizes)} distinct (orig -> new) sizes; most common:")
    for (orig, new), n in sorted(sizes.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {orig[0]}x{orig[1]} -> {new[0]}x{new[1]}: {n}")


if __name__ == "__main__":
    main()
