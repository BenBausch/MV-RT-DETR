"""Characterize the BUU-LSPINE annotation format and how it maps onto RT-DETR boxes.

The release ships no format spec: the CSVs hold labeled *edge lines*, not boxes.
This script reproduces every number in docs/spine_box_encoding.md -- the structural
checks that justify `SpineDetection`'s fixed row->level indexing, and the cost of
reducing the annotated quadrilateral to an axis-aligned box.

Sections 1-6 need only stdlib + Pillow. Section 7 traces a single box through the
real pipeline and additionally needs torch + a config.

Usage:
    <python> tools/scripts/analyze_spine_dataset.py --root /mnt/data/abi/datasets/BUU-LSPINE_400
    # with the end-to-end trace:
    <python> tools/scripts/analyze_spine_dataset.py \
        --root /mnt/data/abi/datasets/BUU-LSPINE_400 \
        -c configs/rtdetrv2/rtdetrv2_r50vd_spine_base.yml --trace AP/0005-F-025Y0.jpg

Run from the rtdetrv2_pytorch/ directory.
"""
import argparse
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from PIL import Image

VIEWS = ['AP', 'LA']
N_LUMBAR = 5          # L1..L5, one (upper, lower) edge pair each
DIAGNOSIS = {0: 'none', 1: 'left laterolisthesis', 2: 'right laterolisthesis',
             3: 'anterolisthesis', 4: 'retrolisthesis'}


def head(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def read_rows(csv_path):
    """[(x1, y1, x2, y2, diagnosis), ...] -- one labeled edge line per row."""
    rows = []
    for line in Path(csv_path).read_text().splitlines():
        parts = line.strip().split(',')
        if len(parts) < 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in parts[:4])
        diag = int(float(parts[4])) if len(parts) > 4 and parts[4].strip() else 0
        rows.append((x1, y1, x2, y2, diag))
    return rows


def stems(root, view):
    return sorted((root / view).glob('*.jpg'))


def quantiles(vals, label, fmt='{:.3f}'):
    v = sorted(vals)
    q = lambda f: v[min(int(len(v) * f), len(v) - 1)]
    parts = [('min', v[0]), ('p10', q(0.10)), ('median', statistics.median(v)),
             ('p90', q(0.90)), ('max', v[-1])]
    print(f'  {label:26s} ' + '  '.join(f'{k} {fmt.format(x)}' for k, x in parts))


def hull_area(points):
    """Convex-hull area (monotone chain). Order-independent, unlike the shoelace
    formula on the raw point order -- an edge line drawn right-to-left would
    otherwise fold the quad into a bow-tie and report a near-zero area."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return 0.0

    def half(ps):
        out = []
        for p in ps:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) > 0:
                    break
                out.pop()
            out.append(p)
        return out

    h = half(pts)[:-1] + half(pts[::-1])[:-1]
    n = len(h)
    return abs(sum(h[i][0] * h[(i + 1) % n][1] - h[(i + 1) % n][0] * h[i][1]
                   for i in range(n))) / 2


# -- 1. inventory ------------------------------------------------------------

def section_inventory(root):
    head('1. INVENTORY')
    by_patient = {}
    for view in VIEWS:
        imgs = stems(root, view)
        missing = [p.name for p in imgs if not p.with_suffix('.csv').exists()]
        print(f'  {view}: {len(imgs)} images, {len(missing)} without a .csv')
        for p in imgs:
            by_patient.setdefault(p.stem[:4], set()).add(view)
    both = [p for p, v in by_patient.items() if len(v) == len(VIEWS)]
    print(f'  {len(by_patient)} unique patient ids, {len(both)} with both views')
    print(f'  view digit: AP stems end in "0", LA in "1" -> '
          f'{all(p.stem.endswith("0") for p in stems(root, "AP"))}, '
          f'{all(p.stem.endswith("1") for p in stems(root, "LA"))}')


# -- 2. row structure --------------------------------------------------------

def section_rows(root):
    head('2. ROW STRUCTURE  (is the row -> vertebra-level mapping fixed?)')
    for view in VIEWS:
        counts, fields = Counter(), Counter()
        for img in stems(root, view):
            rows = read_rows(img.with_suffix('.csv'))
            counts[len(rows)] += 1
            for line in img.with_suffix('.csv').read_text().splitlines():
                if line.strip():
                    fields[len(line.split(','))] += 1
        print(f'  {view}: rows/file {dict(counts)}   fields/row {dict(fields)}')
    print(f'\n  => AP = {N_LUMBAR} (upper, lower) pairs. LA = the same plus one '
          f'unpaired trailing row (the S1 upper edge).')


# -- 3. ordering -------------------------------------------------------------

def section_ordering(root):
    head('3. ORDERING  (does row 2k/2k+1 really mean level k, top to bottom?)')
    for view in VIEWS:
        reversed_files, swapped_pairs, opposite_dirs, total = [], [], 0, 0
        for img in stems(root, view):
            rows = read_rows(img.with_suffix('.csv'))
            mid_y = [(r[1] + r[3]) / 2 for r in rows]
            if mid_y[-1] < mid_y[0]:
                reversed_files.append(img.name)
            for k in range(N_LUMBAR):
                upper, lower = rows[2 * k], rows[2 * k + 1]
                total += 1
                if mid_y[2 * k + 1] < mid_y[2 * k]:
                    swapped_pairs.append(f'{img.name} pair {k}')
                if (upper[2] - upper[0]) * (lower[2] - lower[0]) < 0:
                    opposite_dirs += 1
        print(f'  {view}: bottom-to-top files {len(reversed_files)}, '
              f'lower-before-upper pairs {len(swapped_pairs)}/{total}, '
              f'edge pairs drawn in opposite directions {opposite_dirs}/{total}')
        for x in swapped_pairs:
            print(f'        {x}')
    print('\n  => no file is reverse-ordered, so a fixed row -> level mapping is safe.\n'
          '     The min/max hull over all 4 points makes within-pair order irrelevant.')


# -- 4. diagnosis labels -----------------------------------------------------

def section_diagnosis(root):
    head('4. DIAGNOSIS COLUMN  (5th field -- auxiliary, not the detection class)')
    for view in VIEWS:
        by_value, by_row = Counter(), Counter()
        for img in stems(root, view):
            for i, r in enumerate(read_rows(img.with_suffix('.csv'))):
                if r[4]:
                    by_value[r[4]] += 1
                    by_row[i] += 1
        print(f'  {view}: values ' + ', '.join(
            f'{v}={n} ({DIAGNOSIS.get(v, "unknown")})' for v, n in sorted(by_value.items())))
        print(f'      on row indices {dict(sorted(by_row.items()))} '
              f'(odd index = a lower edge)')


# -- 5 & 6. geometry ---------------------------------------------------------

def section_geometry(root, input_size):
    head('5. IMAGE + BOX GEOMETRY')
    for view in VIEWS:
        aspect, rel_w, rel_h = [], [], []
        for img in stems(root, view):
            iw, ih = Image.open(img).size
            aspect.append(ih / iw)
            rows = read_rows(img.with_suffix('.csv'))
            for k in range(N_LUMBAR):
                upper, lower = rows[2 * k], rows[2 * k + 1]
                xs = [upper[0], upper[2], lower[0], lower[2]]
                ys = [upper[1], upper[3], lower[1], lower[3]]
                rel_w.append((max(xs) - min(xs)) / iw)
                rel_h.append((max(ys) - min(ys)) / ih)
        print(f'  {view}:')
        quantiles(aspect, 'image H/W', '{:.2f}')
        quantiles(rel_w, 'box width / image width')
        quantiles(rel_h, 'box height / image height')
        print(f'    -> at {input_size}x{input_size}: median box '
              f'{statistics.median(rel_w) * input_size:.0f} x '
              f'{statistics.median(rel_h) * input_size:.0f} px, smallest '
              f'{min(rel_w) * input_size:.0f} x {min(rel_h) * input_size:.0f} px')

    head('6. QUAD -> AXIS-ALIGNED BOX  (what the reduction costs)')
    for view in VIEWS:
        fill, tilt = [], []
        for img in stems(root, view):
            rows = read_rows(img.with_suffix('.csv'))
            for k in range(N_LUMBAR):
                upper, lower = rows[2 * k], rows[2 * k + 1]
                quad = [(upper[0], upper[1]), (upper[2], upper[3]),
                        (lower[2], lower[3]), (lower[0], lower[1])]
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                box = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if box > 0:
                    fill.append(hull_area(quad) / box)
                tilt.append(abs(math.degrees(
                    math.atan2(upper[3] - upper[1], upper[2] - upper[0]))) % 180)
        print(f'  {view}:')
        quantiles(fill, 'quad area / box area')
        quantiles(tilt, 'upper endplate tilt (deg)', '{:.1f}')


# -- 7. pipeline trace -------------------------------------------------------

def section_trace(config, file_name, split):
    head(f'7. PIPELINE TRACE  ({file_name}, first box)')
    import torch  # noqa: F401  (only needed for this section)
    from src.core import YAMLConfig
    from src.data.dataset.spine_dataset import SpineDetection

    cfg = YAMLConfig(config)
    dataset = (cfg.train_dataloader if split == 'train' else cfg.val_dataloader).dataset
    match = [i for i, j in enumerate(dataset.ids)
             if dataset.coco.loadImgs(j)[0]['file_name'] == file_name]
    if not match:
        print(f'  {file_name} is not in the {split} split of this config')
        return
    idx = match[0]
    img_id = dataset.ids[idx]
    rec = dataset.coco.loadImgs(img_id)[0]
    print(f"  image {rec['file_name']}  {rec['width']}x{rec['height']}  (split={split})")

    rows = SpineDetection._read_rows(Path(dataset.img_paths[idx]).with_suffix('.csv'))
    print('\n  (a) CSV rows 0,1 = the two edge LINES of level 0:')
    for r in rows[:2]:
        print(f'        {r}')

    ann = dataset.coco.loadAnns(dataset.coco.getAnnIds(imgIds=img_id))[0]
    print(f"\n  (b) COCO bbox   [x,y,w,h] abs px : "
          f"{[round(v, 2) for v in ann['bbox']]}   category_id={ann['category_id']}")

    _, target = dataset.load_item(idx)          # pre-transform
    boxes = target['boxes']
    print(f'  (c) ConvertCocoPolysToMask -> xyxy abs px : '
          f'{[round(v, 2) for v in boxes[0].tolist()]}')
    print(f'      wrapper={type(boxes).__name__}  format={boxes.format}  '
          f'canvas_size={boxes.canvas_size}')

    img, target = dataset[idx]                  # post-transform
    print(f'  (d) transforms -> {[round(v, 4) for v in target["boxes"][0].tolist()]}'
          f'   (image tensor {tuple(img.shape)}, wrapper='
          f'{type(target["boxes"]).__name__})')
    print('\n  The train pipeline ends in normalized cxcywh (what the criterion wants);\n'
          '  val/test stop at xyxy pixels and det_engine rescales GT at eval time.')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('/mnt/data/abi/datasets/BUU-LSPINE_400'))
    ap.add_argument('--input-size', type=int, default=1024,
                    help='model input resolution, for the box-size readout')
    ap.add_argument('-c', '--config', help='config to use for the section-7 trace')
    ap.add_argument('--trace', default='AP/0005-F-025Y0.jpg',
                    help='root-relative image whose first box is traced')
    ap.add_argument('--trace-split', default='train', choices=['train', 'val'])
    args = ap.parse_args()

    if not args.root.exists():
        raise SystemExit(f'dataset root not found: {args.root}')

    print(f'BUU-LSPINE analysis -- root {args.root}')
    section_inventory(args.root)
    section_rows(args.root)
    section_ordering(args.root)
    section_diagnosis(args.root)
    section_geometry(args.root, args.input_size)
    if args.config:
        section_trace(args.config, args.trace, args.trace_split)
    else:
        print('\n(pass -c <config> to also run section 7, the pipeline trace)')


if __name__ == '__main__':
    main()
