# BUU-LSPINE — Annotation Format and Box Encoding

How the BUU-LSPINE lumbar-spine dataset stores its labels, and how `SpineDetection`
turns them into the boxes RT-DETR consumes.

Source:
- [spine_dataset.py](../rtdetrv2_pytorch/src/data/dataset/spine_dataset.py) — the loader
- [analyze_spine_dataset.py](../rtdetrv2_pytorch/tools/scripts/analyze_spine_dataset.py) — reproduces every number below
- [visualize_spine_pairs.py](../rtdetrv2_pytorch/tools/visualization/visualize_spine_pairs.py) — AP/LA side-by-side GT renders

Dataset: `/mnt/data/abi/datasets/BUU-LSPINE_400/` — 400 patients × 2 views (AP, LA) = 800 images.

---

## TL;DR

**The dataset contains no bounding boxes.** It stores labeled *line segments* along
vertebral endplates. A vertebra body is implicitly the **quadrilateral** spanned by its
upper and lower edge lines, and the loader reduces that quad to an axis-aligned box.

There are therefore two distinct translations, and only the first is ours:

| stage | what happens | where |
|---|---|---|
| A | 2 edge lines (4 points) → axis-aligned COCO `[x,y,w,h]` | `SpineDetection._read_csv` / `_to_coco_box` |
| B | COCO box → normalized `cxcywh` tensor | stock repo machinery, unchanged |

The reduction in stage A discards endplate tilt. In the lateral view the box is on
median ~42% larger than the vertebra it encloses (quad fills 0.703 of its box); see
[§4](#4-what-the-reduction-costs).

---

## 1. What the CSV actually encodes

Each image `<view>/<stem>.jpg` has a sibling `<stem>.csv`. The release ships no format
spec; the structure below was established by checking all 800 files (§ 2–3 of the
analysis script).

Every row is **one labeled edge line** — the two endpoints of a segment drawn along a
vertebral endplate, in absolute pixels:

```
x1, y1, x2, y2, diagnosis
1065.694, 352.9687, 1266.615, 324.4597, 0     # row 0 = L1 UPPER endplate
1079.27,  488.7259, 1274.76,  473.7926, 0     # row 1 = L1 LOWER endplate
```

Rows run **top to bottom in (upper, lower) pairs**, one pair per lumbar level:

```
 row 0,1 -> L1     row 2,3 -> L2     row 4,5 -> L3     row 6,7 -> L4     row 8,9 -> L5
 row 10  -> S1 upper edge only   (LA files only)
```

Verified over the full release:

| | rows per file | fields per row |
|---|---|---|
| AP | `10` in 400/400 files | `5` in all 4000 rows |
| LA | `11` in 400/400 files | `5` in all 4400 rows |

So AP = 5 pairs; LA = the same 5 pairs plus one **unpaired trailing row**, the upper
edge of S1. All 400 patients have both views and every image has a CSV.

### The 5th field is a diagnosis, not a class

`diagnosis` marks a spinal slip and is carried onto the annotation dict as `diagnosis`
for downstream use — it is **not** the detection class. It is sparse and, as the release
documents, attached to *lower* edge rows (odd indices):

| view | values | rows carrying them |
|---|---|---|
| AP | `1`=left laterolisthesis (3), `2`=right laterolisthesis (7) | idx 3, 5, 7 |
| LA | `3`=anterolisthesis (57), `4`=retrolisthesis (9), `21`=**undocumented** (7) | idx 1, 3, 5, 7, 9, **10** |

Two anomalies worth knowing: the value `21` appears in 7 LA files and matches no
documented code, and it sits on row index 10 — the S1 edge, not a lumbar lower edge.
Both are left as-is rather than guessed at.

---

## 2. Stage A — edge lines → axis-aligned box

### Why a fixed row → level mapping is safe

The loader indexes levels positionally (`rows[2k], rows[2k+1]` → level `k`). That is only
valid if no file is ordered bottom-to-top. Checked across all 800 files:

| view | bottom-to-top files | lower-listed-before-upper pairs | edge pairs drawn in opposite directions |
|---|---|---|---|
| AP | 0 | 1 / 2000 (`0244-M-025Y0.csv`, L5) | 0 / 2000 |
| LA | 0 | 0 / 2000 | 1 / 2000 |

No file is reverse-ordered, so positional indexing holds. The single lower-before-upper
pair is handled for free by the construction below.

### The construction

Take the **hull of all four endpoints** — not "upper row gives the top, lower row gives
the bottom". That makes the result independent of row order *within* a pair, which is
exactly what absorbs `0244-M-025Y0.csv`, and it stays correct for tilted vertebrae where
one endplate's endpoint is higher than the other's.

```python
# spine_dataset.py — one box per level, from its two edge lines
for k in range(n_levels):
    upper, lower = rows[2 * k], rows[2 * k + 1]
    xs = [upper[0], upper[2], lower[0], lower[2]]
    ys = [upper[1], upper[3], lower[1], lower[3]]
    box = self._to_coco_box(xs, ys, img_w, img_h)
    if box is not None:
        yield k, box, (lower[4] or upper[4])      # class idx, box, diagnosis

@staticmethod
def _to_coco_box(xs, ys, img_w, img_h):
    """Axis-aligned hull of the points, clipped to the image. None if degenerate."""
    x0, x1 = max(min(xs), 0.0), min(max(xs), float(img_w))
    y0, y1 = max(min(ys), 0.0), min(max(ys), float(img_h))
    if x1 <= x0 or y1 <= y0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]
```

Worked example — L1 of `AP/0005-F-025Y0.jpg` (1960×2428):

```
xs = [1065.694, 1266.615, 1079.27, 1274.76]   ->  x ∈ [1065.69, 1274.76]
ys = [ 352.969,  324.460,  488.726,  473.793] ->  y ∈ [ 324.46,  488.73]
COCO bbox = [1065.69, 324.46, 209.07, 164.27]     # x, y, w, h
```

### S1 is synthesized

S1 has only an upper edge, so its lower two points are **fabricated**: the upper edge
copied downward by `s1_height_scale` × the L5 body height measured in that same image,
then clipped. Its width and top are annotated; its height is derived.

```python
s1 = rows[-1]
upper, lower = last_pair                                   # L5's two edges
l5_height = abs((lower[1] + lower[3]) / 2 - (upper[1] + upper[3]) / 2)
dy = self.s1_height_scale * l5_height
xs = [s1[0], s1[2]]
ys = [s1[1], s1[3], s1[1] + dy, s1[3] + dy]
```

Consequences: S1 exists only in LA images (AP files have no S1 row), so 320 of 640
training images lack the class — the assignment head's dustbin covers that. Set
`include_s1: False` with `num_classes: 5` to drop it.

### Output

The boxes go into an in-memory `faster_coco_eval.COCO` index built in `__init__`. That
is deliberate: it is the exact structure `CocoDetection` produces, so `load_item`,
`ConvertCocoPolysToMask`, the evaluator and `get_coco_api_from_dataset` all work
untouched. `SpineDetection` is a drop-in.

---

## 3. Stage B — COCO box → RT-DETR tensor

None of this is spine-specific; it is the stock path every dataset in this repo takes.

```
[1065.69, 324.46, 209.07, 164.27]      COCO xywh, absolute px
  │
  │  ConvertCocoPolysToMask            xywh -> xyxy, clamp to image,
  │    (coco_dataset.py)               drop degenerate boxes
  ▼
[1065.69, 324.46, 1274.76, 488.73]     xyxy, absolute px
  │
  │  convert_to_tv_tensor              wrap as torchvision BoundingBoxes
  │    (_misc.py, called in load_item) format=XYXY, canvas_size=(2428, 1960)
  ▼
  │  Resize(1024, 1024)                tv_tensor => coords rescale WITH the image
  ▼
[ 556.77, 136.84,  666.00, 206.12]     xyxy, px at model input resolution
  │
  │  ConvertBoxes(cxcywh, normalize)   box_convert, then / [W, H, W, H]
  │    (_transforms.py:101-120)
  ▼
[0.5971, 0.1675, 0.1067, 0.0677]       normalized cxcywh  ->  criterion
```

Two load-bearing details:

**The tv_tensor wrapper is what makes boxes follow the image.** Wrapping in
`BoundingBoxes` with a `canvas_size` is what lets `Resize`, `SanitizeBoundingBoxes` and
the augmentations transform boxes and pixels together. Without it the boxes would stay
at original-image scale while the image was resized.

**The final format is dictated by the loss.** `RTDETRCriterionv2.loss_boxes` requires
targets in "(center_x, center_y, w, h), normalized by the image size", matching the
model's `pred_boxes`.

### Train/val asymmetry

`ConvertBoxes` runs **only in the train pipeline**. Val/test targets stop at xyxy pixels
at input resolution, and `det_engine` rescales them by `orig_size / input_size` at eval
time to meet the postprocessor's original-scale predictions.

This is easy to trip over: `ConvertBoxes`'s normalize step (`inpt / tensor`) returns a
plain tensor, **dropping the `BoundingBoxes` wrapper and its `.format`**. Any downstream
code that tries to read the box convention off the tensor will silently mis-decode train
boxes as xyxy. `visualize_spine_pairs.py` therefore reads the convention off the config
transform list instead:

```python
def box_convention(dataset):
    fmt, normalized = 'xyxy', False
    for op in getattr(dataset._transforms, 'transforms', []):
        if isinstance(op, ConvertBoxes):
            fmt = (op.fmt or fmt).lower()
            normalized = op.normalize
    return fmt, normalized
```

---

## 4. What the reduction costs

The CSV encodes an **oriented quad**; RT-DETR consumes an **axis-aligned box**. The tilt
is discarded. To quantify it, compare the convex-hull area of the annotated quad against
the area of its axis-aligned box, over all 4000 vertebrae:

| view | quad area / box area (min / p10 / median / p90) | upper endplate tilt, deg (median / p90 / max) |
|---|---|---|
| AP | 0.468 / 0.741 / **0.883** / 0.950 | 2.6 / 8.2 / 32.9 |
| LA | 0.435 / 0.562 / **0.703** / 0.889 | 10.2 / 23.1 / 178.1\* |

\* the 178° outlier is the one LA file whose two edge lines are drawn in opposite
directions; harmless for a min/max hull.

In the lateral view the box is on median ~30% empty, because lumbar lordosis tilts the
endplates (median ~10°, p90 ~23°). Adjacent boxes therefore overlap visibly around
L4/L5. This is inherent to axis-aligned detection, not a conversion bug, but it caps
achievable IoU and is the first thing to revisit if localization turns out to be the
bottleneck. **The orientation information is preserved on disk** — an oriented-box or
endplate-keypoint head could use it later; it is lost only in what this pipeline consumes.

Use the convex hull, not the shoelace formula on raw point order: if one edge line is
drawn right-to-left the quad folds into a bow-tie and shoelace reports a near-zero area.
That artifact produced a bogus `min 0.020` on a first pass.

```python
def hull_area(points):
    """Convex-hull area (monotone chain). Order-independent."""
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
```

### Resolution is not a constraint

At 1024×1024 input the median box is 172×72 px (AP) / 144×90 px (LA), and the smallest
in the release is 97×37 px (AP) / 86×42 px (LA). Ample for an R50 detector — no need for
a larger input.

Image aspect ratios vary widely though (H/W 0.80–2.39, median 1.30), so the square
1024² resize squashes some images noticeably. A portrait input size is a reasonable
follow-up experiment.

---

## 5. Reproducing this

```bash
cd rtdetrv2_pytorch
../.venv/bin/python tools/scripts/analyze_spine_dataset.py \
    --root /mnt/data/abi/datasets/BUU-LSPINE_400 \
    -c configs/rtdetrv2/rtdetrv2_r50vd_spine_base.yml
```

Sections 1–6 are the structural and geometric checks (stdlib + Pillow only). Section 7
traces one real box through every stage of the live pipeline and needs the config:

```
(a) CSV rows 0,1 = the two edge LINES of level 0:
      (1065.694, 352.9687, 1266.615, 324.4597, 0)
      (1079.27, 488.7259, 1274.76, 473.7926, 0)
(b) COCO bbox   [x,y,w,h] abs px : [1065.69, 324.46, 209.07, 164.27]   category_id=0
(c) ConvertCocoPolysToMask -> xyxy abs px : [1065.69, 324.46, 1274.76, 488.73]
    wrapper=BoundingBoxes  format=BoundingBoxFormat.XYXY  canvas_size=(2428, 1960)
(d) transforms -> [0.5971, 0.1675, 0.1067, 0.0677]   (image tensor (3, 1024, 1024), wrapper=Tensor)
```

For a visual check of the same thing, `visualize_spine_pairs.py --stage raw` draws the
boxes on the original image and `--stage transformed` draws what the dataloader actually
yields, AP and LA side by side per patient.
