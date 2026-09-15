"""BUU-LSPINE (lumbar spine X-ray) detection dataset for RT-DETR.

Same contract as ``CocoDetection`` / ``OrthoDetection``: ``__init__`` builds an
in-memory COCO index (``self.coco``, ``self.ids``) so the rest of the pipeline --
transforms, collate, det_engine, per-tooth (here: per-vertebra) P/R/F1 metrics --
works unchanged. The only difference is the source format: this reads the native
BUU-LSPINE release directly, no conversion step.

Layout
------
``<root>/AP/<stem>0.jpg`` + ``<stem>0.csv``   (anterior-posterior view)
``<root>/LA/<stem>1.jpg`` + ``<stem>1.csv``   (left-lateral view)

``<stem>`` is ``NNNN-S-AAAY`` (patient id, sex, age); the trailing digit is the
view (0 = AP, 1 = LA). Split files list one image path per line, relative to
``root`` (e.g. ``AP/0005-F-025Y0.jpg``).

Annotation format
-----------------
Each CSV row is a labeled *edge line*: ``x1,y1,x2,y2,diagnosis`` in absolute
pixels. Rows run top to bottom in (upper edge, lower edge) pairs, one pair per
lumbar vertebra:

    row 0,1 -> L1    row 2,3 -> L2   ...   row 8,9 -> L5

so AP files have 10 rows. LA files have an 11th, unpaired row: the *upper* edge
of S1. ``diagnosis`` is nonzero only on the lower-edge row of a slipped vertebra
(AP: 1 = left, 2 = right laterolisthesis; LA: 3 = antero-, 4 = retrolisthesis);
it is carried through onto the annotation for downstream use but is *not* the
detection class.

Boxes
-----
A vertebra body is a quad -- 2 points from its upper edge, 2 from its lower edge.
The box is the axis-aligned hull of those 4 points, which makes it invariant to
the (rare) files where a pair is listed lower-edge-first.

S1 has no lower edge, so its box is synthesized: the upper edge extended downward
by ``s1_height_scale`` times the L5 body height measured in the same image, then
clipped to the image. The extent is therefore fabricated (the width and top are
real); set ``include_s1=False`` with ``num_classes: 5`` to drop the class.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
from pathlib import Path

import torch
import torchvision
from faster_coco_eval import COCO
from PIL import Image

from ._dataset import DetDataset
from .coco_dataset import ConvertCocoPolysToMask
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['SpineDetection']

torchvision.disable_beta_transforms_warning()

# class index -> name. Lumbar bodies first (one box per level, top to bottom),
# then the synthesized sacral base.
LUMBAR_NAMES = ['L1', 'L2', 'L3', 'L4', 'L5']
S1_NAME = 'S1'
SPINE_NAMES = LUMBAR_NAMES + [S1_NAME]

VIEW_NAMES = ['AP', 'LA']


def view_from_path(file_name: str) -> str:
    """'AP/0005-F-025Y0.jpg' -> 'AP'. Falls back to the trailing view digit."""
    parts = Path(file_name).parts
    if len(parts) > 1 and parts[-2].upper() in VIEW_NAMES:
        return parts[-2].upper()
    stem = Path(file_name).stem
    return 'LA' if stem.endswith('1') else 'AP'


def patient_id_from_path(file_name: str) -> str:
    """'AP/0005-F-025Y0.jpg' -> '0005'. Both views of a patient share this id."""
    return Path(file_name).stem[:4]


@register()
class SpineDetection(DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['num_classes']

    def __init__(self, img_folder, split_file, transforms=None, views=None,
                 include_s1=True, s1_height_scale=1.0, single_class=False,
                 return_masks=False, num_classes=None):
        """
        Args:
            img_folder: dataset root (the directory holding ``AP/`` and ``LA/``).
                Relative paths in ``split_file`` resolve against it.
            split_file: ``train.txt``/``val.txt``/``test.txt``, one image path per
                line (absolute, or relative to ``img_folder``).
            transforms: injected Compose (RT-DETR convention, stored as ``_transforms``).
            views: optional subset of ``['AP', 'LA']`` to keep. None = both.
            include_s1: emit the synthesized S1 box (LA images only, since AP files
                carry no S1 edge). Requires ``num_classes >= 6``.
            s1_height_scale: S1 box height as a multiple of the L5 body height in
                the same image.
            single_class: collapse every level to one ``vertebra`` class.
            return_masks: kept for parity with CocoDetection; this dataset is bbox-only.
            num_classes: shared from the global config. Pins the category set to
                ``0..num_classes-1`` so all splits agree regardless of which levels
                each split happens to contain.
        """
        self.img_folder = img_folder
        self.split_file = split_file
        self._transforms = transforms
        self.views = [v.upper() for v in views] if views else list(VIEW_NAMES)
        self.include_s1 = include_s1
        self.s1_height_scale = s1_height_scale
        self.single_class = single_class
        self.return_masks = return_masks
        self.num_classes = num_classes
        self.prepare = ConvertCocoPolysToMask(return_masks)

        root = Path(img_folder)

        self.img_paths = []          # absolute paths used to load pixels
        images, annotations = [], []
        ann_id = 1
        img_id = 0
        for line in self._read_split(split_file):
            img_path = Path(line)
            if not img_path.is_absolute():
                img_path = (root / img_path).resolve()
            if view_from_path(img_path) not in self.views:
                continue
            if not img_path.exists():
                print(f"  WARN missing image, skipping: {img_path}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            img_id += 1
            self.img_paths.append(str(img_path))
            images.append({
                "id": img_id,
                "file_name": os.path.relpath(img_path, root),
                "width": img_w,
                "height": img_h,
            })

            for cls, box, diag in self._read_csv(img_path.with_suffix(".csv"), img_w, img_h):
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0 if single_class else cls,
                    "bbox": box,                       # COCO [x, y, w, h] abs px
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "segmentation": [],
                    "diagnosis": diag,                 # 0 = healthy; see module docstring
                })
                ann_id += 1

        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations,
                        "categories": self._build_categories()}
        coco.createIndex()
        self.coco = coco
        self.ids = [img["id"] for img in images]

    # -- item loading (mirrors CocoDetection.load_item) ----------------------

    def load_item(self, idx):
        img_id = self.ids[idx]
        image = Image.open(self.img_paths[idx]).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = {'image_id': img_id, 'annotations': self.coco.loadAnns(ann_ids)}

        image, target = self.prepare(image, target)
        target['idx'] = torch.tensor([idx])

        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'], key='boxes', spatial_size=image.size[::-1])
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def __len__(self):
        return len(self.ids)

    def extra_repr(self) -> str:
        s = f' img_folder: {self.img_folder}\n split_file: {self.split_file}\n'
        s += f' views: {self.views}\n include_s1: {self.include_s1}\n'
        s += f' single_class: {self.single_class}\n return_masks: {self.return_masks}\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s

    # -- COCO-style category properties (parity with CocoDetection) ----------

    @property
    def categories(self):
        return self.coco.dataset['categories']

    @property
    def category2name(self):
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self):
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self):
        return {i: cat['id'] for i, cat in enumerate(self.categories)}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _read_split(split_file):
        with open(split_file) as f:
            return [ln.strip() for ln in f if ln.strip()]

    @staticmethod
    def _read_rows(csv_path: Path):
        """Parse ``x1,y1,x2,y2,diagnosis`` rows. Returns [] if the file is absent."""
        if not csv_path.exists():
            print(f"  WARN missing annotation, no boxes for: {csv_path}")
            return []
        rows = []
        for line in csv_path.read_text().splitlines():
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            x1, y1, x2, y2 = (float(v) for v in parts[:4])
            diag = int(float(parts[4])) if len(parts) > 4 and parts[4].strip() else 0
            rows.append((x1, y1, x2, y2, diag))
        return rows

    def _read_csv(self, csv_path: Path, img_w, img_h):
        """Yield ``(class_idx, [x, y, w, h] abs px, diagnosis)`` per vertebra."""
        rows = self._read_rows(csv_path)
        n_levels = min(len(rows) // 2, len(LUMBAR_NAMES))
        if len(rows) // 2 > len(LUMBAR_NAMES):
            print(f"  WARN {len(rows)} rows in {csv_path}; keeping the first "
                  f"{len(LUMBAR_NAMES)} vertebrae")

        last_pair = None
        for k in range(n_levels):
            upper, lower = rows[2 * k], rows[2 * k + 1]
            last_pair = (upper, lower)
            xs = [upper[0], upper[2], lower[0], lower[2]]
            ys = [upper[1], upper[3], lower[1], lower[3]]
            box = self._to_coco_box(xs, ys, img_w, img_h)
            if box is not None:
                # a disorder is marked on the lower edge; fall back to the upper.
                yield k, box, (lower[4] or upper[4])

        # LA files carry an unpaired trailing row: the upper edge of S1.
        if self.include_s1 and len(rows) % 2 == 1 and last_pair is not None:
            s1 = rows[-1]
            upper, lower = last_pair
            l5_height = abs((lower[1] + lower[3]) / 2.0 - (upper[1] + upper[3]) / 2.0)
            dy = self.s1_height_scale * l5_height
            xs = [s1[0], s1[2]]
            ys = [s1[1], s1[3], s1[1] + dy, s1[3] + dy]
            box = self._to_coco_box(xs, ys, img_w, img_h)
            if box is not None:
                yield len(LUMBAR_NAMES), box, s1[4]

    @staticmethod
    def _to_coco_box(xs, ys, img_w, img_h):
        """Axis-aligned hull of the points, clipped to the image. None if degenerate."""
        x0, x1 = max(min(xs), 0.0), min(max(xs), float(img_w))
        y0, y1 = max(min(ys), 0.0), min(max(ys), float(img_h))
        if x1 <= x0 or y1 <= y0:
            return None
        return [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)]

    def _build_categories(self):
        if self.single_class:
            return [{"id": 0, "name": "vertebra", "supercategory": "vertebra"}]
        n_cls = self.num_classes
        if n_cls is None:
            n_cls = len(SPINE_NAMES) if self.include_s1 else len(LUMBAR_NAMES)
        return [{"id": c,
                 "name": SPINE_NAMES[c] if c < len(SPINE_NAMES) else str(c),
                 "supercategory": "vertebra"}
                for c in range(n_cls)]
