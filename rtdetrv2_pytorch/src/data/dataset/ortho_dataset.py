"""OrthoTracked (5-view dental, YOLO-bbox) detection dataset for RT-DETR.

Same contract as ``CocoDetection`` (see ``coco_dataset.py``) so the rest of the
pipeline -- transforms, collate, det_engine, CocoEvaluator, the orientation
channel / FDI flip -- works unchanged. The only difference is the source format:
this reads the native dataset directly (no separate yolo2coco conversion step).

The dataset stores one image per view (UP/DOWN/CENTER/LEFT/RIGHT.png) under
``<root>/<patient>/<date>/`` with a sibling YOLO ``.txt`` label
(``cls cx cy w h``, all normalized). The split files ``train.txt``/``val.txt``/
``test.txt`` list one image path per line.

``__init__`` builds an in-memory COCO index (``self.coco``) so the class is a
drop-in replacement for ``CocoDetection``: it exposes ``coco``/``ids`` and the
same ``category2*`` properties, ``OrientationChannel`` can resolve the view from
``self.coco.loadImgs(...)['file_name']``, and ``get_coco_api_from_dataset`` works.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
from pathlib import Path

import torch
import torchvision
import yaml
from faster_coco_eval import COCO
from PIL import Image

from ._dataset import DetDataset
from .coco_dataset import ConvertCocoPolysToMask
from .._misc import convert_to_tv_tensor
from ...core import register

__all__ = ['OrthoDetection']

torchvision.disable_beta_transforms_warning()


@register()
class OrthoDetection(DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['num_classes']

    def __init__(self, img_folder, split_file, transforms=None, data_yaml=None,
                 single_class=False, return_masks=False, num_classes=None):
        """
        Args:
            img_folder: dataset root. Relative image paths in ``split_file`` and
                the COCO ``file_name`` are resolved against this.
            split_file: ``train.txt``/``val.txt``/``test.txt`` (one image path per
                line; absolute or relative to ``img_folder``).
            transforms: injected Compose (RT-DETR convention, stored as ``_transforms``).
            data_yaml: optional ``data.yaml`` for class names (default ``<root>/data.yaml``).
            single_class: collapse all teeth to one ``tooth`` class (tooth/no-tooth).
            return_masks: kept for parity with CocoDetection; this dataset is bbox-only.
            num_classes: shared from the global config. Pins the category set to
                ``0..num_classes-1`` so train/val splits agree regardless of which
                classes each split happens to contain. If None, it is inferred from
                the classes seen in this split unioned with ``data.yaml`` names.
        """
        self.img_folder = img_folder
        self.split_file = split_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.single_class = single_class
        self.num_classes = num_classes
        self.prepare = ConvertCocoPolysToMask(return_masks)

        root = Path(img_folder)
        names = self._load_names(Path(data_yaml) if data_yaml else root / "data.yaml")

        # Read the split, resolve image paths, parse YOLO labels -> COCO records.
        self.img_paths = []          # absolute paths used to load pixels
        images, annotations = [], []
        seen_classes = set()
        ann_id = 1
        for img_id, line in enumerate(self._read_split(split_file), start=1):
            img_path = Path(line)
            if not img_path.is_absolute():
                img_path = (root / img_path).resolve()
            if not img_path.exists():
                print(f"  WARN missing image, skipping: {img_path}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            self.img_paths.append(str(img_path))
            images.append({
                "id": img_id,
                "file_name": os.path.relpath(img_path, root),
                "width": img_w,
                "height": img_h,
            })

            for cls, box in self._read_yolo(img_path.with_suffix(".txt"), img_w, img_h):
                cat = 0 if single_class else cls
                seen_classes.add(cat)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": box,                       # COCO [x, y, w, h] abs px
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        categories = self._build_categories(names, seen_classes)

        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations,
                        "categories": categories}
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
    def _load_names(data_yaml: Path):
        """class index -> name from data.yaml (names: {0: '11', ...} or a list)."""
        if not data_yaml.exists():
            return {}
        with open(data_yaml) as f:
            raw = (yaml.safe_load(f) or {}).get("names", {})
        if isinstance(raw, dict):
            return {int(k): str(v) for k, v in raw.items()}
        if isinstance(raw, list):
            return {i: str(v) for i, v in enumerate(raw)}
        return {}

    @staticmethod
    def _read_yolo(label_path: Path, img_w, img_h):
        """Yield (cls, [x, y, w, h] abs px) for each valid YOLO line."""
        if not label_path.exists():
            return
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:5])
            bw, bh = w * img_w, h * img_h
            if bw <= 0 or bh <= 0:
                continue
            x = cx * img_w - bw / 2.0
            y = cy * img_h - bh / 2.0
            yield cls, [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)]

    def _build_categories(self, names, seen_classes):
        if self.single_class:
            return [{"id": 0, "name": "tooth", "supercategory": "tooth"}]
        if self.num_classes is not None:
            n_cls = self.num_classes
        elif seen_classes or names:
            n_cls = max(seen_classes | set(names)) + 1
        else:
            n_cls = 1
        return [{"id": c, "name": names.get(c, str(c)), "supercategory": "tooth"}
                for c in range(n_cls)]
