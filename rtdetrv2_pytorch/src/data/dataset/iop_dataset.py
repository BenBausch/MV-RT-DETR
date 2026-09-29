"""IOP-Compass (5-view intraoral photos, FDI tooth instances) detection dataset.

https://ditto.ing.unimore.it/iop-compass/ -- 1,000 patients x 5 views, every
visible permanent tooth annotated with an FDI code (11..48), a bounding box and a
polygon contour. CC BY-NC-SA 4.0.

Same contract as ``OrthoDetection`` (see ``ortho_dataset.py``): ``__init__`` builds
an in-memory COCO index (``self.coco``) so transforms, collate, det_engine, the
per-tooth eval and ``get_coco_api_from_dataset`` work unchanged. Class indices
use the same FDI order as OrthoTracked (0-7: 11..18, ..., 24-31: 41..48), so a
model trained on OrthoTracked can be evaluated here directly.

On-disk layout::

    <root>/Patient_<n>/IOP_<View>_<n>.png     View in {Center, Upper, Down, Left, Right}
    <root>/Patient_<n>/IOP_<View>_<n>.json    {"teeth": [{"FDI_NUM", "x_min", ..., "contours"}]}

Split files (``train.txt``/``val.txt``/``test.txt``, one image path per line,
relative to ``img_folder``) are written by ``tools/dataset/make_iop_splits.py``,
which reproduces the official patient-level 700/200/100 split.

Buccal-view convention: IOP-Compass stores the Left/Right buccal photos as direct
(non-mirrored) shots -- the dental midline sits at x~0.82 in Left and x~0.17 in
Right -- while OrthoTracked stores them mirrored (LEFT midline x~0.19, RIGHT
x~0.80). Center/Upper/Down agree between the two. ``mirror_buccal=True`` flips
the Left/Right images (and their boxes/polygons) at load time so they match the
OrthoTracked convention. This is a mirror of the *photo*, not of the anatomy, so
the FDI labels are unchanged (unlike ``RandomHorizontalFlipFDI``). The resulting
orientation-channel zero crossing is stored per image as ``x0`` in the COCO image
record and read by ``OrientationChannel``.

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import json
import os
from pathlib import Path

import torch
import torchvision
from faster_coco_eval import COCO
from PIL import Image

from ._dataset import DetDataset
from .coco_dataset import ConvertCocoPolysToMask
from .._misc import convert_to_tv_tensor
from ..ortho_views import _FDI_ORDER, IOP_VIEW_TO_NAME, VIEW_X0, iop_view_token
from ...core import register

__all__ = ['IOPCompassDetection']

torchvision.disable_beta_transforms_warning()

_FDI_TO_CLASS = {code: i for i, code in enumerate(_FDI_ORDER)}
_BUCCAL = ("Left", "Right")


@register()
class IOPCompassDetection(DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['num_classes']

    def __init__(self, img_folder, split_file, transforms=None, mirror_buccal=True,
                 views=None, single_class=False, return_masks=False, num_classes=None):
        """
        Args:
            img_folder: dataset root containing the ``Patient_<n>`` directories.
            split_file: ``train.txt``/``val.txt``/``test.txt`` (one image path per
                line; absolute or relative to ``img_folder``).
            transforms: injected Compose (RT-DETR convention, stored as ``_transforms``).
            mirror_buccal: flip the Left/Right buccal views to the OrthoTracked
                (mirrored) convention; see the module docstring.
            views: optional subset of IOP view tokens to keep, e.g. ``['Center']``.
                Default: all five.
            single_class: collapse all teeth to one ``tooth`` class (tooth/no-tooth).
            return_masks: also return per-tooth instance masks from the contours.
            num_classes: shared from the global config. Must be 32 (or None) unless
                ``single_class``; the FDI -> class mapping is fixed.
        """
        self.img_folder = img_folder
        self.split_file = split_file
        self._transforms = transforms
        self.mirror_buccal = mirror_buccal
        self.views = tuple(views) if views else tuple(IOP_VIEW_TO_NAME)
        self.return_masks = return_masks
        self.single_class = single_class
        self.num_classes = num_classes
        self.prepare = ConvertCocoPolysToMask(return_masks)

        unknown = set(self.views) - set(IOP_VIEW_TO_NAME)
        if unknown:
            raise ValueError(f"unknown IOP views {sorted(unknown)}; expected {list(IOP_VIEW_TO_NAME)}")
        if not single_class and num_classes not in (None, len(_FDI_ORDER)):
            raise ValueError(f"IOP-Compass has {len(_FDI_ORDER)} FDI classes, got num_classes={num_classes}")

        root = Path(img_folder)
        self.img_paths = []          # absolute paths used to load pixels
        self.mirrored = []           # per-index: flip pixels horizontally on load
        images, annotations = [], []
        ann_id = 1
        for img_id, line in enumerate(self._read_split(split_file), start=1):
            img_path = Path(line)
            if not img_path.is_absolute():
                img_path = (root / img_path).resolve()
            token = iop_view_token(img_path.name)
            if token is None:
                raise ValueError(f"not an IOP-Compass image name: {img_path}")
            if token not in self.views:
                continue
            ann_path = img_path.with_suffix(".json")
            if not img_path.exists() or not ann_path.exists():
                print(f"  WARN missing image or annotation, skipping: {img_path}")
                continue

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            mirror = mirror_buccal and token in _BUCCAL
            x0 = VIEW_X0[IOP_VIEW_TO_NAME[token]]
            if not mirror and token in _BUCCAL:
                x0 = 1.0 - x0        # native IOP buccal: midline on the other side

            self.img_paths.append(str(img_path))
            self.mirrored.append(mirror)
            images.append({
                "id": img_id,
                "file_name": os.path.relpath(img_path, root),
                "width": img_w,
                "height": img_h,
                "view": IOP_VIEW_TO_NAME[token],
                "x0": x0,
                "mirrored": mirror,
            })

            with open(ann_path) as f:
                teeth = json.load(f)["teeth"]
            for tooth in teeth:
                box, polys = self._tooth_geometry(tooth, img_w, mirror)
                if box[2] <= 0 or box[3] <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0 if single_class else _FDI_TO_CLASS[tooth["FDI_NUM"]],
                    "bbox": box,                       # COCO [x, y, w, h] abs px
                    "area": box[2] * box[3],
                    "iscrowd": 0,
                    "segmentation": polys,
                })
                ann_id += 1

        coco = COCO()
        coco.dataset = {"images": images, "annotations": annotations,
                        "categories": self._build_categories()}
        coco.createIndex()
        self.coco = coco
        self.ids = [img["id"] for img in images]

    # -- item loading (mirrors OrthoDetection.load_item) ---------------------

    def load_item(self, idx):
        img_id = self.ids[idx]
        image = Image.open(self.img_paths[idx]).convert("RGB")
        if self.mirrored[idx]:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

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
        s += f' views: {list(self.views)}\n mirror_buccal: {self.mirror_buccal}\n'
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
    def _tooth_geometry(tooth, img_w, mirror):
        """COCO [x, y, w, h] box and flat polygons, mirrored in x if requested."""
        x_min, x_max = float(tooth["x_min"]), float(tooth["x_max"])
        y_min, y_max = float(tooth["y_min"]), float(tooth["y_max"])
        if mirror:
            x_min, x_max = img_w - x_max, img_w - x_min
        box = [x_min, y_min, x_max - x_min, y_max - y_min]

        polys = []
        for contour in tooth.get("contours") or [tooth.get("contour") or []]:
            if len(contour) < 3:
                continue
            flat = []
            for px, py in contour:
                flat += [float(img_w - px if mirror else px), float(py)]
            polys.append(flat)
        return box, polys

    def _build_categories(self):
        if self.single_class:
            return [{"id": 0, "name": "tooth", "supercategory": "tooth"}]
        return [{"id": c, "name": code, "supercategory": "tooth"}
                for c, code in enumerate(_FDI_ORDER)]
