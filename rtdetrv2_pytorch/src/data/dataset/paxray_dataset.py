"""PAX-Ray++ (projected chest anatomy) detection dataset for RT-DETR.

Same contract as ``SpineDetection`` / ``OrthoDetection``: ``__init__`` builds an
in-memory COCO index (``self.coco``, ``self.ids``) so transforms, collate,
det_engine and the per-class (``per_tooth_eval``) P/R/F1 metrics all work unchanged.

Why a dedicated class instead of plain ``CocoDetection``
-------------------------------------------------------
The release is already COCO, but it is not usable as-is for this ablation:

1. **159 classes, deliberately overlapping.** ``spine``, ``thoracic spine`` and
   ``vertebrae T5`` are all annotated over the same pixels, and ribs are indexed
   twice -- side-specific (``anterior 5th rib right``) *and* side-agnostic
   (``anterior 5th rib``, a two-blob union of both sides whose box spans the whole
   thorax at ~5% fill). Training on the full set means duplicate, nested and
   near-meaningless boxes. This class selects a disjoint, one-instance-per-class
   subset and remaps it to a contiguous ``0..num_classes-1`` label space.
2. **Both views live in one file.** ``<study>_frontal.png`` and
   ``<study>_lateral.png`` are DRRs of the same CT; ``views`` filters to one.
3. **The label space must be identical across splits**, including classes a split
   happens not to contain (``vertebrae C2`` appears in 0.5% of frontal images), so
   the mapping is pinned by the selected source ids, never by what was observed.

Class groups
------------
``vertebrae``  source ids 4..27  -- C1..C7, T1..T12, L1..L5 (24 classes)
``ribs``       source ids 36..81 -- anterior/posterior 1st..11th + posterior 12th,
                                    left and right (46 classes)
``ribs_agnostic`` source ids 82..104 -- the side-agnostic rib union. NOT in the
                                    default: each is two disconnected ribs, so its
                                    axis-aligned box is not a detection target.

The default (``['vertebrae', 'ribs']``) gives **70 classes**, every one of which
occurs at most once per image -- the precondition the ``per_tooth_eval`` metrics
and the OT assignment head both assume. Presence genuinely varies (T1..T11 in
100% of frontal images, ``vertebrae C2`` in 0.5%), which is what exercises the
assignment head's dustbin.

Layout
------
``<img_folder>/<study>_<view>.png``          e.g. images_patlas/0001a_frontal.png
``split_file``  a COCO json (annotations/{train,val,test}.json) whose ``file_name``
                entries are bare basenames resolved against ``img_folder``.

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
from ...core import register

__all__ = ['PaxrayDetection']

torchvision.disable_beta_transforms_warning()

VIEW_NAMES = ['frontal', 'lateral']

# Source-id ranges in the PAX-Ray++ release (paxray_label_dictionary.json).
CLASS_GROUPS = {
    'vertebrae': list(range(4, 28)),        # C1..C7, T1..T12, L1..L5
    'ribs': list(range(36, 82)),            # side-specific ribs
    'ribs_agnostic': list(range(82, 105)),  # two-blob unions -- see module docstring
}


def view_from_path(file_name: str) -> str:
    """'0001a_frontal.png' -> 'frontal'. Returns '' if no view suffix is present."""
    stem = Path(file_name).stem
    for view in VIEW_NAMES:
        if stem.endswith('_' + view):
            return view
    return ''


def study_from_path(file_name: str) -> str:
    """'0001a_frontal.png' -> '0001a'. Both views of a study share this."""
    stem = Path(file_name).stem
    view = view_from_path(file_name)
    return stem[: -(len(view) + 1)] if view else stem


@register()
class PaxrayDetection(DetDataset):
    __inject__ = ['transforms', ]
    __share__ = ['num_classes']

    def __init__(self, img_folder, split_file, transforms=None, views=('frontal',),
                 class_groups=('vertebrae', 'ribs'), category_ids=None,
                 min_box_size=1.0, drop_empty=True, return_masks=False,
                 num_classes=None):
        """
        Args:
            img_folder: directory holding the PNGs (``images_patlas``).
            split_file: COCO json for this split (``annotations/train.json`` etc.).
                Named ``split_file`` rather than ``ann_file`` so ``DetSolver``'s
                automatic end-of-training test pass can find it; point
                ``per_tooth_eval.test_split_file`` at ``annotations/test.json``.
            transforms: injected Compose (RT-DETR convention, stored as ``_transforms``).
            views: subset of ``['frontal', 'lateral']`` to keep. Default frontal only.
            class_groups: names from ``CLASS_GROUPS`` selecting the source classes.
            category_ids: explicit list of source ids, overriding ``class_groups``.
            min_box_size: drop boxes thinner than this (px) after clipping.
            drop_empty: skip images left with no annotation after filtering.
            return_masks: decode the RLE masks too. The release's boxes are exactly
                the mask bounding boxes, so this is off by default.
            num_classes: shared from the global config; asserted against the number
                of selected classes so a config typo fails loudly instead of
                silently training on a truncated label space.
        """
        self.img_folder = img_folder
        self.split_file = split_file
        self._transforms = transforms
        self.views = [v.lower() for v in views] if views else list(VIEW_NAMES)
        self.class_groups = list(class_groups) if class_groups else []
        self.min_box_size = min_box_size
        self.drop_empty = drop_empty
        self.return_masks = return_masks
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # -- fixed label space, independent of what this split contains ----------
        if category_ids is not None:
            source_ids = sorted(int(c) for c in category_ids)
        else:
            unknown = set(self.class_groups) - set(CLASS_GROUPS)
            if unknown:
                raise ValueError(f'unknown class_groups {sorted(unknown)}; '
                                 f'available: {sorted(CLASS_GROUPS)}')
            source_ids = sorted({i for g in self.class_groups for i in CLASS_GROUPS[g]})
        if not source_ids:
            raise ValueError('no classes selected: pass class_groups or category_ids')
        self.source_ids = source_ids
        self.source2label = {src: i for i, src in enumerate(source_ids)}
        self.num_classes = num_classes if num_classes is not None else len(source_ids)
        if self.num_classes != len(source_ids):
            raise ValueError(
                f'num_classes={self.num_classes} but {len(source_ids)} classes were '
                f'selected from class_groups={self.class_groups}. Fix num_classes in '
                f'the dataset config -- they must agree for the label space to be valid.')

        with open(split_file) as f:
            raw = json.load(f)
        source_names = {c['id']: c['name'] for c in raw['categories']}
        missing = [i for i in source_ids if i not in source_names]
        if missing:
            raise ValueError(f'{split_file} has no categories {missing}')
        self.label2name = {self.source2label[i]: source_names[i] for i in source_ids}

        # -- select images -------------------------------------------------------
        keep_images = {}
        for img in raw['images']:
            base = os.path.basename(img['file_name'])
            if view_from_path(base) not in self.views:
                continue
            keep_images[img['id']] = img

        # -- select + remap annotations -----------------------------------------
        anns_by_image = {}
        n_dropped_small = 0
        for ann in raw['annotations']:
            if ann['image_id'] not in keep_images:
                continue
            label = self.source2label.get(ann['category_id'])
            if label is None:
                continue
            x, y, w, h = ann['bbox']
            if w < self.min_box_size or h < self.min_box_size:
                n_dropped_small += 1
                continue
            anns_by_image.setdefault(ann['image_id'], []).append((label, [x, y, w, h]))
        del raw

        self.img_paths = []
        images, annotations = [], []
        ann_id = 1
        img_id = 0
        n_empty = 0
        for src_id, img in sorted(keep_images.items(), key=lambda kv: kv[1]['file_name']):
            items = anns_by_image.get(src_id, [])
            if not items and self.drop_empty:
                n_empty += 1
                continue
            base = os.path.basename(img['file_name'])
            path = Path(img_folder) / base
            if not path.exists():
                print(f'  WARN missing image, skipping: {path}')
                continue

            img_id += 1
            self.img_paths.append(str(path))
            images.append({'id': img_id, 'file_name': base,
                           'width': img['width'], 'height': img['height']})
            for label, box in items:
                annotations.append({
                    'id': ann_id, 'image_id': img_id, 'category_id': label,
                    'bbox': box, 'area': box[2] * box[3],
                    'iscrowd': 0, 'segmentation': [],
                })
                ann_id += 1

        coco = COCO()
        coco.dataset = {'images': images, 'annotations': annotations,
                        'categories': self._build_categories()}
        coco.createIndex()
        self.coco = coco
        self.ids = [img['id'] for img in images]

        print(f'[PaxrayDetection] {Path(split_file).name}: {len(images)} images '
              f'({"+".join(self.views)}), {len(annotations)} boxes, '
              f'{self.num_classes} classes'
              + (f', {n_empty} empty images dropped' if n_empty else '')
              + (f', {n_dropped_small} sub-{min_box_size}px boxes dropped'
                 if n_dropped_small else ''))

    # -- item loading (mirrors CocoDetection.load_item) ----------------------

    def load_item(self, idx):
        img_id = self.ids[idx]
        image = Image.open(self.img_paths[idx]).convert('RGB')

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
        s += f' views: {self.views}\n class_groups: {self.class_groups}\n'
        s += f' num_classes: {self.num_classes}\n return_masks: {self.return_masks}\n'
        if self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s

    # -- COCO-style category properties (parity with CocoDetection) ----------

    def _build_categories(self):
        return [{'id': label, 'name': self.label2name[label],
                 'supercategory': self.label2name[label]}
                for label in range(self.num_classes)]

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
