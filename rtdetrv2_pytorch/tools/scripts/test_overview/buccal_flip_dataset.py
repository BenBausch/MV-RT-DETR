"""OrthoTracked test variant with ONLY the buccal (LEFT/RIGHT) views mirrored.

Mirror-convention probe for the IOP-trained models. IOP-Compass stores its Left/Right
buccal photos un-mirrored, OrthoTracked stores them mirrored; IOP training used
``mirror_buccal=True`` to bring IOP to the OrthoTracked convention. This dataset
does the inverse on OrthoTracked: LEFT/RIGHT images (pixels + GT boxes) are flipped
horizontally, i.e. put in the native IOP convention, exactly like
``IOPCompassDetection``'s mirror (a mirror of the photo, not of the anatomy: FDI
labels are unchanged). CENTER/UP/DOWN are passed through untouched.

Every COCO image record gets ``view`` and ``x0``; for flipped buccal views
``x0 = 1 - VIEW_X0[view]`` (the same rule IOPCompassDetection applies to un-mirrored
buccal views), which ``OrientationChannel`` prefers over the filename lookup, so the
orientation channel stays consistent with the flipped pixels.

If an IOP-trained model scores clearly better on the flipped buccal views than on the
original ones, the mirroring convention used in training is mismatched with
OrthoTracked; if it scores worse, the convention is right.
"""
from pathlib import Path

import torch
from PIL import Image

from src.core import register
from src.data._misc import convert_to_tv_tensor
from src.data.dataset.ortho_dataset import OrthoDetection
from src.data.ortho_views import VIEW_X0

__all__ = ['OrthoBuccalFlipDetection']

BUCCAL = ('LEFT', 'RIGHT')


@register()
class OrthoBuccalFlipDetection(OrthoDetection):
    __inject__ = ['transforms', ]
    __share__ = ['num_classes']

    def __init__(self, img_folder, split_file, transforms=None, data_yaml=None,
                 single_class=False, return_masks=False, num_classes=None,
                 flip_views=BUCCAL):
        super().__init__(img_folder, split_file, transforms=transforms, data_yaml=data_yaml,
                         single_class=single_class, return_masks=return_masks,
                         num_classes=num_classes)
        self.flip_views = tuple(flip_views)
        self.mirrored = []
        for img in self.coco.dataset['images']:
            view = Path(img['file_name']).stem.upper()
            mirror = view in self.flip_views
            x0 = VIEW_X0.get(view, 0.5)
            img['view'] = view
            img['x0'] = 1.0 - x0 if mirror else x0
            img['mirrored'] = mirror
            self.mirrored.append(mirror)
            if mirror:
                w = img['width']
                for ann in self.coco.dataset['annotations']:
                    if ann['image_id'] == img['id']:
                        x, y, bw, bh = ann['bbox']
                        ann['bbox'] = [w - x - bw, y, bw, bh]
        self.coco.createIndex()

    def load_item(self, idx):
        img_id = self.ids[idx]
        image = Image.open(self.img_paths[idx]).convert('RGB')
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

    def extra_repr(self) -> str:
        return super().extra_repr() + f' flip_views: {list(self.flip_views)}\n'
