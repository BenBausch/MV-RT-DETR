""""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes
from .._misc import SanitizeBoundingBoxes
from ..ortho_views import build_fdi_flip_map, x0_from_filename

from ...core import register


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = (F.get_size if hasattr(F, 'get_size') else F.get_spatial_size)(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
            
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


@register()
class RandomHorizontalFlipFDI(T.Transform):
    """Horizontal flip for the ortho dataset.

    Flips image + boxes, and because a flip is a left<->right mirror it also:
      - remaps FDI tooth labels across the midline (quadrant 1<->2, 3<->4), and
      - records target['hflip'] so the orientation channel is mirrored to match.
    """
    _transformed_types = (PIL.Image.Image, Image, Video, Mask, BoundingBoxes)

    def __init__(self, p: float = 0.5, num_classes: int = 32) -> None:
        super().__init__()
        self.p = p
        self.register_buffer('fdi_flip_map', torch.tensor(build_fdi_flip_map(num_classes)))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return F.horizontal_flip(inpt)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand(1) >= self.p:
            return sample

        flipped = super().forward(*inputs)
        if isinstance(flipped, (tuple, list)) and len(flipped) > 1 and isinstance(flipped[1], dict):
            target = flipped[1]
            if 'labels' in target and target['labels'].numel():
                fmap = self.fdi_flip_map.to(target['labels'].device)
                target['labels'] = fmap[target['labels']]
            target['hflip'] = not bool(target.get('hflip', False))
        return flipped


@register()
class OrientationChannel(T.Transform):
    """Append a horizontal orientation-encoding channel to the image (CoordConv).

    The channel is a sigmoid in image-x: ~-1 at the left edge, ~+1 at the right,
    crossing 0 at a per-view position (CENTER/UP/DOWN -> 0.5; LEFT -> 0.25;
    RIGHT -> 0.75). When the image was horizontally flipped (target['hflip']) the
    channel is mirrored so it stays consistent with the (flipped) image + labels.

    Must run AFTER ConvertPILImage (operates on the image tensor). The view is
    looked up from the dataset's file name. `enabled=False` is a pass-through.
    """
    def __init__(self, enabled: bool = True, slope: float = 10.0) -> None:
        super().__init__()
        self.enabled = enabled
        self.slope = slope

    def __call__(self, *inputs: Any) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not self.enabled:
            return sample
        image, target, dataset = sample

        h, w = image.shape[-2], image.shape[-1]
        img_id = int(target['image_id'].flatten()[0])
        x0 = x0_from_filename(dataset.coco.loadImgs(img_id)[0]['file_name'])

        # On flip, mirror the zero-position (x0 -> 1-x0) so a flipped LEFT view
        # becomes identical to a native RIGHT view (and vice-versa); CENTER/UP/DOWN
        # are symmetric (x0=0.5) so the channel is unchanged but labels still swap.
        # This keeps the encoding consistent across native and flipped samples.
        if target.get('hflip', False):
            x0 = 1.0 - x0

        xs = torch.linspace(0., 1., w)
        row = 2.0 * torch.sigmoid(self.slope * (xs - x0)) - 1.0   # [w]
        chan = row.view(1, 1, w).expand(1, h, w).to(image.dtype)

        image = torch.cat([image.as_subclass(torch.Tensor), chan], dim=0)
        target.pop('hflip', None)
        return image, target, dataset
