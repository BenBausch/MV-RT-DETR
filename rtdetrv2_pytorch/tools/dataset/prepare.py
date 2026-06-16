import numpy as np
import os
import PIL.Image as Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fdi import FDI_TO_CLASS

ALLOWED_VIEWS = ["CENTER", "LEFT", "RIGHT", "UP", "DOWN"]
ALLOWED_SPLITS = ["train", "val", "test"]


def mask_to_yolo_bbox(mask):
    """
    Convert a binary mask to YOLO bbox coordinates.

    Args:
       mask: Binary numpy array of shape (H, W)

    Returns:
        bbox_coords: Bounding box in YOLO format [cx, cy, w, h], all normalized to [0, 1].
    """
    img_w, img_h = mask.shape[1], mask.shape[0]
    coords = np.where(mask > 0)
    x_min, y_min = np.min(coords[1]), np.min(coords[0])
    x_max, y_max = np.max(coords[1]), np.max(coords[0])
    w, h = x_max - x_min, y_max - y_min
    cx, cy = x_min + w / 2, y_min + h / 2
    n_cx, n_cy, n_w, n_h = cx / img_w, cy / img_h, w / img_w, h / img_h
    return [n_cx, n_cy, n_w, n_h]


def prepare_image(path):
    img = np.array(Image.open(path))
    # Get unique segmenation instances (excluding bg = 0)
    instances = np.unique(img)
    bbox_list = []
    bbox_txt = path.replace("_fdi.png", ".txt")
    with open(bbox_txt, "w") as f:
        for i in instances:
            if i == 0:
                continue
            fdi = int(i)
            if fdi not in FDI_TO_CLASS:
                raise ValueError(f"Unknown FDI code {fdi} in {path}")
            cls = FDI_TO_CLASS[fdi]
            mask = (img == i).astype(bool)
            bbox = mask_to_yolo_bbox(mask)
            bbox_list.append(bbox)
            f.write(f"{cls} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    return img, bbox_list


def prepare_dataset(path_do_dataset):
    for subject in os.listdir(path_do_dataset):
        sub_dir = os.path.join(path_do_dataset, subject)
        if not os.path.isdir(sub_dir):
            continue
        for session in os.listdir(sub_dir):
            if os.path.isdir(os.path.join(sub_dir, session)):
                session_dir = os.path.join(sub_dir, session)
                for v in os.listdir(session_dir):
                    if v.replace("_fdi.png", "") in ALLOWED_VIEWS:
                        img_path = os.path.join(session_dir, v)
                        img, bbox_list = prepare_image(path=img_path)


if __name__ == "__main__":
    path = "/mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_156_subjects"
    prepare_dataset(path)
