"""View identity + flip handling for the OrthoTracked 5-view dataset.

Used by the orientation-encoding input channel and the FDI-aware horizontal flip.

A horizontal flip is a left<->right mirror, so when it fires:
  - the orientation channel must mirror (handled by flipping the channel), and
  - the FDI tooth labels must swap across the midline: quadrant 1<->2 and 3<->4.
"""
import os
import re

# view id assignment (stable order)
VIEW_NAMES = ["CENTER", "DOWN", "LEFT", "RIGHT", "UP"]
VIEW_TO_ID = {n: i for i, n in enumerate(VIEW_NAMES)}
NUM_VIEWS = len(VIEW_NAMES)

# horizontal position of the orientation encoding's zero crossing, per view, as a
# fraction of image width. CENTER/UP/DOWN are centered (0.5); the buccal LEFT/RIGHT
# views shift the zero toward the left (0.25) / right (0.75).
VIEW_X0 = {"CENTER": 0.5, "UP": 0.5, "DOWN": 0.5, "LEFT": 0.25, "RIGHT": 0.75}
VIEW_X0_BY_ID = [VIEW_X0[n] for n in VIEW_NAMES]


# IOP-Compass names its images IOP_<View>_<n>.png; its view tokens map onto ours
# by anatomy (Left = patient-left buccal, like LEFT). Its buccal photos are stored
# un-mirrored, so the x0 above only holds once IOPCompassDetection has mirrored
# them (mirror_buccal=True, the default); the dataset stores the effective x0 per
# image, which OrientationChannel prefers over the filename lookup.
IOP_VIEW_TO_NAME = {"Center": "CENTER", "Upper": "UP", "Down": "DOWN",
                    "Left": "LEFT", "Right": "RIGHT"}
_IOP_NAME_RE = re.compile(r"^IOP_(Center|Upper|Down|Left|Right)_")


def iop_view_token(file_name: str):
    """'Patient_7/IOP_Left_7.png' -> 'Left'. None if not an IOP-Compass name."""
    m = _IOP_NAME_RE.match(os.path.basename(file_name))
    return m.group(1) if m else None


def view_id_from_filename(file_name: str) -> int:
    """'.../07.04.2016/CENTER.png' or '.../IOP_Center_7.png' -> VIEW_TO_ID['CENTER']. -1 if unknown."""
    token = iop_view_token(file_name)
    if token is not None:
        return VIEW_TO_ID[IOP_VIEW_TO_NAME[token]]
    stem = os.path.splitext(os.path.basename(file_name))[0].upper()
    return VIEW_TO_ID.get(stem, -1)


def x0_from_filename(file_name: str) -> float:
    vid = view_id_from_filename(file_name)
    return VIEW_X0_BY_ID[vid] if 0 <= vid < len(VIEW_X0_BY_ID) else 0.5


# FDI class order used by the ortho COCO converter (class index -> FDI code):
#   0-7:  11..18 (upper right)     8-15: 21..28 (upper left)
#  16-23: 31..38 (lower left)     24-31: 41..48 (lower right)
# A left<->right mirror swaps quadrant 1<->2 and 3<->4, keeping the tooth number.
_FDI_ORDER = [f"{q}{n}" for q in (1, 2, 3, 4) for n in range(1, 9)]
_QUAD_MIRROR = {1: 2, 2: 1, 3: 4, 4: 3}


def build_fdi_flip_map(num_classes: int):
    """Return list m with m[i] = mirrored class index (identity if not 32-class)."""
    if num_classes != len(_FDI_ORDER):
        return list(range(num_classes))
    code_to_idx = {c: i for i, c in enumerate(_FDI_ORDER)}
    return [code_to_idx[f"{_QUAD_MIRROR[int(c[0])]}{c[1]}"] for c in _FDI_ORDER]
