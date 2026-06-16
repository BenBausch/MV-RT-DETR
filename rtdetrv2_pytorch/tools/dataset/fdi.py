"""Bijective mapping between contiguous class indices and FDI tooth codes.

YOLO expects class IDs in 0..N-1, but FDI codes are sparse two-digit numbers
(quadrant * 10 + position). Ordering: quadrants 1..8, positions ascending.
"""

FDI_CODES: tuple[int, ...] = (
    # Permanent dentition (32 teeth)
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    # Primary / deciduous dentition (20 teeth)
    51,
    52,
    53,
    54,
    55,
    61,
    62,
    63,
    64,
    65,
    71,
    72,
    73,
    74,
    75,
    81,
    82,
    83,
    84,
    85,
)

FDI_TO_CLASS: dict[int, int] = {fdi: idx for idx, fdi in enumerate(FDI_CODES)}
CLASS_TO_FDI: dict[int, int] = {idx: fdi for idx, fdi in enumerate(FDI_CODES)}
NUM_CLASSES: int = len(FDI_CODES)
