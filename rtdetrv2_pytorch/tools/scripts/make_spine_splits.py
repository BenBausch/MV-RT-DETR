"""Generate patient-level train/val/test splits for the BUU-LSPINE dataset.

The dataset ships no splits. Both views of a patient (``AP/<id>...0.jpg`` and
``LA/<id>...1.jpg``) are highly correlated, so the split is made over the 4-digit
patient id and both views of a patient always land in the same split -- otherwise
val/test leak the training patients' anatomy.

Writes ``train.txt`` / ``val.txt`` / ``test.txt`` into the dataset root, one image
path per line, relative to that root (the format ``SpineDetection`` expects).

Usage:
    <python> tools/scripts/make_spine_splits.py \
        --root /mnt/data/abi/datasets/BUU-LSPINE_400 \
        --ratios 0.8 0.1 0.1 --seed 0
"""

import argparse
import random
from pathlib import Path

VIEW_DIRS = ['AP', 'LA']
SPLIT_NAMES = ['train', 'val', 'test']


def collect(root: Path):
    """patient id -> sorted list of root-relative image paths (AP first, then LA)."""
    by_patient = {}
    for view in VIEW_DIRS:
        for img in sorted((root / view).glob('*.jpg')):
            if not img.with_suffix('.csv').exists():
                print(f'  WARN no annotation for {img.name}, skipping')
                continue
            by_patient.setdefault(img.stem[:4], []).append(
                str(img.relative_to(root)))
    return by_patient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path,
                        help='dataset root holding AP/ and LA/')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VAL', 'TEST'))
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    by_patient = collect(args.root)
    patients = sorted(by_patient)
    if not patients:
        raise SystemExit(f'no annotated images found under {args.root}')

    random.Random(args.seed).shuffle(patients)

    total = sum(args.ratios)
    n_train = round(len(patients) * args.ratios[0] / total)
    n_val = round(len(patients) * args.ratios[1] / total)
    groups = [patients[:n_train],
              patients[n_train:n_train + n_val],
              patients[n_train + n_val:]]

    for name, group in zip(SPLIT_NAMES, groups):
        paths = [p for pid in sorted(group) for p in by_patient[pid]]
        out = args.root / f'{name}.txt'
        out.write_text('\n'.join(paths) + '\n')
        print(f'{name:5s}: {len(group):3d} patients, {len(paths):4d} images -> {out}')


if __name__ == '__main__':
    main()
