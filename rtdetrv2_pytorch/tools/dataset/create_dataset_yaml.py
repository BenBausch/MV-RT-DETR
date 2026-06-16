import argparse
import random
from pathlib import Path

import yaml

from fdi import FDI_CODES

ALLOWED_VIEWS = {"CENTER", "LEFT", "RIGHT", "UP", "DOWN"}
DEFAULT_NAMES = [str(c) for c in FDI_CODES]


def find_subjects(root: Path) -> dict[str, list[Path]]:
    subjects = {}
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sorted(
            p for p in subject_dir.rglob("*.png") if p.stem in ALLOWED_VIEWS
        )
        if images:
            subjects[subject_dir.name] = images
    return subjects


def main():
    ap = argparse.ArgumentParser(
        description="Build a YOLO data.yaml with patient-level train/val/test splits. "
        "Expects --root to contain one subdirectory per subject, each holding that "
        "subject's view images."
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", nargs="+", default=DEFAULT_NAMES)
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1.0")

    root = args.root.resolve()
    subjects = find_subjects(root)
    if not subjects:
        raise SystemExit(f"No subject folders with images found under {root}")

    subject_ids = list(subjects.keys())
    random.Random(args.seed).shuffle(subject_ids)

    n = len(subject_ids)
    n_train = round(n * args.train)
    n_val = round(n * args.val)
    splits = {
        "train": subject_ids[:n_train],
        "val": subject_ids[n_train : n_train + n_val],
        "test": subject_ids[n_train + n_val :],
    }

    list_refs: dict[str, str] = {}
    for split_name, ids in splits.items():
        if not ids:
            continue
        list_path = root / f"{split_name}.txt"
        images = [img for sid in ids for img in subjects[sid]]
        list_path.write_text("\n".join(str(p) for p in images) + "\n")
        list_refs[split_name] = str(list_path)
        print(f"{split_name}: {len(ids)} subjects, {len(images)} images -> {list_path}")

    yaml_path = root / "data.yaml"
    data = {
        "path": str(root),
        **list_refs,
        "names": {i: name for i, name in enumerate(args.names)},
    }
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
