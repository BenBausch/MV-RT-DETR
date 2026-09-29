#!/usr/bin/env python
"""Create the frozen patient-level splits.

    python scripts/create_splits.py --config configs/dataset_final_1000.yaml
    python scripts/create_splits.py --config ... --seeds 3

With ``--seeds N`` the eligible patients are carved into N rotating hold-out blocks,
so no patient is ever a validation or test patient for more than one seed.  Seed 1
must reproduce the partition already on disk; the script refuses to continue if its
hash disagrees, because every checkpoint and stored prediction depends on it.

Fails loudly when the number of eligible patients does not match the requested
split sizes, so a silently short cohort can never reach an experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# MV-RT-DETR: adapter.py and splits.py were moved next to this script from the
# IOP-Compass repo (src/iop_compass/data/), so import them from here.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapter import discover_patients, load_dataset_config  # noqa: E402
from splits import (  # noqa: E402
    SplitError,
    check_seed_capacity,
    eligibility,
    make_splits,
    split_paths,
    write_splits,
)


def report_eligibility(patients, cfg) -> None:
    eligible, excluded = eligibility(patients, cfg)
    print(f"[create_splits] eligible={len(eligible)}", file=sys.stderr)
    by_reason: dict[str, int] = {}
    for reason in excluded.values():
        by_reason[reason.split(":")[0]] = by_reason.get(reason.split(":")[0], 0) + 1
    for reason, count in sorted(by_reason.items()):
        print(f"[create_splits]   excluded {reason}: {count}", file=sys.stderr)


def reference_hash(out_dir: Path) -> str | None:
    """``split_hash`` of the unsuffixed splits.json, if one was written before."""
    path, _ = split_paths(out_dir, None)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("split_hash")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="number of rotating disjoint hold-out blocks to write (default 1)",
    )
    ap.add_argument(
        "--allow-seed1-drift",
        action="store_true",
        help="write seed 1 even if its hash differs from the existing splits.json "
        "(invalidates every stored checkpoint and prediction)",
    )
    ap.add_argument(
        "--report-eligibility-only",
        action="store_true",
        help="print the eligible/excluded breakdown without writing splits",
    )
    ap.add_argument(
        "--txt-out",
        default=None,
        help="MV-RT-DETR: also write train/val/test.txt (image paths relative to the "
        "dataset root, as read by IOPCompassDetection) to this directory",
    )
    args = ap.parse_args()

    cfg = load_dataset_config(args.config)
    patients = discover_patients(cfg, load_annotations=False)
    out_dir = Path(args.out) if args.out else cfg.derived_root

    eligible, _ = eligibility(patients, cfg)
    try:
        check_seed_capacity(len(eligible), cfg, args.seeds)
    except SplitError as exc:
        print(f"[create_splits] ERROR: {exc}", file=sys.stderr)
        report_eligibility(patients, cfg)
        return 2

    assignments = []
    for seed_index in range(1, args.seeds + 1):
        try:
            assignments.append(make_splits(patients, cfg, seed_index=seed_index))
        except SplitError as exc:
            print(f"[create_splits] ERROR: {exc}", file=sys.stderr)
            report_eligibility(patients, cfg)
            return 2

    if args.report_eligibility_only:
        print(json.dumps(assignments[0].to_dict(), indent=1)[:4000])
        return 0

    expected = reference_hash(out_dir)
    if expected is not None and assignments[0].hash() != expected:
        message = (
            f"[create_splits] seed 1 hash {assignments[0].hash()} disagrees with the "
            f"partition already recorded in {split_paths(out_dir, None)[0]} "
            f"({expected})"
        )
        if not args.allow_seed1_drift:
            print(f"{message}; refusing to write", file=sys.stderr)
            print(
                "[create_splits] every stored checkpoint and prediction was produced "
                "against the recorded partition; pass --allow-seed1-drift only if you "
                "intend to invalidate them",
                file=sys.stderr,
            )
            return 3
        print(f"{message}; continuing because --allow-seed1-drift was given")

    # Disjointness is a property of the whole set, so assert it here rather than
    # trusting the rotation arithmetic.
    for i, a in enumerate(assignments):
        for j, b in enumerate(assignments[i + 1 :], start=i + 1):
            shared = (set(a.val) | set(a.test)) & (set(b.val) | set(b.test))
            if shared:
                print(
                    f"[create_splits] ERROR: seeds {a.seed_index} and {b.seed_index} share "
                    f"{len(shared)} val/test patients: {sorted(shared)[:5]}",
                    file=sys.stderr,
                )
                return 4

    for assignment in assignments:
        json_path, csv_path = write_splits(
            assignment, out_dir, seed_index=assignment.seed_index
        )
        print(
            f"[create_splits] seed_index={assignment.seed_index} "
            f"seed={assignment.seed} train={len(assignment.train)} "
            f"val={len(assignment.val)} test={len(assignment.test)} "
            f"excluded={len(assignment.excluded)}",
            flush=True,
        )
        print(f"[create_splits]   split_hash={assignment.hash()}")
        print(f"[create_splits]   wrote {json_path} and {csv_path}")

    if expected is None or args.allow_seed1_drift:
        # Seed 1 is the canonical splits.json that the audit, the manifest, the
        # aggregation and the freeze step read.  Refreshing it whenever seed 1 changes
        # is the whole point of --allow-seed1-drift: leaving the old file behind would
        # let those four read a partition no seed actually uses, which is worse than
        # the drift the flag was invoked to accept.
        json_path, csv_path = write_splits(assignments[0], out_dir, seed_index=None)
        verb = "wrote" if expected is None else "refreshed"
        print(f"[create_splits] {verb} canonical {json_path} and {csv_path}")

    if args.txt_out:
        write_txt_splits(assignments, patients, cfg, Path(args.txt_out))

    return 0


def write_txt_splits(assignments, patients, cfg, txt_dir: Path) -> None:
    """MV-RT-DETR: one image path per line, all views of each patient, per split.

    Seed index 1 writes train/val/test.txt; other seeds get a _seed<k> suffix.
    """
    by_id = {p.patient_id: p for p in patients}
    txt_dir.mkdir(parents=True, exist_ok=True)
    for assignment in assignments:
        suffix = "" if assignment.seed_index == 1 else f"_seed{assignment.seed_index}"
        for name in ("train", "val", "test"):
            lines = [
                cfg.rel(by_id[pid].images[view].image_path)
                for pid in getattr(assignment, name)
                for view in cfg.views_required
            ]
            path = txt_dir / f"{name}{suffix}.txt"
            path.write_text("\n".join(lines) + "\n")
            print(f"[create_splits]   wrote {path} ({len(lines)} images)")


if __name__ == "__main__":
    raise SystemExit(main())
