"""Standalone evaluation entry point.

Runs validation/test for a trained checkpoint, equivalent to the old
`train.py --test-only -r <ckpt>` but as a dedicated script. It builds the config,
instantiates the task solver, and calls `solver.val()` (loads the checkpoint,
runs COCO + ultralytics eval on the val_dataloader, writes eval.pth /
confusion matrix to the output dir).

If `-r` is omitted, the checkpoint is auto-detected from the config's output_dir
(best.pth, falling back to last.pth) -- same as tools/scripts/collect_ultralytics_metrics.py.

Run from `rtdetrv2_pytorch/` (env is the uv `.venv`):
    # auto-detect best.pth from the config's output_dir:
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tools/scripts/eval.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_ortho.yml
    # or point at an explicit checkpoint:
    CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python tools/scripts/eval.py \
        -c configs/rtdetrv2/rtdetrv2_r18vd_ortho.yml \
        -r output/rtdetrv2_r18vd_ortho/best.pth

To evaluate on the test split instead of val, point the val_dataloader at the
test split, e.g. add:
    -u val_dataloader.dataset.split_file=<root>/test.txt

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

# repo root (rtdetrv2_pytorch) = two levels up from tools/scripts/, so `src` resolves
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse
from pathlib import Path

from src.misc import dist_utils
from src.core import YAMLConfig, yaml_utils
from src.core.yaml_utils import load_config
from src.solver import TASKS


def find_checkpoint(out_dir: Path) -> Path:
    """Auto-detect a run's checkpoint: best.pth, else last.pth.

    Mirrors tools/scripts/collect_ultralytics_metrics.py:find_checkpoint.
    """
    for name in ("best.pth",):
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no best.pth/last.pth in {out_dir}")


def main(
    args,
) -> None:
    """main"""
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
            ]
            and v is not None
        }
    )

    # no checkpoint given -> auto-detect best.pth/last.pth from the run's output_dir
    if not args.resume:
        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(load_config(args.config, {})["output_dir"])
        )
        ckpt = find_checkpoint(out_dir)
        print(f"no -r given; auto-detected checkpoint: {ckpt}")
        update_dict["resume"] = str(ckpt)

    cfg = YAMLConfig(args.config, **update_dict)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.val()

    dist_utils.cleanup()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # priority 0
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        help="checkpoint to evaluate (e.g. output/<exp>/best.pth). "
        "If omitted, auto-detected from the config output_dir "
        "(best.pth, else last.pth).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        help="device",
    )
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--output-dir", type=str, help="output directoy")

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument(
        "--print-method", type=str, default="builtin", help="print method"
    )
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    main(args)
