"""Per-tooth (FDI) detection metrics + duplicate-class rate across runs.

COCO mAP is a poor headline for this task: each FDI tooth class occurs at most
once per image, so the metric that matters is "did we localize+label each present
tooth", plus "did we ever predict the same tooth class twice". COCO mAP does not
credit the uniqueness constraint that the OT assignment head enforces, and (with
score_threshold=0.0) penalizes the assignment model for force-emitting absent teeth.

This script evaluates one or more configs, each with its OWN postprocessor (the
intended inference scheme), then reports at a common downstream score threshold:
  - per-FDI and macro/micro precision / recall / F1  (greedy class-aware IoU@0.5)
  - duplicate-class rate: images where some FDI class is predicted >= 2x, and the
    total number of surplus (duplicate) predictions. The assignment head should be 0.

Run from rtdetrv2_pytorch/:
    <python> tools/scripts/per_tooth_metrics.py \
        -c configs/rtdetrv2/rtdetrv2_r50vd_ortho_base.yml \
           configs/rtdetrv2/rtdetrv2_r50vd_ortho_orient.yml \
           configs/rtdetrv2/rtdetrv2_r50vd_ortho_assign.yml \
        --thresholds 0.3 0.5

A checkpoint is auto-detected per config (best.pth in its output_dir) unless -r is given.

Different Aggregation Schemas:
    - micro = pool TP/FP/FN across classes, then compute once
    - macro = per-class metric, unweighted mean
    - weighted = per-class metric, mean weighted by support
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.misc.per_tooth_eval import METRIC_KEY, compute_metrics
from src.solver import TASKS


def find_checkpoint(out_dir):
    for name in ("best.pth", "last.pth"):
        p = Path(out_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no best.pth/last.pth in {out_dir}")


@torch.no_grad()
def collect_predictions(config, resume, split=None):
    """Build the solver for `config`, load the checkpoint, run the eval set, and
    return a list of per-image dicts: {pred_labels, pred_scores, pred_boxes(xyxy,
    orig scale), gt_labels, gt_boxes(xyxy, orig scale)} -- all numpy/cpu.

    `split` optionally overrides the val_dataloader split file (e.g. point at
    test.txt to evaluate the test set)."""
    out_dir = load_config(config, {})["output_dir"]
    ckpt = resume or str(find_checkpoint(out_dir))
    cfg = YAMLConfig(config, resume=ckpt)
    if split is not None:
        cfg.yaml_cfg["val_dataloader"]["dataset"]["split_file"] = split
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()  # loads checkpoint + builds val_dataloader

    module = solver.ema.module if solver.ema else solver.model
    module.eval()
    postprocessor = solver.postprocessor
    device = solver.device
    names = (
        getattr(getattr(solver.val_dataloader, "dataset", None), "category2name", None)
        or {}
    )

    records = []
    for samples, targets in solver.val_dataloader:
        samples = samples.to(device)
        targets = [
            {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
            for t in targets
        ]
        outputs = module(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        ih, iw = samples.shape[-2], samples.shape[-1]
        for t, r in zip(targets, results):
            ow, oh = t["orig_size"].tolist()
            gtb = t["boxes"].float()  # xyxy at model-input resolution
            if gtb.numel():
                s = torch.tensor(
                    [ow / iw, oh / ih, ow / iw, oh / ih], device=gtb.device
                )
                gtb = gtb * s
            records.append(
                dict(
                    pred_labels=r["labels"].cpu().numpy().astype(int),
                    pred_scores=r["scores"].cpu().numpy().astype(float),
                    pred_boxes=r["boxes"].cpu().numpy().astype(float),
                    gt_labels=t["labels"].cpu().numpy().astype(int),
                    gt_boxes=gtb.cpu().numpy().astype(float),
                )
            )
    return records, {int(k): v for k, v in names.items()}


def plot_quadrant_grid(summaries, names_by_run, metric, thr, out_path):
    """4x8 grid of bar charts: row = FDI quadrant (1-4), col = tooth position (1-8),
    bars = configs. Bar height is the chosen per-tooth metric (f1/precision/recall)."""
    key = METRIC_KEY[metric]
    runs = list(summaries.keys())
    short = [r.replace("rtdetrv2_r50vd_ortho_", "") for r in runs]
    any_names = next(iter(names_by_run.values()))
    name2id = {str(v): k for k, v in any_names.items()}
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(4, 8, figsize=(26, 13), sharey=True)
    x = np.arange(len(runs))
    for qi, q in enumerate([1, 2, 3, 4]):
        for pi, pos in enumerate(range(1, 9)):
            ax = axes[qi, pi]
            fdi = q * 10 + pos
            cid = name2id.get(str(fdi))
            vals = [summaries[r]["per_class"].get(cid, {}).get(key, 0.0) for r in runs]
            ax.bar(x, vals, color=colors[: len(runs)], width=0.8)
            for xi, v in zip(x, vals):
                ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
            ax.set_title(f"FDI {fdi}", fontsize=10)
            ax.set_ylim(0, 1.08)
            ax.set_xticks([])
            if pi == 0:
                ax.set_ylabel(f"Quadrant {q}\n{metric}", fontsize=10)

    fig.legend(
        handles=[Patch(facecolor=colors[i], label=short[i]) for i in range(len(runs))],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=len(runs),
        fontsize=12, frameon=False,
    )
    fig.suptitle(f"per-tooth {metric}  (score thr={thr:.2f}, IoU@0.5)", y=0.99, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote plot -> {out_path}")


def main(args):
    iou_thr = args.iou
    csv_rows = []
    agg_rows = []
    runs = {}
    for cfg_path in args.config:
        tag = Path(cfg_path).stem
        print(f"\n>>> evaluating {tag} ({cfg_path})", flush=True)
        records, names = collect_predictions(cfg_path, args.resume, split=args.split)
        runs[tag] = (records, names)

    for thr in args.thresholds:
        print("\n" + "=" * 78)
        print(f"SCORE THRESHOLD = {thr:.2f}   (IoU match = {iou_thr:.2f})")
        print("=" * 78)
        print(
            f"{'run':<34}{'micro-F1':>9}{'macro-F1':>9}{'wgt-F1':>9}"
            f"{'micro-P':>9}{'micro-R':>9}{'dup-imgs':>12}{'dups':>7}"
        )
        summaries = {}
        for tag, (records, names) in runs.items():
            m = compute_metrics(records, thr, iou_thr)
            summaries[tag] = m
            dup_pct = 100.0 * m["imgs_with_dup"] / m["n_imgs"] if m["n_imgs"] else 0.0
            dup_str = f"{m['imgs_with_dup']}/{m['n_imgs']} ({dup_pct:.1f}%)"
            print(
                f"{tag:<34}{m['micro_f1']:>9.3f}{m['macro_f1']:>9.3f}{m['weighted_f1']:>9.3f}"
                f"{m['micro_p']:>9.3f}{m['micro_r']:>9.3f}{dup_str:>12}{m['total_dups']:>7}"
            )
            agg_rows.append(dict(
                run=tag, threshold=thr, iou=iou_thr,
                micro_p=round(m["micro_p"], 4), micro_r=round(m["micro_r"], 4),
                micro_f1=round(m["micro_f1"], 4),
                macro_p=round(m["macro_p"], 4), macro_r=round(m["macro_r"], 4),
                macro_f1=round(m["macro_f1"], 4),
                weighted_p=round(m["weighted_p"], 4), weighted_r=round(m["weighted_r"], 4),
                weighted_f1=round(m["weighted_f1"], 4),
                tp=m["tp"], fp=m["fp"], fn=m["fn"],
                correct=m["correct"], misloc=m["misloc"],
                miscls=m["miscls"], missed=m["missed"],
                imgs_with_dup=m["imgs_with_dup"], n_imgs=m["n_imgs"],
                dup_img_pct=round(dup_pct, 1), total_dups=m["total_dups"],
            ))

        # recall-loss decomposition: where do the missed teeth go?
        sup0 = sum(next(iter(summaries.values()))[k] for k in ("correct", "misloc", "miscls", "missed"))
        print(f"\n  recall-loss decomposition (of {sup0} present teeth)   "
              f"correct=TP; misloc+miscls+missed=FN:")
        print(f"  {'run':<34}{'correct':>14}{'misloc':>14}{'misclass':>14}{'missed':>14}")
        for tag, m in summaries.items():
            sup = m["correct"] + m["misloc"] + m["miscls"] + m["missed"]
            cells = [
                f"{m[k]} ({100 * m[k] / sup:.1f}%)" if sup else f"{m[k]}"
                for k in ("correct", "misloc", "miscls", "missed")
            ]
            print(f"  {tag:<34}" + "".join(f"{c:>14}" for c in cells))

        if args.per_class:
            for tag, m in summaries.items():
                names = runs[tag][1]
                print(f"\n  --- {tag}: per-FDI (thr={thr:.2f}) ---")
                print(
                    f"  {'FDI':>6}{'support':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'dup':>5}"
                    f"{'corr':>6}{'misloc':>7}{'miscls':>7}{'missed':>7}{'P':>7}{'R':>7}{'F1':>7}"
                )
                for c, d in sorted(m["per_class"].items()):
                    fdi = names.get(c, str(c))
                    print(
                        f"  {str(fdi):>6}{d['support']:>8}{d['tp']:>5}{d['fp']:>5}{d['fn']:>5}"
                        f"{d['dup']:>5}{d['correct']:>6}{d['misloc']:>7}{d['miscls']:>7}{d['missed']:>7}"
                        f"{d['p']:>7.2f}{d['r']:>7.2f}{d['f1']:>7.2f}"
                    )

        if args.save_csv:
            for tag, m in summaries.items():
                names = runs[tag][1]
                for c, d in sorted(m["per_class"].items()):
                    csv_rows.append(dict(
                        run=tag, threshold=thr, iou=iou_thr,
                        fdi=names.get(c, str(c)), class_id=c,
                        support=d["support"], tp=d["tp"], fp=d["fp"], fn=d["fn"],
                        dup=d["dup"], correct=d["correct"], misloc=d["misloc"],
                        miscls=d["miscls"], missed=d["missed"], precision=round(d["p"], 4),
                        recall=round(d["r"], 4), f1=round(d["f1"], 4),
                    ))

        if args.plot:
            names_by_run = {tag: runs[tag][1] for tag in summaries}
            stem, ext = os.path.splitext(args.plot)
            out_path = f"{stem}_thr{thr:.2f}{ext or '.png'}"
            plot_quadrant_grid(summaries, names_by_run, args.plot_metric, thr, out_path)

    if args.save_csv and csv_rows:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote per-class metrics ({len(csv_rows)} rows) -> {args.save_csv}")

    if args.save_agg_csv and agg_rows:
        with open(args.save_agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"wrote aggregated metrics ({len(agg_rows)} rows) -> {args.save_agg_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", nargs="+", required=True, help="one or more config yamls"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="explicit checkpoint (only sensible with a single -c); "
        "otherwise best.pth auto-detected per config",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="override the val_dataloader split file (e.g. point at test.txt to "
        "evaluate the test set instead of val)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.3, 0.5],
        help="downstream score thresholds to report at",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for a TP match"
    )
    parser.add_argument(
        "--per-class", action="store_true", help="also print per-FDI table"
    )
    parser.add_argument(
        "--save-csv", type=str, default=None,
        help="write per-class metrics (all runs x thresholds) to this CSV path",
    )
    parser.add_argument(
        "--save-agg-csv", type=str, default=None,
        help="write aggregated run-level metrics (micro/macro/weighted P/R/F1 + dup "
             "stats, one row per run x threshold) to this CSV path",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="write a 4x8 (quadrant x position) per-tooth bar plot to this path "
             "(threshold is appended to the filename, one plot per threshold)",
    )
    parser.add_argument(
        "--plot-metric", choices=list(METRIC_KEY), default="f1",
        help="which per-tooth metric to plot (default: f1)",
    )
    main(parser.parse_args())
