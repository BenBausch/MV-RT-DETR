"""Lightweight Weights & Biases integration for the RT-DETR solver.

Opt-in via config:

    use_wandb: True
    wandb:
      project: rtdetr-ortho
      name: ~            # defaults to the output_dir folder name
      mode: online       # online | offline | disabled
      tags: [r18, ortho]

Everything here is a no-op unless wandb is installed, `use_wandb` is true, and
we are the main process — so it never interferes with DDP or a plain run.
W&B keeps a single global run, so the rest of the code just calls these
module-level functions without threading an object around.

NOTE: This logs metrics ONLY. No images are ever uploaded — the dataset is
confidential patient data and must not leave the machine.
"""

from . import dist_utils
import wandb
import os
import numpy as np

_RUN = None
_CLASS_NAMES = None  # {category_id: name} for nicer per-class / box labels


def is_active() -> bool:
    return _RUN is not None


def setup(cfg, class_names=None):
    """Initialize a wandb run from the YAML config. Returns the run or None."""
    global _RUN, _CLASS_NAMES
    if _RUN is not None:
        return _RUN

    yaml_cfg = getattr(cfg, "yaml_cfg", {}) or {}
    if not yaml_cfg.get("use_wandb", False):
        return None
    if not dist_utils.is_main_process():
        return None

    wcfg = yaml_cfg.get("wandb", {}) or {}
    name = wcfg.get("name") or (
        os.path.basename(str(cfg.output_dir).rstrip("/")) if cfg.output_dir else None
    )
    _RUN = wandb.init(
        project=wcfg.get("project", "rtdetr"),
        name=name,
        dir=str(cfg.output_dir) if cfg.output_dir else None,
        config=yaml_cfg,
        mode=wcfg.get("mode", "online"),
        tags=wcfg.get("tags", None),
        resume="allow",
    )
    _CLASS_NAMES = class_names

    # Two independent x-axes: per-iteration train curves and per-epoch val curves.
    _RUN.define_metric("train/global_step")
    _RUN.define_metric("train/*", step_metric="train/global_step")
    _RUN.define_metric("epoch")
    _RUN.define_metric("val/*", step_metric="epoch")
    _RUN.define_metric("val_AP_per_class/*", step_metric="epoch")

    print(f"[wandb] logging to run '{_RUN.name}' (project '{_RUN.project}')")
    return _RUN


def log(data: dict):
    """Log a dict. Include an x-axis key ('train/global_step' or 'epoch')."""
    if _RUN is None:
        return
    wandb.log(data)


def log_train(data: dict, global_step: int):
    if _RUN is None:
        return
    payload = {f"train/{k}": v for k, v in data.items()}
    payload["train/global_step"] = global_step
    log(payload)


def log_summary(data: dict):
    if _RUN is None:
        return
    for k, v in data.items():
        _RUN.summary[k] = v


# ---- COCO metric names (order matches COCOeval.stats) ----
_COCO_STATS = [
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def log_coco_stats(stats, epoch: int, prefix: str = "val"):
    """stats: the 12-element list from coco_eval['bbox'].stats."""
    if _RUN is None or stats is None:
        return
    data = {f"{prefix}/{k}": float(v) for k, v in zip(_COCO_STATS, stats)}
    data["epoch"] = epoch
    log(data)


def log_per_class_ap(coco_eval, epoch: int, prefix: str = "val_AP_per_class"):
    """Per-category AP@[.5:.95] from a finished COCOeval (`bbox`)."""
    if _RUN is None or coco_eval is None:
        return
    try:
        precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
        cat_ids = coco_eval.params.catIds
        data = {}
        for k, cid in enumerate(cat_ids):
            p = precision[:, :, k, 0, -1]  # area=all, maxDet=last
            p = p[p > -1]
            ap = float(np.mean(p)) if p.size else float("nan")
            name = (_CLASS_NAMES or {}).get(cid, str(cid))
            data[f"{prefix}/{name}"] = ap
        data["epoch"] = epoch
        log(data)
    except Exception as e:  # never let logging crash training
        print(f"[wandb] per-class AP logging skipped: {e}")


def log_ultralytics(metrics: dict, epoch: int, prefix: str = "val_ul"):
    """Ultralytics-equivalent metrics dict (from UltralyticsDetMetrics.compute())."""
    if _RUN is None or not metrics:
        return
    data = {
        f"{prefix}/mAP50": metrics.get("map50", 0.0),
        f"{prefix}/mAP50-95": metrics.get("map", 0.0),
        f"{prefix}/precision": metrics.get("mp", 0.0),
        f"{prefix}/recall": metrics.get("mr", 0.0),
        f"{prefix}/mean_iou": metrics.get("mean_iou", 0.0),
    }
    for cid, d in (metrics.get("per_class") or {}).items():
        name = (_CLASS_NAMES or {}).get(cid, str(cid))
        data[f"{prefix}_per_class/{name}_mAP50"] = d["ap50"]
        data[f"{prefix}_per_class/{name}_mAP50-95"] = d["ap"]
    data["epoch"] = epoch
    log(data)


def finish():
    global _RUN
    if _RUN is not None:
        wandb.finish()
        _RUN = None
