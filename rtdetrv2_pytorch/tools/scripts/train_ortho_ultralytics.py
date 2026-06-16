"""Train an Ultralytics YOLO detector on the OrthoTracked dataset, set up to be
comparable to the native RT-DETR runs (same splits, same val images, 1024 res).

Use --single-cls for the tooth/no-tooth experiment (collapses all 32 tooth
classes to one), matching configs/rtdetrv2/rtdetrv2_r50vd_ortho_binary*.yml.

CONFIDENTIALITY: the dataset is patient data. Ultralytics' built-in W&B logger
uploads training mosaics and val prediction *images*. We DISABLE it and attach a
metrics-only callback instead, so nothing but scalars leaves the machine.

Example (binary, comparable to RT-DETR v2 binary):
  python tools/train_ortho_ultralytics.py --single-cls --model yolo11l.pt \
    --imgsz 1024 --epochs 120 --batch 16 --device 2 --wandb
"""
import argparse
import os

DATA_DEFAULT = "/mnt/data/abi/datasets/Orthodontics/Lux/Annotated/Seg_only/OrthoTracked_v9/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11l.pt",
                    help="COCO-pretrained model. yolo11l ~= RT-DETR R50 'L' tier; "
                         "use rtdetr-l.pt / rtdetr-x.pt for Ultralytics' RT-DETR.")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--single-cls", action="store_true",
                    help="tooth/no-tooth: collapse all classes to one")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="output/yolo")
    ap.add_argument("--name", default=None)
    ap.add_argument("--wandb", action="store_true", help="metrics-only W&B logging")
    ap.add_argument("--wandb-project", default="rtdetr-ortho")
    args = ap.parse_args()

    # --- Disable Ultralytics' built-in integrations that would upload IMAGES ---
    from ultralytics import settings, YOLO, RTDETR
    settings.update({"wandb": False, "comet": False, "clearml": False,
                     "mlflow": False, "dvc": False})

    name = args.name or (os.path.splitext(os.path.basename(args.model))[0]
                         + ("_ortho_binary" if args.single_cls else "_ortho_pertooth"))

    # RT-DETR models use the RTDETR class; everything else is YOLO.
    ModelClass = RTDETR if "rtdetr" in os.path.basename(args.model).lower() else YOLO
    print(f"loading {args.model} via {ModelClass.__name__}")
    model = ModelClass(args.model)

    # --- metrics-only W&B (no images) ---
    if args.wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name="yolo_" + name,
                         tags=["ultralytics", os.path.splitext(args.model)[0],
                               "tooth-no-tooth" if args.single_cls else "per-tooth",
                               f"res{args.imgsz}"],
                         config=vars(args))

        def _log_metrics(trainer):
            data = {}
            for k, v in (trainer.metrics or {}).items():
                try:
                    data[k] = float(v)
                except (TypeError, ValueError):
                    pass
            if getattr(trainer, "lr", None):
                for k, v in trainer.lr.items():
                    data[f"lr/{k}"] = float(v)
            data["epoch"] = trainer.epoch
            wandb.log(data)

        model.add_callback("on_fit_epoch_end", _log_metrics)
        model.add_callback("on_train_end", lambda trainer: wandb.finish())

    model.train(
        data=args.data,
        single_cls=args.single_cls,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=name,
        exist_ok=True,
        plots=True,        # plots are saved locally only (never uploaded)
        verbose=True,
    )


if __name__ == "__main__":
    main()
