"""Collect Ultralytics-equivalent eval metrics for several runs into one YAML.

Give it multiple config ymls; for each it auto-detects the run's best checkpoint
(output_dir/best.pth, else last.pth), evaluates on the chosen split, and gathers
the full UltralyticsDetMetrics output (overall P/R/mAP50/mAP50-95/mean-IoU,
per-class table, and the confusion matrix). Results for all runs are written to
one YAML.

Each config is evaluated in its OWN subprocess: the configs have different
architectures (3- vs 4-channel stem, assignment head) and the repo's config
registry mutates global state, so building several in one process is unsafe.

  python tools/collect_ultralytics_metrics.py \
    configs/rtdetrv2/rtdetrv2_r50vd_ortho_base.yml \
    configs/rtdetrv2/rtdetrv2_r50vd_ortho_flip.yml \
    configs/rtdetrv2/rtdetrv2_r50vd_ortho_view.yml \
    --split test --device cuda:0 --out output/ultralytics_metrics.yml
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

# entry script sits two levels under the repo root (tools/scripts/); add the repo
# root for `src` and tools/visualization for the shared visualizer helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', '..'))
sys.path.insert(0, os.path.join(_HERE, '..', 'visualization'))

import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageDraw

from src.core import YAMLConfig
from src.core.yaml_utils import load_config
from src.solver import DetSolver
from src.solver.det_engine import evaluate
from src.misc.ultralytics_eval import write_confusion
from src.data.ortho_views import x0_from_filename, VIEW_NAMES
# reuse the quadrant color coding + font already used by the flip visualizer
from visualize_ortho_flip import QUAD_COLORS, _quad_color, load_font

MONTAGE_SUBJECTS = 4
MONTAGE_SEED = 0          # fixed -> same subjects across every config
MONTAGE_VIEWS = ["CENTER", "DOWN", "LEFT", "RIGHT", "UP"]
_FONT_BOX = load_font(150)  # FDI box labels (drawn at full image resolution)
_FONT_HDR = load_font(64)   # view / subject headers (drawn on the montage canvas)


def find_checkpoint(out_dir: Path) -> Path:
    for name in ('best.pth', 'last.pth'):
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f'no best.pth/last.pth in {out_dir}')


@torch.no_grad()
def _predict_draw(path, module, postprocessor, tf, in_ch, device, names, conf=0.5):
    """Run the model on one image; draw predicted boxes colored by FDI quadrant."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    data = tf(im)[None].to(device)
    if in_ch == 4:                                   # append orientation channel
        _, _, H, W = data.shape
        x0 = x0_from_filename(str(path))
        rowv = 2.0 * torch.sigmoid(10.0 * (torch.linspace(0, 1, W) - x0)) - 1.0
        data = torch.cat([data, rowv.view(1, 1, 1, W).expand(1, 1, H, W).to(device)], dim=1)

    res = postprocessor(module(data), torch.tensor([[w, h]], device=device))[0]
    keep = res["scores"] >= conf
    drw = ImageDraw.Draw(im)
    fh = _FONT_BOX.size
    for lab, box in zip(res["labels"][keep].tolist(), res["boxes"][keep].tolist()):
        name = names.get(int(lab), str(int(lab)))
        color = _quad_color(name)
        x0b, y0b, x1b, y1b = (float(v) for v in box)
        drw.rectangle([x0b, y0b, x1b, y1b], outline=color, width=10)
        tw = drw.textlength(name, font=_FONT_BOX)          # top-right corner of the box
        drw.text((max(0, x1b - tw), max(0, y0b - fh - 2)), name, fill=color, font=_FONT_BOX)
    return im


def _resize_w(im, cw):
    return im.resize((cw, max(1, round(cw * im.height / im.width))))


def make_montage(module, postprocessor, cfg, ds, names, out_prefix, device, cell_w=1920):
    """One image PER subject: 5 views in a row, predictions color-coded by quadrant.
    Returns the list of written PNG paths. Subjects are the same across configs
    (fixed seed). Cells keep their full aspect ratio, so nothing is cropped.
    cell_w defaults to the native image width (1920) -> full-resolution montage.
    """
    root = Path(ds.img_folder)
    sessions = {}                                    # 'patient/date' -> {VIEW: file_name}
    for im in ds.coco.dataset["images"]:
        fn = im["file_name"]
        session = "/".join(Path(fn).parts[:-1])
        sessions.setdefault(session, {})[Path(fn).stem.upper()] = fn

    keys = sorted(sessions)
    pick = random.Random(MONTAGE_SEED).sample(keys, min(MONTAGE_SUBJECTS, len(keys)))

    sz = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    in_ch = int(cfg.yaml_cfg.get("PResNet", {}).get("in_channels", 3))
    tf = T.Compose([T.Resize((sz[0], sz[1])), T.ToTensor()])

    out_prefix = Path(out_prefix)
    header = _FONT_HDR.size + 8
    out_paths = []
    for i, session in enumerate(pick):
        cells = []
        for view in MONTAGE_VIEWS:
            fn = sessions[session].get(view)
            if fn and (root / fn).exists():
                cell = _predict_draw(root / fn, module, postprocessor, tf, in_ch, device, names)
            else:
                cell = Image.new("RGB", (1920, 1080), (40, 40, 40))
            cells.append(_resize_w(cell, cell_w))

        row_h = max(c.height for c in cells)          # tallest cell -> no cropping
        canvas = Image.new("RGB", (cell_w * len(cells), header + row_h), (255, 255, 255))
        drw = ImageDraw.Draw(canvas)
        drw.text((4, 4), session, fill=(0, 0, 0), font=_FONT_HDR)
        for c, (view, cell) in enumerate(zip(MONTAGE_VIEWS, cells)):
            drw.text((c * cell_w + cell_w // 2 - 36, 4), view, fill=(0, 0, 0), font=_FONT_HDR)
            canvas.paste(cell, (c * cell_w, header))   # top-aligned, full height kept

        dst = f"{out_prefix}_{i + 1}_{session.replace('/', '_')}.png"
        canvas.save(dst, quality=92)
        out_paths.append(dst)
    return out_paths


def eval_one(config: str, split: str, device: str, visualize: bool = False) -> dict:
    """Evaluate a single config's best checkpoint; return a JSON-able result."""
    base = load_config(config, {})                      # fresh dict (avoid mutable default)
    out_dir = Path(base['output_dir'])
    ckpt = find_checkpoint(out_dir)

    # native OrthoDetection loader: select the split by swapping the YOLO split
    # file (val.txt -> {split}.txt), mirroring tools/scripts/eval.py.
    val_split = base['val_dataloader']['dataset']['split_file']
    split_file = val_split.replace('val.txt', f'{split}.txt')
    print(f'[{out_dir.name}] ckpt={ckpt.name} split={split} file={Path(split_file).name}')

    cfg = YAMLConfig(
        config,
        resume=str(ckpt),
        tuning=None,                                    # skip pretrained re-download
        device=device,
        val_dataloader={'dataset': {'split_file': split_file}},
    )
    solver = DetSolver(cfg)
    solver.eval()                                       # builds model+loader, loads ckpt

    module = solver.ema.module if solver.ema else solver.model
    names = getattr(solver.val_dataloader.dataset, 'category2name', None)
    stats, _ = evaluate(module, solver.criterion, solver.postprocessor,
                        solver.val_dataloader, solver.evaluator, solver.device,
                        ul_names=names)
    # ortho runs report per-tooth P/R/F1 (+ recall-loss decomposition and the
    # one-box-per-class duplicate rate); COCO/Ultralytics mAP was dropped for them.
    pt = stats.get('per_tooth', {}) or {}
    nm = names or {}

    per_class = {}
    for cid, d in (pt.get('per_class') or {}).items():
        per_class[str(nm.get(cid, cid))] = {
            'support': int(d['support']),
            'precision': round(float(d['p']), 4),
            'recall': round(float(d['r']), 4),
            'f1': round(float(d['f1']), 4),
            'correct': int(d['correct']), 'misloc': int(d['misloc']),
            'miscls': int(d['miscls']), 'missed': int(d['missed']), 'dup': int(d['dup']),
        }

    confusion = pt.get('confusion')
    nc = len(confusion) if confusion is not None else 0
    support = int(pt.get('tp', 0)) + int(pt.get('fn', 0))

    # write per-run confusion CSV + heatmap PNG into the run's output dir
    if confusion is not None:
        write_confusion(confusion, nm, out_dir)

    result = {
        'config': config,
        'checkpoint': str(ckpt),
        'split': split,
        'images': int(pt.get('n_imgs', 0)),
        'instances': support,
        'overall': {
            'micro_p': round(float(pt.get('micro_p', 0.0)), 4),
            'micro_r': round(float(pt.get('micro_r', 0.0)), 4),
            'micro_f1': round(float(pt.get('micro_f1', 0.0)), 4),
            'macro_f1': round(float(pt.get('macro_f1', 0.0)), 4),
            'weighted_f1': round(float(pt.get('weighted_f1', 0.0)), 4),
        },
        'recall_loss': {
            'support': support,
            'correct': int(pt.get('correct', 0)), 'misloc': int(pt.get('misloc', 0)),
            'miscls': int(pt.get('miscls', 0)), 'missed': int(pt.get('missed', 0)),
        },
        'duplicate_class': {
            'imgs_with_dup': int(pt.get('imgs_with_dup', 0)),
            'n_imgs': int(pt.get('n_imgs', 0)),
            'surplus_boxes': int(pt.get('total_dups', 0)),
        },
        'per_class': per_class,
    }
    if confusion is not None:
        result['confusion_labels'] = [str(nm.get(i, i)) for i in range(nc)]
        result['confusion'] = [[int(v) for v in row] for row in confusion]
        result['confusion_png'] = str(out_dir / 'confusion_matrix.png')

    if visualize:
        pngs = make_montage(module, solver.postprocessor, cfg,
                            solver.val_dataloader.dataset, names or {},
                            str(out_dir / f'montage_{split}'), solver.device)
        result['montage_pngs'] = pngs
        print(f'  montages -> {len(pngs)} files in {out_dir}')
    return result


def run_name(config: str) -> str:
    base = load_config(config, {})
    return Path(base.get('output_dir', config)).name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('configs', nargs='*', help='one or more config ymls')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='output/ultralytics_metrics.yml')
    ap.add_argument('--no-confusion', action='store_true', help='omit confusion matrices')
    ap.add_argument('--visualize-montages', action='store_true',
                    help='render a 4-subject x 5-view prediction montage per run '
                         '(same subjects across all configs)')
    # internal: evaluate exactly one config and dump JSON (subprocess worker)
    ap.add_argument('--_single', help=argparse.SUPPRESS)
    ap.add_argument('--_json', help=argparse.SUPPRESS)
    args = ap.parse_args()

    # ---- worker mode: one config -> json ----
    if args._single:
        res = eval_one(args._single, args.split, args.device,
                       visualize=args.visualize_montages)
        with open(args._json, 'w') as f:
            json.dump(res, f)
        return

    # ---- driver mode: spawn a clean subprocess per config, aggregate ----
    if not args.configs:
        ap.error('provide at least one config yml')
    runs = {}
    for config in args.configs:
        name = run_name(config)
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tf:
            jpath = tf.name
        cmd = [sys.executable, os.path.abspath(__file__), '--_single', config,
               '--_json', jpath, '--split', args.split, '--device', args.device]
        if args.visualize_montages:
            cmd.append('--visualize-montages')
        print(f'\n=== evaluating {name} ({config}) ===')
        ret = subprocess.run(cmd)
        if ret.returncode != 0 or not os.path.getsize(jpath):
            print(f'  !! failed: {name} (skipped)')
            runs[name] = {'config': config, 'error': 'eval failed'}
            continue
        with open(jpath) as f:
            res = json.load(f)
        os.unlink(jpath)
        if args.no_confusion:
            res.pop('confusion', None)
            res.pop('confusion_labels', None)
        runs[name] = res

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        yaml.safe_dump({'split': args.split, 'runs': runs}, f, sort_keys=False, default_flow_style=None)
    print(f'\nwrote {out}')

    # compact comparison table (per-tooth metrics)
    print(f"\n{'run':38}{'microP':>8}{'microR':>8}{'microF1':>9}"
          f"{'macroF1':>9}{'wF1':>8}{'dupImgs':>9}{'surplus':>9}")
    for name, r in runs.items():
        o = r.get('overall')
        if o:
            dc = r.get('duplicate_class', {})
            print(f"{name:38}{o['micro_p']:>8.3f}{o['micro_r']:>8.3f}{o['micro_f1']:>9.3f}"
                  f"{o['macro_f1']:>9.3f}{o['weighted_f1']:>8.3f}"
                  f"{dc.get('imgs_with_dup', 0):>9d}{dc.get('surplus_boxes', 0):>9d}")
        else:
            print(f"{name:38}{'FAILED':>8}")


if __name__ == '__main__':
    main()
