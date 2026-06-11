"""Plot a confusion_matrix.csv (written by the eval) as a heatmap PNG.

Cells are colored by row-normalized rate (each GT row sums to 1, so the diagonal
shows per-class top-1 accuracy) and annotated with raw counts.

  python tools/plot_confusion.py output/<run>/confusion_matrix.csv
  # -> writes output/<run>/confusion_matrix.png
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_matrix(path):
    rows = list(csv.reader(open(path)))
    pred_labels = rows[0][1:]
    gt_labels = [r[0] for r in rows[1:]]
    mat = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    return mat, gt_labels, pred_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to confusion_matrix.csv")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside csv)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    mat, gt_labels, pred_labels = load_matrix(args.csv)
    n = len(gt_labels)
    norm = mat / mat.sum(axis=1, keepdims=True).clip(min=1)

    acc = mat.trace() / mat.sum() if mat.sum() else 0.0
    title = args.title or f"{Path(args.csv).parent.name}  (top-1 class acc {acc:.1%})"

    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(7, n * 0.42)))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(pred_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(gt_labels, fontsize=7)
    ax.set_xlabel("predicted class"); ax.set_ylabel("ground-truth class")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                        fontsize=6, color="white" if norm[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized rate")
    fig.tight_layout()

    out = args.out or str(Path(args.csv).with_suffix(".png"))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
