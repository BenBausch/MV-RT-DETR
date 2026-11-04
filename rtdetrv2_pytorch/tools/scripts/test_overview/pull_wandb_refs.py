"""Pull (read-only) test summary metrics from wandb for the test-overview report.

Two groups of runs in project luxiai/rtdetr-ortho:
  in-domain  : models trained ON OrthoTracked_301_montages, scored on its test split
               (checkpoints no longer on disk -- wandb is the only record)
  iop        : the 4 IOP-Compass runs evaluated here; their in-training test/micro_f1
               is the sanity-check target for our re-evaluation

Nothing is written to wandb: only wandb.Api() reads. Output:
  output/test_overview/tables/wandb_refs.csv            totals per run
  output/test_overview/tables/wandb_refs_per_class.csv  test_per_class/* per run

Run from rtdetrv2_pytorch/:  ../.venv/bin/python tools/scripts/test_overview/pull_wandb_refs.py
"""
import csv
import sys
from pathlib import Path

import wandb

PROJECT = 'luxiai/rtdetr-ortho'
RUNS = [
    # (group, model key, run id)
    ('in_domain', 'rtdetr_base', '65r6eoqm'),
    ('in_domain', 'rtdetr_ot', '7phpl8in'),
    ('in_domain', 'yolo_base', 'v822skvc'),
    ('in_domain', 'yolo_ot', 'vye9w4qm'),
    ('iop', 'rtdetr_base', 'lpfjtz4i'),
    ('iop', 'rtdetr_ot', 'pbgy3102'),
    ('iop', 'yolo_base', 'kc5s4imy'),
    ('iop', 'yolo_ot', 'g62pnsud'),
]
TOTAL_KEYS = ['test/micro_p', 'test/micro_r', 'test/micro_f1', 'test/macro_f1',
              'test/macro_p', 'test/macro_r', 'test/correct_frac', 'test/dup_img_pct',
              'test/source_epoch', 'best/micro_f1', 'best/epoch']
OUT = Path('output/test_overview/tables')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    rows, per_class = [], []
    for group, model, rid in RUNS:
        run = api.run(f'{PROJECT}/{rid}')
        s = dict(run.summary)
        row = dict(group=group, model=model, run_id=rid, run_name=run.name, state=run.state)
        for k in TOTAL_KEYS:
            row[k] = s.get(k)
        other = sorted(k for k in s if k.startswith('test/') and k not in TOTAL_KEYS)
        for k in other:
            v = s.get(k)
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)
        for k in sorted(s):
            if k.startswith('test_per_class/') and isinstance(s[k], (int, float)):
                per_class.append(dict(group=group, model=model, run_id=rid, key=k, value=s[k]))
        print(f'{group:9s} {model:12s} {rid} {run.name:45s} test/micro_f1={row["test/micro_f1"]}')
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(OUT / 'wandb_refs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / 'wandb_refs_per_class.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['group', 'model', 'run_id', 'key', 'value'])
        w.writeheader()
        w.writerows(per_class)
    print(f'wrote {OUT / "wandb_refs.csv"} ({len(rows)} runs), per-class rows: {len(per_class)}')


if __name__ == '__main__':
    sys.exit(main())
