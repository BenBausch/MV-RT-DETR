"""Upload the base-vs-assign test comparison to wandb as one summary run.

The two training runs already log their own val curves and their end-of-training
test metrics. This adds a third, tiny run holding the side-by-side table produced by
``per_tooth_metrics.py`` -- both models at identical thresholds -- so the ablation can
be read without opening a CSV.

Usage (invoked by run_ablation.sh):
    <python> tools/paxray-plusplus/log_comparison_wandb.py \
        --agg-csv output/paxray_test_aggregate.csv \
        --per-class-csv output/paxray_test_per_class.csv
"""
import argparse
import csv
import os
import sys

import wandb

PROJECT = 'rtdetr-paxray'


def read_csv(path):
    if not path or not os.path.isfile(path):
        print(f'  missing {path}, skipping')
        return [], []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return (list(rows[0]) if rows else []), rows


def to_table(fields, rows):
    table = wandb.Table(columns=fields)
    for r in rows:
        table.add_data(*[r.get(c, '') for c in fields])
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg-csv', required=True)
    ap.add_argument('--per-class-csv', default=None)
    ap.add_argument('--project', default=PROJECT)
    ap.add_argument('--name', default='paxray_frontal_base_vs_assign_test')
    args = ap.parse_args()

    agg_fields, agg_rows = read_csv(args.agg_csv)
    if not agg_rows:
        sys.exit(f'no rows in {args.agg_csv}; nothing to log')
    pc_fields, pc_rows = read_csv(args.per_class_csv)

    run = wandb.init(project=args.project, name=args.name,
                     tags=['paxray', 'frontal', '70cls', 'comparison', 'test'],
                     config={'split': 'test', 'iou': 0.5})
    run.log({'test/aggregate': to_table(agg_fields, agg_rows)})
    if pc_rows:
        run.log({'test/per_class': to_table(pc_fields, pc_rows)})

    # Headline: best micro-F1 per run across the swept thresholds, so the summary
    # does not hinge on the (miscalibrated for Sinkhorn) 0.5 operating point.
    run_key = next((k for k in ('run', 'config', 'name') if k in agg_fields), None)
    f1_key = next((k for k in ('micro_f1', 'f1_micro', 'micro/f1') if k in agg_fields), None)
    thr_key = next((k for k in ('threshold', 'score_thr', 'thr') if k in agg_fields), None)
    if run_key and f1_key:
        best = {}
        for r in agg_rows:
            try:
                f1 = float(r[f1_key])
            except (TypeError, ValueError):
                continue
            name = r[run_key]
            if name not in best or f1 > best[name][0]:
                best[name] = (f1, r.get(thr_key, ''))
        for name, (f1, thr) in best.items():
            tag = os.path.basename(str(name)).replace('.yml', '')
            run.summary[f'best_micro_f1/{tag}'] = f1
            run.summary[f'best_micro_f1_threshold/{tag}'] = thr
            print(f'  {tag}: best micro-F1 {f1:.4f} @ thr {thr}')
    else:
        print(f'  could not identify run/f1 columns in {agg_fields}; '
              f'tables uploaded without a headline summary')
    run.finish()
    print('wandb comparison run uploaded')


if __name__ == '__main__':
    main()
