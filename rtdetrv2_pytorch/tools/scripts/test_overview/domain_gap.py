"""IOP-trained vs OrthoTracked-trained (in-domain, from wandb) on the OrthoTracked test split.

Joins output/test_overview/tables/per_class.csv (IOP-trained models, ortho_test, fixed
0.5 threshold) with wandb_refs_per_class.csv (in-domain runs, same test split and
threshold) and writes:

  domain_gap_by_type.csv   macro F1 per tooth type (incisor/canine/premolar/molar/wisdom)
                           and pooled recall-loss decomposition, in-domain vs IOP-trained
  domain_gap_per_class.csv per-FDI F1 of both, and the difference

Run after pull_wandb_refs.py and analyze.py, from rtdetrv2_pytorch/:
  ../.venv/bin/python tools/scripts/test_overview/domain_gap.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import MODELS, write_csv

TABLES = Path('output/test_overview/tables')
TYPE = {'1': 'incisor', '2': 'incisor', '3': 'canine', '4': 'premolar', '5': 'premolar',
        '6': 'molar', '7': 'molar', '8': 'wisdom'}
BUCKETS = ('correct', 'misloc', 'miscls', 'missed')


def main():
    ref = defaultdict(dict)                                   # (model, fdi) -> metrics
    with open(TABLES / 'wandb_refs_per_class.csv') as f:
        for r in csv.DictReader(f):
            if r['group'] == 'in_domain':
                _, fdi, k = r['key'].split('/')
                ref[(r['model'], fdi)][k] = float(r['value'])
    ours = {}
    with open(TABLES / 'per_class.csv') as f:
        for r in csv.DictReader(f):
            if r['split'] == 'ortho_test' and r['thr_kind'] == 'fixed' and r['class_id'] != '-1':
                ours[(r['model'], r['fdi'])] = r

    per_class, by_type = [], []
    for m in MODELS:
        groups = defaultdict(lambda: dict(f1_ref=[], f1_iop=[], **{f'{b}_ref': 0 for b in BUCKETS},
                                          **{f'{b}_iop': 0 for b in BUCKETS}))
        for (mm, fdi), d in sorted(ref.items()):
            if mm != m or (m, fdi) not in ours:
                continue
            o = ours[(m, fdi)]
            sup_ref = sum(d.get(b, 0) for b in BUCKETS)
            if sup_ref == 0 and int(o['support']) == 0:
                continue
            per_class.append(dict(model=m, fdi=fdi, support=int(o['support']), support_ref=int(sup_ref),
                                  f1_in_domain=d['f1'], f1_iop_trained=float(o['f1']),
                                  delta=float(o['f1']) - d['f1']))
            for g in (TYPE[fdi[1]], 'all'):
                G = groups[g]
                G['f1_ref'].append(d['f1'])
                G['f1_iop'].append(float(o['f1']))
                for b in BUCKETS:
                    G[f'{b}_ref'] += d.get(b, 0)
                    G[f'{b}_iop'] += int(o[b])
        for g in ('incisor', 'canine', 'premolar', 'molar', 'wisdom', 'all'):
            G = groups[g]
            n_ref = sum(G[f'{b}_ref'] for b in BUCKETS)
            n_iop = sum(G[f'{b}_iop'] for b in BUCKETS)
            row = dict(model=m, tooth_type=g, n_teeth=int(n_iop), n_classes=len(G['f1_iop']),
                       macro_f1_in_domain=float(np.mean(G['f1_ref'])),
                       macro_f1_iop_trained=float(np.mean(G['f1_iop'])))
            row['delta'] = row['macro_f1_iop_trained'] - row['macro_f1_in_domain']
            for b in BUCKETS:
                row[f'{b}_pct_in_domain'] = 100 * G[f'{b}_ref'] / n_ref if n_ref else 0.0
                row[f'{b}_pct_iop_trained'] = 100 * G[f'{b}_iop'] / n_iop if n_iop else 0.0
            by_type.append(row)
    write_csv(TABLES / 'domain_gap_per_class.csv', per_class)
    write_csv(TABLES / 'domain_gap_by_type.csv', by_type)
    for r in by_type:
        print(f"{r['model']:12s} {r['tooth_type']:9s} n={r['n_teeth']:5d} F1 in-domain {r['macro_f1_in_domain']:.3f} "
              f"IOP-trained {r['macro_f1_iop_trained']:.3f} (d {r['delta']:+.3f}) | miscls% {r['miscls_pct_in_domain']:.1f} "
              f"-> {r['miscls_pct_iop_trained']:.1f} | missed% {r['missed_pct_in_domain']:.1f} -> {r['missed_pct_iop_trained']:.1f}"
              f" | misloc% {r['misloc_pct_in_domain']:.1f} -> {r['misloc_pct_iop_trained']:.1f}")


if __name__ == '__main__':
    main()
