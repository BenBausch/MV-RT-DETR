"""Carve a patient-disjoint val split out of the shipped PAX-Ray++ train_val file.

The upstream release is kept exactly as published: ``paxray_test.json`` is the test
split, untouched, so results stay comparable to the numbers reported for PAX-Ray++.
The only thing this script fixes is the train/val boundary *inside*
``paxray_train_val.json`` -- the author's own ``prepare_paxraypp_splits.py`` shuffles
**images**, which drops the frontal of a subject in train and its lateral in val for
90.6% of val. We group first, then cut.

The shipped train_val/test boundary already respects subjects: all 14 multi-scan
subjects (0777a..0777f and friends) sit wholly on one side, 4 of them in test -- a
~1-in-400 event under a study-level random split, so the authors clearly grouped by
a real subject id. This script reconstructs that grouping for the train/val cut:

  <NNNN><letter>                   repeat scans of one subject -> key is <NNNN>
  RSNAPE_<studyhash>_<serieshash>  one study per patient -> key is RSNAPE_<studyhash>
  COVID19xx_<n>, RibFrac_<n>       ids unique within the collection -> key is the stem

Names alone are not enough: the same subject also appears under two *different*
collection names (``COVID1910_0038`` and ``COVID1920_038`` are byte-identical). So
after grouping by name we hash every image and union any patients linked by an
identical file (disable with --no-hash-union).

60 md5-identical image groups straddle the shipped train_val/test boundary (the
COVID1910/COVID1920 re-release overlap, ~2.7% of the test split). The test set stays
exactly as published, so the fix is on our side: every train/val subject with an image
byte-identical to a test image is dropped -- the whole subject, including its other
view and aliases (disable with --keep-test-duplicates).

Usage:
    <python> tools/paxray-plusplus/split_train_val.py
    <python> tools/paxray-plusplus/split_train_val.py --val-fraction 0.15 --seed 1
    <python> tools/paxray-plusplus/split_train_val.py --dry-run

Run from the rtdetrv2_pytorch/ directory. Peak RSS is ~12 GB; stdlib only.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict

DEFAULT_ROOT = '/mnt/data/abi/datasets/PAX-RayPlusPlus'
IMAGES_SUBDIR = 'images_patlas'
VIEW_SUFFIXES = ['frontal', 'lateral']

# Default sized so train/val/test land on ~75/10/15 of the whole dataset, given the
# shipped test split is 2,220 of 14,753 images (15.05%).
DEFAULT_VAL_FRACTION = 0.1177

SOURCE_PATTERNS = [
    ('RSNAPE', re.compile(r'^(?P<patient>RSNAPE_[0-9a-f]+)_[0-9a-f]+$')),
    ('COVID19AR', re.compile(r'^(?P<patient>COVID19AR_\d+)$')),
    ('COVID1910', re.compile(r'^(?P<patient>COVID1910_\d+)$')),
    ('COVID1920', re.compile(r'^(?P<patient>COVID1920_\d+)$')),
    ('RibFrac', re.compile(r'^(?P<patient>RibFrac_\d+)$')),
    ('CT', re.compile(r'^(?P<patient>\d+)[a-z]$')),
]


def head(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def study_and_view(file_name):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    for view in VIEW_SUFFIXES:
        if stem.endswith('_' + view):
            return stem[: -(len(view) + 1)], view
    return stem, None


def source_and_patient(study):
    for source, pattern in SOURCE_PATTERNS:
        m = pattern.match(study)
        if m:
            return source, m.group('patient')
    return 'OTHER', study


def group_images(images):
    """patient key -> [image dicts], plus each patient's source collection."""
    head('1. GROUP train_val BY PATIENT')
    patients = defaultdict(list)
    patient_source = {}
    studies = defaultdict(set)
    for img in images:
        study, _ = study_and_view(img['file_name'])
        source, patient = source_and_patient(study)
        patients[patient].append(img)
        patient_source[patient] = source
        studies[patient].add(study)

    print(f'  {len(images)} images -> {len(studies)} studies -> {len(patients)} patients')
    multi = {p: s for p, s in studies.items() if len(s) > 1}
    print(f'  patients with more than one scan: {len(multi)}')
    for p, s in sorted(multi.items())[:5]:
        print(f'    {p}: {sorted(s)}')
    if len(multi) > 5:
        print(f'    ... and {len(multi) - 5} more')
    print(f'\n  {"source":<12}{"patients":>10}{"images":>9}')
    for source, n in Counter(patient_source.values()).most_common():
        n_img = sum(len(v) for p, v in patients.items() if patient_source[p] == source)
        print(f'  {source:<12}{n:>10}{n_img:>9}')
    return dict(patients), patient_source


def union_by_hash(patients, patient_source, images_dir):
    """Union patients that share a byte-identical image, so aliases cannot split."""
    head('2. UNION ALIASED PATIENTS (md5)')
    if not os.path.isdir(images_dir):
        sys.exit(f'ERROR: {images_dir} is not a directory. Pass --images-dir, or '
                 f'--no-hash-union to skip this step.')

    digests, n = {}, 0
    for imgs in patients.values():
        for img in imgs:
            base = os.path.basename(img['file_name'])
            with open(os.path.join(images_dir, base), 'rb') as f:
                digests[base] = hashlib.md5(f.read()).hexdigest()
            n += 1
            if n % 5000 == 0:
                print(f'    hashed {n} images ...', flush=True)
    by_hash = defaultdict(list)
    for base, digest in digests.items():
        by_hash[digest].append(base)

    patient_of = {os.path.basename(img['file_name']): p
                  for p, imgs in patients.items() for img in imgs}
    cross = [sorted(fs) for fs in by_hash.values()
             if len(fs) > 1 and len({patient_of[f] for f in fs}) > 1]
    print(f'  distinct images: {len(by_hash)} of {n}')
    print(f'  duplicate groups linking different patients: {len(cross)}')
    for fs in cross[:5]:
        print(f'    {sorted({patient_of[f] for f in fs})}')
    if len(cross) > 5:
        print(f'    ... and {len(cross) - 5} more')

    parent = {p: p for p in patients}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for fs in cross:
        root = find(patient_of[fs[0]])
        for f in fs[1:]:
            other = find(patient_of[f])
            if other != root:
                lo, hi = sorted((root, other))
                parent[hi] = lo
                root = lo

    merged, merged_source = defaultdict(list), {}
    for p, imgs in patients.items():
        root = find(p)
        merged[root].extend(imgs)
        merged_source.setdefault(root, patient_source[root])
    print(f'  {len(patients)} -> {len(merged)} patient groups '
          f'({len(patients) - len(merged)} absorbed)')
    return dict(merged), merged_source, digests


def split_patients(patients, patient_source, val_fraction, seed, stratify):
    head('3. SPLIT train_val INTO train / val')
    buckets = defaultdict(list)
    for p in patients:
        buckets[patient_source[p] if stratify else '_all'].append(p)

    train, val = [], []
    rng = random.Random(seed)
    for source in sorted(buckets):
        group = sorted(buckets[source])   # sort first so only the seed decides order
        rng.shuffle(group)
        n_val = int(round(val_fraction * len(group)))
        val.extend(group[:n_val])
        train.extend(group[n_val:])

    print(f'  val_fraction {val_fraction} of train_val   seed {seed}   '
          f'stratified {"yes" if stratify else "no"}')
    n_tr = sum(len(patients[p]) for p in train)
    n_va = sum(len(patients[p]) for p in val)
    print(f'\n  {"split":<8}{"patients":>10}{"images":>9}')
    print(f'  {"train":<8}{len(train):>10}{n_tr:>9}')
    print(f'  {"val":<8}{len(val):>10}{n_va:>9}')
    if stratify:
        print(f'\n  {"source":<12}{"train":>8}{"val":>8}')
        for source in sorted(set(patient_source.values())):
            print(f'  {source:<12}'
                  f'{sum(1 for p in train if patient_source[p] == source):>8}'
                  f'{sum(1 for p in val if patient_source[p] == source):>8}')
    return {'train': train, 'val': val}


def hash_test_images(test_path, images_dir):
    """md5 of every image in the shipped test split."""
    with open(test_path) as f:
        test = json.load(f)
    digests = {}
    for img in test['images']:
        base = os.path.basename(img['file_name'])
        with open(os.path.join(images_dir, base), 'rb') as f:
            digests[base] = hashlib.md5(f.read()).hexdigest()
    return digests


def drop_test_duplicates(patients, assigned, digests, test_digests):
    """Remove every train/val subject that has a byte-identical image in the test split.

    The whole subject group goes, not just the matching files: a subject whose frontal
    is duplicated in test is the same person in its lateral view and in any alias, so
    keeping the rest would still leak. Runs after the split, so it only shrinks the
    affected split and leaves the other one exactly as it was.
    """
    head('3b. DROP SUBJECTS DUPLICATED IN THE SHIPPED TEST SET')
    test_set = set(test_digests.values())
    kept, dropped = {}, {}
    for name, group in assigned.items():
        bad = [p for p in group
               if any(digests[os.path.basename(img['file_name'])] in test_set
                      for img in patients[p])]
        bad_set = set(bad)
        kept[name] = [p for p in group if p not in bad_set]
        dropped[name] = sorted(bad)
        n_img = sum(len(patients[p]) for p in bad)
        n_dup = sum(1 for p in bad for img in patients[p]
                    if digests[os.path.basename(img['file_name'])] in test_set)
        print(f'  {name:<6} dropped {len(bad):4d} subjects / {n_img:4d} images '
              f'({n_dup} byte-identical to test, {n_img - n_dup} other views/aliases '
              f'of the same subjects)')
        for p in bad[:5]:
            print(f'    {p}: {sorted(os.path.basename(i["file_name"]) for i in patients[p])}')
        if len(bad) > 5:
            print(f'    ... and {len(bad) - 5} more')
    return kept, dropped


def dropped_file_rows(patients, dropped, digests, test_digests):
    """One row per dropped image: which split and subject it left, and the test
    image(s) it is byte-identical to (empty for a sibling view/alias dropped with it)."""
    test_by_hash = defaultdict(list)
    for base, digest in test_digests.items():
        test_by_hash[digest].append(base)
    rows = []
    for split in ('train', 'val'):
        for subject in dropped.get(split, []):
            for img in sorted(patients[subject], key=lambda im: im['file_name']):
                base = os.path.basename(img['file_name'])
                rows.append({'file_name': base, 'split': split, 'subject': subject,
                             'md5': digests[base],
                             'identical_test_images': ';'.join(
                                 sorted(test_by_hash.get(digests[base], [])))})
    return rows


def build_cocos(train_val, patients, assigned):
    anns_by_image = defaultdict(list)
    for ann in train_val['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    info = train_val.get('info')
    out = {}
    for name, group in assigned.items():
        images = [img for p in group for img in patients[p]]
        images.sort(key=lambda im: im['file_name'])
        meta = dict(info) if isinstance(info, dict) else {'description': info or ''}
        meta['split'] = name
        out[name] = {'info': meta, 'licenses': train_val.get('licenses', []),
                     'categories': train_val['categories'], 'images': images,
                     'annotations': [a for img in images for a in anns_by_image[img['id']]]}
    return out


def verify(splits, patients, digests, test_path, test_digests, n_dropped_images,
           require_test_disjoint):
    head('4. VERIFY')
    group_of = {os.path.basename(img['file_name']): p
                for p, imgs in patients.items() for img in imgs}
    per_split = {}
    for name, coco in splits.items():
        keys = {'patients': set(), 'studies': set(), 'files': set(), 'pixels': set()}
        for img in coco['images']:
            base = os.path.basename(img['file_name'])
            keys['patients'].add(group_of[base])
            keys['studies'].add(study_and_view(base)[0])
            keys['files'].add(base)
            if digests:
                keys['pixels'].add(digests[base])
        per_split[name] = keys

    shared = {k: per_split['train'][k] & per_split['val'][k]
              for k in ('patients', 'studies', 'files', 'pixels')}
    ok = not any(shared.values())
    print(f'  [{"OK " if ok else "LEAK"}] train vs val   '
          f'patients={len(shared["patients"])} studies={len(shared["studies"])} '
          f'files={len(shared["files"])} '
          f'identical-pixels={len(shared["pixels"]) if digests else "n/a"}')
    for p in sorted(shared['patients'])[:5]:
        print(f'        {p}')

    n_imgs = sum(len(c['images']) for c in splits.values())
    total = sum(len(v) for v in patients.values())
    print(f'  images across train+val  : {n_imgs} (train_val had {total}, '
          f'{n_dropped_images} dropped as test duplicates)')
    if n_imgs != total - n_dropped_images:
        print('  !! image count mismatch')
        ok = False
    print(f'  annotations across train+val: '
          f'{sum(len(c["annotations"]) for c in splits.values())}')

    print(f'\n  {"split":<8}{"images":>9}{"anns":>10}{"frontal":>9}{"lateral":>9}'
          f'{"pairs":>8}{"single":>8}')
    for name in ('train', 'val'):
        views, by_study = Counter(), defaultdict(set)
        for img in splits[name]['images']:
            study, view = study_and_view(img['file_name'])
            views[view] += 1
            by_study[study].add(view)
        pairs = sum(1 for v in by_study.values() if len(v & set(VIEW_SUFFIXES)) == 2)
        print(f'  {name:<8}{len(splits[name]["images"]):>9}'
              f'{len(splits[name]["annotations"]):>10}{views["frontal"]:>9}'
              f'{views["lateral"]:>9}{pairs:>8}{len(by_study) - pairs:>8}')

    # The shipped test split itself is never modified; train/val are checked against it.
    print(f'\n  shipped test split ({os.path.basename(test_path)}):')
    with open(test_path) as f:
        test = json.load(f)
    test_bases = {os.path.basename(i['file_name']) for i in test['images']}
    test_patients = {source_and_patient(study_and_view(b)[0])[1] for b in test_bases}
    name_keys = {name: {source_and_patient(study_and_view(f)[0])[1]
                        for f in per_split[name]['files']} for name in ('train', 'val')}
    for name in ('train', 'val'):
        n = len(test_patients & name_keys[name])
        print(f'    [{"OK " if not n else "LEAK"}] shared patients with {name}: {n}')
        ok &= not n
    if test_digests:
        tset = set(test_digests.values())
        for name in ('train', 'val'):
            n = len(tset & per_split[name]['pixels'])
            if n and require_test_disjoint:
                status, note = 'LEAK', ''
                ok = False
            elif n:
                status, note = 'WARN', '   <- kept by --keep-test-duplicates'
            else:
                status, note = 'OK ', ''
            print(f'    [{status}] byte-identical images shared with {name}: {n}{note}')
    print(f'    images: {len(test_bases)}   patients: {len(test_patients)}')
    return ok, len(test_bases)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_ROOT,
                        help='PAX-Ray++ release directory (default: %(default)s)')
    parser.add_argument('--train-val', default=None, help='override paxray_train_val.json')
    parser.add_argument('--test', default=None, help='override paxray_test.json')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (default: <root>/annotations)')
    parser.add_argument('--val-fraction', type=float, default=DEFAULT_VAL_FRACTION,
                        help='fraction of train_val PATIENTS held out for val '
                             '(default: %(default)s, giving ~75/10/15 overall)')
    parser.add_argument('--seed', type=int, default=0, help='shuffle seed (default: 0)')
    parser.add_argument('--no-stratify', action='store_true',
                        help='do not keep the source-collection mix constant')
    parser.add_argument('--no-hash-union', action='store_true',
                        help='skip the md5 pass that unions aliased patients')
    parser.add_argument('--images-dir', default=None,
                        help=f'image directory (default: <root>/{IMAGES_SUBDIR})')
    parser.add_argument('--keep-test-duplicates', action='store_true',
                        help='keep train/val subjects that have a byte-identical image in '
                             'the shipped test split (dropped by default)')
    parser.add_argument('--no-copy-test', action='store_true',
                        help='do not copy paxray_test.json into the output directory')
    parser.add_argument('--dry-run', action='store_true', help='write nothing')
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        sys.exit(f'ERROR: --val-fraction must be in (0, 1), got {args.val_fraction}')
    if args.no_hash_union and not args.keep_test_duplicates:
        sys.exit('ERROR: dropping test duplicates needs the md5 pass; drop '
                 '--no-hash-union or add --keep-test-duplicates')

    tv_path = args.train_val or os.path.join(args.root, 'paxray_train_val.json')
    te_path = args.test or os.path.join(args.root, 'paxray_test.json')
    out_dir = args.out_dir or os.path.join(args.root, 'annotations')
    images_dir = args.images_dir or os.path.join(args.root, IMAGES_SUBDIR)
    for p in (tv_path, te_path):
        if not os.path.isfile(p):
            sys.exit(f'ERROR: {p} not found. Pass --root or --train-val/--test.')

    head('0. LOAD')
    print(f'  {tv_path} ({os.path.getsize(tv_path) / 1e6:.0f} MB) ...', flush=True)
    with open(tv_path) as f:
        train_val = json.load(f)
    print(f'  {len(train_val["images"])} images, {len(train_val["annotations"])} '
          f'annotations, {len(train_val["categories"])} categories')

    patients, patient_source = group_images(train_val['images'])
    digests = None
    if not args.no_hash_union:
        patients, patient_source, digests = union_by_hash(patients, patient_source,
                                                          images_dir)
    assigned = split_patients(patients, patient_source, args.val_fraction, args.seed,
                              not args.no_stratify)

    test_digests = hash_test_images(te_path, images_dir) if digests else None
    dropped = {'train': [], 'val': []}
    if not args.keep_test_duplicates:
        assigned, dropped = drop_test_duplicates(patients, assigned, digests, test_digests)
    n_dropped_images = sum(len(patients[p]) for g in dropped.values() for p in g)

    splits = build_cocos(train_val, patients, assigned)
    ok, n_test = verify(splits, patients, digests, te_path, test_digests,
                        n_dropped_images, not args.keep_test_duplicates)

    head('5. WRITE')
    n_total = sum(len(c['images']) for c in splits.values()) + n_test
    for name in ('train', 'val'):
        pct = 100 * len(splits[name]['images']) / n_total
        print(f'  {name:<6}{len(splits[name]["images"]):>7} images  ({pct:.1f}% of all)')
    print(f'  {"test":<6}{n_test:>7} images  ({100 * n_test / n_total:.1f}% of all, '
          f'shipped split, unchanged)')

    if args.dry_run:
        print('\n  --dry-run: nothing written.')
    elif not ok:
        sys.exit('\n  verification FAILED -- refusing to write.')
    else:
        os.makedirs(out_dir, exist_ok=True)
        for name in ('train', 'val'):
            path = os.path.join(out_dir, f'{name}.json')
            with open(path, 'w') as f:
                json.dump(splits[name], f)
            print(f'\n  {path}  ({os.path.getsize(path) / 1e6:.0f} MB)')
        if not args.no_copy_test:
            dst = os.path.join(out_dir, 'test.json')
            shutil.copyfile(te_path, dst)
            print(f'  {dst}  ({os.path.getsize(dst) / 1e6:.0f} MB, byte copy of '
                  f'{os.path.basename(te_path)})')
        meta = {'train_val_file': tv_path, 'test_file': te_path,
                'val_fraction_of_train_val': args.val_fraction, 'seed': args.seed,
                'stratified': not args.no_stratify,
                'hash_union': not args.no_hash_union,
                'test_split': 'shipped, unchanged',
                'dropped_test_duplicates': dropped,
                'patients': {n: sorted(assigned[n]) for n in ('train', 'val')}}
        if not args.keep_test_duplicates:
            rows = dropped_file_rows(patients, dropped, digests, test_digests)
            drop_path = os.path.join(out_dir, 'dropped_test_duplicates.csv')
            with open(drop_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['file_name', 'split', 'subject',
                                                       'md5', 'identical_test_images'])
                writer.writeheader()
                writer.writerows(rows)
            print(f'  {drop_path}  ({len(rows)} dropped files)')
        meta_path = os.path.join(out_dir, 'split_manifest.json')
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'  {meta_path}')

    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
