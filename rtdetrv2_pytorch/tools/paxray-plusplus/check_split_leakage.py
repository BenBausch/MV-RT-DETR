"""Verify that PAX-Ray++'s two shipped COCO files are a clean train/test partition.

The upstream release (https://huggingface.co/datasets/cmseibold/PAX-RayPlusPlus)
ships ``paxray_train_val.json`` and ``paxray_test.json`` as two independent COCO
dicts. Nothing in the dataset card states whether they were written from a single
id space, so before training we have to answer two *different* questions:

1. **Id collisions** -- do the two files reuse ``image_id`` / annotation ``id``
   values for different images? This is not leakage, but it silently corrupts any
   code that merges the splits or keys a cache on ``image_id``.
2. **Content overlap** -- does the same *radiograph* appear in both files? That is
   real leakage. Two flavours, and the second is the one that bites:
     a. identical ``file_name`` in both splits;
     b. the same *study* in both splits via its other view. PAX-Ray++ images are
        named ``<source>_<case>_{frontal,lateral}.png`` and both views are DRR
        projections of the **same CT volume**, so ``X_frontal`` in train and
        ``X_lateral`` in test is the same patient, same anatomy, same pseudo-label
        source -- leakage even though the filenames differ.

Section 5 additionally replays the upstream 90/10 train/val split
(``2DAnatomyDatasets/src/prepare_data/prepare_paxraypp/prepare_paxraypp_splits.py``,
which shuffles *images* with seed 42) to measure how many studies it tears across
the train/val boundary. That split is the one we would inherit by default, so its
leakage is worth quantifying even though it is not part of the shipped files.

Exit code is 0 when no content overlap is found, 1 otherwise (id collisions alone
are reported as warnings, not failures).

Usage:
    <python> tools/paxray-plusplus/check_split_leakage.py
    <python> tools/paxray-plusplus/check_split_leakage.py \
        --root /mnt/data/abi/datasets/PAX-RayPlusPlus --json-report /tmp/paxray_leak.json

Run from the rtdetrv2_pytorch/ directory. Peak RSS is ~15 GB (both COCO dicts are
held in memory); stdlib only.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

DEFAULT_ROOT = '/mnt/data/abi/datasets/PAX-RayPlusPlus'
IMAGES_SUBDIR = 'images_patlas'
VIEW_SUFFIXES = ['frontal', 'lateral']

# Upstream train/val split parameters, copied from prepare_paxraypp_splits.py.
UPSTREAM_SEED = 42
UPSTREAM_TRAIN_FRACTION = 0.9


def head(title):
    print(f'\n{"=" * 78}\n{title}\n{"=" * 78}')


def study_and_view(file_name):
    """'RibFrac_194_lateral.png' -> ('RibFrac_194', 'lateral').

    The study key is what both views of one CT volume share. Returns view ``None``
    for names that carry neither suffix, so callers can flag them instead of
    silently folding them into a bogus study.
    """
    stem = os.path.splitext(os.path.basename(file_name))[0]
    for view in VIEW_SUFFIXES:
        if stem.endswith('_' + view):
            return stem[: -(len(view) + 1)], view
    return stem, None


def load_coco(path):
    print(f'  loading {path} ({os.path.getsize(path) / 1e6:.0f} MB) ...', flush=True)
    with open(path) as f:
        return json.load(f)


def summarize(name, coco):
    """Per-file counts plus the id/name duplicate check that must hold inside one file."""
    images = coco['images']
    anns = coco['annotations']
    ids = [img['id'] for img in images]
    names = [img['file_name'] for img in images]

    dup_ids = [i for i, c in Counter(ids).items() if c > 1]
    dup_names = [n for n, c in Counter(names).items() if c > 1]

    views = Counter()
    studies = defaultdict(set)
    unknown_view = []
    for img in images:
        study, view = study_and_view(img['file_name'])
        views[view] += 1
        studies[study].add(view)
        if view is None:
            unknown_view.append(img['file_name'])

    ann_image_ids = {a['image_id'] for a in anns}
    orphan_anns = ann_image_ids - set(ids)
    unannotated = set(ids) - ann_image_ids

    print(f'\n[{name}]')
    print(f'  images                  : {len(images)}')
    print(f'  annotations             : {len(anns)}')
    print(f'  categories              : {len(coco["categories"])} '
          f'(ids {min(c["id"] for c in coco["categories"])}..'
          f'{max(c["id"] for c in coco["categories"])})')
    print(f'  image id range          : {min(ids)}..{max(ids)}')
    print(f'  distinct studies        : {len(studies)}')
    for view in VIEW_SUFFIXES:
        print(f'  {view + " images":<24}: {views[view]}')
    if views[None]:
        print(f'  !! images with no view suffix: {views[None]} '
              f'(e.g. {unknown_view[:3]})')
    both = sum(1 for v in studies.values() if len(v & set(VIEW_SUFFIXES)) == 2)
    print(f'  studies with both views : {both} '
          f'({len(studies) - both} single-view)')
    print(f'  anns/image (mean)       : {len(anns) / max(len(images), 1):.1f}')
    if dup_ids:
        print(f'  !! duplicate image ids within file: {len(dup_ids)} '
              f'(e.g. {dup_ids[:5]})')
    if dup_names:
        print(f'  !! duplicate file_names within file: {len(dup_names)} '
              f'(e.g. {dup_names[:5]})')
    if orphan_anns:
        print(f'  !! annotations pointing at unknown image ids: {len(orphan_anns)}')
    if unannotated:
        print(f'  -- images with zero annotations: {len(unannotated)}')

    return {
        'n_images': len(images),
        'n_annotations': len(anns),
        'image_ids': set(ids),
        'file_names': set(names),
        'studies': dict(studies),
        'dup_image_ids': dup_ids,
        'dup_file_names': dup_names,
        'orphan_annotations': len(orphan_anns),
        'unannotated_images': len(unannotated),
    }


def check_id_collisions(train_val, test, tv_sum, te_sum):
    """Id-space overlap. A collision is a correctness hazard, not leakage per se --
    unless the colliding ids also name the same file, which is checked separately."""
    head('2. IDENTIFIER COLLISIONS ACROSS THE TWO FILES')

    shared_img_ids = tv_sum['image_ids'] & te_sum['image_ids']
    tv_by_id = {img['id']: img['file_name'] for img in train_val['images']}
    te_by_id = {img['id']: img['file_name'] for img in test['images']}

    same_file = sorted(i for i in shared_img_ids if tv_by_id[i] == te_by_id[i])
    diff_file = sorted(i for i in shared_img_ids if tv_by_id[i] != te_by_id[i])

    print(f'shared image ids            : {len(shared_img_ids)}')
    print(f'  ... naming the SAME file  : {len(same_file)}   <- leakage, see section 3')
    print(f'  ... naming DIFFERENT files: {len(diff_file)}   <- id-space collision only')
    for i in diff_file[:5]:
        print(f'      id {i}: train_val={tv_by_id[i]!r}  test={te_by_id[i]!r}')
    if len(diff_file) > 5:
        print(f'      ... and {len(diff_file) - 5} more')

    tv_ann_ids = {a['id'] for a in train_val['annotations']}
    te_ann_ids = {a['id'] for a in test['annotations']}
    shared_ann_ids = tv_ann_ids & te_ann_ids
    print(f'shared annotation ids       : {len(shared_ann_ids)}')
    print(f'  train_val ann id range    : {min(tv_ann_ids)}..{max(tv_ann_ids)}')
    print(f'  test      ann id range    : {min(te_ann_ids)}..{max(te_ann_ids)}')

    if shared_img_ids or shared_ann_ids:
        print('\n=> The two files do NOT share one id space. Renumber (or namespace)'
              '\n   ids before merging them, and never key a cache on image_id alone.')
    else:
        print('\n=> Id spaces are disjoint; the files can be merged as-is.')

    return {
        'shared_image_ids': len(shared_img_ids),
        'shared_image_ids_same_file': len(same_file),
        'shared_image_ids_different_file': len(diff_file),
        'shared_annotation_ids': len(shared_ann_ids),
    }


def check_content_overlap(tv_sum, te_sum):
    """The actual leakage test: same image, or same study via its other view."""
    head('3. CONTENT OVERLAP  (this is what decides leakage)')

    shared_names = sorted(tv_sum['file_names'] & te_sum['file_names'])
    print(f'identical file_names in both splits : {len(shared_names)}')
    for n in shared_names[:10]:
        print(f'    {n}')
    if len(shared_names) > 10:
        print(f'    ... and {len(shared_names) - 10} more')

    tv_studies = set(tv_sum['studies'])
    te_studies = set(te_sum['studies'])
    shared_studies = sorted(tv_studies & te_studies)
    print(f'\nstudies (CT volumes) in both splits : {len(shared_studies)} '
          f'of {len(tv_studies)} train_val / {len(te_studies)} test')

    cross_view_only = [s for s in shared_studies
                       if not (tv_sum['studies'][s] & te_sum['studies'][s])]
    print(f'  ... sharing only the OTHER view   : {len(cross_view_only)}'
          f'   <- invisible to a filename check')
    for s in shared_studies[:10]:
        print(f'    {s}: train_val={sorted(tv_sum["studies"][s])} '
              f'test={sorted(te_sum["studies"][s])}')
    if len(shared_studies) > 10:
        print(f'    ... and {len(shared_studies) - 10} more')

    leaked_test_images = sum(len(te_sum['studies'][s]) for s in shared_studies)
    if shared_studies:
        print(f'\n=> LEAKAGE: {leaked_test_images} test images '
              f'({100 * leaked_test_images / max(te_sum["n_images"], 1):.1f}% of the '
              f'test split) come from a study that also appears in train_val.')
        print('   Test metrics on those images are optimistic. Fix by re-splitting on'
              '\n   the study key (file_name minus the _frontal/_lateral suffix).')
    else:
        print('\n=> CLEAN: no radiograph and no study appears in both files.')

    return {
        'shared_file_names': len(shared_names),
        'shared_file_names_examples': shared_names[:20],
        'shared_studies': len(shared_studies),
        'shared_studies_cross_view_only': len(cross_view_only),
        'shared_studies_examples': shared_studies[:20],
        'leaked_test_images': leaked_test_images,
    }


def check_category_consistency(train_val, test):
    head('4. CATEGORY CONSISTENCY')
    tv_cats = {c['id']: c['name'] for c in train_val['categories']}
    te_cats = {c['id']: c['name'] for c in test['categories']}
    print(f'train_val categories: {len(tv_cats)}   test categories: {len(te_cats)}')
    if tv_cats == te_cats:
        print('=> Identical id->name maps; one `num_classes` covers both splits.')
    else:
        only_tv = sorted(set(tv_cats) - set(te_cats))
        only_te = sorted(set(te_cats) - set(tv_cats))
        renamed = sorted(i for i in set(tv_cats) & set(te_cats)
                         if tv_cats[i] != te_cats[i])
        print(f'!! ids only in train_val: {only_tv}')
        print(f'!! ids only in test     : {only_te}')
        print(f'!! ids with a different name: '
              f'{[(i, tv_cats[i], te_cats[i]) for i in renamed[:5]]}')
    return {'identical': tv_cats == te_cats, 'n_categories': len(tv_cats)}


def check_upstream_split(train_val, tv_sum):
    """Replay the upstream image-level 90/10 shuffle and count torn studies.

    prepare_paxraypp_splits.py shuffles ``coco["images"]`` with ``random.Random(42)``
    and cuts at ``round(0.9 * N)``. Because it shuffles *images*, the two views of a
    study can land on opposite sides -- the same leakage as section 3, self-inflicted.
    """
    head('5. UPSTREAM 90/10 TRAIN/VAL SPLIT  (replayed from prepare_paxraypp_splits.py)')

    images = list(train_val['images'])
    random.Random(UPSTREAM_SEED).shuffle(images)
    n_train = int(round(UPSTREAM_TRAIN_FRACTION * len(images)))
    train_imgs, val_imgs = images[:n_train], images[n_train:]

    train_studies = {study_and_view(img['file_name'])[0] for img in train_imgs}
    val_studies = {study_and_view(img['file_name'])[0] for img in val_imgs}
    torn = train_studies & val_studies
    leaked_val = sum(1 for img in val_imgs
                     if study_and_view(img['file_name'])[0] in torn)

    print(f'train: {len(train_imgs)} images / {len(train_studies)} studies')
    print(f'val  : {len(val_imgs)} images / {len(val_studies)} studies')
    print(f'studies split across train and val : {len(torn)} '
          f'({100 * len(torn) / max(len(val_studies), 1):.1f}% of val studies)')
    print(f'val images whose study is in train : {leaked_val} '
          f'({100 * leaked_val / max(len(val_imgs), 1):.1f}% of val)')
    if torn:
        print('\n=> The upstream split leaks: it partitions images, not studies. Build'
              '\n   our splits by grouping on the study key instead.')
    else:
        print('\n=> No study is torn by the upstream split.')

    return {
        'n_train_images': len(train_imgs),
        'n_val_images': len(val_imgs),
        'torn_studies': len(torn),
        'leaked_val_images': leaked_val,
    }


def check_disk(images_dir, tv_sum, te_sum):
    head('6. CROSS-CHECK AGAINST THE EXTRACTED IMAGES')
    if not os.path.isdir(images_dir):
        print(f'-- {images_dir} not found, skipping.')
        return {'skipped': True}

    on_disk = {f for f in os.listdir(images_dir) if f.lower().endswith('.png')}
    referenced = {os.path.basename(n) for n in tv_sum['file_names'] | te_sum['file_names']}
    missing = sorted(referenced - on_disk)
    unused = sorted(on_disk - referenced)

    print(f'png files on disk        : {len(on_disk)}')
    print(f'referenced by the splits : {len(referenced)}')
    print(f'referenced but MISSING   : {len(missing)}  {missing[:5]}')
    print(f'on disk but unreferenced : {len(unused)}  (the "unfiltered" surplus)')
    if unused:
        unused_studies = {study_and_view(f)[0] for f in unused}
        annotated_studies = set(tv_sum['studies']) | set(te_sum['studies'])
        print(f'  ... belonging to {len(unused_studies)} studies, of which '
              f'{len(unused_studies & annotated_studies)} also have annotated images')
    return {'on_disk': len(on_disk), 'referenced': len(referenced),
            'missing': len(missing), 'unreferenced': len(unused)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=DEFAULT_ROOT,
                        help='PAX-Ray++ release directory (default: %(default)s)')
    parser.add_argument('--train-val', default=None,
                        help='override path to paxray_train_val.json')
    parser.add_argument('--test', default=None,
                        help='override path to paxray_test.json')
    parser.add_argument('--images-dir', default=None,
                        help=f'override path to {IMAGES_SUBDIR}/')
    parser.add_argument('--skip-upstream-split', action='store_true',
                        help='skip section 5 (replay of the upstream 90/10 split)')
    parser.add_argument('--json-report', default=None,
                        help='also write the findings to this JSON file')
    args = parser.parse_args()

    tv_path = args.train_val or os.path.join(args.root, 'paxray_train_val.json')
    te_path = args.test or os.path.join(args.root, 'paxray_test.json')
    images_dir = args.images_dir or os.path.join(args.root, IMAGES_SUBDIR)

    head('1. FILES')
    for p in (tv_path, te_path):
        if not os.path.isfile(p):
            sys.exit(f'ERROR: {p} not found. Pass --root or --train-val/--test.')
    train_val = load_coco(tv_path)
    test = load_coco(te_path)
    print(f'  train_val info: {train_val.get("info")!r}')
    print(f'  test      info: {test.get("info")!r}')

    tv_sum = summarize('paxray_train_val.json', train_val)
    te_sum = summarize('paxray_test.json', test)

    report = {
        'train_val_path': tv_path,
        'test_path': te_path,
        'ids': check_id_collisions(train_val, test, tv_sum, te_sum),
        'content': check_content_overlap(tv_sum, te_sum),
        'categories': check_category_consistency(train_val, test),
    }
    if not args.skip_upstream_split:
        report['upstream_split'] = check_upstream_split(train_val, tv_sum)
    report['disk'] = check_disk(images_dir, tv_sum, te_sum)

    head('VERDICT')
    leaks = report['content']['shared_studies'] > 0
    if leaks:
        print(f'FAIL: {report["content"]["shared_studies"]} studies '
              f'({report["content"]["leaked_test_images"]} test images) appear in both '
              f'shipped files.\n      Re-split on the study key before training.')
    else:
        print('PASS: paxray_train_val.json and paxray_test.json are disjoint at the '
              'study level.')
    if report['ids']['shared_image_ids_different_file'] or \
            report['ids']['shared_annotation_ids']:
        print('WARN: the two files reuse ids for different objects -- renumber before '
              'merging.')
    if not args.skip_upstream_split and report['upstream_split']['torn_studies']:
        print(f'WARN: the upstream 90/10 train/val split tears '
              f'{report["upstream_split"]["torn_studies"]} studies; do not reuse it.')

    if args.json_report:
        with open(args.json_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nreport written to {args.json_report}')

    sys.exit(1 if leaks else 0)


if __name__ == '__main__':
    main()
