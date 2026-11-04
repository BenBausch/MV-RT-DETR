# Test overview: IOP-Compass-trained models on IOP and on our proprietary data

## Summary

Four tooth-detection models were trained on the public IOP-Compass dataset and tested on two test
sets: IOP-Compass's own test split, and our proprietary OrthoTracked test split. The main question was
whether a preprocessing step used in training (mirroring the side-view photos) could be what hurts
performance on our data.

- **The mirroring is correct.** Undoing it destroys performance on the side views (F1 0.73 → 0.02),
  so the models rely on it and it matches how our photos are stored.
- **All models lose about 0.12–0.14 F1 moving from IOP-Compass to our data**, and they end up
  0.065–0.085 below models trained on our data directly.
- **The side views lose the most.** Their errors are overwhelmingly teeth given the number of the
  neighbouring tooth (off-by-one along the arch, mostly premolars and canines), not left/right
  confusion.
- **All four models perform about the same.** The OT variants' main practical advantage is that they
  never predict the same tooth twice in an image.

## Background

### The task
Each image is a clinical intraoral photo. The model has to find every visible permanent tooth (a
bounding box) and give it its **FDI number**: two digits, where the first is the quadrant (1 = upper
right, 2 = upper left, 3 = lower left, 4 = lower right, from the patient's point of view) and the
second is the position from the midline (1 = central incisor … 8 = wisdom tooth). That makes 32
classes, and each tooth occurs at most once per image.

Every patient is photographed in **5 standard views**:

| View | What it shows |
|---|---|
| CENTER | frontal view of both arches |
| UP | upper arch from below (occlusal, via mirror) |
| DOWN | lower arch from above (occlusal, via mirror) |
| LEFT | the patient's left side (buccal/lateral), mainly quadrants 2 and 3 |
| RIGHT | the patient's right side, mainly quadrants 1 and 4 |

LEFT and RIGHT are called the **buccal views** below.

### The two datasets
- **IOP-Compass** is a public dataset of 1,000 patients × 5 views with FDI-labelled teeth.
  - The models were trained on its official train split (700 patients) and selected on its val split
    (200 patients).
  - Here they are tested on its test split: 100 patients, 500 images, 8,797 teeth.
- **OrthoTracked_301_montages** is our proprietary dataset.
  - Only its test split is used here: 31 patients, 155 images, 2,372 teeth.
  - None of our data was used to train these four models, so this measures how well they transfer to
    a new clinic and a new camera setup.

### The mirroring question
The two datasets store the side photos differently:
- **IOP-Compass stores them as direct shots.**
- **Our data stores them mirrored**, with the dental midline on the opposite side of the image.

To make the two consistent, training flipped IOP's LEFT/RIGHT photos horizontally (image and boxes)
into our convention. The FDI labels stay the same, because the photo is mirrored but the patient's
anatomy is not.

The concern was that this step might be wrong, or might hurt performance on our side views. This
report tests it directly.

### The four models
Two detector architectures, each with and without an extra **OT (optimal-transport) assignment
head**.

| Short name | Detector | OT head | Notes |
|---|---|---|---|
| RT-DETR base | RT-DETRv2, ResNet-50 | no | plain per-box classification |
| RT-DETR OT | RT-DETRv2, ResNet-50 | yes, 20 prototypes per tooth | also gets a 4th input channel encoding the view orientation |
| YOLO base | YOLO11-L | no | plain per-box classification + NMS |
| YOLO OT | YOLO11-L | yes, 20 prototypes per tooth | removes overlapping boxes before the OT head; no orientation channel |

What the OT head does:
- It looks at all candidate boxes of an image jointly.
- It assigns each tooth number to at most one box (or declares it absent).
- So it can never output the same tooth twice.

The base models classify every box independently and can produce duplicates. All four were trained
with the same recipe for 120 epochs; the checkpoint with the best IOP val score was used.

## How the numbers are computed

- **Matching.** A predicted box counts as correct (true positive) if it has the right FDI number and
  overlaps the ground-truth tooth with IoU ≥ 0.5. Anything else is a false positive; every tooth not
  found correctly is a false negative.
- **Precision (P), recall (R), F1.** Precision is the share of predictions that are correct; recall is
  the share of teeth found correctly; F1 is their harmonic mean.
- **Micro vs macro.** Micro scores count all teeth together, so common teeth dominate. Macro F1 is the
  unweighted average over the 32 tooth classes, so rare teeth such as wisdom teeth count as much as
  incisors.
- **Score threshold.** Every prediction has a confidence score, and only predictions with score ≥ 0.5
  are kept (the same rule as during training and testing).
  - We also swept the threshold from 0.05 to 0.70 and picked the best one per model on IOP val (0.65 /
    0.55 / 0.65 / 0.50).
  - It changed test F1 by at most 0.004, so everything below uses 0.5.
- **Error types.** Each ground-truth tooth that isn't found correctly falls into one of three bins:
  - **mislocalized:** right number, but box overlap below 0.5
  - **misclassified:** a box covers the tooth but carries a different number
  - **missed:** nothing covers it
- **Confidence intervals.** The ranges in brackets are 95% confidence intervals. They come from
  resampling *patients* (not images) 2,000 times, because a patient's 5 views are not independent.
- **Pipeline check.** Recomputing the IOP test scores here reproduced the scores logged at the end of
  training to 4 decimals (0.9354 / 0.9411 / 0.9386 / 0.9410). This confirms the evaluation pipeline
  matches training.

## 1. Is the mirroring correct? (flip test)

**The test.** We took our test set and flipped *only* the LEFT and RIGHT photos back to the
un-mirrored orientation, adjusting boxes and the orientation channel to match; labels unchanged. If
the training convention were wrong, the models should do *better* on these flipped images. As a
control we did the same on IOP: its side photos were left un-mirrored, unlike in training.

The table shows buccal-view F1 (LEFT + RIGHT together) with the images as normally used and with the
side photos flipped.

| Model | Our data, as normally used | Our data, side photos flipped | IOP, as in training | IOP, side photos un-mirrored |
|---|---|---|---|---|
| RT-DETR base | 0.729 | 0.022 | 0.923 | 0.005 |
| RT-DETR OT | 0.739 | 0.018 | 0.927 | 0.003 |
| YOLO base | 0.736 | 0.017 | 0.921 | 0.004 |
| YOLO OT | 0.729 | 0.019 | 0.926 | 0.004 |

**Reading it.** Flipping makes the side views almost useless (F1 ≈ 0.02) on both datasets.

- **What the models rely on:** they decide "left" vs "right" from the image orientation.
- **What that means:** the orientation they learned in training is the one our data has. The
  mirroring is correct and necessary; without it the models would fail on our side views.
- **Sanity check:** the frontal and occlusal views are untouched by the flip, and their predictions
  were bit-identical in both runs.

## 2. Overall results

The first table gives each model's score on both test sets. The last column repeats the flip test
from section 1 as a whole-dataset number.

| Model | IOP test P / R / F1 | macro F1 | Our data P / R / F1 [95% CI] | macro F1 | Our data, side photos flipped F1 |
|---|---|---|---|---|---|
| RT-DETR base | 0.930 / 0.941 / **0.935** | 0.918 | 0.789 / 0.817 / **0.803** [0.76, 0.84] | 0.765 | 0.545 |
| RT-DETR OT | 0.944 / 0.938 / **0.941** | 0.924 | 0.829 / 0.807 / **0.818** [0.77, 0.86] | 0.739 | 0.559 |
| YOLO base | 0.931 / 0.946 / **0.939** | 0.923 | 0.801 / 0.804 / **0.803** [0.76, 0.84] | 0.737 | 0.543 |
| YOLO OT | 0.944 / 0.938 / **0.941** | 0.926 | 0.814 / 0.793 / **0.804** [0.76, 0.84] | 0.746 | 0.545 |

**Reading it.**
- **On IOP-Compass** all four models reach F1 0.935–0.941.
- **On our data** they drop to 0.803–0.818. This is the cost of moving to a new clinic and camera
  setup without any training on our images.
- **Between models** the differences on our data (≤ 0.015) are well inside the confidence intervals
  (about ±0.04), so no model is clearly better.

**Reference: models trained on our data.** How much does training on IOP-Compass instead of our data
cost? The same four architectures were trained earlier on our training split and tested on the same
155 images. Their checkpoints are no longer on disk, so these numbers come from the training logs
(wandb) and exist only as totals, not per view.

| Model trained on our data | Our data P / R / F1 | macro F1 | IOP-trained model minus this |
|---|---|---|---|
| RT-DETR base (`65r6eoqm`) | 0.842 / 0.896 / 0.868 | 0.783 | −0.065 |
| RT-DETR OT (`7phpl8in`) | 0.899 / 0.893 / 0.896 | 0.784 | −0.078 |
| YOLO base (`v822skvc`) | 0.868 / 0.884 / 0.876 | 0.765 | −0.073 |
| YOLO OT (`vye9w4qm`) | 0.894 / 0.884 / 0.889 | 0.812 | −0.085 |

**Reading it.** Training on the public data alone costs 0.065–0.085 F1 on our data compared with
training on our own data, even though IOP-Compass has about 3× as many training images. The domain
gap is real.

**Duplicate predictions.** An image has a duplicate if some tooth number is predicted more than once.
- The OT models never produce duplicates, by construction.
- The base models do so in 8–15% of IOP test images and 19–20% of our test images.
- Duplicates are always wrong for this task, because each tooth exists only once.

## 3. Results per view

These tables break the F1 down by view. This shows where performance is lost: if the side views
suffered disproportionately on our data, that would point to a side-view-specific problem such as
the mirroring.

**IOP-Compass test** (100 images per view), micro F1:

| View | RT-DETR base | RT-DETR OT | YOLO base | YOLO OT |
|---|---|---|---|---|
| CENTER | 0.919 | 0.926 | 0.926 | 0.927 |
| UP | 0.971 | 0.978 | 0.981 | 0.977 |
| DOWN | 0.965 | 0.970 | 0.965 | 0.969 |
| LEFT | 0.919 | 0.919 | 0.914 | 0.923 |
| RIGHT | 0.926 | 0.935 | 0.929 | 0.929 |
| buccal (L+R) | 0.923 | 0.927 | 0.921 | 0.926 |
| non-buccal (C+U+D) | 0.944 | 0.951 | 0.950 | 0.951 |

**Our test set** (31 images per view), micro F1 [95% CI]:

| View | RT-DETR base | RT-DETR OT | YOLO base | YOLO OT |
|---|---|---|---|---|
| CENTER | 0.800 [0.74, 0.85] | 0.827 [0.78, 0.87] | 0.806 [0.75, 0.86] | 0.810 [0.75, 0.86] |
| UP | 0.898 [0.81, 0.96] | 0.899 [0.81, 0.96] | 0.883 [0.79, 0.95] | 0.898 [0.81, 0.96] |
| DOWN | 0.878 [0.80, 0.93] | 0.888 [0.80, 0.95] | 0.858 [0.77, 0.92] | 0.860 [0.79, 0.92] |
| LEFT | 0.717 [0.65, 0.79] | 0.699 [0.62, 0.77] | 0.723 [0.64, 0.79] | 0.713 [0.63, 0.79] |
| RIGHT | 0.740 [0.67, 0.81] | 0.779 [0.71, 0.85] | 0.749 [0.67, 0.82] | 0.744 [0.67, 0.82] |
| buccal (L+R) | 0.729 [0.67, 0.78] | 0.739 [0.67, 0.80] | 0.736 [0.67, 0.79] | 0.729 [0.66, 0.79] |
| non-buccal (C+U+D) | 0.846 [0.78, 0.89] | 0.862 [0.80, 0.91] | 0.840 [0.78, 0.88] | 0.846 [0.79, 0.89] |

**Drop in F1 from IOP-Compass to our data**, with 95% CI:

| Model | buccal views | non-buccal views | all views |
|---|---|---|---|
| RT-DETR base | −0.194 [0.13, 0.26] | −0.098 [0.05, 0.16] | −0.133 |
| RT-DETR OT | −0.188 [0.12, 0.26] | −0.089 [0.04, 0.15] | −0.123 |
| YOLO base | −0.185 [0.12, 0.26] | −0.110 [0.06, 0.17] | −0.136 |
| YOLO OT | −0.197 [0.14, 0.27] | −0.104 [0.06, 0.16] | −0.137 |

**Reading it.**
- **The occlusal views (UP/DOWN) are the easiest** on both datasets.
- **The side views are the hardest** on both datasets, already on IOP-Compass.
- **On our data the side views lose about twice as much F1** as the other views (−0.19 vs −0.10).
- **In relative terms the difference is smaller.** The error rate (1 − F1) grows 3.3–3.7× on the side
  views and 2.8–3.2× on the others, so the side views get worse partly because they were already the
  weakest. Section 4 looks at *what kind* of errors grow.

## 4. What goes wrong on our data

### Which error type grows
The table below shows, for every ground-truth tooth in our test set, what happened to it: correct /
mislocalized / misclassified / missed, in % of teeth. For comparison, on IOP-Compass about 3% of teeth
are misclassified in every view.

| View | RT-DETR base | RT-DETR OT | YOLO base | YOLO OT |
|---|---|---|---|---|
| LEFT | 73 / 8 / 16 / 3 | 68 / 6 / 17 / 9 | 72 / 6 / 15 / 7 | 71 / 7 / 17 / 6 |
| RIGHT | 76 / 7 / 14 / 3 | 76 / 4 / 12 / 8 | 73 / 4 / 12 / 11 | 73 / 6 / 16 / 5 |
| CENTER | 82 / 9 / 8 / 2 | 83 / 7 / 7 / 3 | 82 / 7 / 9 / 3 | 80 / 6 / 8 / 6 |

**Reading it.** The error that grows most is **misclassification**: the tooth is found and boxed, but
gets the wrong number.
- **On our side views** it rises to 12–17% of teeth, about 5× the IOP rate.
- **On the other views** it rises to 7–9%.

Finding teeth (localization) transfers much better than numbering them.

### Which wrong number: neighbour vs left/right
If the mirroring were the problem, the wrong numbers would mostly be **left/right swaps**: the same
tooth on the other side of the mouth, e.g. 16 predicted as 26. If the problem is counting along the
arch, they would be **neighbour shifts**: the adjacent tooth, e.g. 14 predicted as 15. The table
counts both, per 100 teeth, on our test set.

| Model | Side views: neighbour shift | Side views: left/right swap | Other views: neighbour shift | Other views: upper/lower swap |
|---|---|---|---|---|
| RT-DETR base | 13.0 | 1.3 | 5.3 | 2.5 |
| RT-DETR OT | 11.9 | 1.2 | 4.4 | 1.4 |
| YOLO base | 11.1 | 1.3 | 6.1 | 2.4 |
| YOLO OT | 13.5 | 0.9 | 6.0 | 1.4 |

**Reading it.** On our side views, about **85% of misclassifications are neighbour shifts**. Only 6–9%
are left/right swaps.
- **On IOP-Compass** the corresponding rates are ~3 neighbour shifts and ~0.35 left/right swaps per 100
  side-view teeth.
- **Both grow by a similar factor**, so left/right confusion is not singled out.

This confirms section 1: the side-view drop is a **numbering-along-the-arch** problem, not a mirroring
problem.

### Which teeth are hit
The comparison here is against the models trained on our data (macro F1 difference, IOP-trained minus
trained-on-our-data):

| Tooth type | Macro F1 difference |
|---|---|
| premolars | **−0.14 to −0.19** |
| canines | −0.07 to −0.09 |
| incisors | −0.03 to −0.04 |
| molars | −0.03 to −0.05 |

The loss concentrates on the canine–premolar segment, where neighbouring teeth look most alike and an
off-by-one error is easiest to make. That fits the neighbour-shift finding.

### Other observations
- **The side photos are framed differently.** IOP-Compass side photos show more teeth: 17.4–17.8 per
  image, against 13.8–14.1 in our data. They include more of the front teeth of the opposite side.
  - Both datasets agree on which quadrants each view shows: LEFT = quadrants 2/3, RIGHT = 1/4.
  - With fewer reference teeth in the frame, counting along the arch is plausibly harder. This is a
    likely contributor, not proven.
- **On our frontal and occlusal views, neighbour shifts go mostly in one direction.** They point away
  from the midline: for RT-DETR base, 64 shifts towards the back of the mouth vs 16 towards the front.
  On IOP they are balanced. This suggests a systematic counting offset on our images.
- **Upper/lower swaps are concentrated in a few images.** There are 21–37 such errors per model on the
  non-side views. For RT-DETR base, 16 of them come from just 2 DOWN images, 9 from a single one. Those
  images are worth a manual look: this may be an image or label issue rather than a model error.

## 5. Model comparisons

- **OT head vs base:**
  - F1 is equal or slightly higher with the OT head everywhere: on IOP +0.006 (RT-DETR) and +0.002
    (YOLO), on our data +0.015 and +0.001. All of these are within the confidence intervals, so the
    head's F1 advantage is not demonstrated on this test set.
  - Its clear benefit is zero duplicate predictions, against 19–20% of images with duplicates for the
    base models on our data.
  - It trades a little recall for precision.
  - For RT-DETR its macro F1 on our data is lower (0.739 vs 0.765). This is partly driven by the 11
    wisdom teeth; see below.
- **RT-DETR vs YOLO:** no meaningful difference on either test set; all within 0.015 F1.
- **Wisdom teeth (18/28/38/48):** these are rare in training (140–189 examples each, vs ~2,000 for
  other teeth).
  - On IOP test all four models reach F1 0.68–0.97 on them. That includes RT-DETR OT, which scored 0 on
    wisdom teeth early in training.
  - Our test set contains only 11 wisdom teeth (2–4 per class), too few to say anything.

## 6. Caveats

- **Small test set.** Our test set has 31 patients, so per-view numbers are uncertain by roughly ±0.04
  to ±0.08. Differences between models below about 0.04 F1 cannot be resolved with it.
- **The reference models trained on our data** exist only as logged totals. Their checkpoints were
  deleted, so there is no per-view comparison for them. They were also selected on our val split, not
  on IOP val.
- **The two OT models differ in more than the detector.** RT-DETR OT has the orientation input channel
  and YOLO OT does not. This doesn't affect the flip test, which gives the same result for all four.
- **The flip test** mirrors the photo and updates the orientation channel consistently, the same way
  the training loader treats un-mirrored IOP side photos.

## Files

- **Report:** this file, `docs/iop_models_test_overview.md`
- **Tables (CSV):** `rtdetrv2_pytorch/output/test_overview/tables/`
  - `totals.csv`, `per_view.csv`, `per_class.csv`: main metrics
  - `contrasts.csv`: dataset and flip deltas with CIs
  - `swaps.csv`, `neighbour_shift.csv`, `swap_concentration.csv`, `confusion_pairs.csv`: error types
  - `domain_gap_*.csv`: tooth-type gap vs the models trained on our data
  - `threshold_sweep.csv`, `thresholds.csv`: threshold analysis
  - `wandb_refs*.csv`: reference numbers of the models trained on our data
- **Raw predictions:** `rtdetrv2_pytorch/output/test_overview/preds/`, so metrics can be recomputed
  without re-running the models
- **Scripts:** `rtdetrv2_pytorch/tools/scripts/test_overview/`
  - `run_all.sh`: the full pipeline
  - `buccal_flip_dataset.py`: the flip-test dataset
