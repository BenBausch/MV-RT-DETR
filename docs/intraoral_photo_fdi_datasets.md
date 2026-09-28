# Public 2D intraoral-photo datasets with FDI tooth labels

Survey date: 2026-09-25. Scope: 2D intraoral **photographs** (not intraoral scans, not X-rays)
with per-tooth labels in FDI notation (or another tooth atlas).

**Bottom line:** IOP-Compass is the only large, public dataset that labels every tooth with an FDI
number. The few others are small, restricted, or no longer downloadable.

## Recommended: IOP-Compass (AImageLab, University of Modena and Reggio Emilia, 2026)

| | |
|---|---|
| Images | 5,000 PNG photos from 1,000 patients |
| Views | 5 per patient: frontal, left buccal, right buccal, upper occlusal, lower occlusal |
| Labels | Every visible tooth has a contour, a bounding box and an FDI number (11–48); expert-verified; one JSON per image |
| Splits | Patient-disjoint 700 / 200 / 100 (train / val / test), seed 42, defined in the companion repo |
| Source | Images come from Bite2Text (ECCV 2026), two clinics, same camera protocol, clinical mirror used |
| License | CC BY-NC-SA 4.0 (non-commercial) |
| Access | Free account on the Ditto portal; model weights by email to the maintainers |
| Links | [dataset page](https://ditto.ing.unimore.it/iop-compass/) · [GitHub](https://github.com/AImageLab-zip/IOP-Compass) |

On-disk layout (from `configs/dataset_final_1000.yaml` in the repo):

```
Patient_<pid>/IOP_<View>_RawImage_<pid>.png
Patient_<pid>/IOP_<View>_<pid>.json      # View ∈ {Center, Left, Right, Upper, Down}
```

Notes:
- 2,283 distinct image resolutions occur, so resize to a common size before training.
- The repo includes a loader for the annotations (`src/iop_compass/data/adapter.py`), baselines for
  view classification and tooth instance segmentation (SegmentAnyTooth, Mask R-CNN), and a
  correction viewer.
- 5 views per patient, so it fits multi-view training directly.

## Other public options, with limits

| Dataset | Content | Limitation |
|---|---|---|
| **IO150K** (Teeth-SEG, CVPR 2024) | ~800 real intraoral photos, plus 80K images rendered from 3D scans and 70K photos of plaster models. Annotators used FDI notation; masks generated with SAM from orthodontist-clicked tooth centres. | Google Drive link on the [project page](https://zoubo9034.github.io/TeethSEG/) returned 404 in Sept 2026, so contact the authors. The model uses 16 tooth classes, which doesn't obviously map to 32 FDI teeth. [Paper](https://arxiv.org/abs/2404.01013) |
| **FDTooth** ([PhysioNet](https://physionet.org/content/fdtooth/)) | 241 frontal photos (one per patient); bounding boxes on the 12 front teeth, each with an FDI number and a fenestration / dehiscence / normal label | Front teeth only. Needs a credentialed PhysioNet account, CITI training and a data use agreement. [Paper](https://www.nature.com/articles/s41597-025-05348-3) |

## Datasets with FDI labels that are not public

| Dataset | Content | Access |
|---|---|---|
| [SegmentAnyTooth](https://www.sciencedirect.com/science/article/pii/S1991790225000030) | 5,000 photos (1,000 sets, 953 subjects), 5 views, FDI labels | Only the model weights are released (non-commercial license, sign and email an agreement); [code](https://github.com/thangngoc89/SegmentAnyTooth) is MIT |
| [TLNM](https://arxiv.org/abs/2608.06275) (DigiLeap, Finland) | 1,272 smartphone photos, 5 views, FDI polygons | Restricted by Findata (Finnish data-permit authority) |
| [Structure-guided tooth numbering](https://doi.org/10.3390/jimaging12060256) (J. Imaging 2026) | 845 photos gathered from public Kaggle / GitHub sources, FDI bounding boxes | Images are public, FDI annotations only on request |

## Public, but no tooth-number labels

| Dataset | What is labelled |
|---|---|
| [AlphaDent](https://arxiv.org/abs/2507.22512) | 1,320 DSLR photos, 9 pathology classes (abrasion, filling, crown, 6 caries grades); CC BY-SA |
| [Caries dataset, Pakistan](https://zenodo.org/records/14827784) | 6,313 photos, boxes around decayed teeth (d / D) only |
| [SIOP](https://figshare.com/articles/dataset/SIOP_dataset/29761292) | 1,350 periodontal photos, image-quality and gingival-inflammation scores |
| [Gingivitis captioning](https://data.mendeley.com/datasets/3253gj88rr/1) | 1,096 photos, gingival-index scores and captions |
| [Multi-view extraction database](https://zenodo.org/records/16456357) | 215 patients × 8 views, one label per patient (extraction needed or not) |
| [Varying Views of Teeth](https://data.mendeley.com/datasets/6zsnhrds9t/1) | Children's teeth, view / arch labels only |
| [Multimodal oro-dental dataset](https://pubmed.ncbi.nlm.nih.gov/42062311/) | 50,000 photos, 6 anomaly classes and written reports |
| [MetaDent](https://arxiv.org/abs/2604.14866) | Free-text descriptions and VQA pairs |

Not checked: Roboflow Universe has some community "teeth numbering" datasets, but the site blocked
automated requests, so their modality and quality are unknown. Most such datasets are panoramic X-rays.

## Keeping this up to date

Uribe et al. maintain a living list of public dental datasets:
[review paper (JDR 2024)](https://journals.sagepub.com/doi/10.1177/00220345241272052) ·
[GitHub list](https://github.com/sergiouribe/dental_datasets_itu).
