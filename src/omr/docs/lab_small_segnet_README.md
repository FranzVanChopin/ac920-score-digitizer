# Small segmentation net replacing Homr's segnet (2026-10-24)

Goal: shrink the photo front end ("Homr is too big"). First target: Homr's segnet (14.3 M params,
459 GMAC per page), the part that sees the whole page. A drop-in student with the exact same ONNX
interface (input `[N,3,320,320]` float 0..255, output `[N,6,320,320]` logits) is trained by
distillation and swapped into the full camera67 pipeline without touching any pipeline code.

Nothing in the Codex project was modified. The pipeline is the lab copy of the 2026-10-13 handoff
package in `../binarize_ab_20261024/pkg` (verified against its manifest: 8,669 files, 0 errors).

## Data
* 1,200 PDMX pieces (half piano grand staff), excluding every decoder-benchmark / dose piece and
  every title resembling a Codex test-set piece (126 names, regex in `select_render.py`).
  MuseScore 4 renders at 240 dpi -> first two pages -> 1,423 train / 146 val pages (split by piece).
* Each page: geometry jitter (scale, offset, perspective, rotation) on the clean page; teacher
  label = Homr segnet on that clean page (Homr's exact preprocessing: width 1920, CLAHE, 320 tiles);
  student input = same page after camera degradation (paper/ink levels, uneven light, vignette,
  blur, noise, JPEG) -> CLAHE. `make_dataset.py`.
* camera67 is evaluation only (teacher labels in `dataset/camera67` for pixel agreement).

## Students (`train_student.py`)
Depthwise-separable U-Net, BN+ReLU, nearest upsampling. `--input adaptive` puts a fixed local-mean
binarization in front (`ink = x < boxmean31(x) - 10`), so the net only sees a 1-bit image.
15,000 steps, batch 32, AdamW, mild class weights. ONNX built by `export_onnx.py` (BN folded,
verified argmax-identical to torch).

## Results (camera67, full photo -> MusicXML pipeline, same scorer as the handoff's replay)

| segnet | params | GMAC/page | val mIoU | camera67 mIoU vs teacher | notes | Δ vs teacher [95% CI] | better/worse/equal pages | rests | graces |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Homr teacher | 14.32 M | 459 | — | — | 97.56% | — | — | 98.69% | 102/145 |
| S gray | 68 k | 23.5 | 0.898 | 0.882 | 96.93% | −0.64 [−1.58, +0.01] | 10/15/42 | 98.64% | 92 |
| **S binarized** | **68 k** | **23.5** | 0.873 | 0.842 | **97.39%** | **−0.17 [−0.67, +0.21]** | 13/12/42 | 98.57% | **104** |
| M gray | 238 k | — | 0.905 | 0.885 | 97.27% | −0.29 [−1.13, +0.23] | 11/8/48 | 98.78% | 94 |
| M binarized | 238 k | — | 0.884 | 0.851 | 95.89% | −1.67 [−5.02, +0.12] | 14/14/39 | 94.87% | 97 |

* Teacher run reproduces the accepted 16,280/16,687 exactly under the lab hook (ORT 4 threads).
* M binarized lost one whole page (score26 p1: 260 notes recognized, 0 matched — a measure-structure
  shift, the per-measure scorer then misses everything); without it it is ~97.4%.
  score11 p1 is the weak page for every student.
* Pixel IoU against the teacher does not predict end-to-end accuracy (binarized students have lower
  IoU but S binarized is the best end-to-end).
* segnet time per page (desktop x86, 54 tiles): teacher 1.25 s @4 threads / 3.47 s @1; S 0.45 / 0.6 s.
  CPU gains are only ~6x because depthwise convs at full resolution are memory-bound; MACs drop 20x,
  params 216x (66 KB as int8 — fits XCZU4EV on-chip RAM; the teacher's 14 MB does not).

## Caveats
* camera67 is the development set the pipeline's repairs were tuned on, and its photos are clean
  simulations; real camera photos are untested (the handoff pipeline scores ~13% on the real sonata
  photos with the teacher too — not usable as a benchmark yet).
* Differences between students are within noise except page-level catastrophes; one seed each.

## Earlier in the same lab: binarizing the INPUT of the current models (`../binarize_ab_20261024`)
Feeding Otsu / adaptive binarized photos to the unchanged Homr pipeline: 97.56 -> 85.38 / 82.11.
Binarized input only works with a model trained for it (as above).

## Reproduce
```
python select_render.py 1200 3
python make_dataset.py 0 2 & python make_dataset.py 1 2
python train_student.py --name S_bin --width 16,32,48,64,96 --input adaptive --iters 15000
python chain_eval.py S_bin          # builds ONNX with the package runtime, runs + scores camera67
python compare_runs.py S_gray S_bin M_gray M_bin
```
Load rules: at most 2 trainings and 2 pipeline workers at once, ORT 4 threads, BELOW_NORMAL priority,
memory guard at 12 GB available (an mmap data loader once used 45 GB).
