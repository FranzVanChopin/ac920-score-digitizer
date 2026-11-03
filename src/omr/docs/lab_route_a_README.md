# Route A: retrained relation networks on binarized simulated photos (2026-10-26)

Plan and motivation: `PLAN.md`; diagnosis behind it: `../diag_b/README.md`.  Step 1–2 of the plan:
replace Codex's `symbol` and `stroke` networks (trained on 49 clean Verovio pages of the camera67
compositions, one font) by the same architecture trained on many more pages, five engraving fonts,
simulated camera photos and a fixed binarization in front of the network (the user's choice: train on
binarized simulated photos; the camera is not adapted yet and binarization removes most photo clutter).
Drop-in: same ONNX interface (`input` [1,1,160,320] ink = (255-tile)/255, `output` logits at stride 2,
opset 17), same thresholds file format, so nothing else in the pipeline changes.

## Data (`make_renders.py`, `vrender.py`, `svg_labels.py`)
* Pieces: PDMX rated + deduplicated, 1–2 tracks, 12–300 bars: 2,600 picked (1,950 piano, 650
  single-staff instruments); camera67 / board compositions excluded by title; split by piece (hash):
  4% val, 4% test.
* Each page: Verovio 6.3 in one of Leipzig, Bravura, Gootville, Leland, Petaluma; random layout density
  (page width 1800–2600 units), margins, staff / system spacing; raster at 18–26 px per staff space
  (resvg).  Labels from the SVG with Codex's own label code (`prepare_symbols.labels`,
  `relation_data.extract`, copied verbatim into `svg_labels.py`; only imports and font loading changed):
  12 symbol classes (heads, rests, accidentals, dots) as points, 6 stroke classes (stem, beam, barline,
  flag 1–3) as masks.  Half-size masks were tried for speed and rejected (IoU 0.77–0.83 vs full size on
  2 px lines).
* `render_test.png`: one piece in all five fonts with its labels.

## Camera simulation and binarization (`sim.py`, `nets.py`)
* Per page visit one simulated photo: perspective (corner jitter up to 3.5%), rotation +-3 deg, zoom
  0.7–1.2, ink spread / thinning, paper and ink levels, smooth lighting field, shadow edge, vignette, blur
  0.2–1.6, noise 1–8, JPEG 50–95; 30% "mild" like camera67 (rotation 0.3 deg, blur <= 0.4, JPEG >= 88).
* Tiles are cut exactly as `staff_scale_inference.infer` does: the photo resized so the staff space is
  16 px (+-3% estimation error), INTER_AREA, 160x320.
* `BinNet` = Codex's `build_net` (copied, ch 12, 60,288 / 60,210 parameters, same as Codex's models) with
  S_bin's fixed binarization in front: ink = gray < boxmean31(gray) - 10/255 (box mean as two 1-D pools,
  exact for the valid-window mean).  ONNX output equals torch to 1e-6.
* `sim_test.png`: simulated tiles (mild / harsh) with labels and the binarized view the network sees.

## Training (`train.py`) and calibration (`calib.py`)
* Each crop gives both targets; symbol (Codex's focal loss on Gaussian peaks) and stroke (Codex's
  BCE + dice) nets train side by side on the RTX 5080; crops anchored on symbols (class-balanced), on
  stroke pixels, or anywhere; shuffle buffer so a batch mixes many pages.
* Thresholds chosen per class on validation photos exactly as Codex did (symbol F1 within 5.6 px over
  .15–.6, stroke pixel F1 over .2–.7); `calib.py --baseline` scores Codex's current models on the same
  held-out photos.

## End-to-end evaluation (`eval_e2e.py`)
camera67 through `omr_proto/stage_c.py` (unchanged, its MODELS pointed at our folder) -> `proto.py
--pitch-ours --rn 0.5 --csolve` -> camera67 scorer; the board page through the v13 release package
(desktop dry run).  Baseline check: Codex's models give 243/263 on the board page via this path.

## Results

Run r1 (`train.py r1`, 24k steps x 64, 75 min; 3,320 train pages): on simulated held-out photos the new
models reach F1 0.99+ on every symbol class (Codex's models on the same photos: sharp 0.26, filled head
0.80; not a fair comparison, our photos are out of Codex's domain).  End to end:

| symbol / stroke model | link margin | camera67 notes | board page |
|---|---|---:|---:|
| Codex / Codex (v13) | .35 | 89.85% | 243/263 |
| Codex / Codex | .5 | 90.59% | 243/263 |
| **new r1 / Codex** | .35 | 90.11% | **262/263** |
| **new r1 / Codex** | **.5** | **91.18%** | **262/263** |
| new r1 / new r1 | .35 | 21.00% | 262/263 |

* Board page: all 17 sharps and the 1 natural detected (Codex's model: 9 sharps, 7 sharps as naturals);
  the only error left is the F4 of m1 read as E4 (head half-way between two steps).  The gain comes
  from the symbol model alone.
* New stroke model: 46 camera67 pages get a wrong measure count.  It splits a barline next to a beamed
  group into "barline" (lower half) and "stem" (upper half), and calls a long stem that crosses the
  staff a barline; `bar_bounds` needs the barline probability over most of the staff height
  (mean >= .55, >= 45% of samples > .6).  Codex's stroke model was trained on these very compositions.
* Link margin: Codex's `link_heads` only links a head lying at most .35 staff spaces beyond the detected
  stem end.  Codex's symbol model places heads ~0.3 px low/right (measured against exact labels), which
  happened to keep stem-down heads inside; ours are unbiased and half of those heads fell outside
  (camera67: -278 links, +252 stemless heads, their rhythm hint drops to a quarter).  A head centre
  0.25-0.5 spaces beyond the detected stem end is normal geometry, so the margin is a parameter
  (`relation_fast.LINK_MARGIN`, `run_page.py --link-margin`, default .35 = exact v13 output, checked
  on the board photo in check mode); 0.5 helps both model sets.
* r1 symbol model per-tile cost on the desktop +3% (stroke +14%) for the binarization pools.

Stroke run r2 (`CtxNet`: two dilated convolutions at 1/8, 101,874 parameters; anchors stem 3, beam 2,
barline 4, flags 1): camera67 76.19% (20 pages with a wrong measure count, nearly all missing bars),
board page 262/263.  Split by Codex's own symbol/stroke data split (its models trained on 49 camera67
pages and selected thresholds on 5):

| symbol / stroke (margin .5) | held-out 13 p | validation 5 p | Codex-train 49 p | all |
|---|---:|---:|---:|---:|
| Codex / Codex | 94.84 | 90.40 | 86.57 | 90.59 |
| new r1 / Codex | **95.40** | **91.06** | **87.18** | **91.18** |
| new r1 / new r2 | 94.14 | 64.17 | 61.40 | 76.19 |

So camera67 is largely in-sample for Codex's stroke model; on pages it never saw, r2 is 1.3 points
behind.  The missed bars are tile-grid effects: the recognizer cuts the page into fixed 320x160 tiles
(steps 288 x 128), a staff (64 px) is often split between two tile rows, and our crops always kept the
anchor >= 28 px inside the tile; shifting the page by 64 px raises the bar evidence of a missed barline
from mean .27 to .53.  Also found: tiles beyond the photo are padded with 255, which the binarization
turns into a line at the right / bottom image edge (outside the staves, harmless for bar_bounds).

Stroke run r3 = r2 + half the crops on the recognizer's own tile grid (`--grid-share 0.5`), anchors
anywhere in the tile (`--edge-anchors`), 255 padding beyond the photo (`--pad-white`): simulated test
F1 barline .715 -> .872, beam .806 -> .914; camera67 held-out 95.56% (Codex stroke 95.40%) but Codex-train
pages 33.78% (26 pages miss bars), all 66.17%; board 262/263.  Its barline output depends strongly on
where the tile cuts (the same camera67 barline 0.01–1.0; r2 0.77–1.0, `edge_test.py`), and the misses
sit on single-staff parts with whole-rest bars, which the mostly-piano data lacks.  Added 1,462 pages
of single instrument parts cut out of PDMX ensemble scores (`make_renders.py list-parts / render-parts`,
keys `P*`), and stroke run r4 = r2 + parts + edge anchors + white padding (no grid crops), 12 loader
workers (24-thread budget, 65 s per 1,000 steps).

r4: barline evidence 0.97–1.0 wherever the tile cuts (`edge_test.py`); simulated test F1 stem .846,
beam .913, barline .873; camera67 held-out **95.80** (Codex stroke 95.40), validation 90.26 (91.06),
Codex-train 71.18 (87.18), all 83.84; board 262/263.  Best stroke model so far, but still misses bars on
single-staff orchestral pages that Codex's stroke model was trained on.

## Locked combination v14 (`locked_v14/`, MANIFEST.json there)
New symbol model r1 + Codex's stroke / edge / grace models + link margin 0.5:
`run_page.py ... --relation-models locked_v14/relation_models --link-margin 0.5`.
camera67 91.18% (v13 89.85), board page 262/263 (v13 243), board 62.0–62.1 s (v13 62.7).
r4 stays the starting point for the shared network's stroke head.

## Locked combination v15 (`locked_v15/`, MANIFEST.json there; 10-27)
sh8s = shared network sh8 (SharedNetS2D, `runs/sh8`) with 0.2 subtracted from the staff logit
(`staff_offset.py`; why: SHARED_NET.md "sh7 vs sh8" and "sh8s"), its calibrated relation_models, link margin 0.5:
`run_page.py ... --shared locked_v15 --relation-models locked_v15/relation_models --link-margin 0.5`.
camera67 94.35% with `eval_pages.py` (v14 92.04, sh7 92.77, sh4 94.64), no page zeroed; desktop board page
262/263; board time = sh8's (single 31.6–32.4 s, resident 28.6–28.9 s).  Release:
`../../release/TF_Free_OMR_AC920_20261027_v15/` (`../board_trial/make_release_v15.py`); board-checked from its tarball (31.9 s single, 28.8-29.1 s resident, MusicXML = desktop), sealed.

## On the board (10-26 evening, `../board_trial/board_vA/`, `/home/fpga/claude-routeA-20261026`)
v13 release package + `run_page.py --relation-models DIR --link-margin M` + `relation_fast.LINK_MARGIN`:

| run | configuration | page | relation | decode | MusicXML | board page notes |
|---|---|---:|---:|---:|---|---:|
| 1 | v13 flags | 62.7 s | 28.8 | 4.3 | 60f3d82b (= v13) | 243/263 |
| 2 | + margin 0.5 | 61.1 s | 27.7 | 3.8 | 60f3d82b | 243/263 |
| 3 | + new symbol model r1 | 62.1 s | 29.3 | 3.3 | a7f114d4 | **262/263** |
| 4 | repeat of 3 | 62.0 s | 29.2 | 3.4 | a7f114d4 | **262/263** |

The new symbol model costs ~0.5–1 s in relation (binarization pools), decode is a little shorter, the
page stays at ~62 s (run-to-run noise ~1 s).  Run 3's MusicXML equals the desktop result after the
XML declaration.
