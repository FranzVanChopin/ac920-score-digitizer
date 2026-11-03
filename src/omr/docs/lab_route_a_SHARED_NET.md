# Route A step 3: one shared network instead of three (design, 2026-10-26)

## What the three networks cost and feed today (board, v14 page ~62 s)
| network | input frame | runs | board time | consumer |
|---|---|---|---:|---|
| S_bin segnet (68k params, 23.5 GMAC) | Homr frame: page auto-cropped, width 1920, CLAHE; staff space ~12-20 px | 320x320 patches, no overlap (~54) | ~9 s ONNX (+2.6 s patch merge / argmax) | Homr staff detection (5 masks), our clef/key rules (clefs_keys mask) |
| Codex symbol (60k) | relation frame: photo resized so the median staff space is 16 px | 154 tiles 320x160, 32 px overlap | ~6.5 s | assign_graph / heads, accidentals, rests, dots |
| Codex stroke (60k) | same | 154 tiles | ~6.5 s | Hough stems / bars, beams, flags, bar_bounds |
Plus tile post-processing (per-tile resize, dilate peak NMS, global NMS) ~3-5 s single-core.

The relation frame needs the staff space, which comes from Homr's geometry, which needs the segnet:
so one shared pass has to run in Homr's frame (before any geometry), with symbol / stroke heads
trained over the staff-space range of that frame (12-20 px), and its maps carried into the relation
frame by the known photo <-> Homr-frame transform (autocrop offset + resize factor).

## Network
* Backbone = S_bin's depthwise-separable U-Net (widths 16,32,48,64,96, fixed binarization in front),
  initialised from S_bin's trained weights, so the segnet head starts exactly as production.
* Heads: segnet 6 classes at full resolution (S_bin's own head); symbol 12 + stroke 6 at stride 2
  from the 1/2 decoder level (1x1 convs; Codex's heads are also at stride 2).
* Patches 320x320 with a 16 px halo (352x352 input, centre kept), so symbol / stroke objects are not
  cut at patch borders (route A r3/r4 lesson); ~+21% over S_bin's 23.5 GMAC -> ~28 GMAC per page,
  ~11 s on the A53 instead of ~22 s for the three networks.

## Training (`shared_*.py`)
* Inputs exactly as the pipeline makes them: simulated photo (sim.Photo, full page) -> Homr's autocrop
  (copied) -> width 1920 -> CLAHE(1.0, 8x8) -> 352x352 crops.
* Targets: segnet = S_bin teacher run on the same crop (keeps Homr's staff rules seeing the masks they
  see today); symbol heatmaps / stroke masks = SVG labels mapped into the Homr frame (exact, all
  five fonts plus the single-part pages).
* Losses: S_bin's segnet loss (distillation) + Codex's symbol focal loss + Codex's stroke BCE+dice.

## Integration (run_page, our code only)
* `segmod.inference` (already wrapped to capture the clef mask) runs the shared ONNX once, returns the
  5 masks to Homr and keeps the symbol / stroke maps.
* `relation_fast`'s copy of `infer` turns those maps into what `recognize` expects (symbol points with
  per-class thresholds and Codex's NMS, stroke probabilities as uint8 in the legacy graph frame)
  instead of running 2 x 154 tiles.

## Status (10-26 evening)
* Code: `shared.py` (net, Homr frame, export), `shared_train.py` (sh1: 30k steps x 32, 12 loader workers,
  ~0.3 s/step, data-bound by whole-page photo simulation), `shared_calib.py` (thresholds -> package/),
  `shared_runtime.py` (run time, copied as `sacc/`), hooks `run_page.py --shared DIR` and
  `relation_fast.SHARED` (default None: v13/v14 outputs unchanged, checked on the board: SHA 60f3d82b /
  a7f114d4), `eval_pages.py` (camera67 through run_page: v14 = 91.36%, 2 pages fail in Homr as always).
* sh1 at 3k steps: segnet head agrees with S_bin on 98.95% of pixels (clefs/keys IoU 0.82, recovering);
  the interim checkpoint already runs end to end (board page 237/263 on the board).
* **Board timing** (`board_trial/board_vS`, `/home/fpga/claude-routeA-shared-20261026`, same session):
  v13 62.2 s, v14 62.4 s, **shared 48.7 s** (segment 22.2 -> 24.0, relation 29.7 -> **13.3**); peak RSS
  1284 -> 1784 MiB (ORT arena now off).

## sh1 result (30k steps, 2.3 h)
* Segnet head vs S_bin: 99.30% pixels, IoU stems/rests .86, heads .94, clefs/keys .87, staff .94, symbols .86.
  Simulated test F1: every symbol class .986-.999 (rest_32nd absent), stroke .92-.96.
* camera67 through run_page (`eval_pages.py`), against v14 the same way: common 64 pages **93.14% vs 91.39%**;
  all 67 pages (failures = 0) 91.63 vs 91.36.  Big gains where v14 lost the bar count (score15 +292,
  score05 +101); losses: score13 p1 -154 (the clefs/keys mask lost the small 8 of a treble-8 clef, so the
  clef rule read plain treble), score14 p1 crashed (Homr found a 1.3-space staff fragment at the page edge,
  Codex's bar_bounds failed on it).  Board page 257/263 (v14 262): five heads one step off (one stride-2
  cell of the head heatmap), all 17 sharps right.
* Fixes: `run_page --shared` drops staves shorter than 4 spaces and renumbers systems (score14 p1 -> 245/270,
  v14 243); sub-pixel peak refinement (`shared.json` "subpixel", parabola through the neighbours: peak
  error <= 0.1 px instead of <= 1 px); sh2 = sh1 fine-tuned 12k steps with soft distillation (KL to S_bin's
  softmax) for the segnet head (clefs/keys IoU .87 -> .90 after 1k steps).

## Board round 2 (10-27 morning, `board_trial/board_vS2`, `/home/fpga/claude-routeA-shared2-20261027`)
ORT memory arena off for the shared session, staff-fragment guard, sh1 package:
v14 67.4 s (first run after a reboot, imports 4.8 s), **shared 50.1 / 49.9 s**, main-process peak RSS
**980 MiB** (arena on: 1784; v14 1283), board page 257/263 = desktop (SHA 7e914f74).  Arena off costs
~1.4 s.  Step timers (run3): Homr `inference` (the shared pass) 15.4 s wall, detect_staff 4.4,
resize 1.4, predict_symbols 1.2, braces 0.5; relation 13.4 s.

## sh2 (sh1 + 12k steps soft distillation, sub-pixel decoding)
Segnet head vs S_bin 99.52% (clefs/keys IoU .91, sh1 .87).  Board page **263/263** (v14 262, Homr 263).
camera67 (all 67 pages, failures = 0): v14 91.36, **sh1 + staff guard 93.10**, sh2 91.29.  sh2 - sh1 is
dominated by one page: score32 p2 (-278, one barline between tied chords missed -> 49 of 50 bars, page
scored 0).  score13 p1 is -154 for both: the 8 under a treble-8 clef is still lost from the clefs/keys mask
(S_bin keeps it; treble-8 clefs are rare in the renders).  Next: sh1 with sub-pixel decoding (sh1s) to
separate the decoding from the model; treble-8 renders.

## sh1 with sub-pixel decoding (sh1s) and sh3
* sh1s: camera67 **93.52%** (sh1 grid 93.10, v14 91.36), board page 252/263 (10 heads half-way between steps).
  Head scatter on the board page (|offset from a staff step| > 0.3 half-space): v14 13 heads, sh1 26-28,
  **sh2 9** -> sh2 places heads most precisely (board page 263/263).
* Treble-8 renders: `make_renders.py render-octave` turns the treble clefs of single-instrument parts into
  treble-8 and lowers every pitch an octave (same page apart from the small 8): 914 pages (841 train).
* sh3 = sh2 + 10k steps with the treble-8 pages, segnet weight 1 + soft distillation 10.

## sh3 and the treble-8 rule (10-27)
* sh3: camera67 93.03% (score32 p2 back to 282), board page 258/263 (4 heads half-way between steps).
* score13 p1 (treble-8) was not a network problem: `headers.clef_of` looked for the 8 in a box under the
  clef component's bottom, and the component (shared head; also S_bin in 4 of 13 systems) can include
  the 8, pushing the box below it.  Fallback added: the 8's enclosed holes 1.4-3.8 spaces under the bottom
  line (a treble clef's tail ends in a filled ball, no hole).  camera67: only score13 p1 changes (170/16 ->
  282 notes): v14 91.36 -> **92.04**, sh3 93.03 -> **94.62%**.
* sh4 = sh3 + 10k steps with symbol Gaussians centred at the exact sub-cell position (peak cell kept at 1
  for Codex's focal loss), so the sub-pixel refinement has something to read (board-page borderline heads).

## Speed work on the shared path (10-27 afternoon)
Board profile (`board_trial/board_vP`, sh3, ~50 s): segment 25.9 (shared inference 15.3, detect_staff 4.5,
resize 1.4, predict_symbols 1.1, boxes 0.8, braces 0.5, filter 0.4, autocrop 0.3), relation 13.6
(HoughLinesP 5.3 CPU-s in 2 threads, full-map sigmoid 1.5, photo imread x2 1.1, Codex graph code),
redecode 5.1 (another imread 0.5), decode 3.4 (waiting for the 4 workers), imports 2.0.
Changes (all in `shared_runtime.py` / `run_page.py --shared`, the v13/v14 paths untouched):
* symbol peaks on the logits (sigmoid only at peaks / neighbours), stroke probabilities through a
  1024-entry LUT (<= 1 uint8 level from rint(255 sigmoid)); photo decoded once per imread flag (cached,
  copies returned) -> board **48.2 / 46.8 s** (relation 13.6 -> 11.0), RSS 774-814 MiB, board page
  260/263 (= desktop) (`board_trial/board_vF`).
* full-width strips + argmax inside the ONNX (`shared.export_deploy`, dynamic size): desktop -25%, but no
  change on the A53 (the arithmetic dominates) -> run time uses 352x352 patches again, with
* ink-free patches skipped (adaptive threshold 31 / 10 + integral image, < 30 ink px incl. halo): class
  maps and peaks identical on the tested pages; desktop board page 2.89 -> 2.28 s, a near-empty last page
  2.87 -> 0.45 s (board not yet measured).
* `--fast-resize` (Homr's resize with OpenCV INTER_AREA, ~1.3 s on the board): board page 260 -> 257 with a
  model trained on PIL frames, so `shared_train.py / shared_calib.py --area-resize` train / calibrate on
  INTER_AREA frames and the package's shared.json ("resize": "area") switches run_page to it.

## sh4 (sub-cell targets)
camera67 (with the treble-8 rule, patches + skipping) **94.64%** (sh3 94.62, v14 92.04), board page
**261/263** (sh3 258).  sh5 = sh4 + 8k steps on INTER_AREA frames (`--area-resize`), for `--fast-resize`.

## sh5 (INTER_AREA frames)
camera67 91.16% (sh4 94.64): two page events -- score32 p2 lost one bar again (49/50, page scored 0; also
sh2), score11 p1 crashed in Codex's redecode export (`relation_export.export_relations`: a staff with fewer
measures than its system).  Other pages within +-26 notes of sh4.  Board page 258/263.  run_page now
falls back to decoding the first-pass graph when redecode raises (pages where redecode works unchanged):
score11 p1 0 -> 222/398 (sh4 with redecode ~367).  camera67's all-or-nothing page scoring makes single
checkpoints jumpy; sh4 stays the reference until a variant beats it.

## Where the network spends its time on the A53 (ORT profiler, `board_trial/vO_extra/ort_profile.py`)
8 patches 2.12 s; full-resolution parts ~54%: d0 dw 13.6% + pw 11.8%, stem 12.5%, full-res concat 4.6% and
upsample 4.1%, head 3%, argmax 2%, binarization pool 1.8%; 1/2 level (e1, d1, aux) ~28%; deeper levels ~18%.
`SharedNetHalf` (no d0; segnet head on the 1/2 decoder level, logits upsampled bilinearly): board **1.348 s
vs 2.112 s per 8 patches (-36%)**, i.e. ~13.5 -> ~8.6 s per page.  sh6 = half architecture, from sh5,
15k steps (new segnet head; d0 dropped).

## sh6 (half architecture) and why the frames go back to PIL (10-27 evening)
* sh6: camera67 **93.64%** (sh4 94.64), board page 262/263.  Two causes, separated:
  * score11 p1 (-156) is the INTER_AREA frame, not the weights: sh4's weights with `--fast-resize` crash in
    the same redecode export on that page (`eval_pages/sh4_area_probe`, 101 heads repitched).  So
    `--area-resize` / `--fast-resize` are dropped (worth ~1.3 s on the board, not worth a fragile page).
  * 8 opening meters lost (pages whose first meter was not read: sh4 1, sh6 9): Codex's meter reader
    (`repair_prefix_symbols.detect`, template score >= 0.87 on the staff-flattened photo) is sensitive to the
    staff polylines.  Cross test (`meter_x.py`: staff geometry of one model with the bar bounds of the other)
    -- the meter follows the staff geometry only.  sh6's bottom staff line drifts ~4 px over the clef
    (391 -> 395 px over x 120..230): the bilinearly upsampled 1/2-level logits lose the thin lines there.
* `SharedNetShuffle` (arch "shuffle"): the half model with a pixel-shuffle head (1x1 conv to 6 x 4 logits per
  1/2-level cell, one per pixel of its 2x2 block), initialised from head_h on all four sub-pixels.  Deploy
  ONNX: argmax per sub-pixel at the 1/2 level, then the 2x2 interleave on uint8 (a float DepthToSpace of
  the six full-resolution maps cost +24% on the desktop; this form costs the same as the half head and
  matches torch's pixel_shuffle + argmax exactly).  sh7 = shuffle from sh6, 10k steps, PIL frames.

## sh7 (pixel-shuffle head, PIL frames) and a steadier way to compare models
* camera67 **92.77%** (sh4 94.64), board page **262/263**.  score11 p1 is back (362, PIL frames); the loss is one
  page zeroed: score09 p1 76/77 bars -- system 7's barline at legacy x 1275 is only 37% above threshold in
  sh7's barline map (sh4 80%) and reads as stem there (stem map 243).  Opening meters: still 10 first pages
  without a read meter (sh4 1), now through 0.5-0.7 px shifts of a staff line over the meter (Codex's
  template reader is that sensitive; `meter_x.py` again puts it on the staff geometry); they cost few notes
  (meter from content is mostly right; score00 even +15).
* camera67 scores a page 0 when its bar count is off, so one bar decides ~300 notes.  On the 61 pages that no
  model zeroed: v14 94.93, sh3 94.91, sh4 95.04, sh5 95.25, sh6 95.11, **sh7 94.93%** -- the shared models are
  equal per note; they differ in 0-2 zeroed pages each (v14 05/1 15/1, sh4 06/2, sh5 32/2, sh7 06/2 09/1).
* Board (`board_trial/board_vU`, all exact speed-ups below): sh7 single run **33.7 / 33.4 s** (segment 19.3,
  relation 3.5, redecode 4.5, decode 3.3, imports 2.8; RSS 615 MiB); resident: board page **30.8 / 30.7 s**,
  sparse camera page 18.8 s; all six sh7 board outputs identical (SHA 43ee05f1), also with `--no-prefetch
  --no-cresize` (40.4 s).  sh4 with the same code 38.1 s (SHA 2ab0fd64 as before), v13 flags 60f3d82b.
* s2d (`SharedNetS2D`: binarized input folded 2x2 into 4 channels, stem at 1/2, no full-resolution convolution):
  board 8 patches half 1.36 / shuffle 1.375 / **s2d 1.236 s** (-10%, `board_trial/vO3_extra`).

## sh8 (s2d stem, from sh7, 12k steps)
* camera67 **91.66%**, board page **263/263** (first full score).  Zeroed: 31/2 (system 5 split into two staves,
  the extra one full-width) and 32/2 (a floating 5-line group, x 747-956 / ~9 spaces, between systems 5 and 6):
  Homr's staff detection on sh8's staff masks, not bars.  On the 59 pages no model zeroed: sh4 95.32, sh7 95.01,
  **sh8 94.83**, v14 94.92%.  Opening meters missed on 12 first pages (meter from content).
* Board (`board_trial/board_vW`): single **32.4 / 31.6 s** (sh7 32.7 in the same session), resident board page
  **28.6-28.9 s**, camera page 17.2 / 17.6 s (sh7 29.5-29.7 / 18.5-18.7); outputs identical within each model.
* Resident memory with `malloc_trim(0)` between pages: RSS at page start sh7 335 -> 411 -> 496 -> 500 -> 498 MiB,
  sh8 334 -> 394 -> 449 -> 493 -> 546 (still rising at page 6); peak ~1.0 GB after 6 pages.  Desktop (Windows)
  levels at ~388 MiB, so it looks like glibc arenas (new prefetch threads every page) -- next: MALLOC_ARENA_MAX=2
  and a 20-page run.

## sh7 vs sh8: where the speed and the accuracy differences come from (`sh78_diag.py`)
* Speed: the stem is the only difference.  Conv MACs per 352x352 patch: sh7 408M (stem 67M at full resolution),
  sh8 371M (stem 30M at 1/2), -9%; the full-resolution 16-channel tensors go too.  Board resident (`board_vW`):
  segment 18.9 -> 18.0 s, every other stage within 0.1 s (relation 2.79 / 2.76, redecode 4.08 / 4.17, decode
  3.32 / 3.28).
* Accuracy: synthetic validation is the same (teacher agreement 0.9949 both, symbol / stroke F1 within 0.003).
  camera67's 185-note gap: 142 from which pages are zeroed (sh7: 09/1 316 + 06/2 30 notes; sh8 fixes both and
  zeroes 32/2 322 + 31/2 201), 43 from the rest (sh8 worse on 20 pages, better on 16; sign test p = 0.62).
* Class maps on the 67 camera67 frames vs sh4 (full-resolution d0 + head):
  * 1 px shift stability (pixels whose class changes when the frame moves 1 px, shifted back): staff dy sh4 2.8%,
    sh7 3.6%, sh8 3.3%; stems_rests dx 4.7 / 6.5 / 6.1%; clefs_keys 3.5 / 9.5 / 8.6%.  sh8 is steadier than sh7
    on 67/67 pages (the s2d stem sees all four phases; sh7's stride-2 depthwise conv drops three of four).  The
    loss against sh4 is the 1/2-level head that both share.
  * Staff class: sh8 labels slightly more pixels staff where sh4 does not (1.59% vs sh7 1.48% of staff px, on
    64/67 pages; missing equal 1.63%).  All three models mark ledger lines as staff (the teacher's labels); on
    sh8's two zeroed pages its extra staff pixels are +9% / +11% over sh7's, and Homr's detect_staff builds a
    five-line group from stacked ledger lines (32/2: the chords at x 747-956; 31/2: the fff passage above
    system 5).  sh7's zeroed page is a different event (a barline stroke at 37% of threshold).
  * Not separated: architecture vs the 12k extra training steps (no control run).

## sh8s: sh8 with a staff-logit offset (`staff_offset.py`, `staff_sweep.py`)
* `staff_offset.py SRC DST DELTA` subtracts DELTA from the staff class's four sub-pixel biases of head_ps in the
  deploy ONNX (= a logit offset before the in-graph argmax); delta 0 gives identical maps, the aux output is
  untouched, only staff pixels can change class.  Same graph and cost as sh8.
* Class maps vs sh4 on the 67 frames (`eval_pages/staff_sweep.json`): no single offset brings sh8 closer to sh4 --
  staff px / sh4 0.9996 at 0, extra + missing 3.22% at 0, 3.34% at 0.1, 3.58% at 0.2 (it trades extra for
  missing; sh8's total is already sh4's, its extra pixels sit on ledger lines).  On the two zeroed pages the
  extra staff pixels fall 4812 / 5190 -> 3219 / 3407 at 0.2 (sh7 4402 / 4667).
* camera67 end to end (offsets chosen after the diagnosis, on camera67 itself -- two values tried):

  | model | camera67 | zeroed pages | 63 common pages | desktop board page |
  |---|---|---|---|---|
  | sh8 | 91.66% | 31/2, 32/2 | 94.77% | 263/263 |
  | sh8 + 0.1 | 93.32% | 31/2 | 94.71% | 262/263 |
  | **sh8 + 0.2 (sh8s)** | **94.35%** | none | 94.76% | 262/263 (F#4 -> G#4, one head) |
  | sh7 | 92.77% | 06/2, 09/1 | 95.04% | 262/263 |
  | sh4 | 94.64% | 06/2 | 95.42% | 261/263 |

  The gain is the two pages; elsewhere 9 pages better / 12 worse by 1-7 notes.  The two real_sonata pages
  (set "real") cannot tell the models apart (every model 71-78/277 and 3-6/425).  Higher offsets were not tried
  so as not to tune further on camera67.
* **Locked as v15** (10-27): `locked_v15/` (MANIFEST.json), release `release/TF_Free_OMR_AC920_20261027_v15/`.

## Exact speed-ups of the shared path (10-27 evening, `run_page.py --shared`, `shared_runtime.py`)
Board profile of sh6 (`board_trial/board_vT`, run6p): relation 12.3 s profiled, of which the two stems / bars
`strokes()` (HoughLinesP on the full legacy-size maps) kept the stage waiting 5.9 s, the stroke maps + peaks
1.5 s, ORT session creation 1.1 s (recognize() builds symbol / stroke / edge sessions; the shared path never
runs the first two), `map_photo` 0.5 s (it repeats Homr's autocrop).  Changes, none of them changes a value:
* **prefetch**: right after the segnet pass, background threads compute the symbol peaks, the six stroke
  maps and Codex's stems / bars `strokes()` (the same installed function, same thresholds read as
  recognize() reads them), overlapping Homr's single-threaded staff rules.  The relation stage takes them
  only if the photo shape, Homr's autocrop box (captured at the segnet call as map_photo computes it),
  the thresholds and the strokes function are the ones it would use, else computes them as before
  (`--no-prefetch` forces that).  HoughLinesP uses its own fixed-seed RNG, so the thread does not matter.
* **lazy, kept ORT sessions**: Codex's OnnxModel creates its session on first use and keeps it for the
  process (key: resolved path + sha256 of the bytes; same options as Codex's); edge / grace are created in a
  background thread during the segment stage.
* **autocrop reuse**: map_photo's autocrop returns Homr's captured box when the image equals (array_equal)
  the one Homr cropped.
Checks: board page MusicXML + frontend / relation graph + staff / geometry identical to the old code (desktop);
camera67 with sh4: **65/65 pages MusicXML identical** to `eval_pages/sh4h` (`eval_pages/sh4_pf`,
`sh4_pf2`; prefetch used on all 65; the 2 others stop in Homr as before); v13 flags (no --shared): SHA
d01213b3 and all intermediates unchanged.  Desktop relation 0.85 -> 0.42 s.

* **Codex's geometry index** around recognize() and redecode(): `route_trials.relation_geometry_index.geometry_index`
  (active in Codex's own board runtime, `board_relation_runtime.enhance`) caches np.array(polyline) per staff line
  (read-only, marshal signature checked on every hit); `staff_step` / `indexed_array` otherwise rebuilt the
  arrays 271 x 5 times per stage (~0.7 s profiled in each of relation and redecode on the board).
* **grayscale photo decode at page start**: the relation stage reads the photo a second time with the grayscale
  flag (0.53 s on the board for the PNG); the same `imread(path, 0)` now runs in a background thread when the
  page starts and the cache waits for it.
  Checks for these two: board page all intermediates identical; camera67 with sh4 65/65 MusicXML identical to
  `sh4h` (`eval_pages/sh4_pf3`).
* **C++ Pillow resize** (`pilresize.cpp`, `pil_resize.py`, in the package as `sacc/`): Homr's resize to width
  1920 (Pillow 12.3.0 `Image.resize`, bicubic: double coefficients -> 22-bit fixed point, horizontal pass, then
  vertical) re-implemented with the same integer pixel passes, run on 4 row ranges in Python threads (ctypes
  releases the GIL).  Pillow's aarch64 build may fuse multiply-adds in the coefficient arithmetic: with the box
  starting at 0 the centres / bounds are one rounded product either way, the bicubic polynomial may differ in
  its last bits, so any fixed-point tap within 1e-6 of its rounding boundary sends the image to Pillow
  (~1% of size pairs).  Desktop: 278/278 crops identical (71 photos x autocrop + 3 random views), 6 left to
  Pillow, 0 different; 21.7 -> 4.8 s over the set.  Installed only on the shared path with PIL frames
  (`--no-cresize` keeps Pillow).  Board check: `board_trial/vU_extra/resize_check.py`.

## Resident mode and ink-free patch skipping on the board (`board_trial/board_vR`)
* `run_page.py --serve`: set-up once (imports, patches, sessions; the patches wrap functions, so they must
  not be applied twice), then one page per stdin line `PHOTO<TAB>OUT[<TAB>METER]`, a JSON report per page.
  Single-page mode unchanged (v13 flags: desktop SHA d01213b3 as before).
* Board, sh3 package with skipping: single run **46.4 s** (segment 25.7 -> 23.8); resident: ready after 2.7 s,
  then the board page **42.9 / 42.5 s** per page and a sparser camera67 page 26.3 s; the three board-page
  outputs identical (SHA 0d3defd7, 258/263).  v14: ~62 s -> about -31% per page.

## Acceptance
Each head offline against its current counterpart (masks vs S_bin, symbol / stroke F1 on simulated
held-out photos), then camera67 + board page end to end, then board time.  Target: accuracy of v14
(91.18% / 262), page time ~62 -> ~47 s.
