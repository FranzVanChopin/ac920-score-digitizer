# Transformer-free OMR prototype (2026-10-24, Stage A)

Question: replace Homr's transformer (60 M params) by small symbol models + our 0.3 M-param
measure decoder — how much camera67 accuracy is kept?

Route: Codex's relation graph (its small ONNX models: symbol 12 classes, stroke 6 classes, head–stem
edge, grace; ~0.13 M params total) -> `extract.py` (measures, events, hints) -> production decoder
(beam + exact DP + pitch-swap repair + column-joint DP + repeat consensus; fold model that never
trained on the piece) -> MusicXML -> the camera67 scorer (`pilot.evaluate`).

**Stage A still uses Homr for two inputs:** staff geometry (Homr segnet + rules, no transformer;
the small segnet student can replace it, see `../small_segnet`) and the clef/key context the graph
used to turn head positions into pitches (Homr transformer headers). Meter is NOT taken from Homr:
graph meter glyphs, else inferred from bar content.

## Extraction rules (`extract.py`)
* measures = graph barline bounds per system; groups by x; graces skipped
* hints: hollow head -> half (stem) / whole (no stem); filled -> quarter + max(beams, flags); rests
  by glyph; `dotted`; printed tuplet number -> warp
* chord merging: stemless groups join a stemmed (or stemless same-value) group within 1.3 staff spaces
* measure rests: a staff whose only event is a whole-rest glyph -> onset 0, whole bar (engravers
  centre it; the decoder corpus has no 3/4 measure rests)
* x: head column (stem side) -> `column_snap` -> 0..1000 per measure
* meter: graph meter glyphs; with none read so far, the bar length most single-voice staff-measures'
  hints add up to (16 orchestral-part pages had no meter glyph and silently got 4/4 before this)

## Results (camera67, 67 pages, 16,687 notes)

| arm | notes | rests |
|---|---:|---:|
| Homr pipeline (accepted) | 97.56% | 98.69% |
| Codex rule decode of the same graph | 80.64% | 88.69% |
| **ours (graph events + our decoder)** | **87.80%** | **89.29%** |

Progress while fixing extraction: 74.3 (first run, 5 pages) -> +measure rests -> +chord merging ->
+meter from content = 87.80 on all pages.

By material: piano/other 20 pages 90.4% (Homr 98.1); string parts 8 pages 77.2% (95.5);
orchestral parts 39 pages 86.2% (97.2).

Where the remaining ~9.8 points go (ours): pitch-in-measure 94.5, +duration 90.1, +onset 87.8.
* 3 pages (464 notes, ~2.8 points) score 0: the graph finds 1–2 measures fewer than the reference
  (single-staff parts; multi-measure rests are the likely cause).
* events: 8,725 of 9,009 reference events matched (96.8%); 284 missed, 350 extra or wrong pitch.
* durations on matched events: visual hint 92.8%, Codex rules 92.8%, ours 93.2%
  (decoder fixed 257, broke 220, both wrong 367 — 167 of those implicit triplets).
* 94% of measures have a complete event set; there the decoder only lifts 93.5% -> 94.4%.
  On the simulator it lifts 85% hints to ~99%: real hint errors are CORRELATED (a whole beam group
  missing its tuplet number or a beam), the simulator's are independent per event, so the decoder
  trusts three agreeing wrong hints.

## Stage B (2026-10-25): clef/key without the Homr transformer (`headers.py`, `pitch.py`)
* clef: small segnet student's clefs/keys class, leftmost tall component at the staff start, classified
  by top/bottom against the lines; octave '8' read from image ink under treble/bass clefs (banjo).
  An 'above' check fired on 66 ordinary clefs and was dropped.  Clef + octave: 588/596 systems.
* key: Codex symbol-model sharps/flats in a contiguous run after the clef ending >= 1.5 spaces before
  the first event; "fewer of the same kind than the previous system" = a miss; a run of <= 2 zero
  systems between two equal keys is filled.  Carrying the key over every zero broke 82 real changes to
  C and was dropped; counting clefs/keys-mask components was dropped (meter digits leak into it).
  Key: 481/488 systems (the reference itself disagrees with the print on some viola systems).
* pitch: position on our clef (+ octave mark), alter from Codex's own per-head explicit accidental
  (graph pitch field 4), else the bar's earlier accidental, else our key.  A naive "sign left of the
  head" rule lost 310 notes on chromatic chords (Twilight) and was replaced.
* result: **86.87%** (Stage A 87.80): replacing Homr's clef/key costs 0.93 points.

## Stage C (2026-10-25): staff geometry from the small segnet (`stage_c.py`)
Geometry = Homr's rule-based staff detection on the 68k-param student's masks (the S_bin pipeline
run), map_photo, then the production relation path (recognize at staff scale 16 with placeholder
headers -> our headers from its symbol nodes -> redecode with OUR headers and the production
refinements).  Homr's retry heuristics only rewrite its XML, never the staff polylines, so no
transformer output is consumed.  Result: **86.73%** (rests 88.46%).

| stage | Homr parts still used | notes |
|---|---|---:|
| Homr pipeline | everything | 97.56% |
| A | segnet (geometry) + transformer headers (clef/key) | 87.80% |
| B | segnet (geometry) | 86.87% |
| **C** | **none of its networks (rule code only)** | **86.73%** |

## Decoder retrained with real front-end noise (2026-10-25)
`noise_stats.py` aligns graph hints to the reference on camera67 and measures group-level error rates,
leave-fold-out (rates for fold F use only pieces outside F) -> Deepseek workspace
`real_noise_rates_camera67.json`.  Key finding: **32% of tuplet groups lose their tuplet marking as a
whole group** (203 groups, 4 partial); the iid simulator loses all three notes of a triplet with
p ~ 0.003.  Other rates: whole beam group one value long 1.2%, single beam-group notes one beam
short 2.5% / unattached (quarter) 3.1%, heads 1.6%, rests 5.2%, dots missed 0.7%.

Decoder side (Deepseek workspace, all opt-in, default path verified bit-identical): `src/real_noise.py`
(groups + noise), `visual.synthesise(real_rates=)`, `run_experiment.make_train_candidates(real_rates=,
real_mix=)`, `two_stage --real-noise-mix M` (+ cache tag `|rn<M><sha8>`).  `proto.py --rn 0.5`.

| models | frozen CV (iid sim) | camera67 A | B | C |
|---|---:|---:|---:|---:|
| production | 97.04 | 87.80 | 86.87 | 86.73 |
| real-noise mix 0.5 | 96.45 (−0.59, t −1.27; any-field +0.43) | 89.50 (+1.70 [+0.60,+3.06]) | 88.57 | 88.50 (+1.77 [+0.74,+2.92], 31/6 pages) |

On folds 0–1 pages the decoder's own effect: duration correct 94.8 -> 96.2% (hint 95.2), fixed/broken
hints 82/91 -> 103/70, triplet errors 25 -> 14.  Trade-off: the simulator benchmark drops a little.

## Further extraction fixes (2026-10-25)
* missed barlines (`repair_bars`): a staff's hints adding up to exactly two bars that split exactly at
  an event boundary with >= 2 events per half, or a whole rest beside other events that alone fill a
  bar, is cut in two; an event-less last 'measure' of a system (courtesy key/meter) is dropped.  Both
  conditions were tightened after false cuts on two-voice bars (score02, score06).  Measure count now
  matches on 65/67 pages (score30 recovered; score05, score15 remain).
* key: a page's first system continues the previous page's key for the zero-run fill.
* mid-system clefs by shape: tried, fired on accidental clusters (Stage B 88.6 -> 74.2), OFF.

mix 1.0 (every realisation real noise): frozen CV collapses 97.04 -> 61.39 (it never sees iid errors)
while camera67 Stage C gains only +0.17 [−0.32, +0.70] over mix 0.5 -> **keep mix 0.5**.
A clef-area filter (heads within 3 staff spaces of the staff start are the clef; an event-less first
'measure' before a start-repeat bar is dropped) raised Stage C 88.94 -> 89.77%.

**Current transformer-free result (Stage C, mix-0.5 models): 89.77% notes, 93.44% rests**
(Homr 97.56 / 98.69).  Remaining on aligned pages: accidentals 2.0 points, other pitch 1.2, step 0.6,
octave 0.25, missed/extra 0.5; durations ~2.9; onsets ~1.6; two misaligned pages ~2.5.

## Next steps (expected gain order)
1. Retrain the decoder with realistic correlated hint noise (group-level implicit tuplets and missed
   beams, small drop/spurious rates), noise rates measured on PDMX renders, not on camera67.
2. Multi-measure rests / measure count on single-staff parts (~2.8 points on 3 pages).
3. Stage B: clef/key from the symbol model + a small classifier instead of Homr headers.
4. Stage C: staff geometry from the small segnet student; then the whole front end is small models.

## Files
`extract.py` graph -> events/candidates · `proto.py` decode + MusicXML (ours, codex) ·
`score.py` camera67 scorer + paired delta vs Homr · `diagnose.py` event-level hint/decoder analysis ·
`dump.py` per-measure disagreement dump.  Output: `runs_A/<id>/<page>.{ours,codex}.musicxml`.
