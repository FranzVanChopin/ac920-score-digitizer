# Exact, faster relation stage (2026-10-25)

Goal: cut the board's `relation` stage (~40 s of the 89 s page in v7b) without changing any output
byte.  The stage is Codex's `route_trials.relation_decode.recognize` (board build of 10-24, which
differs from the desktop handoff copy: tests use a read-only copy of the board's program,
`../board_trial/board_codex_app/app`, as the bundle).  `run_page.py --fast-relation` calls
`relation_fast.install(rd)`; Codex's files are never edited.

## Where the time went (board cProfile, v7 run16; 42 s profiled)
* ONNX Runtime 13.9 s: 154 symbol + 154 stroke tiles of 320x160, one after another, 4 intra-op
  threads (42 ms/tile)
* `testset3_stroke_group_cache.strokes` 10.7 s for 2 calls (stems, bars): `HoughLinesP` 9.2 s
  (single-threaded), the group statistics `np.mean` 92k calls
* the rest ~17 s profiled: graph decoding (`decode_staff`, `make_groups`, `beam_relations`,
  `bar_bounds`), photo reads, tile post-processing (resize, dilate, NMS)

## Board micro-benchmark of the tile inference (`bench_ort.py`, run on the A53)
| variant (per model, 154 tiles) | time | logits vs original |
|---|---:|---|
| original: intra-op 4, sequential | 6.54 s | - |
| intra-op 1, sequential | 22.5 s | bitwise identical |
| intra-op 1, 4 tiles in parallel | 6.24 s | bitwise identical |
| intra-op 2, 2 in parallel | 6.29 s | bitwise identical |
| original session, 2 runs in flight | 7.99 s | bitwise identical |

The 4 intra-op threads already scale 3.4x, so tile parallelism gains ~5%: the inference itself is
not where the exact speed-up is.  (Useful fact: these models give bitwise identical logits for
any thread count on the A53.)

## The replacements and why each is exact
| replaced | how | why identical |
|---|---|---|
| `testset3_stroke_group_cache.strokes` | the loop that assigns Hough segments to stroke groups runs in C++ (`relgeo.cpp`, `hr_stroke_groups`); mask, Hough, sort, the final polyfit/confidence stay Codex's code | same filters, same comparisons in the same order, first matching group; Codex recomputes a group's x as `np.mean` of the segment centres after every append: centres are half-integers, so every partial sum is exact in float64 whatever the order and `np.mean` = exact sum / n with one rounding, which C++ computes; min / max of integers kept incrementally |
| `link_heads` (copy of Codex's code) | the stem's x mean and sorted y are computed once per stem instead of once per (head, stem) pair (73,640 pairs on the board page); the distance screen runs on arrays; edge features + edge model on the survivors in stem order | same numbers (numpy float64 per stem, then the same float64 comparisons elementwise); the screen is the exact negation of Codex's `continue` test, NaN included; falls back if a stem's numbers are not float64 |
| `beam_relations` (copy) | the per-row test `mean(prob[1,y,l:r] >= t) >= .32` for all rows of the block at once | a bool row's mean is (count as float64) / width, which `block.sum(axis=1) / width` computes; left/right do not depend on y |
| `assign_graph` (copy, gray=None path, v13) | `staff_distance(staff, x, y)` for all (node, staff) pairs from 2 array `np.interp` calls per staff, used by the per-group cost and the per-node eligible-staff choice | array np.interp returns per element what the scalar calls return (same interval search result, same slope formula) when the polyline x is strictly increasing (else that staff uses the scalar calls); max / divide / sums stay python floats in the original order |
| `bar_bounds` (copy, v13) | the three full-page arrays (vertical-ink dilation, bar-likelihood dilation, horizontal-ink dilation) computed on a crop around the sampled points | dilation is local (margin >= kernel radius + 1, real image borders stay borders); remap is translation-exact: float32 `x - x0` is exact for integer x0 <= x, and OpenCV's subpixel index `round(32 x)` shifts by the even integer `32 x0` |
| `recognize`: the staff-scale inference block (source patch of 5 lines, sha256-guarded) | `infer` copied with the stroke tiles first; the stems and bars `strokes()` run in two threads while the symbol tiles run | the three computations are pure functions of their inputs (separate ONNX sessions, no shared state; `HoughLinesP` uses its own fixed-seed RNG and releases the GIL), so running them in another order or concurrently cannot change a value |

Guards: nothing is installed unless the sources of `recognize`, `infer` and `strokes` have the
reviewed sha256 (`SOURCES`), the 5-line block is found exactly once and the C++ library loads.
`HOMR_FAST_CHECK=1` also runs Codex's original `infer` + `strokes` and `strokes` itself and raises
on any difference (arrays bitwise, stroke lists with ==).

## Results
* desktop: `verify_relation.py` 70/70 identical (runs/verify1 = v8, verify2 = v8b; 2 pages stop in
  Homr before relation, identically); check mode passed on every page.
* board (`../board_trial/board_v8/`): relation 40.5 (v7b flags, run20) -> 36.1 (v8, run21/22) ->
  **30.0 / 30.1 s (v8b, run25/26)**; page 89.7 -> 78.9 s; check-mode runs (run23, run27) passed;
  MusicXML, geometry, frontend and relation graph SHAs unchanged.
* v8c (rejected): numpy separable max filter for the symbol peaks was slower than 12 `cv2.dilate`
  per tile on the A53 (symbol tiles 9.8 -> 10.7 s), the stroke-tile repetition saved 0.17 s;
  reverted, `relation_fast.py` is the v8b version.
* next idea, not implemented: prefetch the next tile's inference in a thread during Codex's
  per-tile post-processing (`bench_phases.py` shows ~5 s of single-core post-processing).

## Redecode (v10, `redecode_fast.py`, `run_page.py --fast-redecode`)
Board profile of redecode (14.4 s, v8b): `beam_chains` 4.8 s, `head_width` 3.7, prefix digit templates
2.8, tuplet number check 2.4 (`continuous_beams`, `_stem_end`), `_stem_end` 1.3 (profiled seconds).

| replaced | how | why identical |
|---|---|---|
| `repair_unison_voices._stem_end` | C++ pixel walk (`hr_stem_end`) | same loops; python `round` = half-even (`nearbyint`); `max(ends)` over (length, col, last) |
| `repair_tuplets.continuous_beams` | C++ (`hr_continuous_beams`), own stem ends | same order of checks; python slice semantics; IndexError cases fall back |
| `relation_grace.head_width` | C++ (`hr_head_width`, `hr_head_width_q`) | quantile 1 = max of the rows; other quantiles by numpy's linear method (virtual index `n*q + (1-q) - 1`, floor, gamma, lerp; tested = numpy on 30k random lists) |
| `relation_voices.beam_chains` | C++ (`hr_beam_chains`) | numpy linspace = `i*step + start`, last = stop (tested); window crops with python slice semantics; same union order |
| `repair_prefix_symbols._classify`, `repair_tuplets.classify` | all templates scored as one array operation; template dilations computed once | integer counts and the same float64 divisions per template; same ranking code |

All helpers are rebound in every module that imported them by name; sources are sha256-guarded.
Desktop: 70/70 photos relation graph, relation MusicXML and final MusicXML identical (`runs/rverify2`,
check mode).  Board (`board_v10/`): redecode 14.3 -> 6.6 s, page 74.8 -> 66.2 s, outputs identical,
check-mode run on the A53 passed.  v10b (head_width median in C++): 70/70 identical (`runs/rverify3`),
board redecode 6.6 -> 5.2 s (run43/44), check-mode run45 passed.

## Files
`relation_fast.py`, `redecode_fast.py`, `relgeo.cpp` (`build_win.bat` -> `relgeo.dll`, `build_racc.sh` ->
`librelgeo.so`), `bench_ort.py` (board tile benchmark), `srchash.py` (source hashes),
`verify_relation.py` (70 photos, board Codex build), `sync_dev.sh` (refresh `pkg_dev/`).
