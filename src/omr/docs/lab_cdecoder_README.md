# Decoder stage A in C++ (2026-10-25)

`decoder.cpp` is a port of the production stage-A beam search (`search.solve` with the PROD config:
beam 16, k_attr 5, warp_top 3, max_expansions 4000, pair scorer w 0.25, close cost 0.5, terminal
tau/dev/iso/hint/stem/staff terms) including the 306k-param picker MLP (`pickers.LearnedPicker`,
two hidden layers, trust anchoring), the pair-successor logistic scorer, `_attr_options`, the greedy
x-order baseline and the anytime `_greedy_complete` fallback.  dp_refine, pitch-swap repair,
column-joint DP and repeat consensus stay in Python.

* `decoder.cpp` -> `decoder.dll` (`build_win.bat`, MSVC `/O2 /fp:precise`) or `libdecoder.so`
  (`build_cdec.sh`, `g++ -O3 -ffp-contract=off`; no fused multiply-add, no fast-math)
* `cdecoder.py`: `CSolver(mlp, pair_model, warp_vocab).solve(cm, n_beats)`, same return format as
  `search.solve`; `.pick(...)` exposes the picker for debugging
* `verify.py [scores] [--raw]`: every camera67 measure (Stage C graphs), Python vs C++
* `trace_one.py score page m_index`: per-depth beam frontier of both, first divergence
* `paired.py RUN_A RUN_B`: paired per-page note delta of two scored camera67 runs
* `lowpri.py`: run a script at BELOW_NORMAL priority
* `DEC_TRACE=<file>` makes the library append every frontier (debug only)

## Exactness: 2475/2475 camera67 measures identical
Same voices, onsets, durations, divisions, dots and warps; same number of expanded nodes; final
loss equal within 1e-9.  Getting there exposed two quirks of the **Python** decoder (the C++ side
follows the reference semantics):

1. **numpy argsort is not stable on this desktop** (AVX-512, x86-simd-sort), not even for 5
   elements: `pickers.cand_features` ranks candidates by `np.argsort(np.argsort(xs))`, so two events at
   the same x get an arbitrary rank (e.g. xs `[492, 1000, 0, 0, 1000]` -> ranks `[2, 3, 1, 0, 4]`).
   The result depends on the CPU's SIMD sort, so the board's numpy may differ.  C++ breaks ties by cid
   (= `kind="stable"`).  Measures with tied x (two voices on one stem column) were 17 of the 18
   initial disagreements.
2. **`PairSuccModel.score_vec` caches candidates by `id(cm)`** and never checks the object; after a
   measure is garbage-collected its id is reused by a later measure, which then gets the old
   measure's candidates as pair features.  Where all measures stay alive (two_stage CV, the board's
   decode_page) this never happens; in `proto.py` (per score) and in verify it did.  Two identical
   `proto.py` runs differed by 1 note because of it.
3. The anytime fallback `_greedy_complete` calls `_attr_options` without `dur_vals`, so it never gets
   the gap-filling `fit` option; C++ copies that (1 measure).

`verify.py` by default makes Python stable (`kind="stable"` in pickers/search) and clears the pair
cache per measure; `--raw` compares with the Python as it is (99.27%, the 18 measures above).

## dp_refine and column joint in C++ (2026-10-25 night, v9)
* `dec_refine` = `two_stage.dp_refine` for the production call (iso_w 0, exact_x 0): per event the
  picker MLP for row 0 only (`pick(..., row0_only=true)`, the same arithmetic as stage A),
  `attr_dp.build_options` (all d/o/w, -(D+O+W), +0.0 where python adds canon_pen 0, +hint_w off the
  hint, stable sort, top 40) and `attr_dp.voice_dp` (1/12-tick cells, strict `<`, first option wins,
  same backtrack; unreachable cells skipped, inf + c never wins).  `CSolver.refine`.
* `dec_column_joint` = `column_joint.column_joint_refine`: the dict-based DP reproduced with
  insertion-ordered layers (first state wins ties), stable cost pruning to 1000 states (new order =
  sorted order), `min((cost, key))` tie-break on (t0, accs, placed), parent indices for backtracking;
  `_xkey` column keys and the (x, onset) voice order computed in python.  `CSolver.column_joint`.
* `verify_refine.py`: 2475/2475 camera67 measures identical (python 13.8 s -> c++ 2.9 s);
  `verify_cj.py`: 66/66 triggered measures identical (21 solved).  `proto.py --csolve` Stage C:
  decode 23 -> 11 s, every MusicXML byte-identical to the stage-A-only C++ run.
* board (v9, run35/36): decode 8.7 -> 4.4 s, MusicXML SHA unchanged.

## Where the board decode time goes (2026-10-26)
Single process on the board page (`board_trial/v12_extra/bench_decode3.py`): 13.5 s = stage A 10.9 +
dp_refine 1.8 + column joint 0.4; python glue < 0.2 s.  Profiling build (`-DDEC_PROF`, per-step
timers, `dec_prof()`): the picker MLP is 10.3 of 14.2 s (26k candidate rows, ~0.3M float64
multiply-adds each, ~0.8 GMAC/s on one A53 core: compute-bound, the 4 worker processes give 4.2 s).
* tried: batched MLP rows (`forward_rows`, W2 read once per pick instead of once per row): bitwise
  identical, but no gain on the board (not memory-bound); kept because dp_refine / column joint now
  share it (`attr_rows0`).
* pick cache in `dec_solve`: pick() depends only on (used set, acc, last x, closed voices, current
  length); nodes reaching the same state reuse the result (a copy, pair scores are added after).
  camera67: 43.7% of picks hit; 2475/2475 measures identical (answers, expansions, losses).

## Speed
* camera67 stage A alone (desktop, 1 thread): Python 59.2 s, C++ 5.5 s (**10.8x**)
* whole camera67 decode (`proto.py` Stage C, 4370 measures, dp_refine etc. still Python): 77 s -> 23 s
* board page decode: see `../board_trial/README.md`

## Accuracy (camera67, Stage C, mix-0.5 models)
| arm | notes | rests |
|---|---:|---:|
| Python as is (`runs_C_py`, rerun of `runs_C_rn` 89.77) | 89.78% | 93.44% |
| Python + pair-cache guard (`--pairfix`, `runs_C_pyfix`) | 89.80% | 93.44% |
| **C++ stage A (`--csolve`, `runs_C_cs`)** | **89.85%** | **93.46%** |

C++ vs Python: +12 notes, pages 3 up / 1 down / 63 same (CI [-0.006, +0.174] points): no loss.
