# Exact, faster Homr staff geometry (2026-10-25)

Goal: cut the board's `segment` stage (38.8 s of the 102 s page in v6), which is mostly Homr 0.7.0's
rule-based staff detection in pure Python, without changing a single output byte.

`homr_fast.install()` (called by `run_page.py --fast-homr`) replaces Homr functions at run time;
the installed Homr package and Codex's files are not edited.  Homr 0.7.0 is byte-identical on the
desktop handoff runtime and on the board (sha256 of every `homr/*.py` compared), so desktop checks
carry over.  Codex's `homr_staff_recovery` / `homr_ledger_geometry` wrappers call Homr through
module attributes and therefore use the replacements too.

## Where the time went (board cProfile of v6, run11; profiled times are ~1.5x real)
* `detect_staff` 24.5 s: `find_raw_staffs_by_connecting_line_fragments` 12.2 (282 anchors, each
  re-connecting all fragments in its zone and merging staffs), `find_staff_anchors` 10.3
* `_can_shapes_possibly_touch` 538k calls, 12.8 s (two ABC `isinstance` checks per call = 1.2M
  `__instancecheck__`), called by `is_intersecting` (295k) and `is_overlapping` (243k); only 7,281
  pairs survive the screen into the polygon test
* `combine_noteheads_with_stems` 5.5 s (every notehead against every stem)
* `is_overlapping_extrapolated` 164k calls (inside `connect_staff_lines`)
* `StaffLineSegment` / `RawStaff` construction and merging (sort + min/max over long fragment
  lists, `fragment not in list` scans, contour re-stacking with `np.array(list_of_points)`)
* `Staff.get_at` (`min(grid, key=...)`, 1M lambda calls)

## The replacements and why each is exact
| replaced | how | why identical |
|---|---|---|
| `AngledBoundingBox._can_shapes_possibly_touch` | per-type cache of the two isinstance answers | same expression for the distance and the comparison |
| `find_staff_anchors` (copy of Homr's code) | candidates for `line.is_intersecting(thickened)` pre-screened in C++ `hg_screen`; Homr's own `is_intersecting` on the survivors, in list order | the screen keeps every pair with `dist <= major1 + major2` plus a 1e-9 relative margin (a superset), so the exact test sees every pair that can pass |
| `connect_staff_lines` | grouping in C++ `hg_connect` | same visiting order (stable reverse sort + pop = ascending x, equal x in reverse input order), same float operations (`-ffp-contract=off`, `size//2` = `floor(size/2)` for sizes >= 0), `tan` computed by Python's `math.tan`; result groups sorted stably by the first fragment's y |
| `StaffLineSegment.merge` | `fragment not in list` -> set of box tuples | `AngledBoundingBox.__eq__`/`__hash__` are those of `.box` (checked per type) |
| `StaffLineSegment.__init__` | per-fragment numbers cached on the fragment, sort/min/max via `map` | same values in the same order; boxes are only assigned in constructors (grep), so caches never go stale |
| `get_staff_for_anchor`, `find_raw_staffs_...` | fragment-set inclusion on cached frozensets of boxes; zone filter with numpy | set membership of fragments = membership of their boxes (same hash, same ==); float >= int comparisons are exact in numpy |
| `RawStaff.__init__` | concatenate the (K,1,2) contour arrays instead of splitting into points and re-stacking | same array (values, dtype, shape) for `minAreaRect` and `.contours` |
| `combine_noteheads_with_stems` (copy) | stems pre-screened per thickened notehead (`hg_screen`) | first overlapping stem in stem order, as the original `break` |
| `is_overlapping_with_any` | others pre-screened (cache keyed by the boxes' identities) | any() does not depend on order |
| `Staff.get_at` | `np.argmin(abs(xs - x))` | argmin returns the first minimum, as `min()` does; cache rebuilt if the grid list changes (no code edits grids in place) |
| `calculate_edges_of_rotated_rectangle` (v11) | plain float arithmetic, results wrapped as np.float64 | only when centre and size are python numbers, where numpy computes the same IEEE float64 operations (`cx + -hw` kept as Homr writes it); otherwise the original |
| `RotatedBoundingBox.get_center_extrapolated` (v11) | `np.tan(angle / 180 * pi)` cached per box | same expression, boxes never reassigned |
| `break_wide_fragments` (v11) | contour split by numpy indexing (stable argsort for Homr's stable `sorted`) | same arrays; Homr's quirk kept: the right part's first point is appended to the left part and, again, to the right part |
| `resample_staff_segment` (v16) | the per-x loop in C++ (`hg_resample`): `StaffLineSegment.get_at` bounds and `get_center_extrapolated` inputs computed in Python by Homr's expressions, then Homr's checks and gap filling in order; `StaffPoint`s built without `__init__`, `average_unit_size` = `np.mean(np.diff(y))` in C++ | same IEEE operations (`-ffp-contract=off`, tan from `np.tan`); numpy adds fewer than 8 values in order starting from 0.0 (checked: `[1e16, 1, 1]`, `[-0.0]`), so both means are a plain loop; only for 5 lines, python-number boxes, Homr's own `get_at` and a plain `StaffPoint` class |
| `find_raw_staffs_...` (v16) | the RawStaff of an anchor that is merged at once is not built (only its `.lines` are read), the connected lines no anchor line matches are not made into segments; feats of all fragments computed once per page | the merged staff is built by Homr's `RawStaff.merge` expression (same lines, anchors, staff id); the temporary staff is still built whenever its constructor could raise (no points, contours other than int32 (K,1,2)); `staffs.remove` keeps comparing real RawStaffs, so its box-equality semantics are unchanged |
| `are_lines_crossing` (v16) | all fragment pairs in C++ (`hg_crossing`): the centre-distance screen, then OpenCV's integer branch of `pointPolygonTest` (`hg_ppt`) on the 4-point polygons | `hg_ppt` = `cv2.pointPolygonTest` in 32M point tests on 3.2M polygons (Homr box polygons, small integer polygons, degenerate ones; 0 mismatches, `fuzz_ppt.log`); the distance screen (python: `** 0.5`) is decided only outside a 1e-9 relative margin, else Homr's function runs |
| `create_noise_grid` (v16) | without the debug drawing when no debug image is written (a BGR copy of the page, a rectangle and a putText per grid cell) | the drawing touches only the debug image; grid (Homr's own `create_grid`), mask, counts and `handle_filter_results` unchanged.  A vectorized grid (all full tiles in numpy, or the separable form [1,-2,1] x [1,-2,1] with reflect-101 tile edges) was exact on 5000 random images but slower than OpenCV per tile (26 vs 17 ms), so not used |
| `do_polygons_overlap` | for polygons with >= 64 points together (ellipses), only points within 1 px of the other polygon's bounding box go to `cv2.pointPolygonTest` | a point outside the vertex bounding box is outside the polygon; any() is order-free; 4-point boxes keep Homr's loop (numpy per call costs more there on the A53, seen in the v7 board profile) |

Guards: nothing is installed unless Homr is 0.7.0; each function falls back to Homr's original on
inputs outside what it was written for (other box types, NaN, unhashable boxes, empty lists, a
missing C++ library).

## Verification
* `HOMR_FAST_CHECK=1`: every replaced call also runs Homr's original and raises on any difference
  (identity of fragments/stems/grid points, boxes, contours, min/max).
* `verify_photos.py`: all 70 test photos (camera67 67 + 3 real), `page.geometry.json` of the
  original vs the patched code in check mode: **70/70 identical** (`runs/verify4`; 2 pages stop
  in Homr with "No noteheads found" before staff detection, identically in both arms).
* board: run15 and run19 in check mode on the A53 (ARM numpy/cv2) passed; MusicXML SHA
  `60f3d82b…` and `page.geometry.json` SHA `8d9cd56f…` identical in run1–run19.

## Results
Desktop (`timing.py`, 3 interleaved rounds, board photo + 9 camera67 pages, segment stage incl.
segnet): 22.24 -> 12.95 s summed medians (-42%); board photo 2.22 -> 1.33 s.

Board (warm, same photo, `board_trial/board_v7/`):

| run | homr | total | segment | relation | redecode | decode |
|---|---|---:|---:|---:|---:|---:|
| run12 | original (= v6) | 104.2 | 38.8 | 40.9 | 14.4 | 9.2 |
| run13 / 14 | v7 | 89.7 / 93.5 | 24.9 / 24.8 | 39.5 / 43.3 | 14.4 | 8.7 |
| **run17 / 18** | **v7b** | **90.1 / 88.6** | **23.4 / 23.3** | 40.8 / 39.9 | 14.4 | 9.3 / 8.8 |

Segment 38.8 -> 23.4 s (-40%).  Board profile after v7 (run16, cProfile): `detect_staff` 24.5 ->
8.5 s profiled.  What is left in segment: segnet inference ~6 s, image resize/preprocessing,
`predict_symbols` / `create_rotated_bounding_boxes` (OpenCV + box objects), `resample_staffs`,
`break_wide_fragments`, brace detection, Codex ledger heads.

v11 (box corners, cached tan, break_wide_fragments): 70/70 photos identical (`runs/verify5`); board
segment 23.3 -> 22.2 s (run47/48), check-mode run49 passed, all hashes unchanged.  Wall-clock split
(`OMR_TIMERS=1`): segnet inference 11.6 s (ONNX ~9 s), resize 1.4, detect_staff 4.4, predict_symbols 1.2.

## v16 (10-27)
Route A shared mode (v15 flags), `OMR_TIMERS` on the board photo (desktop, v15 and v16 in the same session):
`detect_staff` 0.20-0.22 -> 0.12 s (`resample_staffs` 0.037 -> 0.015-0.018, `find_raw_staffs` 0.066 -> 0.052,
`find_staff_anchors` 0.064 -> 0.04), `filter_predictions` 0.041-0.058 -> 0.025-0.043.
The v15 board split (`board_vT` run6t, no prefetch) had detect_staff 4.4 s (find_raw 1.5, anchors 1.5,
resample 1.0), i.e. roughly -1.5 s on the board if the A53 scales like the desktop for this code.
`verify_shared.py PKG OUT`: page.geometry.json of Homr's original vs the patched code in check mode, v15
shared flags, 70 photos: `runs/vshared1` (resample) 70/70, `runs/vshared2` (+ find_raw) 70/70, `runs/vshared3`
(+ are_lines_crossing) 70/70, `runs/vshared4` (+ noise filter) 70/70 identical (2 photos stop in Homr with "No
noteheads found" in both arms).  End to end with `board_trial/pkg_v16dev`: camera67 MusicXML 65/65 byte-identical
to v15 (`route_a/eval_pages/v16b` vs `sh8s020`, `route_a/eval_pkg.py`), board page identical.
Board (`board_trial/board_v16`, `OMR_TIMERS`, v15 vs v16 in the same session): detect_staff 4.75 -> 2.85 s
(find_raw 1.59 -> 1.05, find_staff_anchors 1.64 -> 1.03, resample 1.14 -> 0.42), filter_predictions 1.06 -> 0.87;
segment 18.6 -> 16.7 s, page 31.9 -> 29.9 s single, 28.8-29.1 -> 26.7-28.0 s resident.  Check mode on the A53
(run16c) passed; all MusicXML identical to v15.

## Log of this work (2026-10-25)
1. board cProfile of v6 (run11) -> hot list above; Homr 0.7.0 identical desktop/board (sha256).
2. first set (touch, merge, get_at, connect, anchors, combine, any): board photo identical,
   70/70 photos identical (`runs/verify1`); desktop segment 2.11 -> 1.44 s.
3. second set (StaffLineSegment init, anchor lookup, raw staffs, RawStaff init, polygon test):
   70/70 identical (`runs/verify3`).
4. board v7 (run12–16): segment 38.8 -> 24.8 s, outputs identical; profile showed the polygon
   prefilter slower than Homr's loop for 4-point boxes on the A53 -> limited to >= 64 points.
5. v7b: 70/70 identical (`runs/verify4`); board segment 23.4 s, outputs identical.

## Files
`homr_fast.py` (patches), `homrgeo.cpp` (C++ kernels; `build_win.bat` -> `homrgeo.dll`,
`build_hacc.sh` -> `libhomrgeo.so`), `dry.py` (desktop dry run of a board package),
`verify_photos.py`, `verify_shared.py` (route A shared mode), `timing.py`, `profsum.py` (compact cProfile view), `sync_dev.sh` (refresh
`pkg_dev/` from the sources).  `run_page.py --fast-homr` installs it, `--only-segment` stops after
the geometry.  Board: `/home/fpga/claude-tf-free-20261025-v7` (package, both libraries built there),
`-v7b` (copy of v7 with the final `homr_fast.py`), scripts in `-v7-extra`, `-v7b-extra`.

Note: the board package `pkg_v7` contains the v7 `homr_fast.py`; v7b only swapped that file on the
board.  The next `build_pkg.py` run picks up the final version from this folder.
