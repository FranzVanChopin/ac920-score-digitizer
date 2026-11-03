# Optimisation log: transformer-free OMR on the AC920 board

One page = photo -> MusicXML on the AC920 (4x Cortex-A53, Ubuntu 22.04, CPU only), same photo every
time (SD `/Image/00-00.png`, Piano Sonata in C op.4 p1, 22 measures).  Every step below was required
to leave the output **byte-identical** unless stated otherwise; the check is named in each entry.

Where things live (all under `lab/`):

| folder | what |
|---|---|
| `board_trial/` | board package builder (`build_pkg.py`), `run_page.py`, serial tools (`console.py`, `upload.py`, `fetch.py`), board results `board_v*/`, README |
| `cdecoder/` | measure decoder stage A in C++ (`decoder.cpp`, `cdecoder.py`, `verify.py`), README |
| `homr_accel/` | exact faster Homr staff geometry (`homr_fast.py`, `homrgeo.cpp`, `verify_photos.py`), README |
| `omr_proto/` | transformer-free pipeline prototype + camera67 scorer (`proto.py`, `score.py`), README |
| `small_segnet/` | 68k-param segnet student replacing Homr's 14M segnet, README |

Board directories are always new `/home/fpga/claude-*` folders; Codex's directories, `/Image` and
`/Outputs` are never written.  The installed Homr package and Codex's program files are never
edited: all speed-ups are run-time patches installed by our `run_page.py`.

## Timeline (board seconds per page, warm)

| ver | date | total | segment | relation | redecode | decode | change | exactness check |
|---|---|---:|---:|---:|---:|---:|---|---|
| Homr | 10-24 | 345 | | | | | Codex's Homr pipeline (14M segnet + 60M transformer) | reference |
| v2 | 10-25 | 242.3 | 73.4 | 47.2 | 17.7 | 103.0 | transformer-free: S_bin segnet + Homr rules + Codex relation graph + our decoder | camera67 89.77% notes (Homr 97.56) |
| v3 | 10-25 | 168.1 | 70.3 | 47.2 | 17.6 | 32.1 | decode measures in 4 forked processes | MusicXML SHA = v2 |
| v4 | 10-25 | 144.2 | 48.8 | 45.9 | 17.3 | 31.2 | Homr title OCR stubbed (title unused); bytecode cache in `/home/fpga/claude-pycache` | SHA = v2 |
| v5 | 10-25 | 124.0 | 38.8 | 39.3 | 14.4 | 30.6 | Homr box merging: vectorised pair screen; Codex `strokes` / `staff_distance` caches; getrusage instead of RSS thread | desktop geometry/graph/XML identical; board SHA = v2 |
| v6 | 10-25 | 102.1 | 38.8 | 39.3 | 14.4 | 8.8 | decoder stage A (beam search + MLP + pair scorer) in C++ | 2475/2475 camera67 measures identical to reference Python; board SHA = v2 |
| v7 | 10-25 | 89.7 | 24.9 | 39.5 | 14.4 | 8.7 | Homr staff geometry: C++ pair screens + C++ `connect_staff_lines` + exact Python rewrites, 12 functions (`homr_fast.py`, `run_page.py --fast-homr`) | 70/70 test photos geometry identical in check mode (every replaced call compared with Homr's); board check-mode run passed |
| **v7b** | 10-25 | **88.6** | **23.3** | 39.9 | 14.5 | 8.8 | polygon prefilter only for >= 64-point polygons (it was slower than Homr's loop for 4-point boxes on the A53) | same checks; board SHA = v2 |

| v8 | 10-25 | 84.6 | 23.3 | 36.1 | 14.4 | 8.7 | relation: stroke tiles first, stems/bars `strokes()` (HoughLinesP) in 2 threads overlapping the symbol tiles; stroke grouping in C++ (`relation_fast.py`, `run_page.py --fast-relation`) | 70/70 photos frontend graph + XML identical (board Codex build), check mode on board passed |
| **v8b** | 10-25 | **78.9** | 23.4 | **30.0** | 14.4 | 8.7 | + `link_heads` per-stem numbers once (73,640 head×stem pairs), `beam_relations` rows at once | same checks; board SHA = v2 |
| v9 | 10-25 | ~74 | 23.3 | 29.9 | 14.3 | **4.4** | decoder: `dp_refine` and `column_joint_refine` in C++ (`cdecoder`: `dec_refine`, `dec_column_joint`), stage A already C++ | camera67 2475/2475 measures identical (dp_refine), 66/66 triggered column-joint measures identical, Stage C MusicXML byte-identical; board SHA = v2 |
| **v10** | 10-25 | **66.2** | 23.4 | 29.6 | **6.6** | 4.4 | redecode: `_stem_end`, `continuous_beams`, `head_width`, `beam_chains` in C++, template classifiers vectorised (`redecode_fast.py`, `--fast-redecode`) | 70/70 photos relation graph + MusicXML identical (check mode); board check-mode run passed; board SHA = v2 |
| **v10b** | 10-25 | **~65** | 23.3 | 30.2 | **5.2** | 4.4 | + `head_width` for the median (quantile 0.5) in C++, numpy's linear-quantile formula | 70/70 photos identical (`runs/rverify3`); board check-mode run passed; board SHA = v2 |
| **v11** | 10-25 | **~64** | **22.2** | 29.7 | 5.2 | 4.4 | segment: box corners in plain floats (np.float64 results), cached `np.tan` in `get_center_extrapolated`, `break_wide_fragments` by numpy indexing (`homr_fast.py`) | 70/70 photos geometry identical (`homr_accel/runs/verify5`, check mode); board check-mode run passed; board SHA = v2 |
| v12 | 10-26 | ~64 | 22.2 | 30.2 | 5.2 | 4.5 | decoder: MLP rows batched (`forward_rows`, W2 read once per pick) -- bitwise identical but no gain on the board: the MLP is compute-bound, not memory-bound (profiling build: 10.3 of 14.2 s single-process) | camera67 2475/2475 (A), 2475/2475 (refine), 66/66 (joint); board SHA = v2 |
| **v12b** | 10-26 | **~63** | 22.2 | 29.8 | 5.2 | **3.8** | decoder: pick cache in stage A (same (used, acc, last x, closed, length) state -> reuse; 43.7% hits on camera67); stage A single-process 11.3 -> 8.7 s | camera67 2475/2475 identical incl. expansions/losses; board check-mode run passed; board SHA = v2 |
| **v13** | 10-26 | **~61.5** | 22.1 | **28.2** | 5.1 | 3.8 | relation: staff-distance table for `assign_graph` (array np.interp per staff instead of 15.8k scalar calls), `bar_bounds` dilations on crops around the sampled points (`relation_fast.py`) | 70/70 photos frontend graph + XML identical (`relation_accel/runs/verify4`, check mode); board check-mode run passed; board SHA = v2 |
| **v14** | 10-26 | **~62** | 22.2 | 29.2 | 5.0 | 3.4 | ACCURACY step, not exact: route A symbol network (retrained on binarized simulated Verovio photos, same architecture) + `link_heads` margin 0.5 (`--relation-models lab/route_a/locked_v14/relation_models --link-margin 0.5`) | board page 243 -> **262/263** (SHA a7f114d4), camera67 89.85 -> **91.18%**; v13 flags still give SHA 60f3d82b |
| sh1 | 10-27 | 48.7 | | | | | route A shared network (`run_page.py --shared`): one pass of a S_bin-sized network gives Homr's segnet masks + symbol / stroke maps (replaces S_bin + 2 x 154 tiles); not exact, a new model (`lab/route_a/SHARED_NET.md`) | camera67 / board page per model (sh3 94.62% / 258, **sh4 94.64% / 261**) |
| sh3+ | 10-27 | 46.8–48.2 | | 11.0 | | | symbol peaks on logits, stroke probabilities by LUT, photo decoded once per imread flag | board page = desktop (`board_vF`) |
| sh3++ | 10-27 | 46.4 | 23.8 | 11.1 | 4.5 | 3.4 | ink-free 352x352 patches skipped (`board_vR`) | class maps / peaks identical on tested pages |
| serve | 10-27 | 42.5–42.9 | 23.6 | 10.7 | 4.6 | 3.4 | resident `run_page.py --serve` (set-up once, then one page per stdin line) | 3 board-page outputs identical; v13 flags still SHA d01213b3 on the desktop |
| sh6 | 10-27 | **40.0 / 40.2** | **18.0** | 11.2 | 4.6 | 3.4 | half architecture: no full-resolution decoder stage, segnet head at the 1/2 level (network 8 patches 2.11 -> 1.35 s), single runs; RSS 1086 -> 565 MiB (`board_vT`; sh4 in the same session 46.1 / 45.2 s) | new model: camera67 93.64% / 262 -- its staff lines blur at clefs, see SHARED_NET.md; sh7 (pixel-shuffle head) in training |
| sh4 exact | 10-27 | **38.1** | 23.9 | **3.4** | 4.6 | 3.3 | sh4 single run with exact speed-ups of the shared path: symbol peaks / stroke maps / stems + bars HoughLinesP prefetched in background threads during Homr's staff rules, ORT sessions made on first use and kept (edge / grace warmed in the background), map_photo's autocrop reuses Homr's box, Homr's Pillow resize in C++ on 4 row ranges (`board_vU`, 46.1 s before) | board SHA 2ab0fd64 = sh4 before (vT run4); v13 flags SHA 60f3d82b; camera67 65/65 MusicXML identical (desktop); resize 278/278 crops (desktop) + 26/26 (board Pillow) identical |
| **sh7** | 10-27 | **33.7 / 33.4** (resident **30.8 / 30.7**) | 19.3 | 3.5 | 4.5 | 3.3 | sh7 = half architecture with a pixel-shuffle segnet head (6 x 4 logits per 1/2-level cell; deploy ONNX: argmax per sub-pixel, uint8 interleave), PIL frames; same exact speed-ups as above (`board_vU`) | new model: camera67 92.77% (one page zeroed by a missed bar), 94.93% on the 61 pages no model zeroed (sh4 95.04), board page 262/263; six board outputs identical (SHA 43ee05f1) incl. resident and `--no-prefetch --no-cresize` |
| **sh7+** | 10-27 | **33.3 / 32.5** (resident **29.9 / 29.5 / 29.4 / 29.4**) | 19.3 | **2.8** | **4.1** | 3.3 | + Codex's own `geometry_index` (polyline arrays cached, mutation-checked; active in Codex's board runtime) around recognize / redecode; the grayscale photo decode started in a background thread with the page (`board_vV`); resident: camera page 18.6 / 18.4 s; current RSS levels at ~730 MiB over 6 pages, peak 1081 -> `malloc_trim(0)` between resident pages added | camera67 with sh4 65/65 identical (`sh4_pf3`); all 8 board outputs = vU (SHA 43ee05f1 board page, camera page identical to vU) |
| sh8 | 10-27 | **32.4 / 31.6** (resident **28.6–28.9**) | 18.5 | 2.8 | 4.1 | 3.3 | sh8 = sh7 with the stem at the 1/2 level (binarized input folded 2x2 into 4 channels; no full-resolution convolution; network -10%) (`board_vW`) | new model: camera67 91.66% (two pages zeroed by staff-detection events), 94.83% on the common pages (sh7 95.01), board page **263/263**; outputs identical within the model |
| sh8s | 10-27 | = sh8 (same graph, only 4 bias values differ; board not re-timed: in use for the camera) | | | | | ACCURACY step: sh8 with 0.2 subtracted from the staff logit (`route_a/staff_offset.py`, package `route_a/runs/sh8_staff/d020/package`); sh8 marked slightly more ledger-line pixels staff and Homr's detect_staff built extra five-line groups from them (SHARED_NET.md) | camera67 **94.35%** (sh8 91.66, sh7 92.77, sh4 94.64), no page zeroed, 94.76% on the 63 common pages (sh8 94.77); offset 0.1: 93.32%; desktop board page 262/263 (one F#4 -> G#4) |
| **v15** | 10-27 | **31.9** (resident **28.8–29.1**; camera page 17.9–18.0) | | | | | LOCKED sh8s (`lab/route_a/locked_v15`) + every exact speed-up; release `release/TF_Free_OMR_AC920_20261027_v15/` (`make_release_v15.py`): v13's released pkg/ + run_page.py, headers.py, relation_fast.py, sacc/, shared/, `run_board.sh`, resident `run_serve.sh` | desktop from the release's pkg/: board page + 5 camera67 pages identical to the sh8s evaluation, v13 flags = v13 desktop MusicXML (`checks/desktop_dry.json`); board from the tarball alone (`/home/fpga/claude-release-v15-20261027`, 59 files sha-checked): 5 board-page runs SHA f618b20b = desktop up to CRLF, rebuilt libraries = prebuilt; zip sealed (`release/TF_Free_OMR_AC920_20261027_v15.zip`, sha256 17337344) |
| **v16 dev** | 10-27 | **29.9 / 30.8** (resident **26.7–28.0**; camera page 16.6) | **16.7** | 2.9 | 4.1 | 3.3 | exact Homr staff speed-ups in `homr_fast.py` / `homrgeo.cpp`: `resample_staff_segment` loop in C++, `find_raw_staffs` without the RawStaffs / segments Homr builds and drops, `are_lines_crossing` in C++ (OpenCV's integer point-in-polygon branch, fuzz-checked), noise filter without debug drawing (`lab/homr_accel/README.md` v16); resident mode keeps one decoder per process (`board_decode.decoder_for`: weights only read, the pair model's id-keyed lookup emptied per page); `run_page.py` dev knob `OMR_CV_THREADS` | check mode on 70 photos in shared mode 70/70 (`homr_accel/runs/vshared4`); camera67 MusicXML 65/65 identical to v15 (`route_a/eval_pages/v16b`); board page identical; `hg_ppt` = cv2 in 32M point tests; desktop resident run of 7 pages (decoder reused) identical to single runs.  Board (`board_trial/board_v16`, same session as v15): every MusicXML = v15's (board page f618b20b in 12 runs incl. check mode run16c, camera page b7ce99b3); OMR_TIMERS: detect_staff 4.75 -> 2.85 s (find_raw 1.59 -> 1.05, anchors 1.64 -> 1.03, resample 1.14 -> 0.42), filter 1.06 -> 0.87; prefetch contention now ~0.8 s (no prefetch: detect 2.49, filter 0.64, but relation 10.4 s), OpenCV threads 1 / 2 do not help (1: relation +0.5, redecode +0.7) |
| **pipeline (dev)** | 10-27 | 6 board pages resident: **27.0 -> ~23 s/page** (wall 165.6 -> 146.6 s, last page 17.6 s); alternating board / camera pages: wall 145-147 -> 133 s | | | | | `run_page.py --serve --pipeline [--pipeline-threads N]`: when a page's network pass is done, the next queued photo's decode, Homr's autocrop / resize / CLAHE and its network pass run in a background thread (ORT session with N threads, default 2); the page takes a step's result only if that step's input is equal (file sha256, then array_equal), `shared_runtime.infer / adopt`; throughput for several pages, not single-page latency | desktop: 7 resident pages identical to single runs, pages 2-7 reused all four steps; a photo rewritten after its background job is recomputed (output = new photo's); single runs and v13 flags unchanged (d01213b3); ORT output bitwise identical for 1-4 threads (x86 and the A53: 4 threads 7.9 s, 3: 10.1, 2: 13.3, 1: 24.5 s).  Board (`board_trial/board_pl`, `board_pl2`): all 60 pipelined / plain resident pages = v16's MusicXML (f618b20b / b7ce99b3); background 2 threads best (4 threads, or nice +10 via `--pipeline-nice`, slower); the 4 cores are saturated (network ~26 core-s + page ~19 s serial + prefetch + 4 decode processes), so the gain is ~15% per page, not the network's 8 s; peak RSS ~0.97 -> ~1.1 GB |
| INT8 (REJECTED for the CPU) | 10-27 | network on the A53: FP32 7.8 s, q2 8.0 s, q3 8.2 s; page FP32 30.3-31.3 s, q3 30.5-30.6 s | | | | | NOT exact: static INT8 of the v15 network (`lab/route_a/int8/`): QDQ, per-channel weights, binarization and heads float, calibration on simulated val pages (not camera67) | camera67: q3 (Percentile) **93.72%** (FP32 94.35%; one page zeroed, the other 64 pages +104 notes), q2 (MinMax) 91.91%; desktop board page q3 262/263.  Board (`board_trial/board_int8`): no speed-up although the graph runs 27 QLinearConv (ORT's INT8 kernels are not faster than FP32 on the A53 for these small depthwise / 1x1 convolutions); kept only for a later PL/DPU path |
| **album mode (dev)** | 10-29 | board, 6 Moonlight pages (≈510 notes each): first 46.2 s, then 33-37 s/page (segment 21.3 -> 10.5-13.5 s via the pipeline front job; relation / redecode / decode 5 / 8.6 / 10 s = the denser music, board page 3 / 4.4 / 3.2 s); peak RSS 1.09 GB | | | | | `board_trial/album/` (README there): reads Codex a4_album1's `Grey/SS-PP.bmp` + `Meta/SS-PP.json` (complete:true), turns the sensor-orientation image upright (`--rotate ccw`), feeds `run_page.py --serve --pipeline` (2 pages in flight), joins each song's pages (`merge_pages.py`), resumes by sha256, `--watch`, `--page-timeout` restart.  `run_page.py`: METER `@prev` (serve), and a FIX: in `--pipeline` the stdin reader thread held `sys.stdin`'s buffer lock while blocked on an open pipe, so the forked decode workers (multiprocessing closes sys.stdin first) deadlocked: the thread now reads the original object and sys.stdin is /dev/null (10-27 tests fed stdin from a file, so EOF released the lock) | single run unchanged (grey_1 byte-identical, 621db90a); desktop album 9 pages, page = single run, merged songs measure-for-measure equal to their pages, music21 reads them, timeout path tested; board pages differ from desktop by ARM/x86 float rounding (same input sha, same staff lines, staff_space ~1e-5), correct notes vs PDMX 1392 board / 1384 desktop over 4 pages |
| **v16 (release, sealed)** | 10-30 | board, deployed from `pkg_v16.tar.gz` alone: single 31.0 s (cold 35.6 s), resident pipeline 20.6-21.0 s per board page and 10.3-14.4 s per camera67 photo from page 2 (peak RSS 1.16 GB); album dense Moonlight pages 33-37 s | | | | | `release/TF_Free_OMR_AC920_20261030_v16` (`board_trial/make_release_v16.py`): v15's pkg + run_page.py (pipeline, `@prev`, stdin-deadlock fix), board_decode.py, hacc v16 (C++ staff geometry), shared_runtime.py, `album/` (omr_album fixed ccw + orientation_suspect, merge_pages + engrave, check_engrave, run_album.sh), run_serve.sh with --pipeline | desktop: board page and 5 camera67 pages = v15 bytes, v13 flags d01213b3, pipeline 5 pages = single runs, album 9 pages (00-00 = single run 621db90a, engrave 63/63); board: MANIFEST 63/63, f618b20b / b7ce99b3, all 6 pipelined pages = their single runs, album 6 pages clefs 100%, engrave 63/63, board-built libdecoder / librelgeo / libpilresize = v15 prebuilt |
| **album auto-rotate** | 10-30 | — | | | | | FIX of a silent failure: the one-card rig holds the camera turned 180° vs the 10-29 lens rig, so Codex's config `rotate: ccw` fed every page upside down (staves and 21 bar lines still found, pitches mirrored, no clef / key found: 3/511 notes vs PDMX; `cw` 340/511).  `omr_album.py`: default `--rotate ccw` (the user fixed the convention: page top at the right edge of the sensor image and of the capture preview, to be marked there by Codex), a page whose systems mostly have no clef found gets `orientation_suspect: true` + a warning; `--rotate auto` (option) recognizes such a page again turned by 180°, keeps the attempt with more clefs, uses its orientation for later pages; pages record `rotate_used` / `clef_ratio` and are redone when --rotate changes | desktop: mixed album (2 one-card photos + 1 lens-rig photo) all turned right (clefs on 100% of systems), rerun skips all; Codex's patch anchor kept; review `board_trial/studio_review/README.md` |
| **album engrave** | 10-30 | — | | | | | song files (`SS.musicxml`, what Codex's player renders) get display notation: clefs + key signature (page majority of headers.json), written accidentals (per measure / staff against the key), note types / dots / triplet time-modification, beams by beat with secondary beams, whole-measure rests, part name Piano (`album/engrave.py`, used by `merge_pages.py`; `--plain` off).  `page.musicxml` untouched | `check_engrave.py`: sounding events identical in 21/21, 21/21, 63/63, 123/123 measures, 0 broken beam groups; Verovio (Codex's renderer, copy): MIDI notes identical (1020 events; only key-signature / track-name meta added), page 1 holds 5 measures instead of 4; music21 parses (4 sharps) |
| v8c | 10-25 | 79.0 | 23.4 | 30.3 | 14.4 | 8.7 | REJECTED: numpy 5x5 max filter for symbol peaks (slower than 12 cv2.dilate on the A53), stroke-tile repeat (-0.17 s) | identical, but no gain; reverted to v8b |

MusicXML SHA of every board run so far: `60f3d82b…` (run1–run32); frontend/relation graph SHAs
`592b5da8…` / `f0bc583a…` in run17–run32; `page.geometry.json` SHA
`8d9cd56f…` in every run that kept it (run1, run9–run19).  Run-to-run noise on the board: relation
39.4–43.3 s with identical inputs, so totals vary by ~±2 s.

Current command on the board (v13 = release, below):
`run_page.py --photo P --out NEW_DIR --bundle CODEX_APP --threads 4 --csolve --fast-homr --fast-relation --fast-redecode`
with `PYTHONPYCACHEPREFIX=/home/fpga/claude-pycache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1`,
after `sh cdec/build_cdec.sh`, `sh hacc/build_hacc.sh` and `sh racc/build_racc.sh` once in the
package directory (the release's `run_board.sh` does all of this).

## Release v13 (10-26): `release/TF_Free_OMR_AC920_20261026_v13` (+ `.zip`, `.zip.sha256`)
* Built by `lab/board_trial/build_pkg.py v13 0.5` (fresh package from the current lab sources) and
  `lab/board_trial/make_release.py build|seal`.  README in Chinese inside the release.
* The board's v13 directory had been assembled by overlays since pkg_v10, so its files mix CRLF
  (from the v10 archive) and LF (overlays): the fresh package differed in 6 files only by line
  endings.  `make_release.py` keeps each file's board bytes -> 41/41 files byte-identical to
  `/home/fpga/claude-tf-free-20261026-v13` (`checks/package_vs_board.json`); added only
  `run_board.sh`; `MANIFEST.json` rebuilt (the board's was still pkg_v10's).
* Checks: desktop, same package, original (no flags) vs all patches: MusicXML, geometry, frontend
  and relation graph byte-identical (`checks/desktop_dry.json`; desktop vs board MusicXML differs
  only in the Windows line ending after the XML declaration, intermediate JSON differs across
  x86/aarch64 in the last float digits, as always).  Board: `pkg_v13.tar.gz` pushed into the new
  `/home/fpga/claude-release-v13-20261026`, `sh run_board.sh /Image/00-00.png ...-run1`: the three
  libraries compiled there are byte-identical to the v13 ones (`prebuilt/aarch64`), page 62.1 s
  (129 s with the first-time compile), all four output SHAs = v2 (`checks/board_release_run.json`).
* Where the rest goes (README of the release): networks ~22 s of the 61.5 s, Python/OpenCV ~39 s
  spread over many 0.3–1 s items; exact ports of all of it + resident process -> ~40–45 s at best on
  the A53 with these networks.

## After the release: accuracy diagnosis and route A (10-26)
Exact speed-ups are near their end (~40–45 s at best with these networks).  Route B diagnosis
(`lab/diag_b`, README there): board page 12 of 14 pitch errors are sharps the symbol network misses or
reads as naturals (9/17 sharps found on this engraving); camera67's 10.15 lost points are mostly
front-end (accidentals/key 1.8, rhythm hints 1.1, clef 0.65, position 0.6, missed heads 0.45, two
pages zeroed by a measure-count mismatch 2.5) with ~1.1 points of decoder choices.  Codex's symbol /
stroke networks were trained on 49 clean Verovio pages of one font.  Plan for route A
(`lab/route_a/PLAN.md`): label renderer on PDMX with several fonts -> drop-in retrain of symbol/stroke
(accuracy) -> one shared page network (speed).

Route A first results (10-26, `lab/route_a/README.md`; training on binarized simulated photos as the
user asked): the retrained symbol network (same architecture, S_bin's binarization inside, 3,320
Verovio pages in 5 fonts) with Codex's stroke network -> board page 243 -> 262/263, camera67 89.85 ->
90.11%; with `link_heads`' margin 0.5 (new `run_page.py --link-margin`, default .35 keeps v13 exact)
91.18% (Codex's models alone with 0.5: 90.59%).  The retrained stroke network confused barlines with
stems on camera67 (21%); later stroke runs r2–r4 in the route A README.
On the board (`board_trial/board_vA`, `/home/fpga/claude-routeA-20261026`): v13 flags with the new
`run_page.py` / `relation_fast.py` still give SHA 60f3d82b (62.7 s); margin 0.5 alone keeps the board page
identical (61.1 s); new symbol model + margin 0.5 -> board page **262/263** at **62.1 / 62.0 s** (relation
+0.5–1 s for the in-model binarization, decode -0.5 s), same MusicXML as the desktop.

## v6: decoder stage A in C++ (`lab/cdecoder`)
* What: `search.solve` with the production config ported to `decoder.cpp` (picker MLP 306k params,
  pair scorer, attribute options, greedy baseline and fallback); dp_refine / column joint /
  consensus stay in Python.  `run_page.py --csolve`, `proto.py --csolve`.
* Exactness: `cdecoder/verify.py` compares every camera67 measure (2475): voices, onsets,
  durations, attributes, number of expanded nodes and final loss all identical.
* Python quirks found on the way (Python decoder left as it is, by decision of 10-25):
  1. `np.argsort` is not stable on AVX-512 desktops even for 5 elements -> same-x events get a
     CPU-dependent x rank in `pickers.cand_features` (C++ breaks ties by cid);
  2. `PairSuccModel.score_vec` caches by `id(cm)` -> stale pair features after an id is reused
     (never on the board, where all measures stay alive; it did hit `proto.py`);
  3. `_greedy_complete` never offers the gap-filling option (C++ copies this).
* Result: camera67 stage A 59.2 -> 5.5 s (x10.8); board decode 31.7 -> 8.8 s; camera67 notes
  89.78 (Python) -> 89.85 (C++, +12 notes, n.s.).

## v7 / v7b: Homr staff geometry (`lab/homr_accel`)
* Why: in v6, `segment` was 38.8 s; the board profile put 24.5 s (profiled) in Homr's
  `detect_staff`, and 538k `_can_shapes_possibly_touch` calls of which only 7,281 pairs reached
  the polygon test, i.e. almost all time was pair screening and object bookkeeping in Python.
* What (run-time patches, Homr and Codex files untouched): C++ superset screens in front of Homr's
  exact `is_intersecting` / `is_overlapping` (`find_staff_anchors`, `combine_noteheads_with_stems`,
  `is_overlapping_with_any`); `connect_staff_lines` grouping in C++; set/frozenset instead of
  list scans for fragment equality; cached per-fragment numbers; contour arrays concatenated
  instead of re-stacked; `Staff.get_at` via argmin; ABC isinstance replaced by a per-type cache;
  bounding-box prefilter for large polygons.  Each with its exactness argument in the README.
* Verification: `HOMR_FAST_CHECK=1` runs the original next to every replaced call and raises on
  a difference; 70/70 test photos identical (camera67 + real), board check-mode runs passed,
  board output SHAs unchanged.
* Result: segment 38.8 -> 23.4 s; page ~104 -> ~89 s.
* Lesson: the A53 punishes small numpy calls (~10–20 µs each); a vectorised path only pays off
  above a size threshold that has to be measured on the board, not on the desktop.

## v8 / v8b: relation stage (`lab/relation_accel`, README there)
* Board profile: ONNX tiles 13.9 s (154 symbol + 154 stroke tiles, 4 threads), HoughLinesP 9.2 s
  in Codex's `strokes`, 73,640 (head, stem) pairs in `link_heads` each rebuilding the stem's numbers.
* Tile inference itself can't be sped up exactly: 1/2/4 threads x parallel tiles all bitwise
  identical but at most 5% faster (`bench_ort.py`).  The gain came from overlapping (stems/bars
  Hough during the symbol tiles) and from removing repeated per-pair work.
* Result: relation 40.5 -> 30.0 s, page ~89 -> ~79 s; outputs identical everywhere.
* Board phase timings (`bench_phases.py`, sequential): stroke tiles 8.2 s, symbol tiles 9.8 s,
  stems strokes 7.7 s (Hough 6.8), bars 2.9 s.
* Not done yet (next idea): prefetch the next tile's inference in a thread while Codex's
  post-processing of the current tile runs (cores idle ~1.7 s + 3.3 s per page); estimated ~2 s.
  Also: prebuild the 3 ONNX sessions during segment (~1 s), `bar_bounds` full-image dilations.

## Feasibility of < 10 s per page (10-25 evening, `lab/int8_test`, README there)
* PL: the loaded design is a camera/video pipeline (MIPI CSI-2 RX, demosaic, VPSS, frame buffers,
  HDMI, PL DDR4); no DPU, no XRT/zocl/Vitis AI runtime.  The FPGA manager can load bitstreams and
  overlays at run time, so a DPU is possible but needs a new Vivado design + runtime + compiled
  models (days).
* INT8 on the A53 (ORT static QDQ): symbol / stroke x1.38 (13.0 -> 9.4 s per page), segnet x0.77
  (slower; depthwise convs, no int8 dot-product instructions on ARMv8.0).
* INT8 symbol + stroke accuracy: camera67 90.09 -> 91.96% but driven by one page (+297 notes);
  without it neutral (+16 notes); board page 243 -> 239/263.  Board: page 78.4 -> 74.8 s
  (relation 29.9 -> 25.4).  NOT adopted (output changes; user's decision).  Switch:
  `run_page.py --onnx-dir DIR` (models_int8_rel in `lab/int8_test`).
* Consequence: CPU-only, the three networks alone need ~15 s per page even after INT8, so < 10 s
  needs either a DPU or less network work (one shared network, fewer tiles, retraining), plus all
  Python stages in C++.

## Tooling: Ethernet transfer (10-25, after v7b)
The user connected the board by Ethernet (PC 192.168.92.1/30 on "以太网 2", board eth1
192.168.92.2/30 via NetworkManager profile `omr-direct-92`).  `lab/board_trial/net.py push|pull`
moves files over it (one-shot `netxfer.py` listener on the board, sha256-checked, new
`/home/fpga/claude-*` directories only, no ssh/passwords); commands stay on the serial console.
Package upload 174 s -> 1.3 s; whole run directory download ~0.5 s.  Test copy left on the board:
`/home/fpga/claude-net-test-pkg7` (= pkg_v7).

## Segment wall-clock split on the board (v11, `OMR_TIMERS=1`, run47/48)
segment 22.2 s = Homr `load_and_preprocess_predictions` 14.5 (of which the segnet `inference`
11.6: ~9 s ONNX Runtime for ~6-7 batches of 8 patches at ~1.4 s, plus patch extraction, merging and
argmax; `resize_image` 1.4 PIL; noise filter 0.4; autocrop 0.3) + `detect_staff` 4.4 (anchors 1.5,
raw staffs 1.5, resample ~1.0) + `predict_symbols` 1.2 + braces 0.6 + notes/break/combine ~1.
The ONNX part is the floor without changing the network; the Python geometry (~7 s) needs a full C++
staff-detection port (anchors + raw staffs + resample + boxes as one unit) for the next big step.

## Next candidates (by board seconds, v10b, page ~65 s)
* relation 30 s: ONNX tiles 13 s + Hough (fixed unless INT8 / a new network); Python ~10 s
  (bar_bounds, decode_staff, cost/staff_distance, tile post-processing), tile prefetch (~2 s),
  ONNX session prebuild (~1 s)
* segment 23 s: segnet ~6 s; Homr staff detection still ~7 s profiled (raw staffs, anchors,
  polygon tests), resample_staffs, image resize, contour boxes, predict_symbols
* redecode 5.2 s, decode 4.4 s (per-measure python glue, consensus), imports 2 s
* structural: a resident process (models and modules loaded once) removes imports + session
  creation (~3-4 s per page) for a camera device

Current command on the board (v8b, `/home/fpga/claude-tf-free-20261025-v8b`): add
`--fast-relation` to the v7b command; libraries built there (`cdec`, `hacc`, `racc`).
