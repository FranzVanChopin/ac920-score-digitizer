# Transformer-free OMR on the real AC920 board (2026-10-25)

Same photo and board as Codex's trials (SD `/Image/00-00.png`, SHA 55557b5c…, Piano Sonata in C
op.4 p1, measures 1–22; 4x A53, Ubuntu 22.04, runtime `/home/fpga/ac920-omr-runtime/venv`, Python
3.12.13 / numpy 2.5.3 / cv2 4.14 / ORT 1.30, all CPU, no PL).

Pipeline (`run_page.py`, package built by `build_pkg.py`): small segnet student S_bin (68k params,
swapped in for Homr's segnet) + Homr's rule-based staff detection, stopped before the transformer ->
Codex map_photo + recognize (staff scale 16) -> our clefs/key from the captured segnet mask and
symbol nodes -> Codex redecode with our headers -> our extraction + measure decoder (real-noise
mix-0.5 fold-0 MLP, pair scorer, column joint, repeat consensus) -> MusicXML.  No Homr transformer,
reference or ground truth is read.  Codex's program directory is imported read-only with
PYTHONDONTWRITEBYTECODE=1; everything of mine lives in new `/home/fpga/claude-*` directories.

## Results on the board

| run | total | segment | relation | headers | redecode | decode | peak RSS (main proc.) |
|---|---:|---:|---:|---:|---:|---:|---:|
| run1, decode 1 process | 242.3 s | 73.4 | 47.2 | 0.1 | 17.7 | 103.0 | 1255 MiB |
| **run2, decode 4 processes** | **168.1 s** | 70.3 | 47.2 | 0.1 | 17.6 | **32.1** | 1275 MiB |

run2's MusicXML is byte-identical to run1's (SHA 60f3d82b…) and, after the XML declaration, to the
desktop dry run.  Codex's same-photo same-board numbers (docs/ac920_sbin_board_trial_20261024.md):
Homr pipeline worker 344.9 s (command 359.7 s), Homr with the S_bin segnet 266.8 s (304.0 s).  My
168 s is process start to MusicXML written, without their asset verification / preflight / atomic
SD commit, so the comparison is close but not identical in scope.

Accuracy on this page (reference m1–22, camera67 scorer): **243/263 ordinary notes exact (92.4%)**,
pitch-in-measure 249, rests 4/4, graces 0/8 (our writer has no graces yet); Homr pipeline 263/263.
All 20 misses are pitch (14) or follow-on; no onset/duration errors.

## Speeding it up (same day), all outputs byte-identical to run1

| version | total | segment | relation | redecode | decode | what changed |
|---|---:|---:|---:|---:|---:|---|
| v2 | 242.3 s | 73.4 | 47.2 | 17.7 | 103.0 | first board run, decoder single process |
| v3 | 168.1 s | 70.3 | 47.2 | 17.6 | 32.1 | measures decoded by 4 forked processes |
| v4 (warm) | 144.2 s | 48.8 | 45.9 | 17.3 | 31.2 | Homr title OCR replaced by an empty stub (rapidocr never imported); bytecode cached under PYTHONPYCACHEPREFIX=/home/fpga/claude-pycache (nothing written into Codex dirs) |
| **v5 (warm)** | **124.0 s** | 38.8 | 39.3 | 14.4 | 30.6 | exact Python patches: Homr `_merge_groups_optimized` with a vectorised pair screen (1.25M Python pair checks -> numpy + Homr's own is_overlapping on survivors); Codex `strokes` group stats cached per group; `staff_distance` polyline arrays cached; peak RSS from getrusage instead of a /proc sampling thread that competed for the GIL |
| **v6 (warm)** | **102.1 s** | 38.8 | 39.3 | 14.4 | **8.8** | decoder stage A (beam search + picker MLP + pair scorer) in C++ (`../cdecoder`; `libdecoder.so` compiled on the board, g++ 11.4 `-O3 -ffp-contract=off`; `run_page.py --csolve`); dp_refine / column joint / consensus still Python |

| v7 (warm) | 89.7 s | 24.9 | 39.5 | 14.4 | 8.7 | Homr staff geometry exact patches + C++ kernels (`../homr_accel`, `run_page.py --fast-homr`) |
| **v7b (warm)** | **88.6 s** | **23.3** | 39.9 | 14.5 | 8.8 | polygon prefilter limited to >= 64-point polygons (`board_v7/`, run17–19) |

| v8 (warm) | 84.6 s | 23.3 | 36.1 | 14.4 | 8.7 | relation: Hough overlapped with symbol tiles, C++ stroke grouping (`../relation_accel`, `--fast-relation`) |
| **v8b (warm)** | **78.9 s** | 23.4 | **30.0** | 14.4 | 8.7 | + link_heads / beam_relations (`board_v8/`, run25–28) |

| v9 (warm) | ~74 s | 23.3 | 29.9 | 14.3 | 4.4 | decoder dp_refine + column joint in C++ (`../cdecoder`) |
| v10 (warm) | 66.2 s | 23.4 | 29.6 | 6.6 | 4.4 | redecode helpers in C++ / vectorised (`../relation_accel/redecode_fast.py`, `--fast-redecode`) |
| v10b (warm) | ~65 s | 23.3 | 30.2 | 5.2 | 4.4 | + head_width median in C++ (`board_v10/`, run43–45) |
| v11 (warm) | ~64 s | 22.2 | 29.7 | 5.2 | 4.4 | segment: box corners, cached tan, break_wide_fragments (run47–49; `/home/fpga/claude-tf-free-20261025-v11`) |
| v12b (warm) | ~63 s | 22.2 | 29.8 | 5.2 | 3.8 | decoder pick cache (`-v12bb`, run53–55) |
| **v13 (warm)** | **~61.5 s** | 22.1 | **28.2** | 5.1 | 3.8 | relation: staff-distance table, cropped bar_bounds (`/home/fpga/claude-tf-free-20261026-v13`, run57–59) |

Full history with the exactness check of every step: `../../OPTIMIZATION_LOG.md`.

**Release v13 (10-26):** `../../release/TF_Free_OMR_AC920_20261026_v13/` (+ `.zip`), README in
Chinese.  Made by `build_pkg.py v13 0.5` + `make_release.py build`, checks, `make_release.py seal`;
package files byte-identical to the board's v13 directory (41/41), plus `run_board.sh` (builds the
three libraries, runs one page).  Re-checked from the tarball in the new board directory
`/home/fpga/claude-release-v13-20261026` (`board_release_v13/`: 62.1 s, all SHAs = v2, libraries
byte-identical to v13's) and on the desktop (original vs patched identical).  Run results pulled
back: `board_v13/` (run57–59, the v13 libraries in `board_v13/libs_aarch64`).

**Release v15 (10-27, board-checked and sealed):** `../../release/TF_Free_OMR_AC920_20261027_v15/`, README in
Chinese (`release_v15/`, with `run_board.sh` and the resident `run_serve.sh`).  `make_release_v15.py lock`
(-> `../route_a/locked_v15`, sh8s), `build`, `check`, later `seal`.  The package is v13's released pkg/ (=
the board's v13 directory, on which every route A board run was layered) with run_page.py, lab/headers.py,
racc/relation_fast.py and the new sacc/ from their masters, + shared/ (`checks/package_vs_v13.json`; the
masters of all other files still equal v13's).  `relation_accel/pkg_dev` is not the base: its cdec/ and
lab/board_decode.py are 10-25 copies (only its DLLs are current; outputs are the same, the C++ refine is
exact).  Desktop check from the release's pkg/: board page + 5 camera67 pages identical to the sh8s
evaluation, v13 flags still the v13 desktop MusicXML (`checks/desktop_dry.json`).  To do on the board: deploy
`pkg_v15.tar.gz` into `/home/fpga/claude-release-v15-20261027`, two single runs + a resident run, pull the
results into `results/board_v15/` and the compiled `libpilresize.so` into `prebuilt/aarch64/`, then seal.

**Release v16 (10-30, board-checked and sealed):** `../../release/TF_Free_OMR_AC920_20261030_v16/`, README in
Chinese (`release_v16/`).  `make_release_v16.py build | check | board | seal`: v15's released pkg/ with
run_page.py, lab/board_decode.py, hacc/homr_fast.py + homrgeo.cpp, sacc/shared_runtime.py from their masters, the
new `album/` and a pipelined `run_serve.sh`.  Desktop: board page and 5 camera67 pages = v15's bytes, v13 flags
d01213b3, pipeline = single runs, album test card (engrave 63/63).  Board (`v16rel_extra/board_runs_v16rel.sh`,
pulled into `board_v16rel/`): deployed from `pkg_v16.tar.gz` alone into `/home/fpga/claude-release-v16-20261030`,
single 31.0 s, pipelined 20.6-21.0 s per board page, every pipelined page = its single run, album 6 pages
clefs 100%.  Codex's Score Studio switches to it by pointing `recognizer` in `/etc/ac920-onecard.json` there
(README "给 Codex").

**v16 dev (10-27, board-measured in `board_v16/`):** `pkg_v16dev/` (= the v15 release pkg + win64 DLLs, refreshed from the
masters by `sync_v16dev.sh`) carries the v16 Homr staff speed-ups (`../homr_accel/README.md`).  Staged board test
`v16_extra/` (`v16_extra.tar.gz`: `board_runs_v16.sh`, `homr_fast.py`, `homrgeo.cpp`, `run_page.py`,
`board_decode.py` with the decoder kept across resident pages): after
`net.py push ../../release/TF_Free_OMR_AC920_20261027_v15/pkg_v15.tar.gz /home/fpga/claude-release-v15-20261027`
and `net.py push v16_extra.tar.gz /home/fpga/claude-v16-20261027-extra`, the script checks the v15 release from
its tarball (single x2, resident 6 pages), builds v16 in `/home/fpga/claude-v16-20261027` (single x2, check
mode, resident 6 pages; MusicXML must equal v15's) and runs `OMR_TIMERS` with / without prefetch and with
`OMR_CV_THREADS=1/2` (contention of the prefetch threads with Homr's main thread on 4 cores).
Result (`v16_extra/summarize_v16.py board_v16`): v15 from its tarball 31.9 s single (36.6 s first run after a cold
boot), resident 28.8-29.1 s; v16 29.9 / 30.8 s, resident 26.7-28.0 s, camera page 18.0 -> 16.6 s; all MusicXML
identical (board page f618b20b, = desktop up to CRLF).  Prefetch contention ~0.8 s; limiting OpenCV's threads does
not help.  `make_release_v15.py board` copied the v15 runs into the release, then `seal`.

v6 on the board (`board_v6/`, runs 8–10 back to back): Python stage A 126.7 s (decode 31.7 s), C++
103.9 / 102.1 s (decode 8.8 / 8.8 s).  Decode-only benchmark on run9's graph: 1 process 100.5 s ->
24.5 s (4.1x), 4 processes 31.2 s -> 8.7 s (3.6x).  All ten board runs (run1–run10) and the four
benchmark outputs have the same MusicXML SHA 60f3d82b….  camera67 with the C++ stage A: 89.85% vs
Python 89.78% (`../cdecoder/README.md`).

v5 peak RSS: main process 1283 MiB, decode workers <= 254 MiB each.  Homr pipeline on the same photo:
345 s (Codex) -> transformer-free v5 is 2.8x faster.  Each patch was checked on the desktop
(geometry JSON, relation graph and MusicXML byte-identical) and on the board (MusicXML SHA unchanged).
The board's Codex program is a newer (10-24) build than the desktop handoff copy; `strokes` is
identical and `staff_distance` has the same formula there (it routes through Codex's own optional
geometry cache, which is off by default).

A53 profile of v4 (cProfile on the board, `run_page.py --profile`): Homr pair screening 1.25M calls
was the top item of `segment`; `relation` = ORT symbol/stroke tiles ~13 s + staff_distance/np.array
~7 s + stroke grouping means ~5 s; `redecode` spread over grace head width, digit templates, tuplet
repair, beam chains.

## Can the PL side help?  Not much for this pipeline
Neural inference is only ~25 s of the page (segnet ~6–11 s, symbol/stroke ~13 s, decoder MLP ~15% of
decode); the rest is Python control flow on the PS.  Even an infinitely fast PL for all networks saves
~15–20% (Amdahl).  Codex's audit also found S_bin's largest INT8 activation (4.69 MiB) exceeds the
XCZU4EV BRAM+URAM (~2.25 MiB), so a PL segnet needs DDR tiling.  PL becomes worthwhile only if the
pipeline turns network-dominated (e.g. a small net emitting events directly).

## Where the remaining ~89 s go (v7b)
* segment 23 s: segnet ~6 s, image preprocessing, OpenCV contour boxes, staff resampling, braces.
* relation ~40 s, redecode 14 s, decode 9 s: unchanged from v6.

## Where the 102 s went (v6)
* segment 39 s, relation 39 s, redecode 14 s: unchanged from v5 (below).
* decode 8.8 s (4 processes): C++ stage A ~2 s; the rest is dp_refine, column-joint DP and repeat
  consensus, still Python.

## Where the 124 s went (v5)
* segment 39 s: small segnet ~6 s + Homr's staff detection algorithms (fragment connection, merging).
* relation 39 s: ORT symbol/stroke tiles ~13 s + Codex graph building.
* decode 31 s (4 processes): mostly Python beam search; exact memoisation would save ~10%; a smaller
  beam would save more but changes accuracy (needs a camera67 check first).
* redecode 14 s.

## Ethernet file transfer (2026-10-25)
Direct cable PC <-> board: PC NIC "以太网 2" (ASIX USB) 192.168.92.1/30, set up by the user (never
change the PC's network config); board eth1 192.168.92.2/30 (NetworkManager profile `omr-direct-92`,
made by the user).  `net.py` moves files over it, commands still go over the serial console:

* `net.py push ARCHIVE.tar.gz /home/fpga/claude-NEW` (new directories only, like `upload.py`)
* `net.py pull LOCAL_DIR /remote/path ...` (files or directories, names relative to their parent)

For each transfer the console starts `/home/fpga/claude-net/netxfer.py` (stdlib, system python3),
which accepts one connection on 192.168.92.2:5101 for at most 120 s and checks the sha256; no ssh,
no passwords, nothing persistent.  Measured: pkg_v7 (1.7 MB) push 1.3 s (serial `upload.py`:
174 s), pull 2.4 s (board gzip), round trip byte-identical; a run directory (16 files) 0.5 s.
`net.py install` re-uploads `netxfer.py` over serial if the folder is ever missing.

## Files
`net.py` / `netxfer.py` (Ethernet transfer, above),
`console.py` (probe/login/run over COM4; login read from local `board_login.json`), `upload.py`
(Codex's UART receiver protocol, `receiver.py` copied unchanged), `fetch.py` (gzip+base64 over the
console, SHA-verified), `run_page.py`, `board_decode.py`, `build_pkg.py`, `make_release.py` (+ `release_v13/`:
README and `run_board.sh` of the release), `make_release_v15.py` (+ `release_v15/`).  Results: `board_run1/`,
`board_run2/` (report.json, page.musicxml, log), desktop dry runs `dry_v1/`, `dry_v2/`.
Board directories: `/home/fpga/claude-tf-free-20261025` (v2), `-v3` (parallel decode), `-run1`, `-run2`;
v6: `-v6` (package + `cdec/libdecoder.so`), `-v6-extra` (`board_runs_v6.sh`, `bench_decode.py`), `-run8..10`.

**Resident multi-page pipeline (10-27, desktop-verified, board pending):** `run_page.py --serve --pipeline
[--pipeline-threads N]` (`shared_runtime.Shared.infer / adopt`): once a page's network pass is done, the next queued
photo's decode, Homr's autocrop / resize / CLAHE and its network pass run in a background thread; the page takes each
step only if its input is equal (photo sha256, array_equal), else computes it.  Desktop: 7 resident pages identical to
single runs, pages 2-7 took all four steps; a photo rewritten after its background job gives the new photo's output.
Staged board test `pl_extra/` (`pl_extra.tar.gz`: `board_runs_pl.sh`, `ort_threads_check.py`, `run_page.py`,
`shared_runtime.py`): `/home/fpga/claude-pl-20261027` = copy of the v16 board directory + these files; 6-page
resident runs without / with the pipeline (2 or 4 background threads, repeated), wall time per run, SHAs; ORT output
for 1-4 threads compared on the A53.  Push: `net.py push pl_extra.tar.gz /home/fpga/claude-pl-20261027-extra`, run
`sh /home/fpga/claude-pl-20261027-extra/board_runs_pl.sh` in the background.
Board result (`board_pl/`, `board_pl2/` with `pl2_extra/board_runs_pl2.sh`, `/home/fpga/claude-pl-20261027`,
`-pl2-20261027`): ORT output on the A53 bitwise identical for 1-4 threads; every MusicXML = v16's.  6 board pages:
27.0 s/page without, ~23 s/page with `--pipeline --pipeline-threads 2` (wall 165.6 -> 146.6 s; the last page 17.6 s,
no next job competing); 4 background threads or `--pipeline-nice 10` slower.  Alternating board / camera pages: wall
145-147 -> 133 s (a board page after a quick camera page waits for its background pass).  Peak RSS ~0.97 -> ~1.1 GB.
The four cores are saturated, so further throughput needs less CPU work per page (INT8 network, more exact ports).
