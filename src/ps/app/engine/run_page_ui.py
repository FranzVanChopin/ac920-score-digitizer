"""Transformer-free photo -> MusicXML for one page (AC920 A53 or desktop dry run).

Stages (each timed with a monotonic clock, peak RSS sampled every 0.1 s):
  segment   Homr preprocessing + the 68k-param small segnet (instead of Homr's 14M segnet) +
            Homr's rule-based staff detection; stops BEFORE Homr's transformer
  relation  Codex map_photo -> recognize (staff scale 16, placeholder headers)
  headers   our clefs/key from the captured segnet mask + symbol nodes
  redecode  Codex redecode with OUR headers and the production refinements
  decode    our extraction + measure decoder (real-noise model) + repeat consensus
  write     MusicXML
No Homr-transformer output, reference score or ground truth is read.
Usage: run_page.py --photo P --out DIR --bundle CODEX_BOARD_DIR [--threads 4] [--meter 3/4] [--csolve] [--fast-homr] [--fast-relation] [--fast-redecode]
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path('/home/fpga/claude-release-v16-20261030')
T0 = time.perf_counter()


def rss_mb():
    try:
        import resource                              # Linux: ru_maxrss in KiB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20


class Peak:
    """Peak RSS from the kernel (getrusage ru_maxrss): exact and free.  A 0.1 s /proc sampling
    thread was used before; on the A53 it held the GIL often enough to slow the main thread."""

    @property
    def peak(self):
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            try:
                import psutil
                return psutil.Process().memory_info().peak_wset / 2**20
            except Exception:  # noqa: BLE001  desktop package runtime without psutil
                return 0.0

    @property
    def children(self):
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
        except ImportError:
            return 0.0


def _patch_box_merging():
    """Exact, faster homr.bounding_boxes._merge_groups_optimized for singleton groups.

    Homr compares all n^2/2 box pairs in Python (1.25M calls on one page), and each pair is first
    screened by AngledBoundingBox._can_shapes_possibly_touch: centre distance > sum of the major
    axes -> no overlap.  Here that screen runs vectorised (with a 1e-6 superset margin) and only the
    surviving pairs go through Homr's own is_overlapping.  The output only depends on which pairs
    overlap (components ordered by their smallest index, boxes in index order), so it is identical.
    """
    import numpy as np
    import homr.bounding_boxes as bb
    original = bb._merge_groups_optimized

    def fast(groups):
        if not groups or any(len(g) != 1 for g in groups) or \
                not all(isinstance(g[0], bb.AngledBoundingBox) for g in groups):
            return original(groups)
        boxes = [g[0] for g in groups]
        n = len(boxes)
        cx = np.array([b.box[0][0] for b in boxes], float)
        cy = np.array([b.box[0][1] for b in boxes], float)
        mj = np.array([max(b.box[1]) for b in boxes], float)
        uf = bb.UnionFind(n)
        for i in range(n - 1):
            d = np.sqrt((cx[i + 1:] - cx[i]) ** 2 + (cy[i + 1:] - cy[i]) ** 2)
            for j in np.nonzero(d <= mj[i] + mj[i + 1:] + 1e-6)[0] + i + 1:
                if boxes[i].is_overlapping(boxes[int(j)]):
                    uf.union(i, int(j))
        merged = {}
        for i in range(n):
            merged.setdefault(uf.find(i), []).extend(groups[i])
        return list(merged.values())

    bb._merge_groups_optimized = fast


def _patch_relation(rd):
    """Exact, faster versions of two hot Codex helpers in route_trials.relation_decode.

    strokes(): for every Hough segment it recomputed np.mean / min / max over every existing group;
    groups only change when a segment is appended, so those three values are cached per group and
    recomputed (with the same np.mean) on append -> identical numbers, identical grouping.
    staff_distance(): converted the same two polylines to arrays on every call; cached per staff.
    """
    import cv2
    import numpy as np

    def strokes(prob, threshold, space=16., minimum=1.6):
        mask = (prob >= round(threshold * 255)).astype('uint8')
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((round(minimum * space), 1), np.uint8))
        cv2.setRNGSeed(7123)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 720, threshold=round(minimum * space * .65),
                                minLineLength=round(minimum * space), maxLineGap=round(.3 * space))
        groups, stats = [], []                       # stats[k] = (gx, lo, hi) of groups[k]
        for x1, y1, x2, y2 in sorted(lines.reshape(-1, 4).tolist() if lines is not None else [],
                                     key=lambda p: -abs(p[3] - p[1])):
            if abs(y2 - y1) < minimum * space or abs(x2 - x1) > max(2, .08 * abs(y2 - y1)):
                continue
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            x = (x1 + x2) / 2
            selected = None
            for k, (gx, lo, hi) in enumerate(stats):
                if abs(x - gx) < .35 * space and y1 < hi + .4 * space and y2 > lo - .4 * space:
                    selected = k
                    break
            if selected is None:
                groups.append([[x1, y1, x2, y2]])
                selected = len(groups) - 1
                stats.append(None)
            else:
                groups[selected].append([x1, y1, x2, y2])
            g = groups[selected]
            stats[selected] = (np.mean([(p[0] + p[2]) / 2 for p in g]), min(p[1] for p in g), max(p[3] for p in g))
        result = []
        for group in groups:
            xy = np.array(group).reshape(-1, 2)
            lo = int(xy[:, 1].min())
            hi = int(xy[:, 1].max())
            slope, intercept = np.polyfit(xy[:, 1], xy[:, 0], 1)
            a, b = float(slope * lo + intercept), float(slope * hi + intercept)
            yy = np.arange(lo, hi + 1)
            xx = np.clip(np.rint(slope * yy + intercept).astype(int), 0, prob.shape[1] - 1)
            result.append({'id': len(result), 'xy': [[a, float(lo)], [b, float(hi)]],
                           'confidence': float(prob[yy, xx].mean() / 255),
                           'bbox': [int(min(a, b)), lo, max(1, int(abs(b - a)) + 1), hi - lo + 1]})
        return result

    cache = {}

    def staff_distance(staff, x, y):
        key = id(staff)
        hit = cache.get(key)
        if hit is None or hit[0] is not staff['lines'][0]['polyline'] or hit[1] is not staff['lines'][-1]['polyline']:
            hit = (staff['lines'][0]['polyline'], staff['lines'][-1]['polyline'],
                   np.array(staff['lines'][0]['polyline']).T, np.array(staff['lines'][-1]['polyline']).T)
            cache[key] = hit
        top = np.interp(x, *hit[2])
        bottom = np.interp(x, *hit[3])
        return max(top - y, y - bottom, 0) / staff['space_px']

    rd.strokes = strokes
    rd.staff_distance = staff_distance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--serve", action="store_true",
                    help="resident: set up once, then one page per stdin line 'PHOTO<TAB>OUT[<TAB>METER]', a JSON report per page; "
                         "METER '@prev' = the meter of the page this service wrote last (the next page of the same piece)")
    ap.add_argument("--pipeline", action="store_true",
                    help="--serve with --shared: once a page's network pass is done, the next queued photo's decode, "
                         "Homr's crop / resize / CLAHE and its network pass run in a background thread; the page takes "
                         "each result only if that step's input is equal (same outputs as without)")
    ap.add_argument("--pipeline-threads", type=int, default=2,
                    help="ONNX Runtime threads of the background network pass (0: the page session's own)")
    ap.add_argument("--pipeline-nice", type=int, default=0,
                    help="Linux: nice increment of the background thread and of its ONNX Runtime threads (the session "
                         "is made in that thread, its pool threads inherit it), so the current page keeps its CPU")
    ap.add_argument("--bundle", type=Path, required=True, help="Codex board program dir (route_trials, relation_models)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--meter", default=None, help="meter carried from a previous page, e.g. 3/4")
    ap.add_argument("--profile", action="store_true", help="cProfile every stage into <out>/prof_<stage>.txt")
    ap.add_argument("--csolve", action="store_true", help="decoder stage A in C++ (cdec/libdecoder.so, build_cdec.sh)")
    ap.add_argument("--only-segment", action="store_true", help="stop after page.geometry.json (benchmarks/checks)")
    ap.add_argument("--onnx-dir", type=Path, default=None,
                    help="replacement ONNX models (same file names, e.g. INT8 segnet_student/symbol/stroke)")
    ap.add_argument("--only-relation", action="store_true", help="stop after the relation stage (benchmarks/checks)")
    ap.add_argument("--fast-redecode", action="store_true",
                    help="exact faster redecode helpers (racc/redecode_fast.py + librelgeo.so)")
    ap.add_argument("--fast-relation", action="store_true",
                    help="exact faster relation stage (racc/relation_fast.py + librelgeo.so, build_racc.sh)")
    ap.add_argument("--shared", type=Path, default=None,
                    help="route A shared network folder (shared.onnx): one pass replaces the segnet and the "
                         "symbol / stroke tiles (needs --fast-relation; thresholds via --relation-models)")
    ap.add_argument("--fast-resize", action="store_true",
                    help="route A: Homr's resize to width 1920 with OpenCV INTER_AREA instead of PIL bicubic (needs --shared)")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="route A: compute the stroke maps / strokes in the relation stage instead of in background "
                         "threads during Homr's staff rules (same output; for timing comparisons)")
    ap.add_argument("--no-cresize", action="store_true",
                    help="route A: Homr's resize with Pillow itself instead of sacc/pil_resize (same bytes; for timing)")
    ap.add_argument("--relation-models", type=Path, default=None,
                    help="relation model folder (symbol/stroke/edge/grace + scores) instead of BUNDLE/relation_models")
    ap.add_argument("--link-margin", type=float, default=None,
                    help="head-beyond-stem-end margin of link_heads in staff spaces (Codex .35; needs --fast-relation)")
    ap.add_argument("--fast-homr", action="store_true",
                    help="exact faster Homr staff geometry (hacc/homr_fast.py + libhomrgeo.so, build_hacc.sh)")
    a = ap.parse_args()
    if not a.serve and (a.photo is None or a.out is None):
        ap.error("--photo and --out are required (or --serve)")
    if a.shared is not None and not a.fast_relation:
        ap.error("--shared needs --fast-relation")
    if a.fast_resize and a.shared is None:
        ap.error("--fast-resize needs --shared")
    if a.link_margin is not None and not a.fast_relation:
        ap.error("--link-margin needs --fast-relation")
    if a.pipeline and not (a.serve and a.shared is not None):
        ap.error("--pipeline needs --serve and --shared")
    peak = Peak()
    times = {}
    mark = [time.perf_counter()]
    cur = {"out": a.out}                                # the page being processed (lap's profile files)

    prof = [None]

    def lap(name):
        now = time.perf_counter()
        times[name] = round(now - mark[0], 3)
        mark[0] = now
        print(f"[{now - T0:8.2f}s] {name} {times[name]:.2f}s  rss_peak {peak.peak:.0f} MiB", flush=True)
        if cur.get("photo") is not None:
            print(json.dumps(dict(ui_stage=name, photo=str(cur["photo"]))), flush=True)
        if a.profile:
            import cProfile
            import io
            import pstats
            if prof[0] is not None:
                prof[0].disable()
                s = io.StringIO()
                pstats.Stats(prof[0], stream=s).sort_stats("cumulative").print_stats(70)
                pstats.Stats(prof[0], stream=s).sort_stats("tottime").print_stats(50)
                (cur["out"] / f"prof_{name}.txt").write_text(s.getvalue(), encoding="utf-8")
                prof[0].dump_stats(str(cur["out"] / f"prof_{name}.pstats"))     # raw stats, for callers
            prof[0] = cProfile.Profile()
            prof[0].enable()

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    sys.path[:0] = [str(a.bundle), str(HERE / "lab"), str(HERE / "dec"), str(HERE / "cdec"), str(HERE / "hacc"), str(HERE / "racc")]
    import onnxruntime as ort
    student = str(HERE / "weights/segnet_student.onnx")
    _orig = ort.InferenceSession.__init__

    def _init(self, path_or_bytes, *args, **kw):          # swap Homr's segnet, cap threads
        if isinstance(path_or_bytes, (str, os.PathLike)) and os.path.basename(str(path_or_bytes)).startswith("segnet_"):
            path_or_bytes = student
        if isinstance(path_or_bytes, (str, os.PathLike)) and a.onnx_dir is not None:
            alt = a.onnx_dir / os.path.basename(str(path_or_bytes))
            if alt.exists():
                path_or_bytes = str(alt)
        if not args and "sess_options" not in kw:
            so = ort.SessionOptions()
            so.intra_op_num_threads = a.threads
            so.inter_op_num_threads = 1
            kw["sess_options"] = so
        _orig(self, path_or_bytes, *args, **kw)
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._omr_name = os.path.basename(str(path_or_bytes))[:-5]

    ort.InferenceSession.__init__ = _init
    capture = os.environ.get("ORT_CAPTURE_DIR")          # dev: save model inputs for INT8 calibration
    if capture:
        import numpy as _np
        _run, seen = ort.InferenceSession.run, {}

        def _capture_run(self, output_names, input_feed, *args, **kw):
            name = getattr(self, "_omr_name", "model")
            k = seen[name] = seen.get(name, -1) + 1
            if name.startswith("segnet") or (name in ("symbol", "stroke") and k % 8 == 0):
                for key, val in input_feed.items():
                    _np.save(Path(capture) / f"{name}__{Path(cur['photo']).stem}__{k}.npy", val)
            return _run(self, output_names, input_feed, *args, **kw)
        ort.InferenceSession.run = _capture_run
        import homr.segmentation.inference_segnet as _seg               # Homr feeds its segnet via IO binding
        _seg_run = _seg.Segnet.run

        def _capture_seg(self, input_data):
            k = seen["segnet"] = seen.get("segnet", -1) + 1
            if k % 2 == 0:
                _np.save(Path(capture) / f"segnet_student__{Path(cur['photo']).stem}__{k}.npy", input_data.astype(_np.float32))
            return _seg_run(self, input_data)
        _seg.Segnet.run = _capture_seg
    # The page title is never used: replace Homr's title OCR (a RapidOCR thread with three ONNX
    # models) by an already-finished empty result, before homr.main imports it.
    import concurrent.futures
    import types
    import homr
    _title = types.ModuleType("homr.title_detection")

    def _no_title(_debug, _staff):
        f = concurrent.futures.Future()
        f.set_result("")
        return f

    _title.detect_title = _no_title
    _title.download_ocr_weights = lambda *args, **kw: None
    sys.modules["homr.title_detection"] = _title
    homr.title_detection = _title
    _patch_box_merging()
    fast_homr = None
    if a.fast_homr:
        import homr_fast
        fast_homr = homr_fast.install()
        print("homr_fast:", fast_homr, flush=True)
    import homr.segmentation.inference_segnet as segmod
    captured = {}
    _inf = segmod.inference

    def _capture(image_org, *args, **kw):                 # keep the student's clefs/keys mask
        out = _inf(image_org, *args, **kw)
        captured["gray"] = image_org.copy()
        captured["clefs_keys"] = out[4]
        return out

    segmod.inference = _capture
    shared_net = None
    _photo_cache = {}
    if a.shared is not None:                            # route A: one shared pass instead of S_bin + symbol + stroke
        sys.path.insert(0, str(HERE / "sacc"))
        import shared_runtime
        cfg = json.loads((a.shared / "shared.json").read_text(encoding="utf-8")) if (a.shared / "shared.json").exists() else {}
        shared_net = shared_runtime.Shared(a.shared / "shared.onnx", a.threads, subpixel=bool(cfg.get("subpixel")))
        models_dir = a.relation_models or a.bundle / "relation_models"
        # Codex's stroke thresholds, read as recognize() reads them (the relation stage checks they are the same)
        rth = [v["threshold"] for v in json.loads((models_dir / "stroke_scores.json").read_text(encoding="utf8"))
               ["reports"]["validation"].values()]
        import homr.main as _hm0
        _autocrop = _hm0.autocrop

        def _capture_crop(image):                       # Homr's crop box, found as map_photo finds it
            crop = captured.get("front_crop", _autocrop)(image)
            off = crop.ctypes.data - image.ctypes.data if np.shares_memory(crop, image) else None
            captured["crop"] = None if off is None else (
                image.shape[:2], [int((off % image.strides[0]) // image.strides[1]), int(off // image.strides[0])],
                [crop.shape[1], crop.shape[0]], image)
            return crop

        _hm0.autocrop = _capture_crop

        def _shared_inference(image_org, *args, **kw):
            gray = image_org if image_org.ndim == 2 else cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
            f = captured.get("front")
            if f is not None and _equal(gray, f["fgray"]):          # --pipeline: computed in the background
                captured["front_used"].append("network")
                out = shared_net.adopt(gray.shape, f["cls"], f["aux"])
            else:
                out = shared_net.segnet(gray)
            captured["gray"] = image_org.copy()
            captured["clefs_keys"] = out[4]
            if not a.no_prefetch and captured.get("crop") is not None and a.fast_relation:
                from route_trials import testset3_stroke_group_cache as _sgc
                raw_shape, crop_xy, crop_wh, _img = captured["crop"]
                shared_net.prefetch(raw_shape, crop_xy, crop_wh, rth, _sgc.strokes)
            if getattr(shared_net, "warm", None) is not None:
                shared_net.warm()
            if a.pipeline:
                pipe_after_network()                    # the page's network pass is done: the next one may start
            return out

        segmod.inference = _shared_inference
        import cv2 as _cv2                              # the photo is decoded once per flag, not 3-4 times
        _imread = _cv2.imread

        def _cached_imread(path, flags=_cv2.IMREAD_COLOR):
            key = (str(path), flags)
            if key not in _photo_cache:
                _photo_cache[key] = _imread(path, flags)
            v = _photo_cache[key]
            if hasattr(v, "result"):                   # decoded in the background at page start (same call)
                v = _photo_cache[key] = v.result()
            return None if v is None else v.copy()

        _cv2.imread = _cached_imread

        def _predecode(photo):
            """The relation stage's grayscale decode (a second imread flag), started with the page."""
            import concurrent.futures as _cf
            ex = _cf.ThreadPoolExecutor(1)
            _photo_cache[(str(photo), _cv2.IMREAD_GRAYSCALE)] = ex.submit(_imread, str(photo), _cv2.IMREAD_GRAYSCALE)
            ex.shutdown(wait=False)
        shared_net.predecode = _predecode
        if a.fast_resize or cfg.get("resize") == "area":          # the package says how its model was trained
            import homr.main as _hm
            from homr.resize import calc_target_image_size

            def _resize_area(image_arr):
                tw, th = calc_target_image_size(image_arr.shape[1], image_arr.shape[0])
                if (tw, th) == (image_arr.shape[1], image_arr.shape[0]):
                    return image_arr
                return _cv2.resize(image_arr, (tw, th), interpolation=_cv2.INTER_AREA)

            _hm.resize_image = _resize_area
        elif not a.no_cresize:
            # Pillow's bicubic resize with its two passes in C++ on row ranges in 4 threads (same bytes; sacc/pil_resize.py)
            import pil_resize
            import homr.main as _hm
            captured["pil_resize"] = pil_resize.install(_hm)
        if a.pipeline:
            # --pipeline: the next page's front end (photo decode, Homr's autocrop / resize / CLAHE, the network pass)
            # computed in a background thread.  The page takes a step's result only if that step's input equals
            # the one the background used (file sha256 for the decode, array_equal after it), so every value is what
            # the page would compute itself; ORT's output does not depend on its thread count (shared_runtime).
            import homr.main as _hm1
            from homr import color_adjust as _ca
            _front_resize, _front_clahe = _hm1.resize_image, _ca.apply_clahe
            bg = {}

            def _bg_net():
                """the background session, made in the background thread itself (after its nice increment)"""
                if "net" not in bg:
                    if a.pipeline_nice and hasattr(os, "setpriority"):
                        tid = threading.get_native_id()
                        os.setpriority(os.PRIO_PROCESS, tid, os.getpriority(os.PRIO_PROCESS, tid) + a.pipeline_nice)
                    bg["net"] = shared_net if a.pipeline_threads <= 0 else shared_runtime.Shared(
                        a.shared / "shared.onnx", a.pipeline_threads)
                return bg["net"]

            def _front_job(photo):
                t0 = time.perf_counter()
                data = Path(photo).read_bytes()
                color = _imread(str(photo), _cv2.IMREAD_COLOR)
                if color is None or Path(photo).read_bytes() != data:       # unreadable, or written meanwhile
                    return None
                gray_photo = _imread(str(photo), _cv2.IMREAD_GRAYSCALE)
                crop = _autocrop(color)
                box = None if crop is color else (
                    (crop.ctypes.data - color.ctypes.data) % color.strides[0] // color.strides[1],
                    (crop.ctypes.data - color.ctypes.data) // color.strides[0], crop.shape[1], crop.shape[0])
                if crop is not color and not (np.shares_memory(crop, color) and np.array_equal(
                        crop, color[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])):
                    return None                                          # not a view of the photo: not reused
                resized = _front_resize(crop)
                pre = _front_clahe(resized)
                fgray = pre if pre.ndim == 2 else _cv2.cvtColor(pre, _cv2.COLOR_BGR2GRAY)
                cls, aux = _bg_net().infer(fgray)
                return dict(sha=hashlib.sha256(data).hexdigest(), color=color, gray_photo=gray_photo, box=box,
                            crop=crop, resized=resized, pre=pre, fgray=fgray, cls=cls, aux=aux,
                            seconds=round(time.perf_counter() - t0, 3))

            def _equal(x, y):
                return (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape
                        and x.dtype == y.dtype and np.array_equal(x, y))

            def _crop_reuse(image):
                f = captured.get("front")
                if f is not None and _equal(image, f["color"]):
                    captured["front_used"].append("autocrop")
                    if f["box"] is None:
                        return image
                    x, y, w, h = f["box"]
                    return image[y:y + h, x:x + w]
                return _autocrop(image)

            def _resize_reuse(image_arr):
                f = captured.get("front")
                if f is not None and _equal(image_arr, f["crop"]):
                    captured["front_used"].append("resize")
                    return f["resized"].copy()
                return _front_resize(image_arr)

            def _clahe_reuse(image):
                f = captured.get("front")
                if f is not None and _equal(image, f["resized"]):
                    captured["front_used"].append("clahe")
                    return f["pre"].copy()
                return _front_clahe(image)

            _hm1.resize_image, _ca.apply_clahe = _resize_reuse, _clahe_reuse
            captured["front_crop"] = _crop_reuse
    import cv2
    import numpy as np
    import hashlib
    # set-up that patches functions: once per process (wrapping twice would chain the wrappers)
    from homr.main import ProcessingConfig, detect_staffs_in_image
    from homr_ledger_geometry import install_extended_ledger_detection
    from homr_staff_recovery import install_staff_recovery
    from homr_with_geometry import describe_geometry
    recovery = install_staff_recovery()
    install_extended_ledger_detection()
    config = ProcessingConfig(False, False, False, False, -1, False, False, False)
    from route_trials.map_staffs import map_photo
    import route_trials.relation_decode as _rd
    _patch_relation(_rd)
    fast_relation = None
    if a.fast_relation:
        import relation_fast
        fast_relation = relation_fast.install(_rd)
        if a.link_margin is not None:
            if "link_heads" not in fast_relation.get("installed", []):
                raise SystemExit("--link-margin needs relation_fast's link_heads, which was not installed")
            relation_fast.LINK_MARGIN = a.link_margin
            fast_relation["link_margin"] = a.link_margin
        print("relation_fast:", fast_relation, flush=True)
    fast_redecode = None
    if a.fast_redecode:
        import redecode_fast
        fast_redecode = redecode_fast.install()
        print("redecode_fast:", fast_redecode, flush=True)
    from route_trials.relation_decode import recognize
    from route_trials.relation_v2 import redecode
    import headers as H
    models = a.relation_models or a.bundle / "relation_models"
    import board_decode as BD
    import contextlib
    geo_index = contextlib.nullcontext
    if shared_net is not None:
        # Codex's scoped polyline-array cache (route_trials.relation_geometry_index, active in Codex's own board
        # runtime around recognize): same np.array per polyline, read-only, mutation-checked on every hit
        from route_trials.relation_geometry_index import geometry_index as geo_index
        # Codex's OnnxModel with the session made on first use and kept for the process: recognize() constructs
        # symbol / stroke sessions the shared path never runs (~1 s on the A53 per page), and edge / grace are
        # warmed in a background thread during the segment stage.  Same file bytes, same options -> same session.
        import threading
        import route_trials.runtime_inference as _ri
        _ri_init, _sessions, _sess_lock = _ri.OnnxModel.__init__, {}, threading.Lock()

        def _session_of(path):
            p = Path(path).resolve()
            key = (str(p), hashlib.sha256(p.read_bytes()).hexdigest())
            with _sess_lock:
                if key not in _sessions:
                    m = object.__new__(_ri.OnnxModel)
                    _ri_init(m, path)
                    _sessions[key] = (m.session, m.input)
                return _sessions[key]

        class _LazyOnnxModel(_ri.OnnxModel):
            def __init__(self, path):
                self._omr_path = path

            def __getattr__(self, name):
                if name not in ("session", "input"):
                    raise AttributeError(name)
                self.session, self.input = _session_of(self._omr_path)
                return self.__dict__[name]

        _ri.OnnxModel = _LazyOnnxModel

        def _warm_sessions():
            for name in ("edge.onnx", "grace.onnx"):
                try:
                    _session_of(models / name)
                except Exception:  # noqa: BLE001  made again (and raising there) at first use
                    pass
        shared_net.warm = lambda: threading.Thread(target=_warm_sessions, daemon=True).start()

        import route_trials.map_staffs as _ms
        _ms_autocrop = _ms.autocrop

        def _reuse_crop(image):     # map_photo repeats Homr's autocrop on the same photo: the captured box if equal
            c = captured.get("crop")
            if c is not None and image.shape == c[3].shape and image.dtype == c[3].dtype and np.array_equal(image, c[3]):
                (x0, y0), (cw, ch) = c[1], c[2]
                return image[y0:y0 + ch, x0:x0 + cw]
            return _ms_autocrop(image)
        _ms.autocrop = _reuse_crop
    lap("imports")

    if os.environ.get("OMR_CV_THREADS"):               # dev: OpenCV's own thread pool size (board contention test)
        cv2.setNumThreads(int(os.environ["OMR_CV_THREADS"]))
    timers = {}
    if os.environ.get("OMR_TIMERS") == "1":            # dev: wall-clock the main Homr steps (real, unprofiled)
        import functools
        import importlib

        def _timed(modname, attr):
            mod = importlib.import_module(modname)
            f = getattr(mod, attr)

            @functools.wraps(f)
            def g(*args, **kw):
                t = time.perf_counter()
                try:
                    return f(*args, **kw)
                finally:
                    k = f"{modname.split('.')[-1]}.{attr}"
                    timers[k] = round(timers.get(k, 0.0) + time.perf_counter() - t, 3)
            setattr(mod, attr, g)
        for modname, attr in (("homr.main", "load_and_preprocess_predictions"), ("homr.main", "predict_symbols"),
                              ("homr.main", "break_wide_fragments"), ("homr.main", "combine_noteheads_with_stems"),
                              ("homr.main", "detect_staff"), ("homr.main", "add_notes_to_staffs"),
                              ("homr.main", "find_braces_brackets_and_grand_staff_lines"),
                              ("homr.main", "prepare_brace_dot_image"), ("homr.main", "create_rotated_bounding_boxes"),
                              ("homr.staff_detection", "find_staff_anchors"),
                              ("homr.staff_detection", "find_raw_staffs_by_connecting_line_fragments"),
                              ("homr.staff_detection", "resample_staffs"),
                              ("homr.staff_detection", "predict_other_anchors_from_clefs"),
                              ("homr.segmentation.inference_segnet", "inference"), ("homr.main", "resize_image"),
                              ("homr.main", "filter_predictions"), ("homr.main", "autocrop")):
            try:
                _timed(modname, attr)
            except (ImportError, AttributeError):
                pass

    def _rss_now():
        try:
            return round(int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20, 1)
        except (OSError, ValueError, AttributeError):   # not Linux
            return None

    last_meter = [None]                                  # --serve: "beats/beat-type" of the page written last

    def run_one(photo, out, meter_arg):
        """One page: photo -> out/page.musicxml (+ report.json); returns the report."""
        out.mkdir(parents=True, exist_ok=False)          # never overwrite an earlier run
        cur.update(out=out, photo=photo)
        _photo_cache.clear()
        cur["rss_start"] = _rss_now()
        captured["front"], captured["front_used"] = None, []
        if a.pipeline:                                   # the background front end of this photo, if there is one
            f = pipe_take(photo)
            if f is not None:
                captured["front"] = f
                _photo_cache[(str(photo), _cv2.IMREAD_COLOR)] = f["color"]
                _photo_cache[(str(photo), _cv2.IMREAD_GRAYSCALE)] = f["gray_photo"]
        if shared_net is not None and not a.no_prefetch and captured["front"] is None:
            shared_net.predecode(photo)
        t_page = time.perf_counter() if a.serve else T0
        try:
            return _page(photo, out, meter_arg, t_page)
        finally:
            captured["front"] = None
            if a.pipeline:
                with pipe["cv"]:
                    pipe["past_network"] = False         # idle: a new photo goes straight to the page's own pass

    def _page(photo, out, meter_arg, t_page):
        # ---- segment: small segnet + Homr rule-based staff detection (no transformer)
        staffs, image, debug, title_future = detect_staffs_in_image(str(photo), config)
        geometry = describe_geometry(staffs, image.shape)
        geometry["staff_recovery"] = recovery
        geo_path = out / "page.geometry.json"
        geo_path.write_text(json.dumps(geometry), encoding="utf-8")
        try:
            title_future.result(60)
        except Exception:  # noqa: BLE001  the title is not used
            pass
        lap("segment")
        if timers:
            print("segment timers:", json.dumps(dict(sorted(timers.items(), key=lambda t: -t[1]))), flush=True)
        if a.only_segment:
            (out / "report.json").write_text(json.dumps(dict(times=times, total_s=round(time.perf_counter() - t_page, 3),
                                                               homr_fast=fast_homr), indent=1), encoding="utf-8")
            return

        # ---- relation: Codex map_photo + recognize with placeholder headers
        staff_geo = map_photo(photo, geo_path)
        if shared_net is not None:
            # a staff only a few spaces long (a stray fragment at the page edge, seen once with the shared segnet head)
            # makes Codex's bar_bounds crash on an empty sample range: drop staves shorter than 4 spaces
            keep = [s for s in staff_geo["staves"] if s["lines"][0]["end"][0] - s["lines"][0]["start"][0] >= 4 * s["space_px"]]
            if len(keep) < len(staff_geo["staves"]):
                print(f"dropped {len(staff_geo['staves']) - len(keep)} staff fragment(s) shorter than 4 spaces", flush=True)
                # renumber systems 1..n and staff indices: later stages map rows to systems by position
                ids = {sy: k for k, sy in enumerate(sorted({s["system"] for s in keep}), 1)}
                for k, s in enumerate(keep):
                    s["system"], s["index"] = ids[s["system"]], k
                staff_geo["staves"] = keep
        staff_path = out / "page.staff.json"
        staff_path.write_text(json.dumps(staff_geo), encoding="utf-8")
        if shared_net is not None:
            relation_fast.SHARED = lambda raw, stv, rth_, strokes_: shared_net.relation(
                raw.shape, stv, staff_geo["crop_xy"], staff_geo["crop_size_px"], rth_, strokes_)
        staves = staff_geo["staves"]
        systems = sorted({s["system"] for s in staves})
        per = {sy: sum(1 for s in staves if s["system"] == sy) for sy in systems}
        ph = out / "placeholder.musicxml"
        H.write_header_xml(ph, [dict(system=sy, clefs={k: ("treble" if k == 1 else "bass") for k in range(1, per[sy] + 1)},
                                     key=0) for sy in systems], (4, 4))
        front = out / "frontend.musicxml"
        with geo_index():
            recognize(photo, staff_path, ph, models / "symbol.onnx", models / "symbol_scores.json", models / "stroke.onnx",
                      models / "stroke_scores.json", models / "edge.onnx", models / "edge_scores.json", front,
                      backend="onnx", staff_scale=16.0)
        first = json.loads(front.with_suffix(".graph.json").read_text(encoding="utf-8"))
        lap("relation")
        if a.only_relation:
            (out / "report.json").write_text(json.dumps(dict(times=times, total_s=round(time.perf_counter() - t_page, 3),
                                                               homr_fast=fast_homr, relation_fast=fast_relation),
                                                          indent=1), encoding="utf-8")
            return

        # ---- headers from the captured mask (mapped from Homr's preprocessed frame to the photo frame)
        pre_w = geometry["preprocessed_size"][0]
        left, top = staff_geo["crop_xy"]
        sx = staff_geo["crop_size_px"][0] / pre_w
        W, Hh = staff_geo["image_size_px"]
        scale = 1.0 / sx
        ids = np.zeros((int(round(Hh * scale)) + 1, int(round(W * scale)) + 1), np.uint8)
        gray = np.full(ids.shape, 255, np.uint8)
        ck, g = captured["clefs_keys"], captured["gray"]
        oy, ox = int(round(top * scale)), int(round(left * scale))
        hh, ww = min(ck.shape[0], ids.shape[0] - oy), min(ck.shape[1], ids.shape[1] - ox)
        ids[oy:oy + hh, ox:ox + ww][ck[:hh, :ww] > 0] = 3
        gray[oy:oy + hh, ox:ox + ww] = (g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY))[:hh, :ww]
        hdr = H.page_headers(photo, staff_path, first, None, prev_key=None, maps=(ids, scale, gray))
        (out / "headers.json").write_text(json.dumps(hdr, indent=1), encoding="utf-8")
        lap("headers")

        # ---- redecode with our headers (production refinements)
        import hashlib
        ours = out / "headers.musicxml"
        H.write_header_xml(ours, hdr, (4, 4))
        proposals = out / "proposals.json"
        proposals.write_text(json.dumps({"image_sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
                                         "coordinate_scale": 16 / 19.8, "predictions": first.get("head_proposals", []),
                                         "reference_supplied_to_recognizer": False}), encoding="utf-8")
        relation = out / "relation.musicxml"
        try:
            with geo_index():
                redecode(photo, staff_path, ours, front.with_suffix(".graph.json"), relation, "all", root=models,
                         backend="onnx",
                         refinement={"rescue": proposals, "attach_graces": True, "voice_continuity": True,
                                     "join_grace_chords": True, "visual_tuplets": True, "local_tuplet_spans": True,
                                     "tuplet_crossings": True, "expanded_rhythm": True, "identity_repairs": True})
            graph = json.loads(relation.with_suffix(".graph.json").read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001  e.g. Codex's own MusicXML export when staves differ in bar count
            # decode the first-pass graph (no redecode refinements) instead of losing the page
            print(f"redecode failed ({e!r}); decoding the first-pass graph", flush=True)
            graph = first
            stats_redecode = repr(e)
        lap("redecode")

        # ---- our extraction + decoder
        import board_decode as BD
        if meter_arg == "@prev":                       # --serve: carried from the page written last (None: 4/4 + inference)
            meter_arg = last_meter[0]
        meter = tuple(int(v) for v in meter_arg.split("/")) if meter_arg else None
        clib = HERE / "cdec" / ("decoder.dll" if os.name == "nt" else "libdecoder.so") if a.csolve else None
        measures, cms, answers, stats = BD.decode_page(graph, staves, hdr, HERE / "weights", meter, clib=clib)
        if graph is first:
            stats["redecode_failed"] = stats_redecode
        lap("decode")
        BD.write_xml(out / "page.musicxml", measures, cms, answers)
        if a.serve:                                    # the meter this page was written with, for a later '@prev'
            import re
            sig = re.findall(r"<beats>(\d+)</beats>\s*<beat-type>(\d+)</beat-type>",
                             (out / "page.musicxml").read_text(encoding="utf-8"))
            last_meter[0] = "/".join(sig[-1]) if sig else None
        lap("write")
        report = dict(times=times, total_s=round(time.perf_counter() - t_page, 3), peak_rss_mib=round(peak.peak, 1),
                      peak_rss_child_mib=round(peak.children, 1),
                      threads=a.threads, measures=len(measures), stats=stats, photo=str(photo),
                      photo_sha256=hashlib.sha256(photo.read_bytes()).hexdigest(), reference_used=False,
                      transformer_used=False, segnet="small student (68,134 params)",
                      decoder_stage_a="c++" if a.csolve else "python", homr_fast=fast_homr,
                      relation_fast=fast_relation, redecode_fast=fast_redecode,
                      relation_models=str(models), link_margin=a.link_margin, shared=str(a.shared) if a.shared else None)
        if shared_net is not None:
            report["shared_prefetch_used"] = getattr(shared_net, "prefetch_used", None)
            report["rss_now_mib"] = _rss_now()        # current RSS (peak_rss_mib only ever grows in --serve)
            report["rss_start_mib"] = cur.get("rss_start")
            report["pil_resize"] = captured.get("pil_resize")
            if a.pipeline:
                f = captured.get("front")
                report["pipeline"] = dict(front_used=list(captured["front_used"]),
                                          front_s=f["seconds"] if f is not None else None)
        (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report), flush=True)
        return report

    # ---- --pipeline state: queued stdin lines and at most one background front-end job (for the queue's head)
    import collections
    import threading as _th          # (main() imports threading locally in the shared set-up)
    pipe = {"queue": collections.deque(), "cv": _th.Condition(), "eof": False, "past_network": False,
            "next": None, "ex": None}

    def pipe_start_locked():
        """(pipe["cv"] held) the front job of the first queued photo, once the current page's network pass is done"""
        if pipe["past_network"] and pipe["next"] is None and pipe["queue"]:
            photo = pipe["queue"][0][0]
            if pipe["ex"] is None:
                pipe["ex"] = concurrent.futures.ThreadPoolExecutor(1)
            pipe["next"] = (photo, pipe["ex"].submit(_front_job, photo))

    def pipe_after_network():
        with pipe["cv"]:
            pipe["past_network"] = True
            pipe_start_locked()

    def pipe_take(photo):
        """the background result for this photo (waiting for a running job) if its file is still the same bytes"""
        with pipe["cv"]:
            nxt, pipe["next"], pipe["past_network"] = pipe["next"], None, False
        if nxt is None:
            return None
        try:
            res = nxt[1].result()
        except Exception as e:  # noqa: BLE001  the page computes its own front end instead
            print(json.dumps({"pipeline_error": repr(e), "photo": nxt[0]}), flush=True)
            return None
        if res is None or nxt[0] != str(photo) or hashlib.sha256(Path(photo).read_bytes()).hexdigest() != res["sha"]:
            return None
        return res

    if not a.serve:
        run_one(a.photo, a.out, a.meter)
        return

    def _malloc_trim():
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):         # not glibc (desktop)
            pass
    print(json.dumps({"ready": True, "imports_s": times.get("imports")}), flush=True)
    if a.pipeline:
        # a forked decode worker (multiprocessing) closes sys.stdin first; the reader thread below, blocked in a read
        # of an open stdin pipe, holds that object's buffer lock across the fork, so the child's close would wait
        # forever (seen on the board with album mode).  The thread reads the original object, sys.stdin is /dev/null.
        stdin_src, sys.stdin = sys.stdin, open(os.devnull)

        def _reader():                                    # stdin read ahead, so the next photo is known early
            for line in stdin_src:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                with pipe["cv"]:
                    pipe["queue"].append(parts)
                    pipe_start_locked()
                    pipe["cv"].notify()
            with pipe["cv"]:
                pipe["eof"] = True
                pipe["cv"].notify()
        _th.Thread(target=_reader, daemon=True).start()

        def _queued():
            while True:
                with pipe["cv"]:
                    while not pipe["queue"] and not pipe["eof"]:
                        pipe["cv"].wait()
                    if not pipe["queue"]:
                        return
                    parts = pipe["queue"].popleft()
                yield parts
        pages = _queued()
    else:
        pages = (parts for parts in (line.rstrip("\n").split("\t") for line in sys.stdin) if len(parts) >= 2)
    for parts in pages:
        times.clear()
        mark[0] = time.perf_counter()
        try:
            run_one(Path(parts[0]), Path(parts[1]), parts[2] if len(parts) > 2 and parts[2] else None)
        except Exception as e:  # noqa: BLE001  one bad page must not end the service
            print(json.dumps({"error": repr(e), "photo": parts[0]}), flush=True)
        _malloc_trim()                                # hand freed heap back between pages (RSS crept up page by page)


if __name__ == "__main__":
    main()
