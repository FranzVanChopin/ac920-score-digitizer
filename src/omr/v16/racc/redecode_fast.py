"""Exact, faster redecode helpers (Codex board build of 10-24), installed at run time.

install() replaces four pixel-walking helpers by C++ versions (relgeo.cpp) that return exactly what
the originals return; Codex's files are never edited.  Every module that imported a helper by name
(`from repair_unison_voices import _stem_end` in about ten Codex modules) is rebound too.

  repair_unison_voices._stem_end   walk up/down 5 columns next to a head (hr_stem_end)
  repair_tuplets.continuous_beams  stem ends + thick-band runs at 3/7 interior columns
  relation_grace.head_width        widest dark run per row near a head; quantile 1.0 = max of the
                                   rows (what np.quantile(rows, 1.0) returns), other quantiles by
                                   numpy's linear-method formula (virtual index, floor, gamma, lerp)
  relation_voices.beam_chains      line sampling between beam ends: numpy linspace reproduced
                                   (i * step + start, last = stop), python round = half-even
  repair_prefix_symbols._classify, the template scores for all templates in one array operation
  repair_tuplets.classify          (templates are constant: their dilations are computed once);
                                   integer counts and the same float64 divisions as per template

Python's slice semantics at the image border (negative stops) are reproduced.  Inputs outside what
the C++ handles (non-uint8 or non-contiguous images, unexpected field types) use the original.
HOMR_FAST_CHECK=1 also runs the original for every call and raises on any difference.
"""
import ctypes
import hashlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = os.environ.get("HOMR_FAST_CHECK") == "1"
SOURCES = {"_stem_end": "ab668fc08ca6f06a", "continuous_beams": "24ccbd30ecef6726", "head_width": "8b81ce64ba9d84e4",
           "beam_chains": "e6b027830e346e26"}
PREFIX_SHA = "ada7869f359f4264"                   # repair_prefix_symbols._classify (filled from srchash_redecode.py)
TUPLET_SHA = "49f89fa2198936a9"                   # repair_tuplets.classify
_U8 = ctypes.POINTER(ctypes.c_uint8)
_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)
_LL = ctypes.c_longlong


def _sha(f):
    return hashlib.sha256(inspect.getsource(f).replace("\r\n", "\n").encode("utf-8")).hexdigest()[:16]


def _differ(name, a, b):
    raise AssertionError(f"redecode_fast: {name} differs from Codex's original: {a!r} != {b!r}")


def _rebind(orig, new):
    """Point every module-level name bound to `orig` at `new` (from-imports included)."""
    n = 0
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for k, v in list(d.items()):
            if v is orig:
                d[k] = new
                n += 1
    return n


def _img_ok(a):
    return isinstance(a, np.ndarray) and a.dtype == np.uint8 and a.ndim == 2 and a.flags.c_contiguous


def install():
    import repair_tuplets as rt
    import repair_unison_voices as ruv
    from route_trials import relation_grace as rg
    from route_trials import relation_voices as rv

    report = {"installed": [], "skipped": []}
    if getattr(ruv._stem_end, "_redecode_fast", False):
        report["skipped"].append("already installed")
        return report
    funcs = {"_stem_end": ruv._stem_end, "continuous_beams": rt.continuous_beams, "head_width": rg.head_width,
             "beam_chains": rv.beam_chains}
    for name, f in funcs.items():
        if SOURCES[name] is not None and _sha(f) != SOURCES[name]:
            report["skipped"].append(f"{name}: source differs from the reviewed version ({_sha(f)})")
            return report
    path = HERE / ("relgeo.dll" if os.name == "nt" else "librelgeo.so")
    if not path.exists():
        report["skipped"].append("C++ library missing")
        return report
    lib = ctypes.CDLL(str(path))
    lib.hr_stem_end.restype = ctypes.c_int
    lib.hr_stem_end.argtypes = [_U8, _LL, _LL, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                                ctypes.POINTER(_LL)]
    lib.hr_continuous_beams.restype = ctypes.c_int
    lib.hr_continuous_beams.argtypes = [_U8, _LL, _LL, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, _LL,
                                        ctypes.c_double]
    lib.hr_head_width.restype = ctypes.c_double
    lib.hr_head_width.argtypes = [_U8, _LL, _LL, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.hr_head_width_q.restype = ctypes.c_double
    lib.hr_head_width_q.argtypes = [_U8, _LL, _LL, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.hr_beam_chains.restype = None
    lib.hr_beam_chains.argtypes = [ctypes.c_int, _D, _I, _I, _I, _D, _U8, _LL, _LL, ctypes.c_double, _I]

    # ------------------------------------------------------------ _stem_end
    orig_stem = ruv._stem_end
    out2 = (_LL * 2)()

    def stem_end(head, ink, up):
        if not _img_ok(ink):
            return orig_stem(head, ink, up)
        try:
            s, x, y = float(head["staff_space"]), float(head["x"]), float(head["y"])
        except (TypeError, ValueError, KeyError):
            return orig_stem(head, ink, up)
        k = lib.hr_stem_end(ink.ctypes.data_as(_U8), ink.shape[0], ink.shape[1], s, x, y, 1 if up else 0, out2)
        return (int(out2[0]), int(out2[1])) if k else None

    def stem_checked(head, ink, up):
        r, o = stem_end(head, ink, up), orig_stem(head, ink, up)
        if r != o:
            _differ("_stem_end", r, o)
        return r

    # ------------------------------------------------------------ continuous_beams
    orig_cb = rt.continuous_beams

    def continuous_beams(a, b, ink, up, count, maximum_gap=4):
        if not _img_ok(ink):
            return orig_cb(a, b, ink, up, count, maximum_gap)
        try:
            args = (float(a["staff_space"]), float(a["x"]), float(a["y"]), float(b["staff_space"]), float(b["x"]),
                    float(b["y"]), 1 if up else 0, int(count), float(maximum_gap))
            if args[7] != count or args[8] != maximum_gap:
                raise TypeError
        except (TypeError, ValueError, KeyError):
            return orig_cb(a, b, ink, up, count, maximum_gap)
        r = lib.hr_continuous_beams(ink.ctypes.data_as(_U8), ink.shape[0], ink.shape[1], *args)
        if r < 0:
            return orig_cb(a, b, ink, up, count, maximum_gap)
        return bool(r)

    def cb_checked(a, b, ink, up, count, maximum_gap=4):
        r, o = continuous_beams(a, b, ink, up, count, maximum_gap), orig_cb(a, b, ink, up, count, maximum_gap)
        if r != o:
            _differ("continuous_beams", r, o)
        return r

    # ------------------------------------------------------------ head_width
    orig_hw = rg.head_width

    def head_width(image, xy, space=16., quantile=1.):
        if not _img_ok(image) or not isinstance(quantile, (int, float)) or not 0 <= quantile <= 1:
            return orig_hw(image, xy, space, quantile)
        x, y = xy
        if quantile == 1.:
            return float(lib.hr_head_width(image.ctypes.data_as(_U8), image.shape[0], image.shape[1], float(x),
                                           float(y), float(space)))
        return float(lib.hr_head_width_q(image.ctypes.data_as(_U8), image.shape[0], image.shape[1], float(x),
                                         float(y), float(space), float(quantile)))

    def hw_checked(image, xy, space=16., quantile=1.):
        r, o = head_width(image, xy, space, quantile), orig_hw(image, xy, space, quantile)
        if r != o:
            _differ("head_width", r, o)
        return r

    # ------------------------------------------------------------ beam_chains
    orig_bc = rv.beam_chains

    def beam_chains(groups, image, space):
        if not _img_ok(image):
            return orig_bc(groups, image, space)
        n = len(groups)
        try:
            gx = np.array([float(g['x']) for g in groups], dtype=np.float64)
            beams = np.array([1 if g.get('beams') else 0 for g in groups], dtype=np.int32)
            grace = np.array([1 if g.get('grace') else 0 for g in groups], dtype=np.int32)
            rel, off = [], [0]
            for g in groups:
                for e in g.get('beam_relations', []):
                    yr = e['y_range']
                    if len(yr) != 2:
                        raise TypeError
                    rel.append((float(yr[0]), float(yr[1]), float(e['stem_x'])))
                off.append(len(rel))
        except (TypeError, ValueError, KeyError):
            return orig_bc(groups, image, space)
        rel_a = np.ascontiguousarray(np.array(rel, dtype=np.float64).reshape(-1, 3))
        off_a = np.array(off, dtype=np.int32)
        root = np.zeros(max(1, n), dtype=np.int32)
        lib.hr_beam_chains(n, gx.ctypes.data_as(_D), beams.ctypes.data_as(_I), grace.ctypes.data_as(_I),
                           off_a.ctypes.data_as(_I), rel_a.ctypes.data_as(_D), image.ctypes.data_as(_U8),
                           image.shape[0], image.shape[1], float(space), root.ctypes.data_as(_I))
        for i, g in enumerate(groups):
            g['beam_chain'] = int(root[i])

    def bc_checked(groups, image, space):
        import copy
        twin = copy.deepcopy(groups)
        beam_chains(groups, image, space)
        orig_bc(twin, image, space)
        a, b = [g.get('beam_chain') for g in groups], [g.get('beam_chain') for g in twin]
        if a != b:
            _differ("beam_chains", a[:8], b[:8])

    # ------------------------------------------------------------ template classifiers
    import cv2
    import repair_prefix_symbols as rps
    orig_pc, orig_tc = rps._classify, rt.classify
    stacks = {}

    def stack(kind, rows, codes=None):
        """per-template ref / dilated masks, sums and ratios, computed once (templates are constant)."""
        key = (kind, id(rows), tuple(codes) if codes is not None else None)
        hit = stacks.get(key)
        if hit is None or hit[0] is not rows:
            sel = [r for r in rows if codes is None or r['code'] in codes]
            ref = np.stack([r['mask'] > 0 for r in sel]) if sel else np.zeros((0, 64, 48), bool)
            dil = np.stack([cv2.dilate(r['mask'], np.ones((3, 3), np.uint8)) > 0 for r in sel]) if sel else ref
            hit = stacks[key] = (rows, sel, ref, dil, np.array([r['ratio'] for r in sel], dtype=np.float64))
        return hit[1:]

    def prefix_classify(mask, codes):
        """repair_prefix_symbols._classify: the same precision/recall/ratio score for every template,
        as one array operation (integer counts, the same float64 divisions)."""
        ratio = mask.shape[1] / mask.shape[0]
        small = cv2.resize(mask, (48, 64), interpolation=cv2.INTER_NEAREST) > 0
        expanded = cv2.dilate(small.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        sel, ref, dil, ratios = stack("prefix", rps._templates(), codes)
        if not sel:
            return orig_pc(mask, codes)
        precision = (small[None] & dil).sum(axis=(1, 2)) / max(1, small.sum())
        recall = (ref & expanded[None]).sum(axis=(1, 2)) / np.maximum(1, ref.sum(axis=(1, 2)))
        score = (precision + recall) / 2 - 0.35 * np.abs(ratio - ratios)
        scores = {}
        for r, s in zip(sel, score.tolist()):
            n = r['code']; scores[n] = max(scores.get(n, 0), float(s))
        ranking = sorted(scores.items(), key=lambda r: r[1], reverse=True)
        if ranking[0][1] < 0.87 or ranking[0][1] - ranking[1][1] < 0.04:
            return None
        return ranking[0]

    def tuplet_classify(mask):
        """repair_tuplets.classify, vectorised the same way."""
        ratio = mask.shape[1] / mask.shape[0]
        small = cv2.resize(mask, (48, 64), interpolation=cv2.INTER_NEAREST) > 0
        dil = cv2.dilate(small.astype('uint8'), np.ones((3, 3), dtype='uint8')) > 0
        sel, ref, rd, ratios = stack("tuplet", rt.templates())
        if not sel:
            return orig_tc(mask)
        score = ((small[None] & rd).sum(axis=(1, 2)) / max(1, small.sum())
                 + (ref & dil[None]).sum(axis=(1, 2)) / np.maximum(1, ref.sum(axis=(1, 2)))) / 2 \
            - .35 * np.abs(ratio - ratios)
        scores = {}
        for r, s in zip(sel, score.tolist()):
            scores[r['number']] = max(scores.get(r['number'], 0), float(s))
        ranked = sorted(scores.items(), key=lambda p: p[1], reverse=True)
        return ranked[0][0] if ranked[0][1] >= .86 and ranked[0][1] - ranked[1][1] >= .08 else None

    def pc_checked(mask, codes):
        r, o = prefix_classify(mask, codes), orig_pc(mask, codes)
        if r != o:
            _differ("_classify", r, o)
        return r

    def tc_checked(mask):
        r, o = tuplet_classify(mask), orig_tc(mask)
        if r != o:
            _differ("classify", r, o)
        return r

    for name, f in (("_classify", rps._classify), ("classify", rt.classify)):
        if _sha(f) != {"_classify": PREFIX_SHA, "classify": TUPLET_SHA}[name]:
            report["skipped"].append(f"{name}: source differs ({_sha(f)})")
            break
    else:
        for (orig, mod, attr, f) in ((orig_pc, rps, "_classify", pc_checked if CHECK else prefix_classify),
                                     (orig_tc, rt, "classify", tc_checked if CHECK else tuplet_classify)):
            setattr(mod, attr, f)
            report["installed"].append(f"{attr} ({_rebind(orig, f)} bindings)")

    new = {"_stem_end": stem_checked if CHECK else stem_end,
           "continuous_beams": cb_checked if CHECK else continuous_beams,
           "head_width": hw_checked if CHECK else head_width,
           "beam_chains": bc_checked if CHECK else beam_chains}
    for name, f in new.items():
        f._redecode_fast = True
    new["_stem_end"]._redecode_fast = True
    for (name, orig), mod, attr in zip(funcs.items(), (ruv, rt, rg, rv),
                                      ("_stem_end", "continuous_beams", "head_width", "beam_chains")):
        setattr(mod, attr, new[name])
        k = _rebind(orig, new[name])
        report["installed"].append(f"{name} ({k} bindings)")
    return report
