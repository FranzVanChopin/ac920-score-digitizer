"""ctypes wrapper for decoder.cpp: a drop-in for search.solve under the production config.

    dec = CSolver(lib_path, picker.mlp, pair_model, warp_vocab)
    got, stats = dec.solve(cm, n_beats)          # same format as search.solve
"""
import ctypes
import os
from pathlib import Path

import numpy as np

PROD_PARAMS = dict(beam=16, k_attr=5, warp_top=3, d_top_n=4, o_top_n=2, max_expansions=4000, iso_max=1,
                   pretense_tol=15.0, lam_struct=1.0, progress_w=0.5, voice_w=0.0, dev_w=1.5, tau_w=1.0,
                   canon_pen=0.7, iso_w=3.0, iso_tol=0.03, lam_iso=0.0, hint_w=0.5, pair_w=0.25,
                   close_cost=0.5, stem_w=1.0, staff_w=1.0, width=1000.0, lam_hint=0.0)
ORDER = ["beam", "k_attr", "warp_top", "d_top_n", "o_top_n", "max_expansions", "iso_max", "pretense_tol",
         "lam_struct", "progress_w", "voice_w", "dev_w", "tau_w", "canon_pen", "iso_w", "iso_tol", "lam_iso",
         "hint_w", "pair_w", "close_cost", "stem_w", "staff_w", "width", "lam_hint"]

_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)


def default_lib():
    here = Path(__file__).resolve().parent
    return str(here / ("decoder.dll" if os.name == "nt" else "libdecoder.so"))


def params_from_prod(prod, pair_w=0.25, close_cost=0.5):
    """Map prod.PROD (+ the pair/close settings of two_stage) onto the C parameter vector."""
    p = dict(PROD_PARAMS)
    for k in ("beam", "k_attr", "warp_top", "max_expansions", "voice_w", "dev_w", "tau_w", "iso_w",
              "hint_w", "stem_w", "staff_w", "lam_iso"):
        if k in prod:
            p[k] = prod[k]
    p["pair_w"], p["close_cost"] = pair_w, close_cost
    return p


class CSolver:
    def __init__(self, mlp, pair_model, warp_vocab, lib=None, params=None):
        if getattr(mlp, "hidden3", 0):
            raise ValueError("decoder.cpp supports two hidden layers (hidden3 == 0)")
        self.lib = ctypes.CDLL(lib or default_lib())
        self.lib.dec_create.restype = ctypes.c_void_p
        self.lib.dec_solve.restype = ctypes.c_int
        self.lib.dec_solve.argtypes = [ctypes.c_void_p, ctypes.c_int, _I, _D, ctypes.c_double, ctypes.c_int, _D, _D, _D]
        self.lib.dec_free.argtypes = [ctypes.c_void_p]
        self.wv = [tuple(w) for w in warp_vocab]
        keep = []

        def d(a):
            a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
            keep.append(a)
            return a.ctypes.data_as(_D)

        wv = np.ascontiguousarray(np.array(self.wv, dtype=np.int32).ravel())
        keep.append(wv)
        H, H2 = int(mlp.H), int(mlp.hidden2)
        self.model = self.lib.dec_create(
            ctypes.c_int(H), ctypes.c_int(H2), ctypes.c_int(len(self.wv)),
            d(mlp.W1), d(mlp.W1s), d(mlp.b1), d(mlp.W2), d(mlp.b2), d(mlp.w2), d(mlp.W3), d(mlp.b3),
            d(mlp.W4), d(mlp.b4), d(mlp.W5), d(mlp.b5), d(mlp.we), ctypes.c_double(float(np.ravel(mlp.be)[0])),
            d(mlp.beta), d(mlp.wt), ctypes.c_double(float(np.ravel(mlp.bt)[0])), wv.ctypes.data_as(_I),
            d(pair_model.mu), d(pair_model.sd), d(pair_model.w), ctypes.c_double(float(pair_model.b)))
        self.params = np.array([float((params or PROD_PARAMS)[k]) for k in ORDER], dtype=np.float64)

    def __del__(self):
        try:
            self.lib.dec_free(self.model)
        except Exception:  # noqa: BLE001
            pass

    def _pack(self, cm):
        cands = sorted(cm.cands, key=lambda c: c.cid)
        n = len(cands)
        ints = np.zeros((max(1, n), 10), dtype=np.int32)
        xs = np.zeros(max(1, n), dtype=np.float64)
        for i, c in enumerate(cands):
            hw = tuple(c.hint_warp)
            try:
                hwi = self.wv.index(hw)
            except ValueError:
                hwi = -1
            ints[i] = (c.cid, c.staff, 1 if c.is_rest else 0, c.hint_d, c.hint_dots, hw[0], hw[1], hwi,
                       int(getattr(c, "midi", -1)), int(getattr(c, "stem", 0)))
            xs[i] = c.x
        return cands, n, ints, xs

    def pick(self, cm, n_beats, used, acc, last_x, n_closed, cur_len):
        """Debug: the picker's (succ, D, O, W) log-probs for one search state."""
        cands, n, ints, xs = self._pack(cm)
        nr = n - bin(used).count("1")
        succ, D = np.zeros(nr + 1), np.zeros((nr, 7))
        O, W = np.zeros((nr, 3)), np.zeros((nr, len(self.wv)))
        f = self.lib.dec_pick
        f.restype = ctypes.c_int
        f.argtypes = [ctypes.c_void_p, ctypes.c_int, _I, _D, ctypes.c_double, ctypes.c_int, _D, ctypes.c_ulonglong,
                      ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, _D, _D, _D, _D]
        k = f(self.model, n, ints.ctypes.data_as(_I), xs.ctypes.data_as(_D), float(cm.length), int(n_beats),
              self.params.ctypes.data_as(_D), used, float(acc), float(last_x), n_closed, cur_len,
              succ.ctypes.data_as(_D), D.ctypes.data_as(_D), O.ctypes.data_as(_D), W.ctypes.data_as(_D))
        if k < 0:
            raise RuntimeError(f"dec_pick error {k}")
        return succ, D, O, W

    def solve(self, cm, n_beats):
        cands, n, ints, xs = self._pack(cm)
        ev = np.zeros((max(1, n), 7), dtype=np.float64)
        st = np.zeros(5, dtype=np.float64)
        k = self.lib.dec_solve(self.model, n, ints.ctypes.data_as(_I), xs.ctypes.data_as(_D), float(cm.length),
                               int(n_beats), self.params.ctypes.data_as(_D), ev.ctypes.data_as(_D),
                               st.ctypes.data_as(_D))
        if k < 0:
            raise RuntimeError(f"dec_solve error {k}")
        x_of = {c.cid: c.x for c in cands}
        voices = [[] for _ in range(int(st[4]))]
        for v, cid, onset, dur, dd, o, w in ev[:k]:
            voices[int(v)].append({"cid": int(cid), "x": x_of[int(cid)], "onset": float(onset), "dur": float(dur),
                                   "d": int(dd), "dots": int(o), "warp": self.wv[int(w)]})
        stats = {"expanded": int(st[0]), "complete": bool(st[1]), "valid": bool(st[1]),
                 "loss": float(st[2]) if st[1] else None, "depth": int(st[3])}
        return voices, stats


def _refine_api(lib):
    f = lib.dec_refine
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_int, _I, _D, ctypes.c_double, ctypes.c_int, _D, ctypes.c_int, _I, _D,
                  ctypes.c_int, ctypes.c_int, ctypes.c_double, _D]
    return f


def refine(self, cm, got, n_beats, iters=2, k_attr=40, hint_w=0.5):
    """two_stage.dp_refine(cm, got, ...) for the production call (iso_w=0, exact_x=0): a tuple of
    (cid, x, onset, dur, d, dots, warp) tuples, or `got` itself where python returns it unchanged."""
    if not got:
        return got
    cands, n, ints, xs = self._pack(cm)
    rows = [(e["cid"], e["onset"], e["dur"], e["d"], e["dots"], self.wv.index(tuple(e["warp"]))) for v in got for e in v]
    vlen = np.array([len(v) for v in got], dtype=np.int32)
    inp = np.ascontiguousarray(np.array(rows, dtype=np.float64).reshape(-1, 6))
    out = np.zeros_like(inp)
    k = _refine_api(self.lib)(self.model, n, ints.ctypes.data_as(_I), xs.ctypes.data_as(_D), float(cm.length),
                              int(n_beats), self.params.ctypes.data_as(_D), len(got), vlen.ctypes.data_as(_I),
                              inp.ctypes.data_as(_D), int(iters), int(k_attr), float(hint_w), out.ctypes.data_as(_D))
    if k < 0:
        raise RuntimeError(f"dec_refine error {k}")
    if k == 0:
        return got
    x_of = {c.cid: c.x for c in cands}
    res, p = [], 0
    for m in vlen.tolist():
        res.append(tuple((int(r[0]), x_of[int(r[0])], float(r[1]), float(r[2]), int(r[3]), int(r[4]),
                          self.wv[int(r[5])]) for r in out[p:p + m]))
        p += m
    return tuple(res)


CSolver.refine = refine


def column_joint(self, cm, voices, n_beats, k_attr=40, hint_w=0.5, max_states=1000):
    """column_joint.column_joint_refine(cm, voices, pk, wv, canon_ok, k_attr=...) in C++: same tuple
    result or None.  Column keys (_xkey) and the per-voice (x, onset) order are computed here."""
    from column_joint import _xkey
    cid_of = {c.cid: c for c in cm.cands}
    vs = [sorted(v, key=lambda e: (cid_of[e[0]].x, e[2])) for v in voices if v]
    if not vs:
        return None
    col = {x: i for i, x in enumerate(sorted({_xkey(c.x) for c in cm.cands}))}
    if any(col[_xkey(cid_of[v[0][0]].x)] != 0 for v in vs):
        return None
    cands, n, ints, xs = self._pack(cm)
    col_of = np.array([col[_xkey(c.x)] for c in cands], dtype=np.int32)
    vlen = np.array([len(v) for v in vs], dtype=np.int32)
    vcid = np.array([e[0] for v in vs for e in v], dtype=np.int32)
    vdur = np.array([e[3] for v in vs for e in v], dtype=np.float64)
    out = np.zeros((len(vcid), 5), dtype=np.float64)
    f = self.lib.dec_column_joint
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_int, _I, _D, ctypes.c_double, ctypes.c_int, _D, ctypes.c_int, _I, _I, _D,
                  _I, ctypes.c_int, ctypes.c_double, ctypes.c_int, _D]
    k = f(self.model, n, ints.ctypes.data_as(_I), xs.ctypes.data_as(_D), float(cm.length), int(n_beats),
          self.params.ctypes.data_as(_D), len(vs), vlen.ctypes.data_as(_I), vcid.ctypes.data_as(_I),
          vdur.ctypes.data_as(_D), col_of.ctypes.data_as(_I), int(k_attr), float(hint_w), int(max_states),
          out.ctypes.data_as(_D))
    if k < 0:
        raise RuntimeError(f"dec_column_joint error {k}")
    if k == 0:
        return None
    res, p = [], 0
    for v in vs:
        res.append(tuple((e[0], e[1], float(r[0]), float(r[1]), int(r[2]), int(r[3]), self.wv[int(r[4])])
                         for e, r in zip(v, out[p:p + len(v)])))
        p += len(v)
    return tuple(res)


CSolver.column_joint = column_joint
