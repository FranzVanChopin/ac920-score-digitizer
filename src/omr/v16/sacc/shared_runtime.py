"""Shared network at run time (numpy / OpenCV / ONNX Runtime only; copied into the board package as sacc/).

Shared(onnx, threads)
  .segnet(frame)    replaces homr's segnet inference(): one pass over Homr's frame in 320x320 patches with a
                    16 px halo; returns Homr's 5 masks and keeps the symbol / stroke logits (stride 2)
  .relation_maps(raw_shape, staffs, crop_xy, crop_wh)
                    what route_trials.staff_scale_inference.infer returns, from the kept logits instead of
                    2 x 154 tiles: symbol predictions [N,4] (class, x, y, score) and stroke probabilities
                    [6,H,W] uint8, both in the legacy graph frame (photo px x 16/19.8, half-pixel centres)
  .prefetch(raw_shape, crop_xy, crop_wh, rth, strokes) / .relation(..., rth, strokes)
                    run_page starts the peaks, the six maps and Codex's stems / bars strokes() in background
                    threads right after the segnet pass (they overlap Homr's single-threaded staff rules);
                    relation() returns them only for the same photo shape, crop box, thresholds and strokes
                    function, else computes them as before -> same values either way
  .infer(frame) / .adopt(frame_shape, cls, aux)
                    run_page --serve --pipeline: the next page's network computed in a background thread, taken
                    over by the page when its frame is equal (ORT output does not depend on the thread count:
                    checked on the desktop for 1-4 threads)
Symbol peaks as Codex's predict: per class p >= .15 and p == 5x5 max, then same-class duplicates within 6 px
of the 16-px-per-space frame dropped by score.
"""
from __future__ import annotations

import numpy as np
import cv2

FACTOR = 16 / 19.8
WIN, HALO = 320, 16


def infer_frame(run, frame, batch=8, win=WIN, halo=HALO, skip_empty=False):
    """skip_empty: patches whose 352x352 input holds < 30 ink pixels under the network's own binarization
    (31x31 mean - 10) are not run: background class, symbol / stroke logits -20 (margins, gaps, the empty
    rest of a last page: ~11% of a full page's patches)."""
    h, w = frame.shape
    ny, nx = -(-h // win), -(-w // win)
    pad = np.full((ny * win + 2 * halo, nx * win + 2 * halo), 255, np.uint8)
    pad[halo:halo + h, halo:halo + w] = frame
    cls = np.zeros((ny * win, nx * win), np.uint8)
    aux = np.full((18, ny * win // 2, nx * win // 2), -20 if skip_empty else 0, np.float32)
    jobs = [(iy, ix) for iy in range(ny) for ix in range(nx)]
    if skip_empty:
        ink = cv2.adaptiveThreshold(pad, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        integ = cv2.integral(ink)
        n = lambda y0, x0: integ[y0 + win + 2 * halo, x0 + win + 2 * halo] - integ[y0, x0 + win + 2 * halo] \
            - integ[y0 + win + 2 * halo, x0] + integ[y0, x0]  # noqa: E731
        jobs = [(iy, ix) for iy, ix in jobs if n(iy * win, ix * win) >= 30]
    hw, hh = halo // 2, win // 2
    for k in range(0, len(jobs), batch):
        part = jobs[k:k + batch]
        x = np.stack([pad[iy * win: iy * win + win + 2 * halo, ix * win: ix * win + win + 2 * halo] for iy, ix in part])
        x = np.repeat(x[:, None].astype(np.float32), 3, axis=1)
        seg, ax = run(x)
        for (iy, ix), s, a in zip(part, seg, ax):
            c = s if s.ndim == 2 else s.argmax(0)                 # deploy ONNX: class map already
            cls[iy * win:(iy + 1) * win, ix * win:(ix + 1) * win] = c[halo:halo + win, halo:halo + win]
            aux[:, iy * hh:(iy + 1) * hh, ix * hh:(ix + 1) * hh] = a[:, hw:hw + hh, hw:hw + hh]
    return cls[:h, :w], aux[:, :(h + 1) // 2, :(w + 1) // 2]


def infer_frame_strips(run, frame, win=WIN, halo=HALO):
    """As infer_frame, but one full-width strip (320 rows + halo, width padded to a multiple of 16 + halo) per
    run: fewer calls and less halo than 320x320 patches (desktop 1.9 s vs 2.9 s per page); needs an ONNX
    with dynamic height / width.  Objects keep their full horizontal context (no vertical patch borders)."""
    h, w = frame.shape
    ny = -(-h // win)
    wp = -(-w // 16) * 16
    pad = np.full((ny * win + 2 * halo, wp + 2 * halo), 255, np.uint8)
    pad[halo:halo + h, halo:halo + w] = frame
    cls = np.zeros((ny * win, wp), np.uint8)
    aux = np.zeros((18, ny * win // 2, wp // 2), np.float32)
    hw, hh = halo // 2, win // 2
    for iy in range(ny):
        x = np.repeat(pad[None, None, iy * win: iy * win + win + 2 * halo].astype(np.float32), 3, axis=1)
        seg, ax = run(x)
        c = seg[0] if seg.ndim == 3 else seg[0].argmax(0)       # deploy ONNX: class map already
        cls[iy * win:(iy + 1) * win] = c[halo:halo + win, halo:halo + wp]
        aux[:, iy * hh:(iy + 1) * hh] = ax[0, :, hw:hw + hh, hw:hw + wp // 2]
    return cls[:h, :w], aux[:, :(h + 1) // 2, :(w + 1) // 2]


def masks_from_classes(cls):
    """homr's inference() return order: staff, symbols, stems_rests, notehead, clefs_keys."""
    return tuple((cls == k).astype(np.uint8) for k in (4, 5, 1, 2, 3))


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -80, 80)))


def _subpixel(a, b, c):
    """Offset of a parabola's vertex through (-1, a), (0, b), (1, c), within +-0.5."""
    den = a - 2 * b + c
    return np.where(den < 0, np.clip(0.5 * (a - c) / np.where(den < 0, den, -1), -0.5, 0.5), 0.0)


def symbol_peaks(aux, threshold=.15, subpixel=False):
    """[N,4] class, frame x, frame y, score (cell j <-> frame pixel 2j); subpixel: the peak refined by a parabola
    through its neighbours (a stride-2 grid alone puts heads up to one frame pixel off).  Peaks are found on the
    logits (the sigmoid is monotonic), the sigmoid is only evaluated at peaks and their neighbours."""
    out = []
    zt = np.float32(np.log(threshold / (1 - threshold)))
    k5 = np.ones((5, 5), np.uint8)
    for c in range(12):
        z = aux[c]
        yy, xx = np.nonzero((z >= zt) & (z == cv2.dilate(z, k5)))
        if not len(yy):
            continue
        fx, fy = xx.astype(np.float64), yy.astype(np.float64)
        if subpixel:
            zp = np.pad(z, 1, mode="edge")
            fx = fx + _subpixel(_sigmoid(zp[yy + 1, xx]), _sigmoid(zp[yy + 1, xx + 1]), _sigmoid(zp[yy + 1, xx + 2]))
            fy = fy + _subpixel(_sigmoid(zp[yy, xx + 1]), _sigmoid(zp[yy + 1, xx + 1]), _sigmoid(zp[yy + 2, xx + 1]))
        score = _sigmoid(z[yy, xx])
        out.extend([c, 2.0 * x, 2.0 * y, float(v)] for x, y, v in zip(fx, fy, score))
    return np.array(out, np.float32).reshape(-1, 4)


# uint8 probability of a logit in steps of 1/64 over [-8, 8): within one uint8 level of rint(255 * sigmoid)
_LUT = np.rint(255 / (1 + np.exp(-(np.arange(1024) - 512) / 64))).astype(np.uint8)


def prob_u8(z):
    return _LUT[np.clip(np.rint(z * 64) + 512, 0, 1023).astype(np.int16)]


def nms(pred, radius):
    """Codex's greedy duplicate removal (by score, same class, distance < radius), with the kept points
    bucketed on a radius-sized grid so each candidate is only compared with its neighbours (same result)."""
    final, grid, r2 = [], {}, radius * radius
    for p in sorted(pred.tolist(), key=lambda p: -p[3]):
        c, gx, gy = int(p[0]), int(p[1] // radius), int(p[2] // radius)
        if any((p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2 < r2
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for q in grid.get((c, gx + dx, gy + dy), ())):
            continue
        final.append(p)
        grid.setdefault((c, gx, gy), []).append(p)
    return np.array(final, np.float32).reshape(-1, 4)


class Shared:
    def __init__(self, onnx_path, threads=4, subpixel=False):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        o.enable_cpu_mem_arena = False          # as Codex's sessions: the arena kept ~0.5 GB after the segnet pass
        self.sess = ort.InferenceSession(str(onnx_path), o, providers=["CPUExecutionProvider"])
        self.aux = None
        self.frame_shape = None
        self.subpixel = subpixel

    def run(self, x):
        return self.sess.run(None, {"input": np.ascontiguousarray(x, dtype=np.float32)})

    def segnet(self, frame):
        self.cancel()                           # a running prefetch reads self.aux
        # 352x352 patches, ink-free ones skipped (on the A53 strips were not faster: the arithmetic dominates)
        cls, self.aux = infer_frame(self.run, frame, skip_empty=True)
        self.frame_shape = frame.shape
        return masks_from_classes(cls)

    def infer(self, frame):
        """segnet()'s (class map, symbol / stroke logits) without touching the page state: run_page --pipeline computes
        the next page's network in a background thread (with this or another session of the same model)."""
        return infer_frame(self.run, frame, skip_empty=True)

    def adopt(self, frame_shape, cls, aux):
        """segnet() for a frame whose network output (infer() of an equal frame) is already there."""
        self.cancel()
        self.aux = aux
        self.frame_shape = frame_shape
        return masks_from_classes(cls)

    def relation_maps(self, raw_shape, staffs, crop_xy, crop_wh):
        """raw_shape (h, w) of the photo; crop_xy / crop_wh: Homr's autocrop box in photo px."""
        return self._finish(self._peaks(), raw_shape, staffs, crop_xy, crop_wh), \
            self._prob(raw_shape, crop_xy, crop_wh), self._report(raw_shape, crop_xy, crop_wh)

    def _peaks(self):
        return symbol_peaks(self.aux, subpixel=self.subpixel)

    def _geometry(self, raw_shape, crop_xy, crop_wh):
        h, w = raw_shape
        fh, fw = self.frame_shape
        return h, w, fw / crop_wh[0], fh / crop_wh[1], (round(w * FACTOR), round(h * FACTOR))

    def _finish(self, pred, raw_shape, staffs, crop_xy, crop_wh):
        """Peaks (frame px) -> legacy graph frame, then Codex's duplicate removal at the page's staff space."""
        h, w, sx, sy, legacy = self._geometry(raw_shape, crop_xy, crop_wh)
        x0, y0 = crop_xy
        pred = pred.copy()
        # frame -> photo (PIL resize, pixel centres) -> legacy (half-pixel centres)
        to_photo = lambda fx, fy: ((fx + .5) / sx - .5 + x0, (fy + .5) / sy - .5 + y0)  # noqa: E731
        if len(pred):
            px, py = to_photo(pred[:, 1], pred[:, 2])
            pred[:, 1] = (px + .5) * legacy[0] / w - .5
            pred[:, 2] = (py + .5) * legacy[1] / h - .5
        space = float(np.median([s["space_px"] for s in staffs])) * legacy[0] / w
        return nms(pred, 6 * space / 16)

    def _prob(self, raw_shape, crop_xy, crop_wh):
        h, w, sx, sy, legacy = self._geometry(raw_shape, crop_xy, crop_wh)
        x0, y0 = crop_xy
        # legacy pixel u -> photo (u + .5) w / legacy - .5 -> frame -> aux cell = frame / 2
        ax = w / legacy[0]
        ay = h / legacy[1]
        M = np.array([[ax * sx / 2, 0, ((.5 * ax - .5 - x0 + .5) * sx - .5) / 2],
                      [0, ay * sy / 2, ((.5 * ay - .5 - y0 + .5) * sy - .5) / 2]], np.float32)
        prob = np.empty((6, legacy[1], legacy[0]), np.uint8)
        for k in range(6):
            prob[k] = cv2.warpAffine(prob_u8(self.aux[12 + k]), M, legacy, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderValue=0)
        return prob

    def _report(self, raw_shape, crop_xy, crop_wh):
        h, w, sx, sy, legacy = self._geometry(raw_shape, crop_xy, crop_wh)
        fh, fw = self.frame_shape
        return {"mode": "shared", "graph_size": list(legacy), "frame_size": [fw, fh], "frame_scale": [sx, sy],
                "coordinate_space": "photograph pixels scaled by 16/19.8", "reference_supplied_to_recognizer": False}

    # ---- prefetch: what the relation stage needs from the network, computed while Homr's staff rules run
    def prefetch(self, raw_shape, crop_xy, crop_wh, rth, strokes):
        """Start the symbol peaks, the six stroke maps and the stems / bars strokes (Codex's strokes(), passed in)
        in background threads.  Inputs are the photo shape and Homr's autocrop box as captured at the segnet call;
        relation() uses the results only if the relation stage asks for exactly these (else recomputes)."""
        import concurrent.futures as cf
        self.cancel()
        key = (tuple(raw_shape), tuple(crop_xy), tuple(crop_wh), tuple(rth), strokes)
        ex = cf.ThreadPoolExecutor(3)
        f_peaks = ex.submit(self._peaks)
        f_prob = ex.submit(self._prob, raw_shape, crop_xy, crop_wh)
        f_stems = ex.submit(lambda: strokes(f_prob.result()[0], rth[0]))
        f_bars = ex.submit(lambda: strokes(f_prob.result()[2], rth[2], minimum=2.8))
        self._early = (key, ex, f_peaks, f_prob, f_stems, f_bars)

    def cancel(self):
        early, self._early = getattr(self, "_early", None), None
        if early is not None:
            early[1].shutdown(wait=True)

    def relation(self, raw_shape, staffs, crop_xy, crop_wh, rth, strokes):
        """(pred, prob, report, stems, bars) as relation_maps + strokes(prob[0]) / strokes(prob[2], minimum=2.8)."""
        early, self._early = getattr(self, "_early", None), None
        key = (tuple(raw_shape), tuple(crop_xy), tuple(crop_wh), tuple(rth), strokes)
        self.prefetch_used = early is not None and early[0] == key
        if self.prefetch_used:
            _key, ex, f_peaks, f_prob, f_stems, f_bars = early
            out = (self._finish(f_peaks.result(), raw_shape, staffs, crop_xy, crop_wh), f_prob.result(),
                   self._report(raw_shape, crop_xy, crop_wh), f_stems.result(), f_bars.result())
            ex.shutdown(wait=True)
            return out
        if early is not None:
            early[1].shutdown(wait=True)
        import concurrent.futures as cf
        pred, prob, report = self.relation_maps(raw_shape, staffs, crop_xy, crop_wh)
        with cf.ThreadPoolExecutor(2) as ex:
            f_stems = ex.submit(strokes, prob[0], rth[0])
            f_bars = ex.submit(strokes, prob[2], rth[2], minimum=2.8)
            return pred, prob, report, f_stems.result(), f_bars.result()
