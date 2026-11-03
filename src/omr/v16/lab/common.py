"""Shared pieces for the small segmentation-net experiment (student for Homr's segnet).

Homr's segnet sees: page -> resize to width 1920 (PIL) -> CLAHE gray (clip 1, 8x8) -> BGR float
0..255 -> non-overlapping 320x320 tiles padded with white -> argmax over 6 classes:
0 background, 1 stems/rests, 2 noteheads, 3 clefs/keys, 4 staff lines, 5 other symbols.
Everything here reproduces that exactly (tiling copied from homr/segmentation/inference_segnet.py).
"""
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parent
PKG = LAB.parent / "binarize_ab_20261024" / "pkg"
TEACHER = PKG / "deploy/board/models/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx"
WIN = 320
CLASSES = ("background", "stems_rests", "noteheads", "clefs_keys", "staff", "symbols")


def homr_resize(img):
    """Width 1920, aspect kept, PIL bilinear-default resize as homr.resize.resize_image."""
    h, w = img.shape[:2]
    if w == 1920:
        return img
    return np.array(Image.fromarray(img).resize((1920, round(h * 1920 / w))))


def clahe(gray):
    return cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(gray)


def tiles(gray):
    """Homr's tiling with step == win: yields (y, x, patch 3x320x320 float32)."""
    h, w = gray.shape
    img = np.repeat(gray[None].astype(np.float32), 3, axis=0)
    for yl in range(0, max(h, WIN), WIN):
        y = min(yl, h - WIN)
        for xl in range(0, max(w, WIN), WIN):
            x = min(xl, w - WIN)
            patch = np.full((3, WIN, WIN), 255, np.float32)
            y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + WIN, h), min(x + WIN, w)
            patch[:, : y1 - y0, : x1 - x0] = img[:, y0:y1, x0:x1]
            yield y, x, patch


def merge(labels, shape):
    """Homr's merge_patches for step == win (later tiles overwrite the averaged overlap the
    same way: Homr averages argmax ids, so we reproduce that, rounding like astype(uint8))."""
    rec = np.zeros(shape, np.float32)
    wt = np.zeros(shape, np.float32)
    idx = 0
    for iy in range(0, shape[0], WIN):
        y = min(iy, shape[0] - WIN)
        y0, y1 = max(y, 0), min(y + WIN, shape[0])
        for ix in range(0, shape[1], WIN):
            x = min(ix, shape[1] - WIN)
            x0, x1 = max(x, 0), min(x + WIN, shape[1])
            rec[y0:y1, x0:x1] += labels[idx][: y1 - y0, : x1 - x0]
            wt[y0:y1, x0:x1] += 1
            idx += 1
    wt[wt == 0] = 1
    return (rec / wt).astype(labels[0].dtype)


def session(path, threads=4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def segment(sess, gray_clahe, batch=8):
    """Run a segnet-shaped ONNX model over a CLAHE page the way Homr does; returns uint8 ids."""
    pats = list(tiles(gray_clahe))
    labels = []
    for i in range(0, len(pats), batch):
        out = sess.run(None, {"input": np.stack([p for _y, _x, p in pats[i:i + batch]])})[0]
        labels += [o.argmax(0).astype(np.int64) for o in out]
    return merge(labels, gray_clahe.shape).astype(np.uint8)


def low_priority():
    """Drop this process to BELOW_NORMAL priority (Windows) so the desktop stays responsive."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
