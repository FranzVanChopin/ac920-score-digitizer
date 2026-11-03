"""Homr's resize_image (PIL Image.resize to width 1920, bicubic) with the two passes of Pillow's resampler in C++
(pilresize.cpp) on row ranges in several threads -- the same bytes as Pillow (desktop: 71 photos, their autocrop
and 3 random crop views each, 278/278 identical, 6 left to Pillow by the tap guard; board: resize_check.py),
~4.5x faster on 4 cores (Pillow runs both passes in one thread).  install(hm, check=True) compares every call.

install(homr_main) replaces homr.main.resize_image when it is still Homr's own function and Pillow is the
reviewed 12.3.0; any image outside what the kernels were written for (not uint8 H x W x 3 contiguous, or a
coefficient near a rounding boundary) goes to Pillow.
"""
from __future__ import annotations

import ctypes
import os
import concurrent.futures as cf
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_I = ctypes.POINTER(ctypes.c_int)
_U8 = ctypes.POINTER(ctypes.c_uint8)
MARGIN = 1e-6
THREADS = 4
_lib_cache = []


def _lib():
    if not _lib_cache:
        path = HERE / ("pilresize.dll" if os.name == "nt" else "libpilresize.so")
        lib = None
        if path.exists():
            lib = ctypes.CDLL(str(path))
            lib.pr_coeffs.restype = ctypes.c_int
            lib.pr_coeffs.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, _I, _I,
                                      ctypes.c_double]
            lib.pr_horiz.restype = None
            lib.pr_horiz.argtypes = [_U8, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _U8, ctypes.c_int,
                                     ctypes.c_int, _I, _I, ctypes.c_int]
            lib.pr_vert.restype = None
            lib.pr_vert.argtypes = [_U8, ctypes.c_int, ctypes.c_int, _U8, ctypes.c_int, ctypes.c_int, ctypes.c_int, _I,
                                    _I, ctypes.c_int]
        _lib_cache.append(lib)
    return _lib_cache[0]


def _coeffs(lib, in_size, out_size):
    kmax = int(np.ceil(2.0 * max(1.0, in_size / out_size))) * 2 + 1
    bounds = np.zeros(2 * out_size, np.int32)
    kk = np.zeros(out_size * kmax, np.int32)
    k = lib.pr_coeffs(in_size, 0.0, float(in_size), out_size, kmax, bounds.ctypes.data_as(_I), kk.ctypes.data_as(_I),
                      MARGIN)
    return (k, bounds, kk) if k > 0 else None


def _ranges(n, parts):
    step = -(-n // parts)
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def resize_rgb(img, w_out, h_out, threads=THREADS):
    """Pillow's Image.fromarray(img).resize((w_out, h_out)) as an array, or None when it must be left to Pillow."""
    lib = _lib()
    if (lib is None or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3 or img.strides[1:] != (3, 1)
            or img.strides[0] < 3 * img.shape[1] or w_out == img.shape[1] or h_out == img.shape[0] or w_out < 1
            or h_out < 1):
        return None                                  # (rows may be strided: autocrop's views of the photo)
    h_in, w_in = img.shape[:2]
    ch, cv = _coeffs(lib, w_in, w_out), _coeffs(lib, h_in, h_out)
    if ch is None or cv is None:
        return None
    kh, bh, kkh = ch
    kv, bv, kkv = cv
    first = int(bv[0])
    last = int(bv[2 * h_out - 2] + bv[2 * h_out - 1])
    bv = bv.copy()
    bv[0::2] -= first
    tmp = np.empty((last - first, w_out, 3), np.uint8)
    out = np.empty((h_out, w_out, 3), np.uint8)
    p_src, p_tmp, p_out = img.ctypes.data_as(_U8), tmp.ctypes.data_as(_U8), out.ctypes.data_as(_U8)
    with cf.ThreadPoolExecutor(threads) as ex:
        list(ex.map(lambda r: lib.pr_horiz(p_src, img.strides[0], r[0], r[1], first, p_tmp, w_out * 3, w_out,
                                           bh.ctypes.data_as(_I), kkh.ctypes.data_as(_I), kh),
                    [(a + first, b + first) for a, b in _ranges(last - first, threads)]))
        list(ex.map(lambda r: lib.pr_vert(p_tmp, w_out * 3, w_out, p_out, w_out * 3, r[0], r[1],
                                          bv.ctypes.data_as(_I), kkv.ctypes.data_as(_I), kv),
                    _ranges(h_out, threads)))
    return out


def install(hm, check=False):
    """Replace homr.main.resize_image (hm) if it is still homr.resize.resize_image; returns a report string."""
    import PIL
    import homr.resize as hr
    if hm.resize_image is not hr.resize_image:
        return "skipped: resize_image already replaced"
    if PIL.__version__ != "12.3.0":
        return f"skipped: Pillow {PIL.__version__} != 12.3.0"
    if _lib() is None:
        return "skipped: libpilresize missing"
    orig = hr.resize_image

    def resize_image(image_arr):
        tw, th = hr.calc_target_image_size(image_arr.shape[1], image_arr.shape[0])
        if tw == image_arr.shape[1] and th == image_arr.shape[0]:
            return image_arr
        out = resize_rgb(image_arr, tw, th)
        if out is None:
            return orig(image_arr)
        if check:
            ref = orig(image_arr)
            if ref.shape != out.shape or not np.array_equal(ref, out):
                raise AssertionError("pil_resize differs from Pillow")
        return out

    hm.resize_image = resize_image
    return "installed"
