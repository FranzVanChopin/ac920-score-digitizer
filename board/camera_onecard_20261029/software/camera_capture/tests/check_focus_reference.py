"""Compare the C firmware metric to the user's unmodified edge_widths function.

Requires numpy/OpenCV and an explicit reference script; does not run that script's
main or write __pycache__ beside it. The DLL must be built from capture_focus.c.
"""
import argparse
import ast
import ctypes as ct
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


class Rect(ct.Structure):
    _fields_ = [(s, ct.c_uint) for s in ("x", "y", "width", "height")]


class Result(ct.Structure):
    _fields_ = [("roi", Rect)] + [(s, ct.c_uint) for s in (
        "p5", "p95", "clip_permille", "edge_pixels", "flags", "tenengrad_x10000",
        "edge_h_milli", "edge_v_milli", "edge_h_count", "edge_v_count", "edges_computed")]


class Workspace(ct.Structure):
    _fields_ = [("gray", ct.c_uint8 * 250000), ("profile", ct.c_uint16 * 500),
                ("edge_hist", ct.c_uint32 * 16001)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = args.reference.read_bytes()
    module = ast.parse(source.decode("utf-8-sig"))
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "edge_widths")
    namespace = {"cv2": cv2, "np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.reference), "exec"), namespace)
    reference = namespace["edge_widths"]
    dll = ct.CDLL(str(args.dll.resolve()))
    dll.capture_focus_measure.argtypes = [ct.POINTER(ct.c_uint8), ct.c_uint, ct.c_uint,
        ct.c_size_t, ct.c_uint, ct.c_int, ct.POINTER(Workspace), ct.POINTER(Result), ct.c_void_p, ct.c_void_p]
    dll.capture_focus_measure.restype = ct.c_int
    workspace = Workspace()
    rng = np.random.default_rng(477)
    reports = []

    def check(g, label, region=0):
        h, w = g.shape
        image = np.empty((h, w, 3), dtype=np.uint8)
        image[:, :, 0] = 7
        image[:, :, 1] = g
        image[:, :, 2] = 249
        digest = hashlib.sha256(image).hexdigest()
        result = Result()
        rc = dll.capture_focus_measure(image.ctypes.data_as(ct.POINTER(ct.c_uint8)),
            w, h, image.strides[0], region, 1, ct.byref(workspace), ct.byref(result), None, None)
        assert rc == 0 and hashlib.sha256(image).hexdigest() == digest
        rect = result.roi
        roi = g[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]
        errors = []
        for axis, count, value in ((0, result.edge_h_count, result.edge_h_milli),
                                    (1, result.edge_v_count, result.edge_v_milli)):
            widths = reference(roi, axis)
            assert count == len(widths), (label, axis, count, len(widths))
            err = abs(value / 1000.0 - float(np.median(widths))) if count else 0
            assert err <= .0011, (label, axis, err)
            errors.append(err)
        gx = cv2.Sobel(roi.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)[1:-1, 1:-1]
        gy = cv2.Sobel(roi.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)[1:-1, 1:-1]
        p5, p95 = np.floor(np.percentile(roi, [5, 95]) + .5).astype(int)
        assert (result.p5, result.p95) == (p5, p95)
        if not result.flags:
            expected = float(np.mean(gx * gx + gy * gy) / (p95 - p5) ** 2 * 10000)
            assert abs(expected - result.tenengrad_x10000) <= .50001
        reports.append(dict(case=label, region=region, roi=[rect.x, rect.y, rect.width, rect.height],
            flags=result.flags, t_x10000=result.tenengrad_x10000, edges=[result.edge_h_count, result.edge_v_count],
            widths_milli=[result.edge_h_milli, result.edge_v_milli], max_error_px=max(errors)))
        return result

    base = np.full((500, 500), 230, np.float64)
    base[250:, :] = 30
    for noise in (0., 2.):
        widths, scores = [], []
        for sigma in (.5, 1., 2., 4.):
            blurred = cv2.GaussianBlur(base, (0, 0), sigma)
            g = np.clip(np.rint(blurred + rng.normal(0, noise, base.shape)), 0, 255).astype(np.uint8)
            r = check(g, f"horizontal_sigma{sigma}_noise{noise}")
            check(g.T.copy(), f"vertical_sigma{sigma}_noise{noise}")
            widths.append(r.edge_h_milli)
            scores.append(r.tenengrad_x10000)
            half = check(np.rint(g / 2).astype(np.uint8), f"half_sigma{sigma}_noise{noise}")
            assert abs(half.tenengrad_x10000 / r.tenengrad_x10000 - 1) <= .05
        assert all(a < b for a, b in zip(widths, widths[1:]))
        assert all(a > b for a, b in zip(scores, scores[1:]))
    # Five distinct patches in one immutable native frame; coordinates match
    # the supplied script and each orientation/blur differs deliberately.
    full = np.full((3040, 4056), 220, np.uint8)
    centers = [(2028, 1520), (676, 506), (3380, 506), (676, 2533), (3380, 2533)]
    for i, (x, y) in enumerate(centers):
        pattern = np.where((np.indices((500, 500))[0] // 32 + np.indices((500, 500))[1] // 48) % 2, 35., 225.)
        full[y-250:y+250, x-250:x+250] = np.rint(cv2.GaussianBlur(pattern, (0, 0), .6 + i * .7)).astype(np.uint8)
    for i in range(5):
        check(full, f"same_frame_region{i}", i)
    report = dict(passed=True, reference=str(args.reference), reference_sha256=hashlib.sha256(source).hexdigest(),
        dll_sha256=hashlib.sha256(args.dll.read_bytes()).hexdigest(), cases=reports,
        c_source_sha256=hashlib.sha256((Path(__file__).parents[1] / "capture_focus.c").read_bytes()).hexdigest(),
        test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_error_px=max(r["max_error_px"] for r in reports),
        scope="Identical pre-CCM G samples only. Does not assert agreement with post-CCM grayscale BMP or absolute optical PSF.")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS focus reference: {len(reports)} cases; max width error={report['max_error_px']:.6f}px; independent Sobel, noise, brightness, five native ROIs")


if __name__ == "__main__":
    main()
