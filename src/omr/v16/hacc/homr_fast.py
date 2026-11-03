"""Exact, faster replacements for the hot geometry of Homr 0.7.0's staff detection.

install() patches Homr's module and class attributes at run time (the installed Homr package and
Codex's files are never modified).  Every replacement returns exactly what the original returns:

  _can_shapes_possibly_touch    same arithmetic; the two ABC isinstance checks (1.2M calls/page on
                                the A53) are replaced by a per-type cache of their answers
  find_staff_anchors            Homr's code, except that the staff fragments a thickened symbol can
                                intersect are pre-screened in C++ (hg_screen, superset margin) and
                                only those go through Homr's own is_intersecting
  connect_staff_lines           grouping in C++ (hg_connect): same visiting order, same float ops,
                                tan() taken from Python's math.tan
  StaffLineSegment.merge        `fragment not in list` (O(n*m) __eq__ calls) -> set of box tuples
  combine_noteheads_with_stems  stems pre-screened per notehead (hg_screen), first hit in stem order
  is_overlapping_with_any       others pre-screened (hg_screen); any() is order-independent
  Staff.get_at                  numpy argmin over the grid x (first minimum, as min() returns)
  StaffLineSegment.__init__     per-fragment numbers cached on the fragment; sort/min/max via map()
  get_staff_for_anchor,         fragment-set inclusion on cached frozensets of boxes (== and hash
  find_raw_staffs_...           of AngledBoundingBox are those of .box); zone filter in numpy
  RawStaff.__init__             contour arrays concatenated instead of split into points and re-stacked
  do_polygons_overlap           only points within 1 px of the other polygon's bounding box go to
                                cv2.pointPolygonTest (the rest are outside); any() is order-free
  v16 (route A, 10-27):
  resample_staff_segment        the per-x loop in C++ (hg_resample), same float ops and order; StaffPoints
                                built without __init__ (average_unit_size from C++, numpy's summation order)
  find_raw_staffs_...           the RawStaff of an anchor merged at once is not built unless its constructor
                                could raise; unmatched connected lines are not made into segments
  are_lines_crossing            fragment pairs in C++ (hg_crossing): distance screen decided outside a margin,
                                OpenCV's integer pointPolygonTest branch (hg_ppt, fuzz-checked against cv2)
  create_noise_grid             without the debug drawing when no debug image is written

HOMR_FAST_CHECK=1 runs the original next to every replacement and raises on any difference.
Nothing is patched if Homr is not 0.7.0 or a patched function's source changed (sha256 guard), or
if the C++ library is missing (then only the pure-Python patches apply).
"""
import ctypes
import hashlib
import inspect
import math
import operator
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHECK = os.environ.get("HOMR_FAST_CHECK") == "1"
MARGIN = 1e-9                       # relative slack of the C++ screen (python rounds sqrt differently)
POLY_MIN = 64                       # do_polygons_overlap prefilter only from this many points (v7 board profile)
STATS = {"checked": 0}

# sha256[:16] of the original functions this module replaces (Homr 0.7.0)
SOURCES = {
    "bounding_boxes.AngledBoundingBox._can_shapes_possibly_touch": None,
    "bounding_boxes.AngledBoundingBox.is_overlapping_with_any": None,
    "staff_detection.find_staff_anchors": None,
    "staff_detection.connect_staff_lines": None,
    "staff_detection.StaffLineSegment.merge": None,
    "note_detection.combine_noteheads_with_stems": None,
    "model.Staff.get_at": None,
    "staff_detection.StaffLineSegment.__init__": None,
    "staff_detection.get_staff_for_anchor": None,
    "staff_detection.find_raw_staffs_by_connecting_line_fragments": None,
    "staff_detection.RawStaff.__init__": None,
    "bounding_boxes.do_polygons_overlap": None,
}

_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)


def _lib():
    name = "homrgeo.dll" if os.name == "nt" else "libhomrgeo.so"
    path = HERE / name
    if not path.exists():
        return None
    lib = ctypes.CDLL(str(path))
    lib.hg_screen.restype = ctypes.c_int
    lib.hg_screen.argtypes = [ctypes.c_int, _D, _D, _D, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                              ctypes.c_double, _I]
    lib.hg_connect.restype = ctypes.c_int
    lib.hg_connect.argtypes = [ctypes.c_int, _D, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int,
                               _I, ctypes.c_int, _I]
    try:
        lib.hg_resample.restype = ctypes.c_int
        lib.hg_resample.argtypes = [ctypes.c_int, _I, _D, ctypes.c_double, _D, ctypes.c_int, _D, _I, _D, _D, _D]
        lib.hg_crossing.restype = ctypes.c_int
        lib.hg_crossing.argtypes = [ctypes.c_int, _I, _D, _D, _D, _I, _I, ctypes.c_double]
    except AttributeError:                    # an older build without hg_resample / hg_crossing
        pass
    return lib


def _sha(obj):
    return hashlib.sha256(inspect.getsource(obj).encode("utf-8")).hexdigest()[:16]


def _differ(name, a, b):
    raise AssertionError(f"homr_fast: {name} differs from Homr's original: {a!r} != {b!r}")


# ------------------------------------------------------------------ geometry arrays for screening
class _Screen:
    """Centres and major axes of a box list, as _can_shapes_possibly_touch reads them."""

    def __init__(self, lib, boxes, kind_of):
        self.lib = lib
        self.n = len(boxes)
        g = np.empty((3, self.n), dtype=np.float64)
        for i, b in enumerate(boxes):
            c, a, _ = b.rotated_box if kind_of(b) == 1 else b.box
            g[0, i], g[1, i], g[2, i] = c[0], c[1], max(a)
        self.g = g
        self.cx, self.cy, self.mj = (np.ascontiguousarray(g[k]) for k in range(3))
        self.pcx, self.pcy, self.pmj = (a.ctypes.data_as(_D) for a in (self.cx, self.cy, self.mj))
        self.out = np.empty(max(1, self.n), dtype=np.int32)
        self.pout = self.out.ctypes.data_as(_I)

    def near(self, centre, axes):
        k = self.lib.hg_screen(self.n, self.pcx, self.pcy, self.pmj, float(centre[0]), float(centre[1]),
                               float(max(axes)), MARGIN, self.pout)
        return self.out[:k].tolist()


def install():
    import homr
    import homr.bounding_boxes as bb
    import homr.main as hm
    import homr.model as model
    import homr.note_detection as nd
    import homr.staff_detection as sd
    from homr import constants

    report = {"installed": [], "skipped": []}
    try:
        from importlib.metadata import version
        hv = version("homr")
    except Exception:  # noqa: BLE001
        hv = getattr(homr, "__version__", "?")
    if hv != "0.7.0":
        report["skipped"].append(f"homr {hv} != 0.7.0")
        return report
    objs = {
        "bounding_boxes.AngledBoundingBox._can_shapes_possibly_touch": bb.AngledBoundingBox._can_shapes_possibly_touch,
        "bounding_boxes.AngledBoundingBox.is_overlapping_with_any": bb.AngledBoundingBox.is_overlapping_with_any,
        "staff_detection.find_staff_anchors": sd.find_staff_anchors,
        "staff_detection.connect_staff_lines": sd.connect_staff_lines,
        "staff_detection.StaffLineSegment.merge": sd.StaffLineSegment.merge,
        "note_detection.combine_noteheads_with_stems": nd.combine_noteheads_with_stems,
        "model.Staff.get_at": model.Staff.get_at,
        "staff_detection.StaffLineSegment.__init__": sd.StaffLineSegment.__init__,
        "staff_detection.get_staff_for_anchor": sd.get_staff_for_anchor,
        "staff_detection.find_raw_staffs_by_connecting_line_fragments": sd.find_raw_staffs_by_connecting_line_fragments,
        "staff_detection.RawStaff.__init__": sd.RawStaff.__init__,
        "bounding_boxes.do_polygons_overlap": bb.do_polygons_overlap,
    }
    if getattr(bb.AngledBoundingBox._can_shapes_possibly_touch, "_homr_fast", False):
        report["skipped"].append("already installed")
        return report
    for name, obj in objs.items():
        if SOURCES[name] is not None and _sha(obj) != SOURCES[name]:
            report["skipped"].append(f"{name}: source changed")
            return report
    lib = _lib()

    # ---------------------------------------------------------------- _can_shapes_possibly_touch
    kinds = {}

    def kind_of(other):
        t = type(other)
        k = kinds.get(t)
        if k is None:
            k = kinds[t] = 1 if isinstance(other, bb.BoundingBox) else 2 if isinstance(other, bb.AngledBoundingBox) else 0
        return k

    orig_touch = bb.AngledBoundingBox._can_shapes_possibly_touch

    def touch(self, other):
        k = kind_of(other)
        if k == 1:
            center2, axes2, _ = other.rotated_box
        elif k == 2:
            center2, axes2, _ = other.box
        else:
            raise ValueError(f"Unknown type {type(other)}")
        center1, axes1, _ = self.box
        major_axis1 = max(axes1)
        major_axis2 = max(axes2)
        distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
        if distance > major_axis1 + major_axis2:
            return False
        return True

    def touch_checked(self, other):
        r = touch(self, other)
        o = orig_touch(self, other)
        if r != o:
            _differ("_can_shapes_possibly_touch", r, o)
        return r

    new_touch = touch_checked if CHECK else touch
    new_touch._homr_fast = True
    bb.AngledBoundingBox._can_shapes_possibly_touch = new_touch
    report["installed"].append("_can_shapes_possibly_touch")

    # ---------------------------------------------------------------- StaffLineSegment.merge
    orig_merge = sd.StaffLineSegment.merge
    eq_types = {}

    def box_eq(f):
        """True if f == g is AngledBoundingBox.__eq__ (box tuple equality) for this type."""
        t = type(f)
        k = eq_types.get(t)
        if k is None:
            k = eq_types[t] = kind_of(f) == 2 and t.__eq__ is bb.AngledBoundingBox.__eq__
        return k

    def merge(self, other):
        staff_lines = self.staff_fragments.copy()
        if not all(box_eq(f) for f in staff_lines) or not all(box_eq(f) for f in other.staff_fragments):
            return orig_merge(self, other)
        try:
            boxes = {f.box for f in staff_lines}     # AngledBoundingBox.__eq__ compares .box
            for fragment in other.staff_fragments:
                if fragment.box not in boxes:
                    staff_lines.append(fragment)
                    boxes.add(fragment.box)
        except TypeError:                            # unhashable box: keep Homr's list scan
            return orig_merge(self, other)
        return sd.StaffLineSegment(self.debug_id, staff_lines)

    def merge_checked(self, other):
        r, o = merge(self, other), orig_merge(self, other)
        if [id(f) for f in r.staff_fragments] != [id(f) for f in o.staff_fragments]:
            _differ("StaffLineSegment.merge", len(r.staff_fragments), len(o.staff_fragments))
        STATS["checked"] += 1
        return r

    sd.StaffLineSegment.merge = merge_checked if CHECK else merge
    report["installed"].append("StaffLineSegment.merge")

    # ---------------------------------------------------------------- StaffLineSegment.__init__
    orig_seg_init = sd.StaffLineSegment.__init__
    v_of = operator.attrgetter("_homr_fast_v")
    first, second, third, fourth, fifth = (operator.itemgetter(k) for k in range(5))

    def frag_v(f):
        """The five numbers StaffLineSegment.__init__ reads, by the same expressions.  Boxes are
        only assigned in constructors, so the cached values never go stale."""
        v = (f.box[0][0], f.center[0] - f.size[0] / 2, f.center[0] + f.size[0] / 2,
             f.center[1] - f.size[1] / 2, f.center[1] + f.size[1] / 2)
        f.__dict__["_homr_fast_v"] = v
        return v

    def seg_init(self, debug_id, staff_fragments):
        if not staff_fragments:
            return orig_seg_init(self, debug_id, staff_fragments)
        try:
            vals = list(map(v_of, staff_fragments))
        except AttributeError:
            vals = [f.__dict__.get("_homr_fast_v") or frag_v(f) for f in staff_fragments]
        keys = list(map(first, vals))
        order = sorted(range(len(keys)), key=keys.__getitem__)       # stable, like sorted(key=box[0][0])
        self.debug_id = debug_id
        self.staff_fragments = list(map(staff_fragments.__getitem__, order))
        self.min_x = min(map(second, vals))                           # same values in the same order
        self.max_x = max(map(third, vals))
        self.min_y = min(map(fourth, vals))
        self.max_y = max(map(fifth, vals))

    def seg_init_checked(self, debug_id, staff_fragments):
        seg_init(self, debug_id, staff_fragments)
        o = sd.StaffLineSegment.__new__(sd.StaffLineSegment)
        orig_seg_init(o, debug_id, staff_fragments)
        a = ([id(f) for f in self.staff_fragments], self.min_x, self.max_x, self.min_y, self.max_y)
        b = ([id(f) for f in o.staff_fragments], o.min_x, o.max_x, o.min_y, o.max_y)
        if a != b:
            _differ("StaffLineSegment.__init__", a[1:], b[1:])

    sd.StaffLineSegment.__init__ = seg_init_checked if CHECK else seg_init
    report["installed"].append("StaffLineSegment.__init__")

    # ---------------------------------------------------------------- fragment-set inclusion
    box_of = operator.attrgetter("box")

    def boxes(seg):
        """frozenset of the fragments' boxes: what set(seg.staff_fragments) compares, since hash and
        == of AngledBoundingBox are those of .box.  None if some fragment type compares otherwise."""
        c = seg.__dict__.get("_homr_fast_boxes")
        if c is None or c[0] is not seg.staff_fragments:
            frags = seg.staff_fragments
            fs = frozenset(map(box_of, frags)) if all(box_eq(f) for f in frags) else None
            c = seg.__dict__["_homr_fast_boxes"] = (frags, fs)
        return c[1]

    def subset(req_seg, seg):
        a, b = boxes(req_seg), boxes(seg)
        if a is None or b is None:
            return set(req_seg.staff_fragments).issubset(set(seg.staff_fragments))
        return a <= b

    # ---------------------------------------------------------------- get_staff_for_anchor
    orig_get_staff = sd.get_staff_for_anchor

    def get_staff_for_anchor(anchor, staffs):
        for staff in staffs:
            for i, anchor_line in enumerate(anchor.staff_lines):
                if subset(anchor_line, staff.lines[i]):
                    return staff
        return None

    def get_staff_checked(anchor, staffs):
        r, o = get_staff_for_anchor(anchor, staffs), orig_get_staff(anchor, staffs)
        if r is not o:
            _differ("get_staff_for_anchor", r, o)
        return r

    sd.get_staff_for_anchor = get_staff_checked if CHECK else get_staff_for_anchor
    report["installed"].append("get_staff_for_anchor")

    # ---------------------------------------------------------------- find_raw_staffs_by_connecting_line_fragments
    orig_raw = sd.find_raw_staffs_by_connecting_line_fragments

    def find_raw_staffs(anchors, staff_fragments):
        """Homr's function; the zone filter runs on a numpy copy of the fragments' centre y and the
        fragment-set tests on the cached box sets."""
        staffs = []
        staff_id = 0
        cy = np.array([f.center[1] for f in staff_fragments], dtype=np.float64)
        for anchor in anchors:
            existing_staff = sd.get_staff_for_anchor(anchor, staffs)
            keep = np.flatnonzero((cy >= anchor.zone.start) & (cy <= anchor.zone.stop)).tolist()
            fragments = list(map(staff_fragments.__getitem__, keep))
            connected = sd.connect_staff_lines(fragments, anchor.average_unit_size)
            staff_lines = []
            for anchor_line in anchor.staff_lines:
                matching_anchor = [line for line in connected if subset(anchor_line, line)]
                if len(matching_anchor) == 1:
                    staff_lines.extend(matching_anchor)
                else:
                    staff_lines.append(anchor_line)
            if existing_staff:
                staffs.remove(existing_staff)
                staffs.append(existing_staff.merge(sd.RawStaff(staff_id, staff_lines, [anchor])))
            else:
                staffs.append(sd.RawStaff(staff_id, staff_lines, [anchor]))
            staff_id += 1
        return staffs

    def sig_raw(staffs):
        return [(s.staff_id, s.box, len(s.anchors), [[id(f) for f in l.staff_fragments] for l in s.lines])
                for s in staffs]

    def raw_checked(anchors, staff_fragments):
        r, o = find_raw_staffs(anchors, staff_fragments), orig_raw(anchors, staff_fragments)
        if sig_raw(r) != sig_raw(o):
            _differ("find_raw_staffs_by_connecting_line_fragments", len(r), len(o))
        STATS["checked"] += 1
        return r

    sd.find_raw_staffs_by_connecting_line_fragments = raw_checked if CHECK else find_raw_staffs
    report["installed"].append("find_raw_staffs_by_connecting_line_fragments")

    # ---------------------------------------------------------------- RawStaff.__init__
    orig_raw_init = sd.RawStaff.__init__

    def raw_init(self, staff_id, lines, anchors):
        """Homr splits every fragment contour (K,1,2) into K arrays of shape (1,2) and stacks them
        again with np.array(list): concatenating the contour arrays gives the same (N,1,2) array."""
        arrs = [f.contours for line in lines for f in line.staff_fragments]
        parts = []
        for c in arrs:
            if not isinstance(c, np.ndarray):
                return orig_raw_init(self, staff_id, lines, anchors)
            if c.size == 0 and c.ndim == 1:          # np.array([]) of a predicted anchor: adds nothing
                continue
            if c.ndim != 3 or c.shape[1:] != (1, 2):
                return orig_raw_init(self, staff_id, lines, anchors)
            if len(c):
                parts.append(c)
        if not parts:
            return orig_raw_init(self, staff_id, lines, anchors)
        pts = np.concatenate(parts)
        box = sd.cv2.minAreaRect(pts)
        super(sd.RawStaff, self).__init__(box, pts.reshape(-1, 2).copy(), staff_id)
        self.staff_id = staff_id
        self.lines = lines
        self.anchors = anchors
        self.min_x = self.center[0] - self.size[0] / 2
        self.max_x = self.center[0] + self.size[0] / 2
        self.min_y = self.center[1] - self.size[1] / 2
        self.max_y = self.center[1] + self.size[1] / 2

    def raw_init_checked(self, staff_id, lines, anchors):
        raw_init(self, staff_id, lines, anchors)
        o = sd.RawStaff.__new__(sd.RawStaff)
        orig_raw_init(o, staff_id, lines, anchors)
        same = (self.box == o.box and self.contours.dtype == o.contours.dtype and self.contours.shape == o.contours.shape
                and np.array_equal(self.contours, o.contours) and np.array_equal(self.polygon, o.polygon)
                and (self.min_x, self.max_x, self.min_y, self.max_y) == (o.min_x, o.max_x, o.min_y, o.max_y))
        if not same:
            _differ("RawStaff.__init__", self.box, o.box)

    sd.RawStaff.__init__ = raw_init_checked if CHECK else raw_init
    report["installed"].append("RawStaff.__init__")

    # ---------------------------------------------------------------- calculate_edges_of_rotated_rectangle
    orig_edges = bb.calculate_edges_of_rotated_rectangle
    f64 = np.float64

    def edges(box):
        """Homr builds four small numpy arrays per box; with python-number inputs numpy computes in
        float64, i.e. the same IEEE operations as plain floats.  Results stay np.float64."""
        (cx, cy), (w, h) = box[0], box[1]
        if not all(type(v) in (float, int) for v in (cx, cy, w, h)):
            return orig_edges(box)
        hw, hh = w / 2, h / 2
        return ((f64(cx - hw), f64(cy - hh)), (f64(cx + -hw), f64(cy + hh)), (f64(cx + hw), f64(cy + -hh)),
                (f64(cx + hw), f64(cy + hh)))

    def edges_checked(box):
        r, o = edges(box), orig_edges(box)
        if r != o or any(type(a) is not type(b) for p, q in zip(r, o) for a, b in zip(p, q)):
            _differ("calculate_edges_of_rotated_rectangle", r, o)
        return r

    bb.calculate_edges_of_rotated_rectangle = edges_checked if CHECK else edges
    report["installed"].append("calculate_edges_of_rotated_rectangle")

    # ---------------------------------------------------------------- get_center_extrapolated
    orig_extra = bb.RotatedBoundingBox.get_center_extrapolated

    def get_center_extrapolated(self, x):
        """the same expression, with np.tan of the (never reassigned) box angle computed once"""
        c = self.__dict__.get("_homr_fast_tan")
        if c is None or c[0] is not self.box:
            c = self.__dict__["_homr_fast_tan"] = (self.box, np.tan(self.box[2] / 180 * np.pi))
        return (x - self.box[0][0]) * c[1] + self.box[0][1]

    def extra_checked(self, x):
        r, o = get_center_extrapolated(self, x), orig_extra(self, x)
        if not (r == o and type(r) is type(o)):
            _differ("get_center_extrapolated", r, o)
        return r

    bb.RotatedBoundingBox.get_center_extrapolated = extra_checked if CHECK else get_center_extrapolated
    report["installed"].append("get_center_extrapolated")

    # ---------------------------------------------------------------- break_wide_fragments
    orig_break = sd.break_wide_fragments

    def break_wide_fragments(fragments, limit=100):
        """Homr's loop with the contour split done by numpy indexing.  Homr appends the first point of
        the right part to the left part, and then the left part's (new) last point -- the same point --
        to the right part; both kept."""
        result = []
        for fragment in fragments:
            remaining_fragment = fragment
            while remaining_fragment.size[0] > limit:
                cont = remaining_fragment.contours
                if not (isinstance(cont, np.ndarray) and cont.ndim == 3 and cont.shape[1:] == (1, 2) and len(cont)):
                    return orig_break(fragments, limit)
                xs = cont[:, 0, 0]
                min_x = xs.min()
                left_mask = xs < min_x + limit
                li, ri = np.flatnonzero(left_mask), np.flatnonzero(~left_mask)
                if len(li) == 0 or len(ri) == 0:
                    break
                li = li[np.argsort(xs[li], kind="stable")]         # sorted(..., key=c[0][0]) is stable
                ri = ri[np.argsort(xs[ri], kind="stable")]
                left = np.concatenate([cont[li], cont[ri[:1]]])
                right = np.concatenate([cont[ri], cont[ri[:1]]])
                result.append(sd.create_rotated_bounding_box(left, remaining_fragment.debug_id))
                remaining_fragment = sd.create_rotated_bounding_box(right, remaining_fragment.debug_id)
            result.append(remaining_fragment)
        return result

    def break_checked(fragments, limit=100):
        r, o = break_wide_fragments(fragments, limit), orig_break(fragments, limit)
        sig = lambda fs: [(f.box, f.contours.shape, f.contours.dtype.str, f.contours.tobytes(), f.debug_id) for f in fs]  # noqa: E731
        if sig(r) != sig(o):
            _differ("break_wide_fragments", len(r), len(o))
        return r

    sd.break_wide_fragments = break_checked if CHECK else break_wide_fragments
    hm.break_wide_fragments = sd.break_wide_fragments                 # main imported it by name
    report["installed"].append("break_wide_fragments")

    # ---------------------------------------------------------------- do_polygons_overlap
    orig_poly = bb.do_polygons_overlap
    ppt = bb.cv2.pointPolygonTest

    def inside_any(points, poly):
        """any(pointPolygonTest(poly, p) >= 0): a point more than 1 px outside the polygon's vertex
        bounding box is outside the polygon, so only the others are tested (in their order)."""
        lo, hi = poly.min(axis=0) - 1.0, poly.max(axis=0) + 1.0
        near = np.flatnonzero((points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0]) &
                              (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]))
        for i in near.tolist():
            if ppt(poly, (float(points[i, 0]), float(points[i, 1])), False) >= 0:
                return True
        return False

    def do_polygons_overlap(poly1, poly2):
        if not (isinstance(poly1, np.ndarray) and isinstance(poly2, np.ndarray) and poly1.ndim == 2
                and poly2.ndim == 2 and poly1.shape[1] == 2 and poly2.shape[1] == 2 and len(poly1) and len(poly2)
                and poly1.dtype.kind in "iu" and poly2.dtype.kind in "iu") or len(poly1) + len(poly2) < POLY_MIN:
            return orig_poly(poly1, poly2)      # small polygons (4-point boxes): numpy costs more on the A53
        return inside_any(poly1, poly2) or inside_any(poly2, poly1)

    def poly_checked(poly1, poly2):
        r, o = do_polygons_overlap(poly1, poly2), orig_poly(poly1, poly2)
        if r != o:
            _differ("do_polygons_overlap", r, o)
        return r

    bb.do_polygons_overlap = poly_checked if CHECK else do_polygons_overlap
    report["installed"].append("do_polygons_overlap")

    # ---------------------------------------------------------------- Staff.get_at
    orig_get_at = model.Staff.get_at
    tol = constants.staff_position_tolerance

    def get_at(self, x):
        """grid x cached per Staff; rebuilt when the grid list, its length or its end points change
        (Homr and Codex build a new Staff instead of editing a grid)."""
        grid = self.grid
        c = self.__dict__.get("_homr_fast_grid")
        if c is None or c[0] is not grid or c[1] != len(grid) or not grid:
            if not grid:
                return orig_get_at(self, x)
            c = (grid, len(grid), np.array([p.x for p in grid], dtype=np.float64), grid[0], grid[-1])
            self.__dict__["_homr_fast_grid"] = c
        if c[3] is not grid[0] or c[4] is not grid[-1] or not math.isfinite(x):
            return orig_get_at(self, x)
        closest_point = grid[int(np.argmin(np.abs(c[2] - x)))]
        if abs(closest_point.x - x) > tol:
            return None
        return closest_point

    def get_at_checked(self, x):
        r, o = get_at(self, x), orig_get_at(self, x)
        if r is not o:
            _differ("Staff.get_at", r, o)
        return r

    model.Staff.get_at = get_at_checked if CHECK else get_at
    report["installed"].append("Staff.get_at")

    # ---------------------------------------------------------------- create_noise_grid (v16)
    import homr.noise_filtering as nfl
    orig_noise_grid = nfl.create_noise_grid
    homr_noise_fns = (nfl.estimate_noise, nfl.create_grid, nfl.apply_noise_filter, nfl.get_neighbors,
                      nfl.handle_filter_results)

    def create_noise_grid(gray, debug):
        """Homr's create_noise_grid without the debug drawing when no debug image is written (a BGR copy of the page,
        a rectangle and a putText per cell; they touch nothing else).  The grid is Homr's own create_grid (OpenCV
        per tile was faster than any vectorized form tried, `README.md`)."""
        if (debug.debug or not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8
                or (nfl.estimate_noise, nfl.create_grid, nfl.apply_noise_filter, nfl.get_neighbors,
                    nfl.handle_filter_results) != homr_noise_fns):
            return orig_noise_grid(gray, debug)
        H, W = gray.shape
        M, N = H // 20, W // 20
        if M < 1 or N < 1:
            return orig_noise_grid(gray, debug)
        grid = nfl.create_grid(gray, M, N)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        filtered_cells, total_cells = 0, 0
        for i, y1 in enumerate(range(0, H, M)):           # apply_noise_filter without the drawing
            for j, x1 in enumerate(range(0, W, N)):
                y2, x2 = y1 + M, x1 + N
                noise = grid[i, j]
                neighbors = nfl.get_neighbors(grid, i, j)
                any_neighbor_above_limit = np.any(np.array(neighbors) > constants.image_noise_limit)
                if noise > constants.image_noise_limit and any_neighbor_above_limit:
                    filtered_cells += 1
                else:
                    mask[y1:y2, x1:x2] = 255
                total_cells += 1
        return nfl.handle_filter_results(filtered_cells, total_cells, mask)

    def noise_checked(gray, debug):
        r, o = create_noise_grid(gray, debug), orig_noise_grid(gray, debug)
        if (r is None) != (o is None) or (r is not None and not (r.dtype == o.dtype and np.array_equal(r, o))):
            _differ("create_noise_grid", r is None, o is None)
        STATS["checked"] += 1
        return r

    nfl.create_noise_grid = noise_checked if CHECK else create_noise_grid
    report["installed"].append("create_noise_grid")

    if lib is None:
        report["skipped"].append("C++ library missing: screens and connect_staff_lines not installed")
        return report

    # ---------------------------------------------------------------- connect_staff_lines
    orig_connect = sd.connect_staff_lines

    def feats(f):
        v = f.__dict__.get("_homr_fast_f")
        if v is None or v[0] is not f.box:
            v = (f.box, float(f.bottom_left[0]), float(f.bottom_right[0]), float(f.box[0][0]), float(f.box[0][1]),
                 float(f.size[0]), math.tan(f.box[2] * math.pi / 180.0), 0.0)
            f.__dict__["_homr_fast_f"] = v
        return v[1:]

    def connect_groups(staff_lines, unit_size, F=None):
        """connect_staff_lines' groups top to bottom, as lists of indices into staff_lines (a fragment can be in
        several groups), or None where Homr's own function must run.  F: the fragments' feats() rows, if the
        caller has them (then it has checked the types and that they are finite)."""
        n = len(staff_lines)
        if F is None:
            if not all(type(f) is bb.RotatedBoundingBox for f in staff_lines):
                return None
            F = np.array([feats(f) for f in staff_lines], dtype=np.float64)
            if not np.isfinite(F).all():
                return None
        max_gap = float(constants.max_line_gap_size(unit_size))
        tol_ = float(constants.tolerance_for_staff_line_detection(unit_size))
        short = float(constants.is_short_line(unit_size))
        if not (math.isfinite(max_gap) and math.isfinite(tol_) and math.isfinite(short)):
            return None
        cap = 8 * n + 64
        lens = np.empty(n, dtype=np.int32)
        idx = np.empty(cap, dtype=np.int32)
        F = np.ascontiguousarray(F)
        k = lib.hg_connect(n, F.ctypes.data_as(_D), max_gap, tol_, short, n, lens.ctypes.data_as(_I), cap,
                           idx.ctypes.data_as(_I))
        if k < 0:
            return None
        flat, groups, p = idx.tolist(), [], 0
        for m in lens[:k].tolist():
            groups.append(flat[p:p + m])
            p += m
        return groups

    def connect(staff_lines, unit_size):
        if len(staff_lines) == 0:
            return []
        groups = connect_groups(staff_lines, unit_size)
        if groups is None:
            return orig_connect(staff_lines, unit_size)
        return [sd.StaffLineSegment(g, [staff_lines[j] for j in idx]) for g, idx in enumerate(groups)]

    def sig_lines(lines):
        return [(s.debug_id, [id(f) for f in s.staff_fragments]) for s in lines]

    def connect_checked(staff_lines, unit_size):
        r, o = connect(staff_lines, unit_size), orig_connect(staff_lines, unit_size)
        if sig_lines(r) != sig_lines(o):
            _differ("connect_staff_lines", sig_lines(r)[:3], sig_lines(o)[:3])
        STATS["checked"] += 1
        return r

    sd.connect_staff_lines = connect_checked if CHECK else connect
    report["installed"].append("connect_staff_lines")

    # ---------------------------------------------------------------- find_raw_staffs, without the dropped objects
    homr_raw_merge = sd.RawStaff.merge

    def raw_init_safe(lines):
        """True when RawStaff(.., lines, ..) cannot raise: raw_init's own path (int32 (K,1,2) contours or empty
        1-D arrays) with at least one point; otherwise the RawStaff is built, to raise where Homr raises."""
        pts = 0
        for line in lines:
            for f in line.staff_fragments:
                v = f.__dict__.get("_homr_fast_ri")
                if v is None or v[0] is not f.contours:
                    c = f.contours
                    if isinstance(c, np.ndarray) and c.ndim == 1 and c.size == 0:
                        k = 0
                    elif isinstance(c, np.ndarray) and c.ndim == 3 and c.shape[1:] == (1, 2) and c.dtype == np.int32:
                        k = len(c)
                    else:
                        k = -1
                    v = f.__dict__["_homr_fast_ri"] = (c, k)
                if v[1] < 0:
                    return False
                pts += v[1]
        return pts > 0

    def find_raw_staffs_light(anchors, staff_fragments):
        """find_raw_staffs (above) without two kinds of objects Homr builds and drops: the RawStaff of an anchor that
        is merged at once into an existing staff (only its .lines are read; built anyway if its constructor could
        raise), and the connected lines no anchor line matches (the matching groups become StaffLineSegments with
        the same debug ids, one object per group as in Homr's `connected` list)."""
        if not all(box_eq(f) for f in staff_fragments) or sd.RawStaff.merge is not homr_raw_merge \
                or sd.RawStaff.__init__ not in (raw_init, raw_init_checked):
            return find_raw_staffs(anchors, staff_fragments)
        staffs = []
        staff_id = 0
        cy = np.array([f.center[1] for f in staff_fragments], dtype=np.float64)
        F_all = None                                  # feats() of every fragment once, rows taken per zone
        if staff_fragments and all(type(f) is bb.RotatedBoundingBox for f in staff_fragments):
            F_all = np.array([feats(f) for f in staff_fragments], dtype=np.float64)
            if not np.isfinite(F_all).all():
                F_all = None
        for anchor in anchors:
            existing_staff = sd.get_staff_for_anchor(anchor, staffs)
            keep_a = np.flatnonzero((cy >= anchor.zone.start) & (cy <= anchor.zone.stop))
            keep = keep_a.tolist()
            fragments = list(map(staff_fragments.__getitem__, keep))
            groups = (connect_groups(fragments, anchor.average_unit_size,
                                     None if F_all is None else np.ascontiguousarray(F_all[keep_a]))
                      if fragments else [])
            reqs = [boxes(anchor_line) for anchor_line in anchor.staff_lines]
            staff_lines = []
            if groups is None or any(r is None for r in reqs):
                connected = sd.connect_staff_lines(fragments, anchor.average_unit_size)
                for anchor_line in anchor.staff_lines:
                    matching_anchor = [line for line in connected if subset(anchor_line, line)]
                    staff_lines.append(matching_anchor[0] if len(matching_anchor) == 1 else anchor_line)
            else:
                gsets, segs = {}, {}
                for anchor_line, req in zip(anchor.staff_lines, reqs):
                    matches = []
                    for g, idx in enumerate(groups):
                        s = gsets.get(g)
                        if s is None:
                            s = gsets[g] = frozenset(fragments[j].box for j in idx)
                        if req <= s:
                            matches.append(g)
                    if len(matches) == 1:
                        g = matches[0]
                        if g not in segs:
                            segs[g] = sd.StaffLineSegment(g, [fragments[j] for j in groups[g]])
                        staff_lines.append(segs[g])
                    else:
                        staff_lines.append(anchor_line)
            if existing_staff:
                staffs.remove(existing_staff)
                if raw_init_safe(staff_lines):          # RawStaff.merge with the new RawStaff's lines and anchors
                    lines = [staff_lines[i].merge(line) for i, line in enumerate(existing_staff.lines)]
                    staffs.append(sd.RawStaff(existing_staff.staff_id, lines, existing_staff.anchors + [anchor]))
                else:
                    staffs.append(existing_staff.merge(sd.RawStaff(staff_id, staff_lines, [anchor])))
            else:
                staffs.append(sd.RawStaff(staff_id, staff_lines, [anchor]))
            staff_id += 1
        return staffs

    def raw_light_checked(anchors, staff_fragments):
        r, o = find_raw_staffs_light(anchors, staff_fragments), orig_raw(anchors, staff_fragments)
        if sig_raw(r) != sig_raw(o):
            _differ("find_raw_staffs_by_connecting_line_fragments", len(r), len(o))
        STATS["checked"] += 1
        return r

    sd.find_raw_staffs_by_connecting_line_fragments = raw_light_checked if CHECK else find_raw_staffs_light

    # ---------------------------------------------------------------- find_staff_anchors
    orig_anchors = sd.find_staff_anchors

    def find_staff_anchors(staff_lines, anchor_symbols, are_clefs=False):
        """homr.staff_detection.find_staff_anchors with a pre-screened is_intersecting list."""
        result = []
        scr = _Screen(lib, staff_lines, kind_of) if staff_lines else None
        for center_symbol in anchor_symbols:
            if are_clefs:
                adjacent = [
                    center_symbol.move_to_x_horizontal_by(-10),
                    center_symbol,
                    center_symbol.move_to_x_horizontal_by(10),
                    center_symbol.move_to_x_horizontal_by(30),
                    center_symbol.move_to_x_horizontal_by(60),
                    center_symbol.move_to_x_horizontal_by(80),
                ]
            else:
                adjacent = [
                    center_symbol.move_to_x_horizontal_by(-10),
                    center_symbol.move_to_x_horizontal_by(-5),
                    center_symbol,
                    center_symbol.move_to_x_horizontal_by(5),
                    center_symbol.move_to_x_horizontal_by(10),
                ]
            for symbol in adjacent:
                estimated_unit_size = round(symbol.size[1] / (constants.number_of_lines_on_a_staff - 1))
                thickened_bar_line = symbol.make_box_taller(estimated_unit_size)
                if scr is None:
                    overlapping_staff_lines = []
                else:
                    c2, a2, _ = thickened_bar_line.box
                    overlapping_staff_lines = [
                        staff_lines[i] for i in scr.near(c2, a2) if staff_lines[i].is_intersecting(thickened_bar_line)
                    ]
                connected_lines = sd.connect_staff_lines(overlapping_staff_lines, estimated_unit_size)
                if len(connected_lines) > constants.number_of_lines_on_a_staff:
                    connected_lines = [
                        line
                        for line in connected_lines
                        if (line.max_x - line.min_x)
                        > constants.is_short_connected_line(estimated_unit_size)
                    ]
                if not len(connected_lines) == constants.number_of_lines_on_a_staff:
                    continue
                if not sd.are_lines_parallel(connected_lines, estimated_unit_size):
                    continue
                if sd.are_lines_crossing(connected_lines):
                    continue
                if not are_clefs and not sd.begins_or_ends_on_one_staff_line(
                    symbol, connected_lines, estimated_unit_size
                ):
                    continue

                result.append(sd.StaffAnchor(connected_lines, symbol))
        return result

    def sig_anchors(anchors):
        return [(a.symbol.box, [[id(f) for f in l.staff_fragments] for l in a.staff_lines], a.average_unit_size,
                 a.zone) for a in anchors]

    def anchors_checked(staff_lines, anchor_symbols, are_clefs=False):
        r = find_staff_anchors(staff_lines, anchor_symbols, are_clefs)
        o = orig_anchors(staff_lines, anchor_symbols, are_clefs)
        if sig_anchors(r) != sig_anchors(o):
            _differ("find_staff_anchors", len(r), len(o))
        STATS["checked"] += 1
        return r

    sd.find_staff_anchors = anchors_checked if CHECK else find_staff_anchors
    report["installed"].append("find_staff_anchors")

    # ---------------------------------------------------------------- resample_staff_segment
    if hasattr(lib, "hg_resample"):
        orig_resample = sd.resample_staff_segment
        seg_tol = constants.staff_line_segment_x_tolerance
        homr_seg_get_at = sd.StaffLineSegment.get_at
        homr_point = sd.StaffPoint                  # (staff_detection imported model.StaffPoint by name)
        # bypassing StaffPoint.__init__ needs a plain class (instance __dict__, object's __new__ / __setattr__)
        point_plain = ("__slots__" not in vars(homr_point) and homr_point.__new__ is object.__new__
                       and homr_point.__setattr__ is object.__setattr__)
        new_point = object.__new__
        num = (float, int)

        def rs_frag(f):
            """lo / hi of StaffLineSegment.get_at's test, bx, by, tan of get_center_extrapolated and the angle, by
            Homr's expressions on python numbers; None for anything else.  Cached (boxes are never reassigned)."""
            v = f.__dict__.get("_homr_fast_rs")
            if v is None or v[0] is not f.box:
                (bx, by), _, ang = f.box
                if not (type(f) is bb.RotatedBoundingBox and all(type(a) in num for a in (*f.center, *f.size, bx, by, ang))
                        and type(f.angle) in num):
                    v = (f.box, None)
                else:
                    t = f.__dict__.get("_homr_fast_tan")
                    t = t[1] if t is not None and t[0] is f.box else np.tan(ang / 180 * np.pi)
                    v = (f.box, (f.center[0] - f.size[0] / 2 - seg_tol, f.center[0] + f.size[0] / 2 + seg_tol, bx, by,
                                 float(t), f.angle))
                f.__dict__["_homr_fast_rs"] = v
            return v[1]

        def rs_staff(staff):
            """(nf, G) of a staff's lines, cached while the lines and their fragment lists are the same objects."""
            lines = staff.lines
            key = (lines, len(lines), tuple(map(id, lines)), tuple(id(line.staff_fragments) for line in lines))
            c = staff.__dict__.get("_homr_fast_rsg")
            if c is not None and c[0][0] is lines and c[0][1:] == key[1:]:
                return c[1]
            out = None
            if all(type(line) is sd.StaffLineSegment for line in lines):
                rows = [rs_frag(f) for line in lines for f in line.staff_fragments]
                if all(r is not None for r in rows) and rows:
                    nf = np.array([len(line.staff_fragments) for line in lines], dtype=np.int32)
                    out = (nf, np.ascontiguousarray(np.array(rows, dtype=np.float64)))
            staff.__dict__["_homr_fast_rsg"] = (key, out)
            return out

        def resample_staff_segment(anchor, staff, axis_range):
            """Homr's generator as a list; the per-x loop runs in C++ (hg_resample) when every input is a python
            number and the staff lines are Homr's own StaffLineSegment / RotatedBoundingBox."""
            xs = list(axis_range)
            x = anchor.symbol.center[0]
            line_fragments = [line.staff_fragments[0] for line in anchor.staff_lines]
            centers = [line.get_center_extrapolated(x) for line in line_fragments]
            previous_point = sd.StaffPoint(x, centers, float(np.mean([line.angle for line in line_fragments])))
            g = rs_staff(staff) if xs else None
            L = len(staff.lines)
            # (StaffPoint.__init__ raises unless len(y) is a multiple of 5: only L == 5 bypasses it)
            unit = anchor.average_unit_size
            if (g is None or sd.StaffLineSegment.get_at is not homr_seg_get_at or L != 5 or not point_plain
                    or sd.StaffPoint is not homr_point or len(previous_point.y) != L or type(unit) not in num or not all(type(v) is int for v in xs)
                    or not all(type(v) in (float, np.float64) for v in previous_point.y)):
                return list(orig_resample(anchor, staff, xs))
            nf, G = g
            prev = np.array(previous_point.y, dtype=np.float64)
            xa = np.array(xs, dtype=np.float64)
            n = len(xs)
            ok = np.zeros(n, dtype=np.int32)
            ys = np.empty((n, L), dtype=np.float64)
            ang = np.empty(n, dtype=np.float64)
            au = np.empty(n, dtype=np.float64)
            if lib.hg_resample(L, nf.ctypes.data_as(_I), G.ctypes.data_as(_D), float(unit), prev.ctypes.data_as(_D), n,
                               xa.ctypes.data_as(_D), ok.ctypes.data_as(_I), ys.ctypes.data_as(_D),
                               ang.ctypes.data_as(_D), au.ctypes.data_as(_D)) != 0:
                return list(orig_resample(anchor, staff, xs))
            # Homr's y values are np.float64 (get_center_extrapolated), its angle a python float; StaffPoint.__init__
            # sets x, y, angle and average_unit_size = np.mean(np.diff(y)) (np.float64, computed in C++)
            pts = []
            for q in np.flatnonzero(ok).tolist():
                p = new_point(homr_point)
                p.x, p.y, p.angle, p.average_unit_size = xs[q], list(ys[q]), float(ang[q]), au[q]
                pts.append(p)
            return pts

        def sig_points(points):
            return [(p.x, type(p.x), [float(v) for v in p.y], [type(v) for v in p.y], p.angle, type(p.angle),
                     float(p.average_unit_size), type(p.average_unit_size), sorted(vars(p))) for p in points]

        def resample_checked(anchor, staff, axis_range):
            xs = list(axis_range)
            r, o = resample_staff_segment(anchor, staff, xs), list(orig_resample(anchor, staff, xs))
            if sig_points(r) != sig_points(o):
                _differ("resample_staff_segment", sig_points(r)[:2], sig_points(o)[:2])
            STATS["checked"] += 1
            return r

        sd.resample_staff_segment = resample_checked if CHECK else resample_staff_segment
        report["installed"].append("resample_staff_segment")

    # ---------------------------------------------------------------- are_lines_crossing
    if hasattr(lib, "hg_crossing"):
        orig_crossing = sd.are_lines_crossing
        homr_seg_overlap = sd.StaffLineSegment.is_overlapping
        homr_box_overlap = bb.AngledBoundingBox.is_overlapping
        num_ = (float, int)

        def cr_frag(f):
            """(cx, cy, major axis, polygon int32 xy) as _can_shapes_possibly_touch / do_polygons_overlap read them,
            or None for anything but a RotatedBoundingBox of python numbers with an int32/int64 (k, 2) polygon."""
            v = f.__dict__.get("_homr_fast_cr")
            if v is None or v[0] is not f.box or v[1] is not f.polygon:
                c, a, _ = f.box
                poly = f.polygon
                ok = (type(f) is bb.RotatedBoundingBox and all(type(t) in num_ for t in (*c, *a))
                      and isinstance(poly, np.ndarray) and poly.ndim == 2 and poly.shape[1] == 2 and len(poly) > 0
                      and poly.dtype in (np.int32, np.int64) and int(np.abs(poly).max()) < 2 ** 30)
                v = (f.box, poly, (float(c[0]), float(c[1]), float(max(a)), poly.astype(np.int32).reshape(-1))
                     if ok else None)
                f.__dict__["_homr_fast_cr"] = v
            return v[2]

        def are_lines_crossing(lines):
            """Homr's pairwise is_overlapping of the lines' fragments in C++ (hg_crossing: same screen, OpenCV's
            integer point-in-polygon test); Homr's own function when a centre distance is within the margin of
            the screen's threshold or an input is not what the kernel was written for."""
            if not (bb.AngledBoundingBox._can_shapes_possibly_touch in (touch, touch_checked)
                    and bb.do_polygons_overlap in (orig_poly, do_polygons_overlap, poly_checked)
                    and sd.StaffLineSegment.is_overlapping is homr_seg_overlap
                    and bb.RotatedBoundingBox.is_overlapping is homr_box_overlap
                    and all(type(line) is sd.StaffLineSegment for line in lines)):
                return orig_crossing(lines)
            rows, ls = [], [0]
            for line in lines:
                for f in line.staff_fragments:
                    r = cr_frag(f)
                    if r is None:
                        return orig_crossing(lines)
                    rows.append(r)
                ls.append(len(rows))
            if not rows:
                return orig_crossing(lines)
            g = np.array([r[:3] for r in rows], dtype=np.float64).T.copy()
            ps = np.zeros(len(rows) + 1, dtype=np.int32)
            ps[1:] = np.cumsum([len(r[3]) // 2 for r in rows])
            xy = np.ascontiguousarray(np.concatenate([r[3] for r in rows]))
            lsa = np.array(ls, dtype=np.int32)
            res = lib.hg_crossing(len(lines), lsa.ctypes.data_as(_I), g[0].ctypes.data_as(_D), g[1].ctypes.data_as(_D),
                                  g[2].ctypes.data_as(_D), ps.ctypes.data_as(_I), xy.ctypes.data_as(_I), MARGIN)
            if res < 0:
                return orig_crossing(lines)
            return res == 1

        def crossing_checked(lines):
            r, o = are_lines_crossing(lines), orig_crossing(lines)
            if r != o:
                _differ("are_lines_crossing", r, o)
            STATS["checked"] += 1
            return r

        sd.are_lines_crossing = crossing_checked if CHECK else are_lines_crossing
        report["installed"].append("are_lines_crossing")

    # ---------------------------------------------------------------- combine_noteheads_with_stems
    orig_combine = nd.combine_noteheads_with_stems

    def combine_noteheads_with_stems(noteheads, stems):
        """homr.note_detection.combine_noteheads_with_stems with pre-screened stems."""
        result = []
        noteheads = sorted(noteheads, key=lambda notehead: notehead.box[0][1])
        used_stems = set()
        scr = _Screen(lib, stems, kind_of) if stems else None
        for notehead in noteheads:
            thickened_notehead = notehead.make_box_thicker(15)
            found_stem = False
            if scr is not None:
                c2, a2, _ = (thickened_notehead.rotated_box if kind_of(thickened_notehead) == 1
                             else thickened_notehead.box)
                for i in scr.near(c2, a2):
                    stem = stems[i]
                    if stem.is_overlapping(thickened_notehead):
                        is_stem_above = stem.center[1] < notehead.center[1]
                        if is_stem_above:
                            direction = nd.StemDirection.UP
                        else:
                            direction = nd.StemDirection.DOWN
                        result.append(nd.NoteheadWithStem(notehead, stem, direction))
                        used_stems.add(stem)
                        found_stem = True
                        break
            if not found_stem:
                result.append(nd.NoteheadWithStem(notehead, None, None))
        return result

    def sig_heads(heads):
        return [(id(h.notehead), id(h.stem) if h.stem is not None else None, h.stem_direction) for h in heads]

    def combine_checked(noteheads, stems):
        r, o = combine_noteheads_with_stems(noteheads, stems), orig_combine(noteheads, stems)
        if sig_heads(r) != sig_heads(o):
            _differ("combine_noteheads_with_stems", len(r), len(o))
        STATS["checked"] += 1
        return r

    nd.combine_noteheads_with_stems = combine_checked if CHECK else combine_noteheads_with_stems
    hm.combine_noteheads_with_stems = nd.combine_noteheads_with_stems      # main imported it by name
    report["installed"].append("combine_noteheads_with_stems")

    # ---------------------------------------------------------------- is_overlapping_with_any
    orig_any = bb.AngledBoundingBox.is_overlapping_with_any
    screens = {}

    def is_overlapping_with_any(self, others):
        if len(others) < 16:
            return orig_any(self, others)
        ids = tuple(map(id, others))
        hit = screens.get(ids)
        if hit is None:                              # keyed by the identities of the boxes, in order;
            if len(screens) >= 8:                    # the cached tuple keeps them alive (no id reuse)
                screens.clear()
            hit = screens[ids] = (tuple(others), _Screen(lib, others, kind_of))
        for i in hit[1].near(self.box[0], self.box[1]):
            if self.is_overlapping(others[i]):
                return True
        return False

    def any_checked(self, others):
        r, o = is_overlapping_with_any(self, others), orig_any(self, others)
        if r != o:
            _differ("is_overlapping_with_any", r, o)
        return r

    bb.AngledBoundingBox.is_overlapping_with_any = any_checked if CHECK else is_overlapping_with_any
    report["installed"].append("is_overlapping_with_any")
    return report
