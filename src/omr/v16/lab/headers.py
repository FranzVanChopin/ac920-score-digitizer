"""Stage B: per-system clefs and key signature from small models only (no Homr transformer).

clef  the small segnet student's clefs/keys class (`../small_segnet`), first tall component in
      the first 5 staff spaces of each staff, classified by its top/bottom relative to the lines:
        treble  extends > 1 space above the top line and > 0.5 below the bottom line
        bass    top within 0.6 spaces of the top line, bottom 2.2..3.8 spaces below it
        alto    top near the top line, bottom near the bottom line (spans the staff)
        tenor   top ~1 space above the top line, bottom ~1 space above the bottom line
key   sharps/flats of Codex's symbol model between the clef and the first event of each staff,
      majority over the staves of the system
meter graph meter glyphs, else bar content (extract.infer_meter), carried across pages
Writes a header MusicXML (one measure per system, as Codex's `contexts` reads it) and a JSON.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

LAB = Path(__file__).resolve().parent
SEG = LAB.parent / "small_segnet"
sys.path.insert(0, str(SEG))
import common as SC  # noqa: E402

CLEFS = {"treble": ("G", 2), "bass": ("F", 4), "alto": ("C", 3), "tenor": ("C", 4)}
# Mid-system clef detection by shape (inline_clefs) is OFF: on camera67 it fired 78 times, 56 of them
# on a single-staff piece that never changes clef, and Stage B fell 88.6 -> 74.2.  The most it could
# recover is ~0.4 points (A vs B "other" pitch errors); a real fix needs a small clef classifier.
INLINE_CLEFS = False


def class_map(photo, sess):
    """Student segnet on Homr's preprocessing; returns (ids at width 1920, scale photo->map,
    the CLAHE gray map it saw)."""
    bgr = cv2.imread(str(photo))
    g = SC.clahe(cv2.cvtColor(SC.homr_resize(bgr), cv2.COLOR_BGR2GRAY))
    return SC.segment(sess, g), 1920.0 / bgr.shape[1], g


def octave_mark(gray, box, s_map):
    """True when the box (map coords) holds an ink blob of an octave '8' (>= 0.6 space tall)."""
    xa, ya, xb, yb = (max(0, int(v)) for v in box)
    ink = (gray[ya:yb, xa:xb] < 128).astype(np.uint8)
    if ink.size == 0 or ink.sum() == 0:
        return False
    n, _l, st, _c = cv2.connectedComponentsWithStats(ink)
    return any(0.6 * s_map <= st[i, cv2.CC_STAT_HEIGHT] <= 1.4 * s_map
               and 0.35 * s_map <= st[i, cv2.CC_STAT_WIDTH] <= 1.2 * s_map for i in range(1, n))


def octave_holes(gray, box, s_map):
    """True when the box (map coords) holds an enclosed hole of an octave '8' (two holes, each ~0.1-0.4
    space): found even when the 8 touches the clef's tail, which ends in a filled ball without a hole."""
    xa, ya, xb, yb = (max(0, int(v)) for v in box)
    ink = (gray[ya:yb, xa:xb] < 128).astype(np.uint8)
    if ink.size == 0 or ink.sum() == 0:
        return False
    contours, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return False
    for c, h in zip(contours, hier[0]):
        if h[3] < 0:                                    # outer contour, not a hole
            continue
        x, y, w, hh = cv2.boundingRect(c)
        if 0.1 * s_map <= w <= 0.6 * s_map and 0.1 * s_map <= hh <= 0.6 * s_map:
            return True
    return False


def _line_y(staff, k, x):
    pts = np.array(staff["lines"][k]["polyline"])
    return float(np.interp(x, pts[:, 0], pts[:, 1]))


def clef_of(staff, ids, scale, gray=None):
    """(name, evidence, octave) for the clef at the start of `staff` (staff.json coordinates);
    octave -1/+1 when a small '8' sits under/over a treble or bass clef (banjo, guitar, tenor
    voice parts), read from the image ink because the segnet class does not include it."""
    s = staff["space_px"]
    x0 = staff["lines"][0]["start"][0]
    top, bot = _line_y(staff, 0, x0 + 2 * s), _line_y(staff, 4, x0 + 2 * s)
    xa, xb = int((x0 - 0.5 * s) * scale), int((x0 + 5.0 * s) * scale)
    ya, yb = int((top - 4 * s) * scale), int((bot + 4 * s) * scale)
    xa, ya = max(0, xa), max(0, ya)
    crop = (ids[ya:yb, xa:xb] == 3).astype(np.uint8)
    if crop.sum() == 0:
        return None, "no clef ink", 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(crop, np.ones((3, 3), np.uint8)))
    cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 2.2 * s * scale]
    if not cand:
        return None, "no tall component", 0
    i = min(cand, key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    t = (ya + stats[i, cv2.CC_STAT_TOP]) / scale
    b = (ya + stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) / scale
    rt, rb = (t - top) / s, (b - top) / s               # staff lines at 0..4
    if rt < -1.0 and rb > 4.5:
        name = "treble"
    elif abs(rt) <= 0.6 and 2.2 <= rb <= 3.8:
        name = "bass"
    elif abs(rt) <= 0.6 and 3.8 < rb <= 4.6:
        name = "alto"
    elif -1.6 <= rt < -0.6 and 2.4 <= rb <= 3.6:
        name = "tenor"
    else:
        name = "treble" if rb - rt > 5.5 else None
    octave = 0
    if gray is not None and name in ("treble", "bass"):
        L, W = xa + stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        tm, bm, sm = ya + stats[i, cv2.CC_STAT_TOP], ya + stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT], s * scale
        # only BELOW: above a clef sit tempo text, measure numbers and the previous staff's notes
        # (an 'above' check fired on 66 ordinary clefs); 8va clefs are rare
        if octave_mark(gray, (L + 0.15 * W, bm + 0.05 * sm, L + 0.85 * W, bm + 1.6 * sm), sm):
            octave = -1
        elif name == "treble" and octave_holes(gray, (L + 0.15 * W, (top + 5.4 * s) * scale,
                                                     L + 0.85 * W, (top + 7.8 * s) * scale), sm):
            # the clef component can swallow the 8 (then bm lies below it and the box above is empty), and the
            # 8 can touch the clef's tail: look for the 8's holes 1.4-3.8 spaces under the bottom line
            octave = -1
    return name, f"top {rt:+.2f} bottom {rb:+.2f}", octave


def inline_clefs(staff, ids, scale):
    """Mid-system clef changes [(x_photo, name)] from the clefs/keys class past the staff start.
    Accidentals share that class but are narrow (< 1.4 spaces); clefs are wide or very tall:
      treble  >= 4.2 spaces tall, reaching above the top line
      bass    2.0..3.6 tall, >= 1.5 wide, top within 0.7 space of the top line
      C       3.2..4.4 tall, >= 1.5 wide (alto if its top is at the top line, tenor a space above)
    Codex's visual reader found 1 such change on all of camera67; Homr's per-measure attributes
    carried the rest in Stage A."""
    s = staff["space_px"]
    x0, x1 = staff["lines"][0]["start"][0], staff["lines"][0]["end"][0]
    xm = (x0 + x1) / 2
    top, bot = _line_y(staff, 0, xm), _line_y(staff, 4, xm)
    xa, xb = int((x0 + 6 * s) * scale), int(x1 * scale)
    ya, yb = int((top - 3 * s) * scale), int((bot + 3 * s) * scale)
    if xb <= xa:
        return []
    crop = (ids[max(0, ya):yb, xa:xb] == 3).astype(np.uint8)
    n, _l, st, _c = cv2.connectedComponentsWithStats(cv2.dilate(crop, np.ones((3, 3), np.uint8)))
    out = []
    for i in range(1, n):
        h, w = st[i, cv2.CC_STAT_HEIGHT] / (s * scale), st[i, cv2.CC_STAT_WIDTH] / (s * scale)
        x = (xa + st[i, cv2.CC_STAT_LEFT]) / scale
        t = (max(0, ya) + st[i, cv2.CC_STAT_TOP]) / scale
        lt, lb = _line_y(staff, 0, x), _line_y(staff, 4, x)
        rt = (t - lt) / s
        rb = rt + h
        if h >= 4.2 and rt < -0.5 and rb > 4.0:
            name = "treble"
        elif 2.0 <= h <= 3.6 and w >= 1.5 and abs(rt) <= 0.7:
            name = "bass"
        elif 3.2 <= h <= 4.4 and w >= 1.5:
            name = "alto" if abs(rt) <= 0.6 else "tenor" if -1.6 <= rt < -0.6 else None
        else:
            name = None
        if name:
            out.append((float(x), name))
    return sorted(out)


def key_of(staff, row, ids, scale, gscale):
    """Key fifths of one staff from Codex's symbol model: the contiguous run of sharps/flats that
    starts within 4 spaces of the staff start and ends >= 1.5 spaces before the first event (so the
    first note's own accidental is not counted).  Mask-based counting was tried and rejected:
    time-signature digits leak into the clefs/keys class (score05) and sharps split."""
    s = staff["space_px"]
    x0 = staff["lines"][0]["start"][0]
    xs = [g["x"] / gscale for g in row["groups"] if not g.get("grace")]
    if not xs:
        return None
    first = min(xs)
    acc = sorted((nd for nd in row["nodes"] if nd["class"] in (8, 9)
                  and x0 < nd["xy"][0] / gscale < first - 1.5 * s), key=lambda nd: nd["xy"][0])
    run = []
    for nd in acc:
        x = nd["xy"][0] / gscale
        if not run and x > x0 + 6.0 * s:
            break
        if run and x - run[-1]["xy"][0] / gscale > 1.6 * s:
            break
        run.append(nd)
    sh, fl = sum(nd["class"] == 8 for nd in run), sum(nd["class"] == 9 for nd in run)
    return min(7, sh) if sh >= fl else -min(7, fl)


def key_of_mask(staff, row, ids, scale, gscale):
    """Rejected variant (kept for the record): count clefs/keys-mask components."""
    s = staff["space_px"]
    x0 = staff["lines"][0]["start"][0]
    xs = [g["x"] / gscale for g in row["groups"] if not g.get("grace")]
    if not xs:
        return None
    first = min(xs)
    top, bot = _line_y(staff, 0, x0 + 4 * s), _line_y(staff, 4, x0 + 4 * s)
    xa, xb = int((x0 - 0.5 * s) * scale), int((first - 0.9 * s) * scale)
    ya, yb = int((top - 2.5 * s) * scale), int((bot + 2.0 * s) * scale)
    if xb <= xa:
        return 0
    crop = (ids[max(0, ya):yb, max(0, xa):xb] == 3).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(crop)
    comps = sorted((i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= 1.4 * s * scale),
                   key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    if not comps:
        return 0
    # the clef is the leftmost tall component (as in clef_of); a sharp is ~2.8 spaces tall too,
    # so height alone cannot tell them apart -- position can
    tall = [i for i in comps if stats[i, cv2.CC_STAT_HEIGHT] >= 2.2 * s * scale]
    body = tall[:1] if tall and stats[tall[0], cv2.CC_STAT_LEFT] <= 2.5 * s * scale else []
    start = stats[body[0], cv2.CC_STAT_LEFT] + stats[body[0], cv2.CC_STAT_WIDTH] if body else 0
    acc = [i for i in comps if stats[i, cv2.CC_STAT_LEFT] >= start and i not in body]
    run = []
    for i in acc:                                     # contiguous run starting right after the clef
        if run and stats[i, cv2.CC_STAT_LEFT] - (stats[run[-1], cv2.CC_STAT_LEFT] + stats[run[-1], cv2.CC_STAT_WIDTH]) > 1.2 * s * scale:
            break
        run.append(i)
    if not run:
        return 0
    lo = (xa + stats[run[0], cv2.CC_STAT_LEFT]) / scale
    hi = (xa + stats[run[-1], cv2.CC_STAT_LEFT] + stats[run[-1], cv2.CC_STAT_WIDTH]) / scale
    nodes = [nd for nd in row["nodes"] if nd["class"] in (8, 9) and lo - 0.5 * s <= nd["xy"][0] / gscale <= hi + 0.5 * s]
    sh, fl = sum(nd["class"] == 8 for nd in nodes), sum(nd["class"] == 9 for nd in nodes)
    if sh == fl:                                      # no vote: shape of the components
        low = sum((cents[i][1] - stats[i, cv2.CC_STAT_TOP]) / stats[i, cv2.CC_STAT_HEIGHT] > 0.58 for i in run)
        flat = low * 2 > len(run)
    else:
        flat = fl > sh
    k = min(7, len(run))
    return -k if flat else k


def _naturals_in_header(staff, row, gscale):
    s = staff["space_px"]
    x0 = staff["lines"][0]["start"][0]
    xs = [g["x"] / gscale for g in row["groups"] if not g.get("grace")]
    first = min(xs) if xs else x0 + 8 * s
    return any(nd["class"] == 10 and x0 < nd["xy"][0] / gscale < first - 1.5 * s for nd in row["nodes"])


def page_headers(photo, staff_json, graph, sess, prev_key=None, maps=None):
    """Per-system clefs (+ octave marks) and key.  Key signatures are reprinted on every system,
    so a system that shows FEWER accidentals of the same kind than the previous one (or none) and
    no naturals cancelling it is read as a detection miss and keeps the previous key; `prev_key`
    carries it across pages of one score."""
    staves = json.loads(Path(staff_json).read_text(encoding="utf-8"))["staves"]
    # `maps` = (ids, scale, gray) already computed by the caller (the board run captures them from
    # Homr's own segnet call instead of running the student a second time)
    ids, scale, gray = maps if maps is not None else class_map(photo, sess)
    gscale = graph.get("coordinate_scale", 16 / 19.8)
    systems = sorted({s["system"] for s in staves})
    incoming = prev_key                              # the previous page's last key, for zero runs
    out = []
    for sysno in systems:
        st = [s for s in staves if s["system"] == sysno]
        rows = sorted([r for r in graph["systems"] if r["system"] == systems.index(sysno) + 1 or r["system"] == sysno],
                      key=lambda r: r["staff"])
        clefs, octs, ev, inl = {}, {}, {}, {}
        for k, s in enumerate(st, start=1):
            name, why, octave = clef_of(s, ids, scale, gray)
            clefs[k], octs[k], ev[k] = name, octave, why
            inl[k] = inline_clefs(s, ids, scale) if INLINE_CLEFS else []
        keys = [key_of(s, r, ids, scale, gscale) for s, r in zip(st, rows)]
        keys = [k for k in keys if k is not None]
        key = max(set(keys), key=keys.count) if keys else 0
        naturals = any(_naturals_in_header(s, r, gscale) for s, r in zip(st, rows))
        if prev_key and not naturals and key * prev_key > 0 and abs(key) < abs(prev_key):
            key = prev_key                           # fewer accidentals of the same kind: a miss
        prev_key = key
        out.append(dict(system=sysno, clefs=clefs, octaves=octs, key=key, inline=inl, evidence=ev))
    # A key cannot be told from a missed one when a system shows no accidentals at all (a real
    # change to C major looks the same; carrying the key broke 82 such systems).  Only a short
    # run of zeros between two systems with the same key is filled.
    keys = [o["key"] for o in out]
    for i in range(len(keys)):
        if keys[i] == 0:
            j = i
            while j < len(keys) and keys[j] == 0:
                j += 1
            left = keys[i - 1] if i > 0 else incoming   # a page's first system continues the last page
            right = keys[j] if j < len(keys) else None
            if j - i <= 2 and left and left == right:
                for t in range(i, j):
                    out[t]["key"] = left
    return out


def write_header_xml(path, systems, meter):
    """One measure per system; Codex's relation_context.contexts reads exactly this."""
    root = ET.Element("score-partwise", version="3.1")
    pl = ET.SubElement(root, "part-list")
    ET.SubElement(ET.SubElement(pl, "score-part", id="P1"), "part-name").text = "Music"
    part = ET.SubElement(root, "part", id="P1")
    prev = {}
    for i, s in enumerate(systems):
        m = ET.SubElement(part, "measure", number=str(i + 1))
        if i:
            ET.SubElement(m, "print", {"new-system": "yes"})
        at = ET.SubElement(m, "attributes")
        if i == 0:
            ET.SubElement(at, "divisions").text = "480"
        ET.SubElement(ET.SubElement(at, "key"), "fifths").text = str(s["key"])
        t = ET.SubElement(at, "time")
        ET.SubElement(t, "beats").text = str(meter[0])
        ET.SubElement(t, "beat-type").text = str(meter[1])
        if i == 0:
            ET.SubElement(at, "staves").text = str(len(s["clefs"]))
        for k, name in sorted(s["clefs"].items()):
            name = name or prev.get(k) or ("treble" if k == 1 else "bass")   # undecided: keep the last
            prev[k] = name
            c = ET.SubElement(at, "clef", number=str(k))
            ET.SubElement(c, "sign").text = CLEFS[name][0]
            ET.SubElement(c, "line").text = str(CLEFS[name][1])
        ET.SubElement(ET.SubElement(m, "note"), "rest", measure="yes")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
