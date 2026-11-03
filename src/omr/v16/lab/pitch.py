"""Stage B pitch: head position + our clef/key (headers.py) + in-measure accidentals.

Replaces the graph's head pitches, which Codex computed with Homr-transformer headers.
  position  half-spaces above the bottom line of the head's own staff (staff.json polylines)
  clef      bottom-line pitch: treble E4, bass G2, alto F3, tenor D3
  key       standard sharp order F C G D A E B, flat order B E A D G C F
  accidental Codex's own per-head decision (4th field of the graph pitch: sharp/flat/natural/None,
            from its symbol model and attachment rules, no transformer); it holds for that step and
            octave until the barline, as in Codex's theory.resolve_accidental.  A naive "sign left
            of the head" rule was tried first and lost 310 notes on chromatic chords (score34).
"""
from __future__ import annotations

import numpy as np

STEPS = "CDEFGAB"
BOTTOM = {"treble": ("E", 4), "bass": ("G", 2), "alto": ("F", 3), "tenor": ("D", 3)}
SHARPS = "FCGDAEB"
FLATS = "BEADGCF"


def _line_y(staff, k, x):
    pts = np.array(staff["lines"][k]["polyline"])
    return float(np.interp(x, pts[:, 0], pts[:, 1]))


def step_of(clef, pos):
    b, o = BOTTOM[clef]
    idx = STEPS.index(b) + o * 7 + pos
    return STEPS[idx % 7], idx // 7


def key_alter(step, fifths):
    if fifths > 0 and step in SHARPS[:fifths]:
        return 1
    if fifths < 0 and step in FLATS[:-fifths]:
        return -1
    return 0


def repitch_row(row, staff, clef, fifths, gscale, octave_shift=0, inline=()):
    """Rewrite every head's `pitch` in one graph staff row (in place).  Returns #changed.
    `octave_shift` is the clef's octave mark (-1 for a treble clef with an 8 below); `inline` lists
    mid-system clef changes [(x_photo, name)] that hold from their x to the end of the system."""
    s = staff["space_px"]
    accs = [(nd["xy"][0] / gscale, nd["xy"][1] / gscale, {8: 1, 9: -1, 10: 0}[nd["class"]])
            for nd in row["nodes"] if nd["class"] in (8, 9, 10)]
    bounds = [b / gscale for b in row["bounds"]]
    heads = []
    for g in row["groups"]:
        for h in g.get("heads") or []:
            heads.append(h)
    heads.sort(key=lambda h: h["xy"][0])
    state = {}                                       # (measure, step, octave) -> alter
    changed = 0
    for h in heads:
        x, y = h["xy"][0] / gscale, h["xy"][1] / gscale
        bottom = _line_y(staff, 4, x)
        pos = int(round((bottom - y) / (s / 2)))
        cur = clef
        for cx, name in inline:
            if cx < x:
                cur = name
        step, octave = step_of(cur, pos)
        octave += octave_shift if cur == clef else 0
        mi = int(np.searchsorted(bounds, x)) - 1
        explicit = (h.get("pitch") or [None] * 4)[3] if len(h.get("pitch") or []) > 3 else None
        alter = {"sharp": 1, "flat": -1, "natural": 0, "double-sharp": 2, "flat-flat": -2}.get(explicit)
        if alter is not None:
            state[(mi, step, octave)] = alter
        else:
            alter = state.get((mi, step, octave), key_alter(step, fifths))
        new = [step, octave, alter, explicit]
        if h.get("pitch") and list(h["pitch"][:3]) != new[:3]:
            changed += 1
        h["pitch"] = new
    return changed


def repitch_graph(graph, staves, headers):
    """Apply per-system clefs/key (headers.page_headers output) to every row of a page graph."""
    gscale = graph.get("coordinate_scale", 16 / 19.8)
    systems = sorted({s["system"] for s in staves})
    hdr = {h["system"]: h for h in headers}
    changed = total = 0
    for row in graph["systems"]:
        sysno = systems[row["system"] - 1] if row["system"] - 1 < len(systems) else row["system"]
        st = [s for s in staves if s["system"] == sysno]
        if row["staff"] - 1 >= len(st):
            continue
        h = hdr.get(sysno)
        clef = (h["clefs"].get(row["staff"]) or h["clefs"].get(str(row["staff"]))) if h else None
        clef = clef or ("treble" if row["staff"] == 1 else "bass")
        fifths = h["key"] if h else 0
        octs = (h or {}).get("octaves") or {}
        shift = int(octs.get(row["staff"], octs.get(str(row["staff"]), 0)) or 0)
        inl = (h or {}).get("inline") or {}
        inline = [tuple(v) for v in (inl.get(row["staff"]) or inl.get(str(row["staff"])) or [])]
        total += sum(len(g.get("heads") or []) for g in row["groups"])
        changed += repitch_row(row, st[row["staff"] - 1], clef, fifths, gscale, shift, inline)
    return changed, total
