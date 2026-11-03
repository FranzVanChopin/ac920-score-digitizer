"""Relation graph (Codex's small symbol/stroke/edge models) -> decoder candidates.

One graph per photographed page.  Each `systems[]` entry is one staff of one system with its
barline x positions (`bounds`) and its event `groups` (a stem with its heads, or a rest).
Only VISUAL evidence becomes a hint for the decoder:
  division  hollow head: half with a stem, whole without; filled head: quarter + beams/flags;
            rests: their glyph class
  dots      the group's `dotted` flag
  warp      a printed tuplet number (`tuplet_ratio`)
Codex's own rule decode (`onset`, `duration`, `voice`) is NOT used by the decoder; it is
kept on the side so the Codex-rules baseline can be written from the same events.
"""
from __future__ import annotations

import collections
import math
import random
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

DEC = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "dec")
if DEC not in sys.path:
    sys.path.insert(0, DEC)
import visual as V  # noqa: E402
from column_snap import event_x, snap_columns  # noqa: E402
from visual import Cand, CandMeasure  # noqa: E402

PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
REST_D = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}          # rest_whole .. rest_32nd


def midi_of(p):
    step, octave, alter = p[0], int(p[1]), int(p[2] or 0)
    return (octave + 1) * 12 + PC[step] + alter


@dataclass
class Event:
    staff: int
    x_px: float
    space_px: float
    is_rest: bool
    d: int
    dots: int
    warp: tuple
    stem: int
    pitches: list                      # [(step, octave, alter)], empty for rests
    heads_x: list                      # [(x_px, midi)]
    codex: dict                        # Codex rule decode: onset / duration / voice


@dataclass
class Measure:
    index: int                         # 0-based on the page
    beats: int
    beat_type: int
    events: list = field(default_factory=list)
    n_staves: int = 1

    @property
    def length(self):
        return int(self.beats * 1920 // self.beat_type)


def _hint(g):
    if "rest" in g:
        return True, REST_D.get(g["rest"]["class"], 2), 0
    heads = g.get("heads") or []
    hollow = sum(h.get("class") == 1 for h in heads) * 2 > len(heads)
    has_stem = g.get("stem") is not None or g.get("direction") is not None
    if hollow:
        return False, (1 if has_stem else 0), 0
    n = max(int(g.get("flags") or 0), int(g.get("beams") or 0))
    return False, min(6, 2 + n), 0


def _warp(g):
    r = g.get("tuplet_ratio")
    if not r:
        return (1, 1)
    a, n = int(r[0]), int(r[1])
    k = math.gcd(a, n) or 1
    return (a // k, n // k)


def _head_x(e):
    return event_x([(hx, mi) for hx, mi in e.heads_x], e.stem, e.x_px) if e.heads_x else e.x_px


def merge_chords(m, reach=1.3):
    """Fuse note groups that are one chord.  Stemless heads (whole notes, or heads the edge model
    left unattached) are separate groups in the graph; within one staff a stemless group joins a
    stemmed group, or another stemless group, whose head column is within `reach` staff spaces
    (a head width plus the displacement of a second).  Opposite stems stay apart: two voices."""
    notes = [e for e in m.events if not e.is_rest]
    rest = [e for e in m.events if e.is_rest]
    changed = True
    while changed:
        changed = False
        for a in notes:
            if a.stem != 0:
                continue
            best, bd = None, None
            for b in notes:
                if b is a or b.staff != a.staff:
                    continue
                dx = abs(_head_x(a) - _head_x(b)) / a.space_px
                if dx <= reach and (bd is None or (b.stem != 0, -dx) > (bd[0], -bd[1])):
                    best, bd = b, (b.stem != 0, dx)
            if best is not None:
                if best.stem == 0 and a.d != best.d:      # two stemless heads of different value
                    continue
                best.pitches = sorted(set(best.pitches) | set(a.pitches), key=midi_of)
                best.heads_x = best.heads_x + a.heads_x
                best.dots = max(best.dots, a.dots)
                notes.remove(a)
                changed = True
                break
    m.events = sorted(notes + rest, key=lambda e: (e.staff, e.x_px))
    return m


def _split_x(m):
    """x (px) of a barline the stroke model missed inside measure m, or None.
    (a) a whole-rest glyph is a measure rest and owns its bar: if its staff has other events, the
        bar boundary lies between them and the rest;
    (b) every staff's hints add up to exactly two bars and each splits into two exact bars at one
        event boundary (a single-voice staff only), e.g. two 'note rest rest' bars of a tuba part."""
    per = collections.defaultdict(list)
    for e in m.events:
        per[e.staff].append(e)
    for evs in per.values():
        evs.sort(key=lambda e: e.x_px)
        wr = [e for e in evs if e.is_rest and e.d == 0]
        others = [e for e in evs if not (e.is_rest and e.d == 0)]
        # a second voice's measure rest shares the bar with the first voice's notes, so the rest
        # only owns a bar of its own when the other events already fill one exactly, alone
        if wr and len(others) >= 2 and abs(sum(V.duration_of(e.d, e.dots, e.warp) for e in others) - m.length) < 1e-6 \
                and len({e.stem for e in others if not e.is_rest and e.stem}) <= 1:
            r = wr[0]
            left = [e.x_px for e in evs if e.x_px < r.x_px]
            right = [e.x_px for e in evs if e.x_px > r.x_px]
            if left and not right:
                return (max(left) + r.x_px) / 2
            if right and not left:
                return (r.x_px + min(right)) / 2
    cuts = []
    for evs in per.values():
        if len({e.stem for e in evs if not e.is_rest and e.stem}) > 1:
            return None
        durs = [V.duration_of(e.d, e.dots, e.warp) for e in evs]
        if abs(sum(durs) - 2 * m.length) > 1e-6:
            return None
        acc = 0.0
        cut = None
        for k, d in enumerate(durs[:-1]):
            acc += d
            # both halves need >= 2 events: a whole note beside a second voice's whole rest or
            # four quarters also adds up to two bars (score02, score06) but is one bar
            if abs(acc - m.length) < 1e-6 and 1 <= k < len(durs) - 2:
                cut = (evs[k].x_px + evs[k + 1].x_px) / 2
        if cut is None:
            return None
        cuts.append(cut)
    return sum(cuts) / len(cuts) if cuts else None


def _split(m, x):
    a = Measure(index=m.index, beats=m.beats, beat_type=m.beat_type, n_staves=m.n_staves,
                events=[e for e in m.events if e.x_px < x])
    b = Measure(index=m.index, beats=m.beats, beat_type=m.beat_type, n_staves=m.n_staves,
                events=[e for e in m.events if e.x_px >= x])
    return [a, b]


def repair_bars(measures):
    """Missed barlines (split, see _split_x) and courtesy signatures after a system's last bar
    (an event-less last 'measure' is dropped).  Re-indexes the page's measures."""
    out = []
    for m in measures:
        x = _split_x(m)
        out.extend(_split(m, x) if x is not None else [m])
    for i, m in enumerate(out):
        m.index = i
    return out


def page_measures(graph, meter):
    """Split the page into measures; `meter` (beats, beat_type) carries in from the previous
    page and is updated by the graph's own meter glyph detections.  Returns (measures, meter)."""
    systems = collections.defaultdict(list)
    for s in graph["systems"]:
        systems[s["system"]].append(s)
    out = []
    for sysno in sorted(systems):
        staves = sorted(systems[sysno], key=lambda s: s["staff"])
        n = min(len(s["bounds"]) - 1 for s in staves)
        changes = {}
        for s in staves:
            for c in s.get("context_changes") or []:
                if c.get("meter"):
                    changes[int(c.get("measure", 0))] = tuple(int(v) for v in c["meter"])
        for i in range(n):
            if i in changes:
                meter = changes[i]
            def _real(s, g):                         # a 'head' inside the clef (first 3 spaces) is the clef
                return not g.get("grace") and g["x"] >= s["bounds"][0] + 3.0 * float(s.get("space_px") or 18.0) * \
                    graph.get("coordinate_scale", 16 / 19.8)
            if (i in (0, n - 1)) and n > 1 and not any(s["bounds"][i] <= g["x"] < s["bounds"][i + 1] and _real(s, g)
                                                        for s in staves for g in s["groups"]):
                continue                             # header before a start-repeat bar / courtesy signature at the end
            m = Measure(index=len(out), beats=meter[0], beat_type=meter[1], n_staves=len(staves))
            for s in staves:
                lo, hi = s["bounds"][i], s["bounds"][i + 1]
                for g in s["groups"]:
                    if g.get("grace") or not (lo <= g["x"] < hi) or not _real(s, g):
                        continue
                    is_rest, d, _ = _hint(g)
                    heads = [] if is_rest else (g.get("heads") or [])
                    pitches = [tuple(h["pitch"][:3]) for h in heads if h.get("pitch")]
                    if not is_rest and not pitches:
                        continue
                    stem = {"up": 1, "down": -1}.get(g.get("direction"), 0)
                    m.events.append(Event(
                        staff=int(s["staff"]), x_px=float(g["x"]), space_px=float(s.get("space_px") or 18.0),
                        is_rest=is_rest, d=d, dots=1 if g.get("dotted") else 0, warp=_warp(g), stem=stem,
                        pitches=pitches,
                        heads_x=[(float(h["xy"][0]), midi_of(h["pitch"])) for h in heads if h.get("pitch")],
                        codex=dict(onset=g.get("onset"), duration=g.get("duration"), voice=g.get("voice"))))
            out.append(merge_chords(m))
    return repair_bars(out), meter


METER_OF_LENGTH = {960: (2, 4), 1440: (3, 4), 1920: (4, 4), 720: (3, 8), 2400: (5, 4), 2880: (6, 4)}


def has_meter_glyph(graph):
    return any(c.get("meter") for s in graph["systems"] for c in (s.get("context_changes") or []))


def infer_meter(measures):
    """Meter from content when no meter glyph was read: the bar length that most single-voice
    staff-measures' hinted durations add up to exactly (no transformer, no header).  None if
    there is too little evidence."""
    votes = collections.Counter()
    for m in measures:
        skip = measure_rests(m)
        per = collections.defaultdict(list)
        for i, e in enumerate(m.events):
            if i not in skip:
                per[e.staff].append(e)
        for evs in per.values():
            stems = {e.stem for e in evs if not e.is_rest and e.stem}
            if len(stems) > 1:                       # two voices on the staff: sums are not bars
                continue
            total = round(sum(V.duration_of(e.d, e.dots, e.warp) for e in evs))
            if total in METER_OF_LENGTH:
                votes[total] += 1
    if not votes or votes.most_common(1)[0][1] < 3:
        return None
    return METER_OF_LENGTH[votes.most_common(1)[0][0]]


def set_meter(measures, meter):
    for m in measures:
        m.beats, m.beat_type = meter
    return measures


def measure_rests(m):
    """Indices of measure rests: a staff whose only event is a whole-rest glyph.  Engravers centre
    it in the bar and it fills any meter, so it is settled here (onset 0, the whole bar) and never
    reaches the decoder, whose corpus has no 3/4 or 6/8 measure rests at all."""
    per = collections.defaultdict(list)
    for i, e in enumerate(m.events):
        per[e.staff].append(i)
    return {ix[0] for ix in per.values() if len(ix) == 1 and m.events[ix[0]].is_rest and m.events[ix[0]].d == 0}


def to_cand_measure(m, piece, seed):
    """Measure -> CandMeasure with column-snapped x normalised to 0..1000 (as real-x training).
    Measure rests are left out (see measure_rests); `gt_eid` indexes m.events."""
    skip = measure_rests(m)
    keep = [i for i in range(len(m.events)) if i not in skip]
    if not keep:
        return None
    evs = [m.events[i] for i in keep]
    xs, staves, stems, pitches = [], [], [], []
    for e in evs:
        tenths = 10.0 / e.space_px
        if e.is_rest:
            x = e.x_px * tenths
        else:
            x = event_x([(hx * tenths, mi) for hx, mi in e.heads_x], e.stem, e.x_px * tenths)
        xs.append(x)
        staves.append(e.staff)
        stems.append(e.stem)
        pitches.append([mi for _hx, mi in e.heads_x])
    col = snap_columns(xs, staves, stems, pitches)
    lo, hi = min(col), max(col)
    order = list(range(len(evs)))
    random.Random(seed).shuffle(order)                 # decoder never sees reading order
    cands = []
    for new_id, i in enumerate(order):
        e = evs[i]
        cands.append((new_id, Cand(cid=new_id, staff=e.staff, x=500.0 if hi <= lo else 1000.0 * (col[i] - lo) / (hi - lo),
                                   is_rest=e.is_rest, hint_d=e.d, hint_dots=e.dots, hint_warp=e.warp, gt_eid=keep[i],
                                   midi=min(p for _hx, p in e.heads_x) if e.heads_x else -1, stem=e.stem)))
    cands = [c for _k, c in sorted(cands, key=lambda t: t[0])]
    # The decoder reads only `gt.beats` (the printed meter numerator, a visible input); a stub
    # with nothing else makes any accidental ground-truth access fail loudly.
    return CandMeasure(piece=piece, length=m.length, cands=cands, gt=SimpleNamespace(beats=m.beats))
