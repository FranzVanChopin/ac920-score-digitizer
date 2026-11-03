"""Group noteheads into rhythmic columns, tolerating displaced noteheads.

The decoder relies on "same column <=> same onset" (column_joint, repetition consensus, the
features), so a front end has to hand it one x per column.  Real engraving does not put
every event of one instant at one x:

  * a chord containing a second shifts one notehead to the other side of the stem (up-stem:
    the normal noteheads are on the LEFT, down-stem: on the RIGHT);
  * two voices that collide on a second or unison on one staff push one of them sideways by
    about one notehead width (1.18 staff spaces = 11.8 MusicXML tenths).

Measured on the PDMX tuning set (pdmx_tune_q45.json, 200 pieces, disjoint from the
benchmark), share of measures whose snapped columns do not match the true onsets:

    first note's x, x within 6 tenths merged                3.24%
    + chord x taken on the stem's normal side               2.00%
    + collided-voice merge (below)                          1.27%   (54 merges, 0 false)

The collided-voice merge joins two ADJACENT columns 9-14 tenths apart when one staff has
an event in each whose noteheads are a second or unison apart (<= 2 semitones) with
opposite stems.  A wider window starts to merge stepwise lines of two voices (16 tenths:
18 false merges; 22 tenths: 205).

Units are MusicXML tenths (1/10 staff space); a camera front end converts with its own
measured staff spacing.
"""
from __future__ import annotations

import collections

SNAP_TENTHS = 6.0               # x this close is one column
COLLIDE_GAP = (9.0, 14.0)       # tenths: one notehead width (11.8) +- 2.5
COLLIDE_PITCH = 2               # semitones: unison or second


def event_x(notes, stem, first_x):
    """One x for an event from its noteheads [(x, midi)], on the stem's normal side."""
    xs = [x for x, _m in notes if x is not None]
    if len(xs) <= 1:
        return first_x
    if stem > 0:
        return min(xs)
    if stem < 0:
        return max(xs)
    cnt = collections.Counter(round(x, 1) for x in xs)
    top = max(cnt.values())
    return min(x for x, n in cnt.items() if n == top)       # stemless: majority, then left


def snap_columns(xs, staves, stems, pitches):
    """Column x (mean of the column's raw x) for every event.

    xs[i]       event x in tenths (event_x)
    staves[i]   staff number
    stems[i]    +1 up / -1 down / 0 none
    pitches[i]  midi numbers of the event's noteheads ([] for a rest)
    """
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    cols = [[order[0]]]
    for i in order[1:]:
        if xs[i] - xs[cols[-1][-1]] <= SNAP_TENTHS:
            cols[-1].append(i)
        else:
            cols.append([i])
    merged = [cols[0]]
    for c in cols[1:]:
        prev = merged[-1]
        gap = min(xs[i] for i in c) - max(xs[i] for i in prev)
        collide = COLLIDE_GAP[0] <= gap <= COLLIDE_GAP[1] and any(
            staves[i] == staves[j] and stems[i] * stems[j] < 0
            and any(abs(p - q) <= COLLIDE_PITCH for p in pitches[i] for q in pitches[j])
            for i in prev for j in c)
        if collide:
            merged[-1] = prev + c
        else:
            merged.append(c)
    out = [0.0] * len(xs)
    for c in merged:
        centre = sum(xs[i] for i in c) / len(c)
        for i in c:
            out[i] = centre
    return out


def apply_real_x(cands, x_real, x_notes):
    """Give synthesised candidates their events' REAL layout x, grouped into columns.

    x_real[eid] is the event's first-note default-x and x_notes[eid] its noteheads'
    [(x, midi)]; candidates are matched through gt_eid (the simulator's own bookkeeping, as
    a front end would match its noteheads).  Normalised like visual._engrave: first column
    at 0, last at 1000.
    """
    xs = [event_x(x_notes[c.gt_eid], c.stem, x_real[c.gt_eid]) for c in cands]
    pitches = [[m for _x, m in x_notes[c.gt_eid] if m >= 0] for c in cands]
    col = snap_columns(xs, [c.staff for c in cands], [c.stem for c in cands], pitches)
    lo, hi = min(col), max(col)
    for c, v in zip(cands, col):
        c.x = 500.0 if hi <= lo else 1000.0 * (v - lo) / (hi - lo)
