"""Correlated hint noise measured on a real visual front end (opt-in; production uses iid noise).

The simulator's iid hint noise (`visual.synthesise`, hint_acc 0.85) corrupts each event on its own.
A real front end fails per GROUP: a printed tuplet number marks one group and the next groups are
left implicit, and a beam the stroke model misses takes every note under it down one value.  On
camera67 the decoder, trained on iid noise, then trusts three agreeing wrong hints
(lab/omr_proto/README.md).  This module defines the groups (shared by the measurement script and
the simulator, so both use one definition) and applies group-level noise with measured rates.

Groups, per voice in onset order:
  tuplet group  consecutive events with the same non-trivial warp, cut at beat boundaries
  beam group    consecutive notes of an eighth or shorter (d >= 3) inside one beat
"""
from __future__ import annotations

import collections

BEAT = 480


def groups(events):
    """(tuplet_groups, beam_groups) as lists of index lists into `events`."""
    by_voice = collections.defaultdict(list)
    for i, e in enumerate(events):
        by_voice[e.voice].append(i)
    tup, beam = [], []
    for idx in by_voice.values():
        idx.sort(key=lambda i: events[i].onset)
        cur_t, cur_b = [], []
        for i in idx:
            e = events[i]
            w = tuple(e.warp)
            prev = events[cur_t[-1]] if cur_t else None
            if w != (1, 1) and prev is not None and tuple(prev.warp) == w and e.onset // BEAT == prev.onset // BEAT:
                cur_t.append(i)
            else:
                if len(cur_t) > 0 and tuple(events[cur_t[0]].warp) != (1, 1):
                    tup.append(cur_t)
                cur_t = [i]
            pb = events[cur_b[-1]] if cur_b else None
            if not e.is_rest and e.d >= 3 and pb is not None and e.onset // BEAT == pb.onset // BEAT:
                cur_b.append(i)
            else:
                if cur_b and not events[cur_b[0]].is_rest and events[cur_b[0]].d >= 3:
                    beam.append(cur_b)
                cur_b = [i] if (not e.is_rest and e.d >= 3) else []
        if cur_t and tuple(events[cur_t[0]].warp) != (1, 1):
            tup.append(cur_t)
        if cur_b:
            beam.append(cur_b)
    return tup, beam


DEFAULT_RATES = dict(
    tuplet_group_miss=0.0,    # P(a tuplet group's warp hint is lost -> (1, 1))
    beam_group_minus=0.0,     # P(a beam group's notes get one value too long, d - 1)
    beam_group_all=0.0,       # P(a beam group's notes lose every beam -> quarter, d = 2)
    beam_group_plus=0.0,      # P(a beam group's notes get one value too short, d + 1)
    beam_note_minus=0.0,      # per note of an unshifted beam group: one beam missed (d - 1, >= 2)
    beam_note_quarter=0.0,    # per note of an unshifted beam group: not attached to its beam (d = 2)
    head_err=0.0,             # per note with d <= 2: d +- 1 (hollow/filled head, stem missed)
    rest_err=0.0,             # per rest: glyph class +- 1
    dot_miss=0.0,             # per dotted event: dot not seen
    dot_spur=0.0,             # per undotted event: a dot is seen
    tuplet_spur=0.0,          # per event outside tuplets: a (3, 2) warp hint appears
)


def apply(events, rng, rates, d_vocab=(0, 1, 2, 3, 4, 5, 6)):
    """Return per-event (hint_d, hint_dots, hint_warp) with correlated noise; truth by default."""
    hd = [e.d for e in events]
    hdots = [e.dots for e in events]
    hw = [tuple(e.warp) for e in events]
    tup, beam = groups(events)
    in_tup = {i for g in tup for i in g}
    in_beam = {i for g in beam for i in g}
    for g in tup:
        if rng.random() < rates["tuplet_group_miss"]:
            for i in g:
                hw[i] = (1, 1)
    for g in beam:
        r = rng.random()
        a, b, c = rates["beam_group_minus"], rates["beam_group_all"], rates["beam_group_plus"]
        for i in g:
            if r < a:
                hd[i] = max(2, hd[i] - 1)
            elif r < a + b:
                hd[i] = 2
            elif r < a + b + c:
                hd[i] = min(max(d_vocab), hd[i] + 1)
            else:                                    # group intact: single notes can still slip
                q = rng.random()
                if q < rates["beam_note_quarter"]:
                    hd[i] = 2
                elif q < rates["beam_note_quarter"] + rates["beam_note_minus"]:
                    hd[i] = max(2, hd[i] - 1)
    for i, e in enumerate(events):
        if e.is_rest:
            if rng.random() < rates["rest_err"]:
                hd[i] = min(max(d_vocab), max(min(d_vocab), hd[i] + rng.choice([-1, 1])))
        elif i not in in_beam and e.d <= 2 and rng.random() < rates["head_err"]:
            hd[i] = min(2, max(0, hd[i] + rng.choice([-1, 1])))
        if e.dots and rng.random() < rates["dot_miss"]:
            hdots[i] = 0
        elif not e.dots and rng.random() < rates["dot_spur"]:
            hdots[i] = 1
        if i not in in_tup and hw[i] == (1, 1) and rng.random() < rates["tuplet_spur"]:
            hw[i] = (3, 2)
    return hd, hdots, hw
