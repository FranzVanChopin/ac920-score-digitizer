"""Visual layer synthesis.

Turns a ground-truth measure into what a (good but imperfect) visual front end
would hand to the decoder:

  * candidate events with a real x position from an engraving-like layout,
  * soft hints for division / dots / time-warp that are wrong some of the time.

The decoder NEVER sees onset, duration or voice identity - only the candidates,
their x, their staff, and the noisy hints.  That is exactly the gap the project's
current pipeline closes too early by serialising straight to MusicXML.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

WHOLE = 1920


@dataclass
class Cand:
    cid: int
    staff: int
    x: float
    is_rest: bool
    hint_d: int
    hint_dots: int
    hint_warp: tuple
    gt_eid: int = -1          # for scoring only; never used by any decoder
    # PITCH of the detected notehead; -1 = not available (rests).  This is the only
    # signal that separates two voices sharing one staff: measured on the source,
    # 86.5% of within-staff multi-voice cases have disjoint pitch ranges, but the
    # voice LABEL does not follow vertical order (only 16.1%), so it must be a
    # feature rather than a rule.
    midi: int = -1
    # STEM DIRECTION: the engraver's own voice marker.  Measured on the
    # frozen corpus, re-grouping by stem recovers the exact within-staff
    # partition 95.53% of the time against 8.03% for chance -- far stronger
    # than pitch ORDER (36.71%).  1 up / -1 down / 0 unknown.
    stem: int = 0


@dataclass
class CandMeasure:
    piece: str
    length: int
    cands: list
    gt: object = None
    order: list = field(default_factory=list)   # shuffled input order


def _engrave(onsets, length, rng, width=1000.0):
    """Monotone but NOT linear x layout.

    Engravers space columns roughly by the DURATION between them, but
    sub-linearly (a half note does not get twice the room of a quarter), with
    collision jitter on top.  That is what makes x a *weak* clock: informative
    about time, but far from a measurement of it.

    The earlier version used purely random spacing increments, which threw away
    the duration signal entirely -- unrealistic, and it made the task harder
    than the real one for no reason.
    """
    uniq = sorted(set(onsets))
    n = len(uniq)
    if n == 1:
        return [width * 0.5 for _ in onsets]
    gaps = []
    for a, b in zip(uniq, uniq[1:]):
        dt = (b - a) / float(max(1, length))
        gaps.append((dt ** 0.6) * rng.uniform(0.85, 1.15))
    tot = sum(gaps)
    pos, acc = {}, 0.0
    for i, t in enumerate(uniq):
        pos[t] = width * (acc / tot)
        if i < n - 1:
            acc += gaps[i]
    pos[uniq[-1]] = width
    return [pos[t] for t in onsets]


def synthesise(gt, rng, *, hint_acc=0.85, drop_rate=0.0, spur_rate=0.0,
               d_vocab=(0, 1, 2, 3, 4, 5, 6), shuffle=True, real_rates=None):
    """Build a CandMeasure from a GTMeasure.

    `real_rates` (a dict, see real_noise.DEFAULT_RATES) replaces the iid hint noise with the
    group-correlated noise measured on a real front end; None keeps the iid path bit-identical.
    """
    evs = gt.events
    keep = [e for e in evs if rng.random() >= drop_rate]
    if not keep:
        keep = [evs[0]]

    xs = _engrave([e.onset for e in keep], gt.length, rng)
    real_hint = None
    if real_rates is not None:
        import real_noise
        rd, rdots, rw = real_noise.apply(evs, rng, real_rates, d_vocab)
        real_hint = {id(e): (rd[i], rdots[i], rw[i]) for i, e in enumerate(evs)}
    cands = []
    for e, x in zip(keep, xs):
        if real_hint is not None:
            hd, hdots, hwarp = real_hint[id(e)]
        elif rng.random() < hint_acc:
            hd, hdots, hwarp = e.d, e.dots, e.warp
        else:
            # upstream got it wrong: pick a neighbouring division class
            hd = e.d + rng.choice([-1, 1])
            hd = hd if hd in d_vocab else e.d
            hdots = rng.choice([0, 1, 2]) if e.dots == 0 else 0
            # the classic failure: an implicit tuplet is simply not marked
            hwarp = (1, 1) if e.warp != (1, 1) else rng.choice([(1, 1), (3, 2)])
        cands.append(Cand(cid=-1, staff=e.staff, x=x, is_rest=e.is_rest,
                          hint_d=hd, hint_dots=hdots, hint_warp=hwarp, gt_eid=e.eid,
                          midi=getattr(e, "midi", -1),
                          stem=getattr(e, "stem", 0)))

    # spurious candidates: a non-existent event the detector hallucinated
    n_spur = int(round(spur_rate * max(1, len(cands))))
    for _ in range(n_spur):
        cands.append(Cand(cid=-1, staff=rng.choice(sorted({c.staff for c in cands})),
                          x=rng.uniform(0.0, 1000.0), is_rest=rng.random() < 0.5,
                          hint_d=rng.choice(list(d_vocab)), hint_dots=0,
                          hint_warp=(1, 1), gt_eid=-1))

    order = list(range(len(cands)))
    if shuffle:
        rng.shuffle(order)
    for new_id, old in enumerate(order):
        cands[old].cid = new_id
    return CandMeasure(piece=gt.piece, length=gt.length,
                       cands=sorted(cands, key=lambda c: c.cid), gt=gt)


def written_ticks(d: int, dots: int) -> float:
    return WHOLE * (2.0 ** -d) * (2.0 - 2.0 ** -dots)


def apply_warp(written: float, warp: tuple) -> float:
    a, n = warp
    return written * (n / a)


def duration_of(d: int, dots: int, warp: tuple) -> float:
    return apply_warp(written_ticks(d, dots), warp)


def hint_duration(c: Cand) -> float:
    return duration_of(c.hint_d, c.hint_dots, c.hint_warp)


def warp_vocab(measures) -> list:
    s = {(1, 1)}
    for m in measures:
        for e in m.events:
            a, n = e.warp
            g = math.gcd(a, n)
            s.add((a // g, n // g))
    return sorted(s, key=lambda w: (w[0] / w[1], w))


def canonical(dur, wv, tol=1e-6):
    """The way a duration of `dur` ticks would actually be written.

    Duration is not a bijection onto (division, dots, time-warp): a dotted half
    (960 * 1.5 = 1440) and a whole note under a 4:3 time-warp (1920 * 3/4 = 1440)
    are the same number of ticks.  Without a canonical form the decoder may
    invent equivalences no engraver would produce, which pollutes the accuracy
    metric with differences that are not real errors.

    Preference: no time-warp first, then simpler division, then fewer dots.
    """
    best = None
    for w in wv:
        plain = 0 if w == (1, 1) else 1
        for d in range(7):
            for o in range(3):
                if abs(duration_of(d, o, w) - dur) <= tol:
                    key = (plain, d, o)
                    if best is None or key < best[0]:
                        best = (key, (d, o, w))
    return best[1] if best is not None else None
