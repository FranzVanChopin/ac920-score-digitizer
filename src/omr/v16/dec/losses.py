"""Global structural loss: the x-tick geometry term ("tick twist", tau).

x is only a WEAK clock.  A coherent voice advances in both x and tick at a
roughly stable rate; an ill assignment produces an abnormal local segment.
   * monotone but abnormal slope  -> penalty in [0, 1)
   * a backwards-going segment    -> penalty > 1  (fatal: pruned immediately)

Computed WITHOUT any ground truth - it is a self-consistency measure.
"""
from __future__ import annotations

EPS = 1e-9


def tick_twist(points) -> float:
    """points: ordered [(x_norm, t_norm), ...] for ONE voice (real events only).

    Do NOT append a synthetic 'end of voice' node: its (x == last x, t larger)
    shape would look like a backwards segment and kill every voice.
    """
    if len(points) < 3:
        return 0.0
    (x0, t0) = points[0]
    (x1, t1) = points[-1]
    if x1 - x0 <= EPS:                  # a voice whose x never advances is nonsense
        return 2.0
    s = (t1 - t0) / (x1 - x0)           # overall slope
    if s <= EPS:
        return 2.0
    worst = 0.0
    for i in range(len(points) - 1):
        ax, at = points[i]
        bx, bt = points[i + 1]
        dx, dt = bx - ax, bt - at
        if dt < -EPS:                   # time runs backwards inside a voice: fatal
            return 2.0
        if dx <= EPS:
            # two events at the same x with time advancing: structurally wrong,
            # but scored as a large finite penalty rather than instant death
            worst = max(worst, 1.5)
            continue
        local = dt / dx
        dev = abs(local / s - 1.0)
        # A FORWARD but abnormal segment must live in [0, 1): only a segment
        # going backwards is fatal.  Leaving `dev` unbounded made ordinary
        # deviation look like a reversal and pruned nearly every path.
        pen = dev / (1.0 + dev)
        if pen > worst:
            worst = pen
    return worst


def voice_points(voice, length, width=1000.0):
    """voice: list of (x, onset, dur) -> normalised (x, t) pairs, real events only."""
    return [(x / width, onset / float(length)) for (x, onset, _dur) in voice]


def structural_loss(voices, length, width=1000.0, picker_dev=0.0,
                    voice_weight=0.02, dev_weight=0.15):
    """L(zeta) over a complete measure solution.

    voices: list of voices, each a list of (x, onset, dur).
    No ground truth is used anywhere in here.
    """
    if not voices:
        return 10.0
    taus = [tick_twist(voice_points(v, length, width)) for v in voices]
    tau = max(taus) if taus else 0.0
    n_extra = max(0, len(voices) - 1)
    return tau + voice_weight * n_extra + dev_weight * picker_dev


def iso_penalty(voices, length, width=1000.0, tol=0.03, use_max=True):
    """Isochrony across voices.

    Voices in piano music share one metric grid, so events that the engraver put
    in the SAME horizontal column must sound at the SAME tick.  A duration error
    inside one voice shifts its onsets and breaks that agreement, which makes
    this a strong ground-truth-free signal for localising a wrong time value --
    exactly the information the geometric tick-twist term alone does not carry.

    Weighted by horizontal proximity so that near-but-distinct columns count less.
    """
    evs = [e for v in voices for e in v]
    n = len(evs)
    if n < 2:
        return 0.0
    tot, cnt, worst = 0.0, 0, 0.0
    for i in range(n):
        xi, ti = evs[i][1], evs[i][2]
        for j in range(i + 1, n):
            dx = abs(xi - evs[j][1]) / width
            if dx >= tol:
                continue
            w = 1.0 - dx / tol
            d = w * (abs(ti - evs[j][2]) / float(length))
            tot += d
            cnt += 1
            if d > worst:
                worst = d
    if not cnt:
        return 0.0
    # The MEAN lets one badly mis-aligned column hide among many good ones; the
    # worst offender is the informative statistic for locating a wrong duration.
    return worst if use_max else tot / cnt

