"""Column-consistent joint value DP: re-solve ALL voices at once when the answer is impossible.

The invariant
-------------
Events that share an x share an onset, distinct x means distinct onsets, and x order is
onset order (37,884 candidate/truth pairs, zero exceptions; see src/joint_dp.py).  This is
observable at inference time -- it needs only the candidates' x and the decoded onsets --
so a production answer that breaks it is KNOWN to be wrong without any ground truth.

Measured on the 95.81 production run (5 folds, 4,100 measures, 2026-10-23):
  * 185 measures fail; scoring truth and answer under the exact simulator
    (visual.synthesise) shows 139 of them (75%) are answers the observations could
    never have produced, and 90 of those break this column invariant;
  * the stage-B DP cannot see it: it solves each voice independently, so two voices can
    each tile [0, length) and still disagree about a shared column.

What this does
--------------
Only when the final answer breaks the invariant, keep its partition and solve the values
of every voice jointly:
  * columns = distinct x, increasing; each column has ONE onset, strictly increasing;
  * a voice's events are taken in x order and its first event must sit in column 0
    (every voice starts at onset 0, which `_engrave` maps to x 0);
  * a voice that still has events left must be sounding through any column it skips;
  * per-event options and costs are exactly dp_refine's (build_options, k_attr=40,
    hint_w), with state features from the production trajectory.
The state is (column onset, per-voice accumulated time, per-voice events placed); voices
present at a column are placed one at a time, and at most `max_states` cheapest states
survive each step (prod.COLUMN_JOINT_MAX_STATES = 1,000).  On all 93 triggered measures
the answers at 1,000, 4,000 and 30,000 are identical; 500 changes 3 of them.

Why src/joint_dp.py reported "1/721 solvable": it prices every (d, dots, warp) option and
returns nothing once 60,000 states are exceeded.  Measured on the 93 measures this stage
triggers on: the old `joint_refine` gave up on the state cap 64 times, found no solution
22 times and returned an answer 7 times.  This version uses the DP's own top-40 options
and keeps the cheapest states instead of giving up -- it solves 74 of the 93, and all 49
whose partition is right.

The replacement is accepted unconditionally: the answer it replaces is impossible, and
a correct answer never triggers (0 of 3,915 perfect measures did).
"""
from __future__ import annotations

import collections
import math

import numpy as np

from attr_dp import SCALE, build_options, cells
from pickers import cand_features, hint_indices, state_features
from prod import COLUMN_JOINT_MAX_STATES, PROD


def _xkey(x):
    return round(float(x), 6)


def column_violation(cm, voices):
    """Name of the broken invariant for a tuple-form answer, or None if it holds.

    `voices` holds (cid, x, onset, dur, d, dots, warp) tuples.  An answer that does not
    cover every candidate has no complete partition to re-solve on, so it never triggers.
    """
    on = {}
    for v in voices:
        for e in v:
            on[e[0]] = int(round(e[2] * SCALE))
    if any(c.cid not in on for c in cm.cands):
        return None
    by_x = collections.defaultdict(set)
    by_on = collections.defaultdict(set)
    for c in cm.cands:
        by_x[_xkey(c.x)].add(on[c.cid])
        by_on[on[c.cid]].add(_xkey(c.x))
    if any(len(s) > 1 for s in by_x.values()):
        return "column-split"
    if any(len(s) > 1 for s in by_on.values()):
        return "column-merge"
    pairs = sorted((_xkey(c.x), on[c.cid]) for c in cm.cands)
    if any(b[1] < a[1] for a, b in zip(pairs, pairs[1:])):
        return "x-order"
    return None


def spacing_residuals(cm, voices, alpha):
    """Engraving-spacing residuals of a tuple-form answer, or None.

    Real engraving spaces consecutive columns roughly as dt ** alpha (alpha = 0.53 fitted on
    PDMX default-x; see pdmx_spacing_fit.py) times a per-measure scale.  Returns
    ([log dx - alpha * log dt - offset per column gap], offset), the offset being the median
    over the measure's own gaps, so a few wrong gaps cannot move it.  None when the answer
    misses a candidate, breaks the column invariant, or has fewer than two gaps.
    """
    on = {}
    for v in voices:
        for e in v:
            on[e[0]] = int(round(e[2] * SCALE))
    if any(c.cid not in on for c in cm.cands) or column_violation(cm, voices) is not None:
        return None
    col_on = {_xkey(c.x): on[c.cid] for c in cm.cands}
    xs = sorted(col_on)
    if len(xs) < 3:
        return None
    raw = []
    for a, b in zip(xs, xs[1:]):
        dt, dx = col_on[b] - col_on[a], b - a
        if dt <= 0 or dx <= 0:
            return None
        raw.append(math.log(dx) - alpha * math.log(dt))
    off = sorted(raw)[len(raw) // 2]
    return [r - off for r in raw], off


def _options(cm, seq, durs, pk, warp_vocab, canon_ok, k_attr):
    """dp_refine's option lists for one voice, in tick CELLS."""
    accs, a = [], 0.0
    for du in durs:
        accs.append(a)
        a += du
    per = []
    for i, c in enumerate(seq):
        st = {"acc": accs[i], "last_x": (seq[i - 1].x if i else 0.0),
              "remaining": seq[i:], "n_closed": 0,
              "cur_voice": [z.cid for z in seq[:i]], "all_cands": cm.cands,
              "n_beats": cm.gt.beats}
        F = cand_features(seq[i:], st, None, cm.length)
        S = state_features(st, cm.length, len(cm.cands), None)
        di, oi, wi = hint_indices(seq[i:], warp_vocab)
        D, O, W = pk.mlp.attr_logprobs_all(F, S, (di, oi, wi))
        opts = build_options(D[0], O[0], W[0], warp_vocab, c, k_attr=k_attr,
                             canon_ok=canon_ok, canon_pen=0.0, hint_w=PROD["hint_w"])
        per.append([(cells(du), d, o, tuple(w), co) for du, d, o, w, co in opts])
    return per


def column_joint_refine(cm, voices, pk, warp_vocab, canon_ok, *, k_attr=40,
                        max_states=COLUMN_JOINT_MAX_STATES, stats=None, spacing=None):
    """Tuple-form voices solved jointly under the column invariant, or None.

    `stats`, if a dict, receives the cost counters the FPGA budget needs:
    `peak_states` (largest surviving layer) and `steps` (layers built).

    `spacing` = (alpha, offset, c, weight) adds an engraving-spacing cost when the DP enters
    a column: weight * log(1 + (eps / c) ** 2), eps = log dx - alpha * log dt - offset.
    Heavy-tailed on purpose: real engraving adds space for accidentals and ornaments (1% of
    PDMX gaps sit > 0.42 above the power law), so an outlier costs a few nats at most.
    """
    cid_of = {c.cid: c for c in cm.cands}
    L = cells(cm.length)
    vs = [sorted(v, key=lambda e: (cid_of[e[0]].x, e[2])) for v in voices if v]
    nv = len(vs)
    if nv == 0:
        return None
    col = {x: i for i, x in enumerate(sorted({_xkey(c.x) for c in cm.cands}))}
    if any(col[_xkey(cid_of[v[0][0]].x)] != 0 for v in vs):
        return None
    seqs = [[cid_of[e[0]] for e in v] for v in vs]
    opts = [_options(cm, seq, [e[3] for e in v], pk, warp_vocab, canon_ok, k_attr)
            for seq, v in zip(seqs, vs)]
    # COMPLETABILITY: reach[v][j][s] is True when the LAST j events of voice v can sum to
    # s cells, so a state is only kept if the rest of every voice can still tile the
    # measure.  Boolean arrays, not Python sets: a set of up to L ~ 23,000 ints per
    # (voice, j) was measured at ~45 MiB of peak heap on dense measures.
    reach = []
    for per in opts:
        n = len(per)
        arrs = [np.zeros(L + 1, dtype=bool) for _ in range(n + 1)]
        arrs[0][0] = True
        for j in range(1, n + 1):
            prev, cur = arrs[j - 1], arrs[j]
            for du in {o[0] for o in per[n - j]}:
                if 0 <= du <= L:
                    cur[du:] |= prev[:L + 1 - du]
        reach.append(arrs)
    members = collections.defaultdict(list)
    for vi, seq in enumerate(seqs):
        for i, c in enumerate(seq):
            members[col[_xkey(c.x)]].append((vi, i))
    xcol = sorted(col)
    if spacing is not None:
        sp_alpha, sp_off, sp_c, sp_w = spacing

    # state = (column onset, per-voice accumulated cells, per-voice events placed)
    start = (-1, (0,) * nv, (0,) * nv)
    frontier = {start: 0.0}
    layers = []                 # one dict per step: state -> (cost, parent, action)
    for ci in range(len(col)):
        pres = members[ci]
        here = {vi for vi, _i in pres}
        nxt = {}
        for (tp, accs, placed), cost in frontier.items():
            t0 = accs[pres[0][0]]
            if t0 <= tp or any(accs[vi] != t0 for vi in here):
                continue
            # a voice skipping this column must still be sounding through it
            if any(vj not in here and placed[vj] < len(seqs[vj]) and accs[vj] <= t0
                   for vj in range(nv)):
                continue
            if spacing is not None and ci > 0:
                eps = math.log(xcol[ci] - xcol[ci - 1]) - sp_alpha * math.log(t0 - tp) - sp_off
                cost = cost + sp_w * math.log1p((eps / sp_c) ** 2)
            key = (t0, accs, placed)
            if key not in nxt or cost < nxt[key][0]:
                nxt[key] = (cost, (tp, accs, placed), None)
        if not nxt:
            return None
        layers.append(nxt)
        frontier = {k: v[0] for k, v in nxt.items()}
        for vi, i in pres:
            rem_after = len(seqs[vi]) - i - 1
            nxt = {}
            for (t0, accs, placed), cost in frontier.items():
                for du, d, o, w, co in opts[vi][i]:
                    a2 = accs[vi] + du
                    if du <= 0 or a2 > L or not reach[vi][rem_after][L - a2]:
                        continue
                    key = (t0, accs[:vi] + (a2,) + accs[vi + 1:],
                           placed[:vi] + (placed[vi] + 1,) + placed[vi + 1:])
                    c2 = cost + co
                    if key not in nxt or c2 < nxt[key][0]:
                        nxt[key] = (c2, (t0, accs, placed), (vi, i, du, d, o, w))
            if not nxt:
                return None
            if len(nxt) > max_states:
                nxt = dict(sorted(nxt.items(), key=lambda kv: kv[1][0])[:max_states])
            layers.append(nxt)
            frontier = {k: v[0] for k, v in nxt.items()}
            if stats is not None:
                stats["peak_states"] = max(stats.get("peak_states", 0), len(nxt))

    if stats is not None:
        stats["steps"] = len(layers)
    done =[(c, k) for k, c in frontier.items() if all(a == L for a in k[1])]
    if not done:
        return None
    key = min(done)[1]
    assign = {}
    for lay in reversed(layers):
        _c, parent, act = lay[key]
        if act is not None:
            vi, i, du, d, o, w = act
            assign[(vi, i)] = (parent[1][vi] / SCALE, du / SCALE, d, o, w)
        key = parent
    return tuple(tuple((e[0], e[1]) + assign[(vi, i)] for i, e in enumerate(v))
                 for vi, v in enumerate(vs))
