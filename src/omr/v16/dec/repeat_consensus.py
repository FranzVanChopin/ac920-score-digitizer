"""Intra-piece repetition consensus: repeated measures of one piece vote on one answer.

Why
---
Music repeats.  On the frozen corpus 47.7% of measures have an "observational twin" in
the same piece: the same multiset of (staff, column rank, pitch, rest) over their
candidates -- everything a front end reports except the noisy time-value hints and the
exact x.  637 of 639 such twin groups really are the same music, but every twin carries
its OWN hint noise and x jitter, so a group is several independent noisy looks at one
measure.  At the 96.50 working point 45 of the 155 failing measures have a twin, and in
31 of them some twin decoded the measure correctly.

What this does
--------------
After every measure of a piece is decoded, group the twins.  Only a group whose answers
DISAGREE is touched (a group that agrees is left alone).  Each distinct answer is
priced on every twin with the stage-B option costs (model log-probabilities + hint
penalty, the same numbers dp_refine and column_joint use), the costs are summed over the
group, and the cheapest answer is offered to every twin.

A twin keeps its own answer when its own evidence prefers that answer by more than
`veto` nats.  This guards against FALSE twins -- same pitches and columns, different
rhythm (2 of 639 groups, e.g. a 2-event measure whose single x gap carries no
information).  Measured on the captured 96.50 answers: no veto 30 fixed / 3 broken;
veto 4 -> 25 fixed / 0 broken (all three broken twins had deltas of 5.1-11.9 nats).
The veto was chosen on the same CV, so it carries a mild selection bias.

Not done: re-solving the values with pooled evidence (one column-joint DP over the summed
option costs).  Measured 31 fixed / 6 broken -- no better than choosing an answer.

Board note: this is the only stage that needs more than one measure.  It works on the
already-decoded answers of a piece (a few hundred bytes per measure), so the per-measure
working-set property of the decoder is unchanged, but output of a measure waits until
its piece (or at least its page) has been decoded.
"""
from __future__ import annotations

import collections

from attr_dp import SCALE
from column_joint import _options, _xkey
from visual import duration_of

ALL_OPTIONS = 10 ** 6       # build_options keeps at most k_attr options: ask for all


def _keys(cm):
    cols = {x: i for i, x in enumerate(sorted({_xkey(c.x) for c in cm.cands}))}
    return {c.cid: (c.staff, cols[_xkey(c.x)], c.midi, c.is_rest) for c in cm.cands}


def _signature(cm):
    return (cm.piece, cm.length, tuple(sorted(_keys(cm).values())))


def _struct(cm, voices):
    """A tuple-form answer in key space: key -> (voice keyset, onset cells, d, dots, warp)."""
    k = _keys(cm)
    out = {}
    for v in voices:
        ks = frozenset(k[e[0]] for e in v)
        for e in v:
            out[k[e[0]]] = (ks, int(round(e[2] * SCALE)), e[4], e[5], tuple(e[6]))
    return out


def _voices(cm, struct):
    """A key-space answer as tuple-form voices of measure `cm`."""
    inv = {v: c for c, v in _keys(cm).items()}
    cid_of = {c.cid: c for c in cm.cands}
    by_v = collections.defaultdict(list)
    for key, (ks, on, d, o, w) in struct.items():
        c = cid_of[inv[key]]
        by_v[ks].append((c.cid, c.x, on / SCALE, duration_of(d, o, w), d, o, w))
    vs = [sorted(v, key=lambda e: (e[1], e[2])) for v in by_v.values()]
    return tuple(tuple(v) for v in sorted(vs, key=lambda v: v[0][1]))


def _cost(cm, struct, pk, warp_vocab, canon_ok):
    """Stage-B option cost of a fixed answer on measure `cm` (lower = better supported)."""
    k = _keys(cm)
    cid_of = {c.cid: c for c in cm.cands}
    tot = 0.0
    for v in _voices(cm, struct):
        seq = [cid_of[e[0]] for e in v]
        per = _options(cm, seq, [e[3] for e in v], pk, warp_vocab, canon_ok, ALL_OPTIONS)
        for c, opts in zip(seq, per):
            want = struct[k[c.cid]][2:]
            hit = [co for _du, d, o, w, co in opts if (d, o, tuple(w)) == want]
            tot += hit[0] if hit else 1e6
    return tot


def repeat_consensus(cms, answers, pk, warp_vocab, canon_ok, *, veto=4.0, stats=None):
    """Return new tuple-form answers for measures `cms` (same order as `answers`).

    `answers[i]` is the tuple-form answer of `cms[i]`.  Measures of different pieces may
    be mixed; grouping is by piece and observation signature.
    """
    out = list(answers)
    groups = collections.defaultdict(list)
    for i, cm in enumerate(cms):
        groups[_signature(cm)].append(i)
    for idx in groups.values():
        if len(idx) < 2:
            continue
        # a twin mapping must be one-to-one, and every answer must cover its measure
        if any(len(set(_keys(cms[i]).values())) != len(cms[i].cands) for i in idx):
            continue
        if any(len({e[0] for v in out[i] for e in v}) != len(cms[i].cands) for i in idx):
            continue
        structs = [_struct(cms[i], out[i]) for i in idx]
        if all(s == structs[0] for s in structs[1:]):
            continue
        if stats is not None:
            stats["groups"] += 1
        uniq = []
        for s in structs:
            if s not in uniq:
                uniq.append(s)
        cost = {}                                    # (member, proposal) -> cost
        for m, i in enumerate(idx):
            for u, s in enumerate(uniq):
                cost[m, u] = _cost(cms[i], s, pk, warp_vocab, canon_ok)
        best = min(range(len(uniq)), key=lambda u: (sum(cost[m, u] for m in range(len(idx))), u))
        for m, i in enumerate(idx):
            own = uniq.index(structs[m])
            if own == best:
                continue
            if cost[m, best] - cost[m, own] >= veto:
                if stats is not None:
                    stats["vetoed"] += 1
                continue
            out[i] = _voices(cms[i], uniq[best])
            if stats is not None:
                stats["rewritten"] += 1
    return out
