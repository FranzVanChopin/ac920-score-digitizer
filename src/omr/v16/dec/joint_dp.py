"""Joint column DP: solve ALL voices at once, subject to the exact column invariant.

The invariant
-------------
Measured on all 37,884 candidate/truth pairs in the frozen corpus: events that
share an x share an **onset** (11,178 pairs, zero exceptions), and x order is
onset order (zero violations).  Equivalently, the accumulated time of every voice
that has an event in a column equals that column's onset.

Measured headroom (results/column_headroom.json): on the dense bucket **72.4% of
the WRONG measures violate this**, 52.2% overall.  A local repair that pinned one
column at a time changed 8 of 721 measures and fixed 1 -- a violation is a
SYMPTOM, so forcing one column to agree just re-tiles the voice into a different,
still-wrong solution.

Why the per-voice DP cannot see it
----------------------------------
Two voices can each tile [0, length) perfectly and still disagree about the
columns.  Solving jointly searches only invariant-respecting assignments -- a
strictly smaller set that still contains the truth -- so wrong-but-valid tilings
are excluded by construction.

Formulation
-----------
Columns are the distinct x values in increasing order.  The state is the vector of
per-voice accumulated times, which is exactly the onset each voice would take if
it placed its next event now.  At a column, every voice that HAS an event there
must already be at the SAME accumulated time (that is the invariant); each then
advances by a valid duration of its own.  Voices without an event at the column
simply wait.

Two earlier attempts are recorded because both failed for instructive reasons:
  * a float version reported "infeasible" on 719/721 measures purely from
    accumulated drift against `round(dur, 6)` dictionary keys;
  * a version that required the same *end* time at a column was simply wrong --
    two voices can start together and end at different times, and the very first
    failing example showed exactly that (durations 1440 vs 2880 from a shared
    onset).
All arithmetic is therefore in INTEGER tick cells (1/12 tick).
"""
from __future__ import annotations

from attr_dp import cells


def joint_solve(voices, length, cost_maps, *, max_states=200000, strict=True):
    """voices    : list of lists of (cid, x), each in x order
    cost_maps  : cost_maps[v][i] = {ticks:int -> (cost, d, dots, warp)}
    Returns (assignments, total_cost, n_states) or (None, inf, n).

    assignments[v][i] = (onset_ticks, dur_ticks, d, dots, warp)
    """
    nv = len(voices)
    if nv == 0:
        return None, float("inf"), 0
    L = cells(length)
    col_of = {}
    for vi, v in enumerate(voices):
        for i, c in enumerate(v):
            col_of.setdefault(int(round(c[1] * 1e6)), []).append((vi, i))
    order = sorted(col_of)
    if not order:
        return None, float("inf"), 0
    # Every voice's first event has onset 0 and _engrave maps onset 0 to x 0, so
    # all first events share the first column.  Verified on all 2,194 polyphonic
    # measures in the frozen corpus.
    if strict and len({vi for vi, _i in col_of[order[0]]}) != nv:
        return None, float("inf"), 0

    INF = float("inf")
    # PRUNE BY COMPLETABILITY.  The state is a product of per-voice accumulated
    # times, which explodes: measured 60,028 states for 2 voices and 512,000 for 3
    # (the cap), so the DP returned nothing for almost every measure.  But at
    # column c the number of events each voice has ALREADY placed is fixed by the
    # partition, hence so is how many remain.  A state can only lead to a solution
    # if `length - acc_v` is expressible as a sum of the durations of that voice's
    # remaining events, so precompute those reachable sums and drop the rest.
    placed_upto = []          # per column: {voice: events placed at columns <= c}
    cnt = {vi: 0 for vi in range(nv)}
    for x in order:
        for vi, _i in col_of[x]:
            cnt[vi] += 1
        placed_upto.append(dict(cnt))
    placed_before = [{vi: 0 for vi in range(nv)}]
    for d in placed_upto[:-1]:
        placed_before.append(dict(d))
    reach = []                # reach[vi][j] = set of sums using the last j events
    for vi, v in enumerate(voices):
        n = len(v)
        sets = [set() for _ in range(n + 1)]
        sets[0] = {0}
        for j in range(1, n + 1):
            opts = cost_maps[vi][n - j]
            prev = sets[j - 1]
            cur = set()
            for s in prev:
                for dur in opts:
                    cur.add(s + dur)
            sets[j] = cur
        reach.append(sets)

    def live(st, ci):
        """`ci` is the column about to be processed; the state is BEFORE it."""
        for vi in range(nv):
            rem = len(voices[vi]) - placed_before[ci][vi]
            if (L - st[vi]) not in reach[vi][rem]:
                return False
        return True

    dp = {(0,) * nv: 0.0}
    if not live((0,) * nv, 0):
        return None, float("inf"), 1
    back = []
    for ci, x in enumerate(order):
        present = col_of[x]
        nxt, bp = {}, {}
        for st, cost in dp.items():
            # THE INVARIANT: everyone with an event here is at the same onset
            t0 = st[present[0][0]]
            if any(st[vi] != t0 for vi, _i in present):
                continue
            # branch over each present voice's valid durations
            choices = [list(cost_maps[vi][i].items()) for vi, i in present]
            for combo in _product(choices):
                tot = cost
                ns = list(st)
                ok = True
                for (vi, _i), (dur, (c, _d, _o, _w)) in zip(present, combo):
                    nv_acc = st[vi] + dur
                    if dur <= 0 or nv_acc > L:
                        ok = False
                        break
                    tot += c
                    ns[vi] = nv_acc
                if not ok:
                    continue
                ns = tuple(ns)
                # check against the NEXT column (ci + 1); if this was the last
                # column the final `all(st == L)` test governs instead
                if ci + 1 < len(order) and not live(ns, ci + 1):
                    continue
                if ns in nxt and nxt[ns] <= tot:
                    continue
                nxt[ns] = tot
                bp[ns] = st
            if len(nxt) > max_states:
                return None, float("inf"), len(nxt)
        if not nxt:
            return None, float("inf"), 0
        dp = nxt
        back.append(bp)

    best, best_cost = None, INF
    for st, cost in dp.items():
        if all(st[vi] == L for vi in range(nv)) and cost < best_cost:
            best, best_cost = st, cost
    if best is None:
        return None, INF, len(dp)

    assign = [[None] * len(v) for v in voices]
    st = best
    for ci in range(len(order) - 1, -1, -1):
        prev_st = back[ci][st]
        for vi, i in col_of[order[ci]]:
            onset = prev_st[vi]
            need = st[vi] - onset
            hit = cost_maps[vi][i].get(need)
            if hit is None:
                return None, INF, len(dp)
            _c, d, o, w = hit
            assign[vi][i] = (onset, need, d, o, w)
        st = prev_st
    return assign, best_cost, len(dp)


def _product(lists):
    """Cartesian product without importing itertools at module scope."""
    import itertools
    return itertools.product(*lists)
