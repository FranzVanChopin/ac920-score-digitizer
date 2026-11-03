"""Undo same-onset voice swaps with pitch -- a post-hoc repair, no model change.

The failure-mechanism analysis found that 54% of the remaining failures are
partition-only, and 6 of 35 are swaps of two notes that share the SAME onset and the
SAME duration on one staff.  Such a swap preserves every voice's exact tiling, so
nothing in the timing can forbid it -- only pitch can tell the two notes apart.

Measured on the source: within a staff, the earlier-listed (lower-indexed) voice is
the UPPER one 85.2% of the time.  So the rule is "higher note -> higher voice".

Safety: a swap is only applied when it provably preserves exact tiling, i.e. both
notes sit at the same onset AND carry the same duration, so each voice's onset
sequence is unchanged.  Everything else is left alone.

The learned model is untouched: `cand_features` still returns 31 dims and never sees
pitch.  This module only reorders already-assigned events.
"""
from __future__ import annotations

import collections


def _tiles(voice, length):
    """True when the voice's events exactly cover [0, length) with no gaps/overlaps."""
    acc = 0
    for e in sorted(voice, key=lambda z: z[2]):
        if abs(e[2] - acc) > 1:
            return False
        acc += e[3]
    return abs(acc - length) <= 1


def pitch_swap_repair(cm, voices, *, max_passes=3, min_voice_gap=0.0):
    """Swap same-onset same-duration notes whose voice contradicts pitch order.

    `voices` is a list of voice lists of 7-tuples (cid, x, onset, dur, d, dots, warp),
    i.e. the tuple form used by two_stage.  Returns a new list; the input is untouched.
    """
    midi = {c.cid: int(getattr(c, "midi", -1)) for c in cm.cands}
    out = [list(v) for v in voices]
    if not any(m > 0 for k, m in midi.items()
               if any(e[0] == k for v in out for e in v)):
        return out
    for _ in range(max_passes):
        # mean pitch of each voice (unknown pitches ignored)
        means = {}
        for i, v in enumerate(out):
            ps = [midi[e[0]] for e in v if midi.get(e[0], -1) > 0]
            if ps:
                means[i] = sum(ps) / len(ps)
        if len(means) < 2:
            return out
        # group events by (onset, dur, staff); only these groups are swappable
        groups = collections.defaultdict(list)
        for i, v in enumerate(out):
            for k, e in enumerate(v):
                st = next((c.staff for c in cm.cands if c.cid == e[0]), -1)
                groups[(e[2], e[3], st)].append((i, k, midi.get(e[0], -1)))
        changed = False
        for _key, items in groups.items():
            if len(items) < 2:
                continue
            known = [it for it in items if it[2] > 0 and it[0] in means]
            if len(known) < 2:
                continue
            for a in range(len(known)):
                for b in range(a + 1, len(known)):
                    ia, ka, pa = known[a]
                    ib, kb, pb = known[b]
                    if ia == ib or pa == pb:
                        continue
                    # current order vs pitch order
                    cur = (means[ia] > means[ib])
                    want = (pa > pb)
                    if cur == want:
                        continue
                    if abs(means[ia] - means[ib]) < min_voice_gap:
                        continue
                    va, vb = list(out[ia]), list(out[ib])
                    ea, eb = va[ka], vb[kb]
                    va[ka] = (eb[0], eb[1], ea[2], ea[3], ea[4], ea[5], ea[6])
                    vb[kb] = (ea[0], ea[1], eb[2], eb[3], eb[4], eb[5], eb[6])
                    if not (_tiles(va, cm.length) and _tiles(vb, cm.length)):
                        continue        # would break exact tiling: leave it alone
                    out[ia], out[ib] = va, vb
                    changed = True
        if not changed:
            break
    return out
