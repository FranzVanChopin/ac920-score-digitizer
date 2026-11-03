"""Exact per-voice attribute DP: take the ORDER from the beam, solve the VALUES exactly.

Why
---
Five separate beam knobs were measured to have zero or negative effect
(k_attr, beam width, dev_w, lam_hint, attr_diversity), and the reason is
structural: a dense measure generates |rem| * k_attr ~ 95 children per step and
the beam keeps 16, so the survivors are almost all "every event takes its top-1
attribute".  Top-1 recall is 95.99% on dense measures, hence 0.9599^18.9 = 0.46 --
a majority of dense measures contain at least one event whose truth is NOT the
top-1, and those measures are unreachable through the beam.

Two already-measured facts point at the fix:
  * with TRUE attributes the same search is 100% correct on dense measures, so
    the ORDER search is good and does not need replacing;
  * the terminal evaluator ranks the true solution strictly better on 100% of
    the misses, so the loss function is right too.

So keep the order and solve the values with an EXACT dynamic program: state =
accumulated ticks, action = attribute option, and the "voice tiles [0, length)
exactly" constraint is enforced by construction rather than by search.  There is
no beam, so every option the model offers is genuinely considered.

Per-voice time is milliseconds: the DP is a handful of shifted vector minimums.
"""
from __future__ import annotations

import math

import numpy as np

from visual import duration_of

SCALE = 12          # 1/12 tick cells; exact for the 2/3, 4/5, 3/4, 7/4 ratios


def cells(ticks: float) -> int:
    return int(round(ticks * SCALE))


def build_options(D, O, W, warp_vocab, c, *, k_attr, canon_ok=None, canon_pen=0.0,
                  hint_w=0.0, dur_bias=0.0, dur_ref=240.0, cx_bonus=0.0,
                  dur_pred=0.0):
    """(dur, d, dots, warp, cost) options for one event, cheapest first.

    `dur_bias` is a log-duration prior, i.e. a CALIBRATION correction.  Measured:
    when the exact DP picks the wrong assignment on dense measures the chosen
    duration is too LONG 71.5% of the time (28.5% too short) -- the signature of
    regression to the mean under uncertainty.  A positive value tilts the cost
    toward shorter values and cancels that bias; it is not a modelling advance,
    just an honest correction for a measured asymmetry.

    `cx_bonus` rewards RHYTHMIC COMPLEXITY (a negative cost per unit), because the
    decoder was measured to suffer a SIMPLICITY BIAS: among the 153 value-only
    failures the ground truth is rhythmically MORE complex than the pick in 73.2% of
    cases and simpler in only 15.7%.  Base rates favour plain durations, so the model
    under-predicts dotted and tupleted values -- precisely the texture of the dense
    pieces.  This is the opposite of the usual "prefer the simple parse" prior.
    """
    outs = []
    bias = 0.0
    for d in range(D.shape[0]):
        for o in range(O.shape[0]):
            for wi in range(W.shape[0]):
                w = tuple(warp_vocab[wi])
                dur = duration_of(d, o, w)
                cost = -(float(D[d]) + float(O[o]) + float(W[wi]))
                if dur_bias:
                    # dur_pred > 0 switches the reference from the global constant to a
                    # PER-EVENT prediction.  The x-column spacing predicts the true
                    # duration with 84.62% exact accuracy (37,498 ground-truth events,
                    # rank correlation 0.894) -- a purely geometric signal the model
                    # cannot see, because it only uses x as a soft feature.
                    _ref = dur_pred if dur_pred > 0 else dur_ref
                    cost += dur_bias * math.log2(max(dur, 1e-6) / _ref)
                if cx_bonus:
                    _cx = float(d) + 2.0 * float(o)
                    if w != (1, 1):
                        _cx += 3.0
                    cost -= cx_bonus * _cx
                if canon_ok is not None and not canon_ok.get((d, o, w), True):
                    cost += canon_pen
                if hint_w:
                    h = (min(max(c.hint_d, 0), 6), min(max(c.hint_dots, 0), 2),
                         tuple(c.hint_warp))
                    if (d, o, w) != h:
                        cost += hint_w
                outs.append((dur, d, o, w, cost))
    outs.sort(key=lambda t: t[4])
    return outs[:max(1, k_attr)]


def iso_penalty_vectors(seq_x, refs, length, *, tol=0.03, width=1000.0,
                        max_pen=1.0):
    """Per-event cost ARRAYS penalising onsets that misalign across voices.

    `refs` is a list of (x, onset) from the OTHER voices.  Engraving puts events
    that sound together in the same column, so an onset far from every reference
    at a nearby x is evidence that the time value is wrong.  This is the one
    strong signal the policy cannot see: it only exists once two voices are
    placed, which is exactly why it belongs in the DP and not in the model.
    """
    L = cells(length)
    accs = np.arange(L + 1) / float(SCALE)
    out = []
    for x in seq_x:
        near = [(rx, rt) for (rx, rt) in refs if abs(rx - x) / width < tol]
        if not near:
            out.append(None)
            continue
        pen = np.full(L + 1, max_pen)
        for rx, rt in near:
            w = 1.0 - abs(rx - x) / (tol * width)
            pen = np.minimum(pen, w * np.abs(accs - rt) / float(length))
        out.append(pen)
    return out


def voice_dp(n_events, length, options, state_pen=None):
    """Exact argmin over assignments whose durations sum to `length`.

    options[i]   = list of (dur, d, dots, warp, cost) for event i, order fixed.
    state_pen[i] = optional array over tick cells: extra cost for ENTERING state
                   `acc` at event i.  A state-dependent signal such as cross-voice
                   isochrony cannot be a scalar, because the penalty depends on
                   the moment the event is placed.

    Returns (assignment, cost) with assignment[i] = (onset, dur, d, dots, warp),
    or (None, inf) when the voice cannot be tiled at all.
    """
    L = cells(length)
    INF = float("inf")
    dp = np.full(L + 1, INF)
    dp[0] = 0.0
    bps = []
    for i in range(n_events):
        if state_pen is not None and state_pen[i] is not None:
            dp = dp + state_pen[i][:L + 1]
        nxt = np.full(L + 1, INF)
        bp_acc = np.full(L + 1, -1, dtype=np.int32)
        bp_opt = np.full(L + 1, -1, dtype=np.int16)
        for j, (dur, _d, _o, _w, c) in enumerate(options[i]):
            sh = cells(dur)
            if sh <= 0 or sh > L:
                continue
            src = dp[:L + 1 - sh]
            cand = src + c
            better = cand < nxt[sh:]
            idx = np.nonzero(better)[0]
            if idx.size:
                nxt[sh + idx] = cand[idx]
                bp_acc[sh + idx] = idx            # parent state == its own index
                bp_opt[sh + idx] = j
        dp = nxt
        bps.append((bp_acc, bp_opt))
        if not np.isfinite(dp).any():
            return None, INF
    if not np.isfinite(dp[L]):
        return None, INF
    out = []
    acc = L
    for i in range(n_events - 1, -1, -1):
        bp_acc, bp_opt = bps[i]
        j = int(bp_opt[acc])
        if j < 0:
            return None, INF
        pa = int(bp_acc[acc])
        dur, d, o, w, _c = options[i][j]
        out.append((pa / SCALE, dur, d, o, tuple(w)))
        acc = pa
    out.reverse()
    return out, float(dp[L])

