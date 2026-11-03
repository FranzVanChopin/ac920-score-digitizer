"""Probability-guided beam search over measure structure.

A measure is decoded by threading the candidate events into voice chains:
  * state  = (used set, completed voices, current voice prefix, accumulated ticks)
  * action = (next candidate u, its division/dots/time-warp)
  * EOS    = legal only when the current voice exactly fills the measure; it
             closes the voice and the next voice restarts at tick 0.

Ground truth is never consulted.  Pruning uses only self-consistency:
  * cumulative log-probability penalty ("pretentiousness")
  * a fatal x-tick reversal inside a voice
  * a reachability bound on the current voice's remaining ticks

If no complete solution is found within the beam, the best PARTIAL solution is
returned instead (flagged complete=False) so partial recovery is still scored.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from losses import iso_penalty, tick_twist, voice_points
from visual import canonical, duration_of, hint_duration

EPS = 1e-6
SCALE = 12          # 1/12 tick cells make 1:1, 2:1, 3:2, 2:3 durations integral


def _cells(dur):
    """Duration in 1/SCALE tick cells, or None if it is not an exact multiple."""
    v = dur * SCALE
    iv = int(round(v))
    return iv if abs(v - iv) < 1e-9 else None


def _reach_mask(opts, table, need):
    """Bitmask of tick sums reachable by choosing, per remaining candidate, one
    of its offered durations -- or skipping that candidate (it belongs to a
    later voice).  Bit k set  <=>  sum k/SCALE is reachable."""
    need_cell = int(round(need * SCALE))
    if need_cell < 0:
        return 0
    cap = (1 << (need_cell + 1)) - 1
    mask = 1                                   # sum 0 is always reachable
    for oi in opts:
        nm = mask                              # skip this candidate
        for (d, o, w, _lp) in oi:
            cell = _cells(table[(d, o, w)])
            if cell is not None and 0 < cell <= need_cell:
                nm |= mask << cell
        mask = nm & cap
        if mask == 0:
            return 0
    return mask


def _bit(mask, need):
    return (mask >> int(round(need * SCALE))) & 1


@dataclass
class Node:
    used: int
    cur: tuple
    acc: int
    voices: tuple
    logp: float
    decisions: int = 0
    tau: float = 0.0
    priority: float = 0.0
    hdev: int = 0          # events consumed so far whose attrs differ from the hint


def _key(node):
    return (node.used, tuple((e[0], e[4], e[5], e[6]) for e in node.cur))


def _popcount(x):
    return bin(x).count("1")


def _tau_voice(voice, length, width):
    if len(voice) < 3:
        return 0.0
    return tick_twist(voice_points([(e[1], e[2], e[3]) for e in voice], length, width))


def _partial_iso(node, length, width, tol):
    """Isochrony visible DURING the search.

    The terminal loss already scores cross-voice alignment, but the beam's
    priority only looked at the policy probability and the geometric twist -- so
    the single most informative consistency signal could not guide the search,
    only rank finished candidates.  This exposes it to the beam.
    """
    other = [e for v in node.voices for e in v]
    if not other or not node.cur:
        return 0.0
    return iso_penalty([node.cur, other], length, width, tol=tol)


def _max_tau(voices, length, width):
    if not voices:
        return 0.0
    return max(_tau_voice(v, length, width) for v in voices)


def _hint_deviation(voices, hint_map):
    """Fraction of events whose chosen time value differs from the upstream hint.

    The policy is anchored to the hint by the trust head, but the EVALUATOR had
    no such term -- so a solution that rewrote correct hints could still win on
    geometry alone.  With perfect hints the answer is literally the hint set, yet
    16% of polyphonic measures were still not perfect; minimal-edit preference is
    the missing Occam term.
    """
    if not hint_map:
        return 0.0
    n = bad = 0
    for v in voices:
        for e in v:
            h = hint_map.get(e[0])
            if h is None:
                continue
            n += 1
            if (e[4], e[5], tuple(e[6])) != h:
                bad += 1
    return bad / n if n else 0.0


def _voice_median_pitch(voice, midi_of):
    """Median pitch of one voice, ignoring rests and unpitched events."""
    ps = [midi_of.get(e[0], -1) for e in voice]
    ps = sorted(p for p in ps if p > 0)
    if not ps:
        return None
    n = len(ps)
    return float(ps[n // 2]) if n % 2 else 0.5 * (ps[n // 2 - 1] + ps[n // 2])


def _xmeas_term(voices, prev_pitches, midi_of):
    """Register-discontinuity cost against the PREVIOUS measure's voices.

    The decoder is called once per measure and carries no state, so nothing in the model
    can know that a voice kept roughly the same register across the barline -- yet that is
    exactly what real engraving does.  Measured on the frozen corpus: 95.7% of voices stay
    within one octave across a barline against a 71.2% chance baseline, median change 0.17
    octave (5,750 shared voices).

    Voices are matched by INDEX, because the encoder numbers them in order and the beam
    builds them in that same order.  A voice with no precedent, or a precedent with no
    voice, is charged a fixed penalty rather than being silently ignored: that is the
    voice-COUNT decision, which is otherwise completely unscored.
    """
    if not prev_pitches:
        return 0.0
    cur = [_voice_median_pitch(v, midi_of) for v in voices]
    cur = [p for p in cur if p is not None]
    tot, n = 0.0, 0
    for i, p in enumerate(cur):
        if i < len(prev_pitches) and prev_pitches[i] is not None:
            tot += min(2.0, abs(p - prev_pitches[i]) / 12.0)
            n += 1
        else:
            tot += 1.0          # a voice the previous measure did not have
            n += 1
    if len(prev_pitches) > len(cur):
        tot += 1.0 * (len(prev_pitches) - len(cur))   # a precedent that vanished
        n += len(prev_pitches) - len(cur)
    return tot / max(1, n)


def _stem_term(voices, stem_of, staff_of):
    """Voices that share a staff with another voice yet mix up- and down-stems.

    When two voices share a staff the engraver writes them with opposite stems: on the
    frozen corpus 98.56% of such voices are stem-uniform (956/970), while 55 of the 85
    partition failures at the 96.11 working point break this.  A staff with ONE voice
    is exempt -- its stems follow pitch, not voice.  Stemless events (rests, whole
    notes) never count.  As a terminal term it only re-ranks complete solutions the
    beam already found, so it cannot invent a partition the model gave no weight to.
    """
    staves = [{staff_of[e[0]] for e in v} for v in voices]
    count = collections.Counter(s for ss in staves for s in ss)
    bad = 0
    for v, ss in zip(voices, staves):
        if any(count[s] > 1 for s in ss) and \
                len({stem_of[e[0]] for e in v if stem_of[e[0]] != 0}) > 1:
            bad += 1
    return float(bad)


def _staff_term(voices, staff_of):
    """Voices whose events sit on more than one staff.

    Ground-truth voices never cross staves (0 of 6,729 on the frozen corpus: a voice's
    identity is (staff, voice) in the encoding), yet at the 96.38 working point 25 of the
    160 failing measures contain one.  Terminal-only, like _stem_term.
    """
    return float(sum(1 for v in voices if len({staff_of[e[0]] for e in v}) > 1))


def _final_loss(voices, length, width, logp, decisions, voice_w=0.02, dev_w=0.15,
                tau_w=1.0, k_est=None, iso_w=1.0, iso_tol=0.03, iso_max=True,
                hint_w=0.0, hint_map=None, xmeas_w=0.0, prev_pitches=None,
                midi_of=None, stem_w=0.0, stem_of=None, staff_of=None, staff_w=0.0):
    tau = _max_tau(voices, length, width)
    # Because every voice must tile the measure exactly, the number of voices is
    # determined by the total duration.  Anchoring to an estimate is right;
    # penalising (k-1) blindly would always prefer the fewest voices.
    if k_est is None:
        kv = 0.0
    else:
        kv = abs(len(voices) - k_est)
    dev = -logp / max(1.0, float(decisions))
    iso = iso_penalty(voices, length, width, tol=iso_tol, use_max=iso_max)
    hd = _hint_deviation(voices, hint_map) if hint_w else 0.0
    xm = (_xmeas_term(voices, prev_pitches, midi_of or {})
          if (xmeas_w and prev_pitches) else 0.0)
    sm = _stem_term(voices, stem_of, staff_of) if (stem_w and stem_of) else 0.0
    xs = _staff_term(voices, staff_of) if (staff_w and staff_of) else 0.0
    return (tau_w * tau + voice_w * kv + dev_w * dev + iso_w * iso
            + hint_w * hd + xmeas_w * xm + stem_w * sm + staff_w * xs)


import bisect


def _nearest_dur(durs, need):
    """Index of the table duration closest to `need` (durs is a sorted list)."""
    i = bisect.bisect_left(durs, need)
    if i <= 0:
        return 0
    if i >= len(durs):
        return len(durs) - 1
    return i if (durs[i] - need) < (need - durs[i - 1]) else i - 1


def _attr_options(d_lp, o_lp, w_lp, c, warp_vocab, k, table=None, need=None,
                  exact_only=False, canon_ok=None, canon_pen=0.0, wide=False,
                  dur_vals=None, dur_keys=None, canon_tbl=None, warp_top=3,
                  d_top_n=4, o_top_n=2,
                  warp_bias=None):
    """Candidate (division, dots, time-warp) configurations for one event.

    Three sources, applied identically to every arm:
      * the policy's top-k configurations plus the upstream hint,
      * one structurally derived configuration whose duration matches the ticks
        still missing from the current voice (a voice MUST fill its measure, so
        this is a constraint of the notation, not a policy advantage),
      * a canonicalisation prior: a configuration that is not the way its own
        duration would be written (e.g. a dotted half spelled as a whole note
        under a 4:3 time-warp) pays a fixed penalty, so tick-equivalent but
        unwritten spellings cannot outrank the real one.
    """
    # Cheap top-k.  Enumerating all 7x3x|W| combinations and sorting them, once
    # per candidate per expansion, dominated the runtime.  Only the leading
    # classes of each head can reach the top k, so restrict the cross product.
    nd, no = len(d_lp), len(o_lp)
    # Prior debiasing.  (1,1) is 78% of the corpus, so its log-prob carries a large prior
    # advantage; subtracting A*log(prior) removes it and tests whether the measured
    # (3,2) -> (1,1) error is a prior artefact rather than a discriminative failure.  A = 0
    # (the default) leaves production bit-identical.
    if warp_bias is not None:
        w_lp = (np.asarray(w_lp, dtype=float) + warp_bias).tolist()
    nw = len(w_lp)
    dk = min(nd, d_top_n)
    ok = min(no, o_top_n)
    # warp_top was hard-coded to 3.  Measured on the frozen corpus, the single largest
    # time-warp error is (3,2) -> (1,1): 185 of 302 warp errors, and (3,2) is WELL supported in
    # training (14,240 events), so it is not a sparsity problem.  One candidate mechanism is
    # that (1,1) dominates the warp head's prior and pushes (3,2) out of the top 3, in which
    # case the correct ratio is never even offered.  Exposing the width tests that directly.
    wk = min(nw, warp_top)
    d_top = np.argsort(-np.asarray(d_lp))[:dk]
    o_top = np.argsort(-np.asarray(o_lp))[:ok]
    w_top = np.argsort(-np.asarray(w_lp))[:wk]
    lp_of = {}
    scored = []
    for d in d_top:
        for o in o_top:
            base = d_lp[d] + o_lp[o]
            for wi in w_top:
                lp = float(base + w_lp[wi])
                lp_of[(int(d), int(o), warp_vocab[int(wi)])] = lp
                scored.append((lp, int(d), int(o), warp_vocab[int(wi)]))

    def cpen(key):
        if canon_ok is None or not canon_pen:
            return 0.0
        return 0.0 if canon_ok.get(key, True) else canon_pen

    fit = None
    if need is not None and need > EPS and dur_vals:
        # nearest table duration by binary search instead of scanning all 147
        # entries on every call -- that loop was 55% of the total runtime
        j = _nearest_dur(dur_vals, need)
        dur, key = dur_vals[j], dur_keys[j]
        if abs(dur - need) <= max(1.0, 0.02 * need):
            fit = (canon_tbl or {}).get(key) or canonical(dur, warp_vocab) or key

    if exact_only:
        if fit is None:
            return []
        return [(fit[0], fit[1], fit[2], lp_of.get(fit, -12.0) - 2.0 - cpen(fit))]

    scored.sort(key=lambda t: -(t[0] - cpen((t[1], t[2], t[3]))))
    out, seen = [], set()
    for lp, d, o, w in scored:
        if (d, o, w) in seen:
            continue
        seen.add((d, o, w))
        out.append((d, o, w, float(lp) - cpen((d, o, w))))
        if len(out) >= k:
            break
    hint = (c.hint_d, c.hint_dots, c.hint_warp)
    if hint not in seen and 0 <= hint[0] < nd and 0 <= hint[1] < no:
        try:
            wi = warp_vocab.index(hint[2])
        except ValueError:
            wi = 0
        hk = (hint[0], hint[1], warp_vocab[wi])
        seen.add(hk)
        out.append((hk[0], hk[1], hk[2], lp_of.get(hk, -12.0) - 1.0 - cpen(hk)))

    # WARP FLIP.  The dominant upstream failure is an erased tuplet: a beamed
    # group of three is reported as plain eighths, which breaks the exact tiling
    # by a fixed amount.  Offering exactly ONE sibling -- the hint's time-warp
    # toggled between "none" and the common tuplet ratio -- gives the search a
    # targeted repair without widening the branch factor the way a full option
    # sweep does (which measurably destroys the already-solved voice structure).
    flip = (1, 1) if tuple(c.hint_warp) != (1, 1) else (3, 2)
    if flip in warp_vocab:
        fk = (hint[0], hint[1], flip)
        if fk not in seen and 0 <= fk[0] < nd and 0 <= fk[1] < no:
            seen.add(fk)
            out.append((fk[0], fk[1], fk[2],
                        lp_of.get(fk, -12.0) - 1.5 - cpen(fk)))

    if fit is not None and fit not in seen:
        out.append((fit[0], fit[1], fit[2],
                    lp_of.get(fit, -12.0) - 2.0 - cpen(fit)))

    if wide:
        # Last-resort widening, used only when nothing above can tile the voice.
        hd = hint_duration(c)
        for f in (2.0, 0.5, 1.5, 2.0 / 3.0, 3.0, 1.0 / 3.0):
            cc = canonical(hd * f, warp_vocab)
            if cc is not None and cc not in seen:
                seen.add(cc)
                out.append((cc[0], cc[1], cc[2],
                            lp_of.get(cc, -12.0) - 3.0 - cpen(cc)))
        for lp, d, o, w in scored:
            if (d, o, w) in seen:
                continue
            seen.add((d, o, w))
            out.append((d, o, w, float(lp) - 1.5 - cpen((d, o, w))))
            if len(out) >= 5 * k:
                break
    return out


def solve(cands, length, picker, warp_vocab, *, width=1000.0, beam=24, k_attr=5,
          pretense_tol=15.0, lam_struct=1.0, n_total=None, progress_w=0.5,
          voice_w=0.02, dev_w=0.15, tau_w=1.0, k_est=None, canon_pen=0.7,
          repair=False, iso_w=1.0, iso_tol=0.03, iso_max=True, lam_iso=0.0,  # 0: see report.md -- exposing isochrony to the beam hurt voice match
          hint_w=0.0, k_wide=0, trust_thr=0.5, lam_hint=0.0, attr_diversity=1, warp_top=3, warp_bias=None,
          d_top_n=4, o_top_n=2,
          pair_w=0.0, pair_score=None, pair_eos=0.0, close_cost=0.0,
          xmeas_w=0.0, prev_pitches=None, midi_of=None,
          stem_w=0.0,       # terminal stem-consistency weight, see _stem_term
          staff_w=0.0,      # terminal cross-staff-voice weight, see _staff_term
          n_beats,          # REQUIRED: no default, so a caller that forgets it fails loudly
          order_guard=False,  # exact invariant, but see the note at the call site
          staff_guard=False,  # a voice only grows on its own staff, see the call site
          n_alt=0,            # >0 also return the best n_alt distinct complete solutions
          max_expansions=0,
          verbose=False, diag=None):
    def bump(k):
        if diag is not None:
            diag[k] = diag.get(k, 0) + 1
    n = len(cands)
    if n == 0:
        return [], {"expanded": 0, "complete": True, "loss": None, "depth": 0}
    n_total = n_total or n
    stem_of = {c.cid: int(getattr(c, "stem", 0)) for c in cands} if stem_w else None
    staff_of = ({c.cid: c.staff for c in cands}
                if (stem_w or staff_w or staff_guard) else None)

    table = {}
    for d in range(7):
        for o in range(3):
            for w in warp_vocab:
                table[(d, o, w)] = duration_of(d, o, w)
    min_one = min(table[(6, 0, w)] for w in warp_vocab)
    if k_est is None:
        k_est = max(1, int(round(sum(hint_duration(c) for c in cands) / float(length))))
    # tick-equivalent but unwritten spellings are penalised, once per duration
    hint_map = {}
    for _c in cands:
        try:
            _wi = warp_vocab.index(_c.hint_warp)
        except ValueError:
            _wi = 0
        hint_map[_c.cid] = (min(max(_c.hint_d, 0), 6), min(max(_c.hint_dots, 0), 2),
                            warp_vocab[_wi])
    canon_ok = {}
    canon_tbl = {}
    for _k, _dur in table.items():
        _c = canonical(_dur, warp_vocab)
        canon_tbl[_k] = _c
        canon_ok[_k] = (_c == _k)
    _pairs = sorted(((d, k) for k, d in table.items()), key=lambda t: t[0])
    dur_vals = [d for d, _k in _pairs]
    dur_keys = [k for _d, k in _pairs]

    start = Node(used=0, cur=(), acc=0, voices=(), logp=0.0)
    start.hdev = 0
    start.priority = 0.0
    frontier = [start]
    seen = {_key(start): 0.0}
    best_partial = start
    expanded, depth = 0, 0

    # ANYTIME baseline.  Install the cheap x-order chain as a complete solution
    # BEFORE searching, so any budget (including zero) returns something usable
    # and extra budget can only improve on it.  Without this the per-measure
    # budget was a cliff: a 200-expansion cap collapsed the perfect-measure rate
    # from 78% to 3% because the beam had not yet reached a complete solution.
    gv, gst = greedy(cands, length, warp_vocab)
    if gst.get("valid") and gv:
        gv_t = tuple(tuple((e["cid"], e["x"], e["onset"], e["dur"],
                            e["d"], e["dots"], e["warp"]) for e in v) for v in gv)
        best_complete = gv_t
        best_loss = _final_loss(gv_t, length, width, 0.0, 1, voice_w=voice_w,
                                dev_w=dev_w, tau_w=tau_w, k_est=k_est,
                                iso_w=iso_w, iso_tol=iso_tol, iso_max=iso_max,
                                hint_w=hint_w, hint_map=hint_map, xmeas_w=xmeas_w,
                                      prev_pitches=prev_pitches, midi_of=midi_of,
                                stem_w=stem_w, stem_of=stem_of, staff_of=staff_of,
                                staff_w=staff_w)
    else:
        best_complete, best_loss = None, float("inf")
    _alts, _alt_seen = [], set()   # populated only when n_alt > 0

    while frontier and depth < n + 6:
        if max_expansions and expanded >= max_expansions:
            break          # hard per-measure budget: bounds the worst case on A53
        depth += 1
        children = []
        for node in frontier:
            expanded += 1
            rem_cands = [c for c in cands if not (node.used >> c.cid) & 1]
            state = {"acc": node.acc,
                     "last_x": node.cur[-1][1] if node.cur else 0.0,
                     "remaining": rem_cands, "n_closed": len(node.voices),
                     "cur_voice": node.cur, "all_cands": cands,
                     "n_beats": n_beats}

            # ---------------- EOS: close the current voice ----------------
            if node.cur and abs(node.acc - length) <= EPS:
                voices = node.voices + (node.cur,)
                if not rem_cands:
                    loss = _final_loss(voices, length, width, node.logp, node.decisions,
                                      voice_w=voice_w, dev_w=dev_w, tau_w=tau_w,
                                      k_est=k_est, iso_w=iso_w, iso_tol=iso_tol,
                                      iso_max=iso_max, hint_w=hint_w,
                                      hint_map=hint_map, xmeas_w=xmeas_w,
                                      prev_pitches=prev_pitches, midi_of=midi_of,
                                      stem_w=stem_w, stem_of=stem_of, staff_of=staff_of,
                                staff_w=staff_w)
                    if loss < best_loss:
                        best_loss, best_complete = loss, voices
                    # MULTI-HYPOTHESIS HOOK.  Keep the best `n_alt` distinct
                    # complete solutions seen anywhere in the search, not just the
                    # winner.  Stage B (the exact DP) can only re-solve values on
                    # the ONE partition it is given, so if the correct partition is
                    # in here but not first, that information is currently thrown
                    # away.  Off by default (n_alt=0) so nothing changes unless a
                    # caller asks for alternatives.
                    if n_alt:
                        key = tuple(tuple((e[0], e[2], e[4], e[5], e[6]) for e in v)
                                    for v in voices)
                        if key not in _alt_seen:
                            _alt_seen.add(key)
                            _alts.append((loss, voices))
                            _alts.sort(key=lambda t: t[0])
                            del _alts[n_alt:]
                else:
                    # close_cost charges for fragmentation.  Closure is otherwise
                    # FREE here (no score is consumed), so the number of voices was
                    # never a scored decision -- which is why terminal oice_w sweeps
                    # did nothing.  A positive cost makes an extra voice have to earn
                    # its place through better fits inside it.
                    ch = Node(used=node.used, cur=(), acc=0, voices=voices,
                              logp=node.logp - close_cost,
                              decisions=node.decisions + 1)
                    ch.tau = _max_tau(voices, length, width)
                    ch.hdev = node.hdev          # carry the running hint-agreement
                    ch.priority = (node.logp - lam_struct * ch.tau
                                   - lam_hint * (ch.hdev / max(1, ch.decisions))
                                   + progress_w * (_popcount(node.used) / n_total))
                    children.append(ch)

            # ---------------- consume one candidate ----------------
            if not rem_cands:
                continue
            trust = None
            if hasattr(picker, "logprobs_both"):
                _res = picker.logprobs_both(rem_cands, state, length, n_total)
                if len(_res) == 5:
                    succ_lp, d_all, o_all, w_all, trust = _res
                else:
                    succ_lp, d_all, o_all, w_all = _res
            else:
                succ_lp = picker.succ_logprobs(rem_cands, state, length, n_total)
                d_all, o_all, w_all = picker.attr_logprobs(rem_cands, state, length, n_total)
            # PAIRWISE VOICE CONTINUATION.  Whether two notes belong to the same voice
            # is a RELATIONAL property; three attempts to add it as per-candidate
            # features all regressed (pitch -1.65, stem -1.38, stem-successor-only
            # -0.89), and a dedicated pair classifier confirms the relation itself is
            # highly informative (AUC 0.961, and whole-partition recovery 83.54% vs
            # this model's 75.21%).  So score the relation ONCE per (last note, next
            # note) and add it to the successor logits -- a re-ranking that cannot
            # break the exact-tiling constraint, because the tiling mask still governs.
            # PAIRWISE VOICE CONTINUATION.  Whether two notes belong to the same voice
            # is a RELATIONAL property; three attempts to add it as per-candidate
            # features all regressed (pitch -1.65, stem -1.38, stem-successor-only
            # -0.89), while a dedicated pair classifier is highly informative (AUC
            # 0.961; whole-partition recovery 83.54% vs this model's 75.21%).  So score
            # the relation once per (current voice, candidate) and add it to the
            # successor logits -- a re-ranking that cannot break the exact-tiling
            # constraint because the tiling mask still governs.
            #
            # NOTE: succ_lp carries an extra end-of-voice entry that NOTHING reads.
            # Voice closure is handled above and is FREE: it is legal only when the
            # voice exactly fills the measure, and it adds nothing to logp.  An
            # earlier attempt to modulate that entry had zero effect for exactly this
            # reason.  Closing a voice is simply not a scored decision here, which is
            # also why sweeping oice_w at the terminal loss never did anything.
            if pair_w and pair_score is not None and node.cur:
                _n = len(rem_cands)
                _ps = np.asarray(pair_score(node.cur, rem_cands, node.acc),
                                 dtype=succ_lp.dtype)
                succ_lp = np.concatenate([succ_lp[:_n] + pair_w * _ps, succ_lp[_n:]])
            node_max_tau = _max_tau(node.voices, length, width)
            need = length - node.acc

            # Options offered to each remaining candidate, computed once.
            opts = []
            wildcard = False
            for i, c in enumerate(rem_cands):
                # TRUST-ADAPTIVE BRANCHING.  Where the trust head says the
                # upstream hint is unreliable, look at more alternatives; where
                # it is confident, look at fewer.  This is deliberately NOT a
                # blanket widening: uniform widening was measured to dilute the
                # beam and destroy the already-solved voice structure.
                _k = k_attr
                if trust is not None and k_wide > k_attr and trust[i] < trust_thr:
                    _k = k_wide
                oi = _attr_options(d_all[i], o_all[i], w_all[i], c, warp_vocab,
                                   _k, table=table, need=need,
                                   exact_only=(len(rem_cands) == 1),
                                   canon_ok=canon_ok, canon_pen=canon_pen,
                                   wide=bool(getattr(picker, "always_wide", False)),
                                   dur_vals=dur_vals, dur_keys=dur_keys, canon_tbl=canon_tbl,
                                   warp_top=warp_top, warp_bias=warp_bias,
                                   d_top_n=d_top_n, o_top_n=o_top_n)
                if not oi:
                    wildcard = True
                opts.append(oi)
                for (_d, _o, _w, _lp) in oi:
                    if _cells(table[(_d, _o, _w)]) is None:
                        wildcard = True

            # EXACT-FILL REACHABILITY.  A voice must tile the measure exactly, so
            # if no subset of the remaining candidates -- using only the options
            # the search will actually consider -- can sum to `need`, this state
            # is a dead end.  Before giving up, widen the options once (repair
            # mode); only prune if even the widened set cannot tile.
            # Skipped (soundly) when some option has a duration that is not an
            # exact multiple of the 1/SCALE tick cell.
            if repair and not wildcard and need > EPS and not _bit(_reach_mask(opts, table, need), need):
                bump("repair")
                opts, wildcard = [], False
                for i, c in enumerate(rem_cands):
                    oi = _attr_options(d_all[i], o_all[i], w_all[i], c, warp_vocab,
                                       k_attr, table=table, need=need,
                                           exact_only=(len(rem_cands) == 1),
                                       canon_ok=canon_ok, canon_pen=canon_pen, wide=True,
                                       dur_vals=dur_vals, dur_keys=dur_keys, canon_tbl=canon_tbl,
                                   warp_top=warp_top, warp_bias=warp_bias,
                                   d_top_n=d_top_n, o_top_n=o_top_n)
                    if not oi:
                        wildcard = True
                    opts.append(oi)
                    for (_d, _o, _w, _lp) in oi:
                        if _cells(table[(_d, _o, _w)]) is None:
                            wildcard = True
                if not wildcard and not _bit(_reach_mask(opts, table, need), need):
                    bump("unreachable")
                    continue

            for i, c in enumerate(rem_cands):
                # EXACT ORDER INVARIANT.  Measured on all 37,884 candidate/truth
                # pairs in the frozen corpus: x_i < x_j implies onset_i <= onset_j
                # with ZERO violations -- engraving lays a voice out left to right,
                # so x order IS onset order.
                #
                # DEFAULT OFF.  It prunes a real 6.5% of children and is a genuine
                # correctness invariant, but on the full 5-fold out-of-fold it
                # costs 0.03 any-field (3.99 -> 4.02) and 0.03 voice match with NO
                # perfect-measure gain.  A single fold had shown "zero change",
                # which is why it was briefly the default -- the same single-fold
                # trap that cost this project eight rounds on the capacity lever.
                if order_guard and node.cur and c.x < node.cur[-1][1] - 1e-9:
                    bump("x_backwards"); continue
                # EXACT STAFF INVARIANT.  A voice never leaves its staff (0 of 6,729
                # ground-truth voices do; the encoding keys a voice by staff), so a
                # voice under construction only accepts candidates on its first event's
                # staff.  The terminal staff_w term can only re-rank complete solutions
                # the beam already holds; this keeps cross-staff ones from being built.
                if staff_guard and node.cur and c.staff != staff_of[node.cur[0][0]]:
                    bump("cross_staff"); continue
                for (d, o, w, attr_lp) in opts[i]:
                    dur = table[(d, o, w)]
                    new_acc = node.acc + dur
                    if new_acc > length + EPS:
                        bump("overshoot"); continue   # overruns the measure: fatal
                    if new_acc < length - EPS:
                        if len(rem_cands) == 1:
                            bump("last_cand_gap"); continue
                        if (length - new_acc) < min_one - EPS:
                            bump("gap_too_small"); continue
                    logp = node.logp + float(succ_lp[i]) + attr_lp
                    # CROSS-MEASURE REGISTER CONTINUITY, charged in the BEAM.
                    # Putting this in the terminal loss changed zero decisions: the beam's
                    # partition is decided by the successor scores and the tiling mask, and
                    # the terminal loss only ranks complete solutions the beam already
                    # produced.  So the signal has to be paid at the moment the decision is
                    # made -- when the FIRST event of a new voice is placed.
                    if (xmeas_w and prev_pitches and midi_of is not None
                            and not node.cur):
                        _vi = len(node.voices)
                        _p = midi_of.get(c.cid, -1)
                        if (_p > 0 and _vi < len(prev_pitches)
                                and prev_pitches[_vi] is not None):
                            logp -= xmeas_w * min(
                                2.0, abs(_p - prev_pitches[_vi]) / 12.0)
                    if (-logp) / max(1, node.decisions + 1) > pretense_tol:
                        bump("pretense"); continue    # pretentiousness budget spent
                    ch = Node(used=node.used | (1 << c.cid),
                              cur=node.cur + ((c.cid, c.x, node.acc, dur, d, o, w),),
                              acc=new_acc, voices=node.voices, logp=logp,
                              decisions=node.decisions + 1)
                    ctau = _tau_voice(ch.cur, length, width)
                    if ctau >= 2.0:
                        bump("tau_fatal"); continue   # time runs backwards: fatal
                    ch.tau = max(ctau, node_max_tau)
                    # PRIORITY MUST SPEAK THE SAME LANGUAGE AS THE FINAL LOSS.
                    # Measured: on 100% of the missed dense measures the TRUE
                    # solution has a strictly lower structural loss, i.e. the
                    # terminal evaluator identifies it perfectly -- but the beam
                    # had already pruned it.  The priority was dominated by `logp`
                    # while the final loss uses tau / voiceness / deviation /
                    # isochrony / hint agreement.  Carrying the hint-agreement
                    # term into the priority closes that gap.
                    hdev = node.hdev + (0 if hint_map.get(c.cid) == (d, o, tuple(w))
                                        else 1)
                    ch.hdev = hdev
                    ch.priority = (logp - lam_struct * ch.tau
                                   - lam_iso * _partial_iso(ch, length, width, iso_tol)
                                   - lam_hint * (hdev / max(1, ch.decisions))
                                   + progress_w * (_popcount(ch.used) / n_total))
                    children.append(ch)

        if not children:
            break
        children.sort(key=lambda x: -x.priority)
        bump("children", )
        if diag is not None:
            diag["children"] = diag.get("children", 0) + len(children) - 1
        # PER-SUCCESSOR DIVERSITY.  Each step generates |rem| * k_attr children
        # (about 19 * 5 = 95 in a dense measure) and the beam keeps only 16.  A
        # global top-16 by priority is almost entirely "every event takes its
        # top-1 attribute", so the lower-ranked attribute options are born and
        # killed in the same step.  That is why k_attr had exactly zero effect,
        # why beam 16 -> 64 bought almost nothing (64 is still well under 95), and
        # why adding hint agreement to the priority changed nothing: none of them
        # touched the fact that the alternatives never survive the cut.
        #
        # Model top-1 recall is 95.99% on dense measures, so with ~19 events a
        # majority of measures contain at least one event whose truth is NOT the
        # top-1.  Those measures are unreachable unless a second-ranked option for
        # that event stays alive.  Keep the best `attr_diversity` children per
        # successor BEFORE the global cut.
        if attr_diversity > 1:
            per_succ = {}
            for ch in children:
                key = ch.cur[-1][0] if ch.cur else -1
                b = per_succ.get(key)
                if b is None:
                    per_succ[key] = [ch]
                elif len(b) < attr_diversity:
                    b.append(ch)
            keep = [ch for b in per_succ.values() for ch in b]
            keep.sort(key=lambda x: -x.priority)
            children = keep
        nxt = []
        for ch in children:
            k = _key(ch)
            if k in seen:
                bump("dup"); continue
            seen[k] = ch.priority
            nxt.append(ch)
            if len(nxt) >= beam:
                break
        frontier = nxt
        for ch in frontier:
            if _popcount(ch.used) > _popcount(best_partial.used):
                best_partial = ch

    if best_complete is None:
        # ANYTIME fallback: greedily finish the best partial state so that a
        # truncated budget still yields a usable, tileable answer.
        comp = _greedy_complete(best_partial, cands, length, picker, warp_vocab,
                                table, canon_ok, canon_pen, n_total, k_est,
                                n_beats)
        if comp is not None:
            best_complete = tuple(comp)
            best_loss = _final_loss(best_complete, length, width, best_partial.logp,
                                    max(1, best_partial.decisions), voice_w=voice_w,
                                    dev_w=dev_w, tau_w=tau_w, k_est=k_est,
                                    iso_w=iso_w, iso_tol=iso_tol, iso_max=iso_max)
    if best_complete is not None:
        voices, complete, loss = best_complete, True, best_loss
    else:
        voices = best_partial.voices + ((best_partial.cur,) if best_partial.cur else ())
        complete, loss = False, None
    stats = {"expanded": expanded, "complete": complete, "valid": complete, "loss": loss,
             "alts": [(l, v) for l, v in _alts],
             "depth": depth, "coverage": _popcount(best_partial.used) / n_total}
    return [[{"cid": e[0], "x": e[1], "onset": e[2], "dur": e[3],
              "d": e[4], "dots": e[5], "warp": e[6]} for e in v] for v in voices], stats


# --------------------------------------------------------------------------- #
# greedy floor: x-order, hints only, no search at all
# --------------------------------------------------------------------------- #
def greedy(cands, length, warp_vocab):
    table = {}
    for d in range(7):
        for o in range(3):
            for w in warp_vocab:
                table[(d, o, w)] = duration_of(d, o, w)
    order = sorted(cands, key=lambda c: c.x)
    voices, cur, acc = [], [], 0
    for c in order:
        key = (c.hint_d, c.hint_dots, c.hint_warp)
        if key not in table:
            key = (c.hint_d, c.hint_dots, (1, 1))
        dur = table.get(key, 480.0)
        if acc + dur > length + EPS and cur:
            voices.append(cur)
            cur, acc = [], 0
        cur.append({"cid": c.cid, "x": c.x, "onset": acc, "dur": dur,
                    "d": key[0], "dots": key[1], "warp": key[2]})
        acc += dur
        if abs(acc - length) <= EPS:
            voices.append(cur)
            cur, acc = [], 0
    truncated = bool(cur)
    if cur:
        voices.append(cur)
    # A greedy chain is only a VALID score if every voice exactly fills the
    # measure.  The search arms are held to that constraint; so is this one,
    # otherwise its advantage on simple measures would be an artefact.
    valid = (not truncated) and all(
        abs(sum(e["dur"] for e in v) - length) <= EPS for v in voices) and bool(voices)
    return voices, {"expanded": 0, "complete": valid, "valid": valid, "loss": None,
                    "depth": len(order), "coverage": 1.0}


















def solve_best(cands, length, picker, warp_vocab, passes, n_beats, **kw):
    """Multi-pass decoding: escalating search budgets / widening, best wins.

    Pass 1 keeps branching narrow so the narrow beam is not diluted.  Later
    passes widen the attribute options (tuplet ratios included) for the measures
    that the cheap pass could not close -- which is where the upstream hint is
    most likely to be wrong.
    """
    best, best_key = None, None
    for cfg in passes:
        merged = dict(kw)
        merged.update(cfg)
        v, st = solve(cands, length, picker, warp_vocab, **merged)
        loss = st["loss"] if st["loss"] is not None else 9e9
        key = (1 if st["complete"] else 0, round(st.get("coverage", 0.0), 3), -loss)
        if best_key is None or key > best_key:
            best_key, best = key, (v, st)
        if st["complete"] and st.get("loss") is not None:
            break
    return best







def _greedy_complete(node, cands, length, picker, warp_vocab, table, canon_ok,
                     canon_pen, n_total, k_est, n_beats):
    """Finish a partial state cheaply so the search is ANYTIME.

    Beam search advances level by level, so a per-measure budget that stops it
    mid-flight used to return almost nothing (perfect-measure rate collapsed
    from 78% to 3% at a 200-expansion cap).  Completing the best partial state
    greedily -- always taking a duration that still leaves the voice tileable --
    gives a valid answer at any budget, which is what a bounded A53 budget needs.
    """
    voices = list(node.voices)
    cur = list(node.cur)
    acc = node.acc
    used = node.used
    for _ in range(len(cands) + 8):
        rem = [c for c in cands if not (used >> c.cid) & 1]
        if not rem:
            if cur and abs(acc - length) <= EPS:
                voices.append(tuple(cur))
            return voices if voices and not cur or abs(acc - length) <= EPS else None
        if cur and abs(acc - length) <= EPS:
            voices.append(tuple(cur))
            cur, acc = [], 0
            continue
        need = length - acc
        state = {"acc": acc, "last_x": cur[-1][1] if cur else 0.0,
                 "remaining": rem, "n_closed": len(voices), "cur_voice": tuple(cur),
                 "all_cands": cands, "n_beats": n_beats}
        succ = picker.succ_logprobs(rem, state, length, n_total)
        d_all, o_all, w_all = picker.attr_logprobs(rem, state, length, n_total)
        order = sorted(range(len(rem)), key=lambda i: -succ[i])
        placed = False
        for i in order:
            c = rem[i]
            opts = _attr_options(d_all[i], o_all[i], w_all[i], c, warp_vocab, 6,
                                 table=table, need=need,
                                 exact_only=(len(rem) == 1),
                                 canon_ok=canon_ok, canon_pen=canon_pen)
            best = None
            for (d, o, w, lp) in opts:
                dur = table[(d, o, w)]
                if dur > need + EPS:
                    continue
                if best is None or dur > best[0]:
                    best = (dur, d, o, w)
            if best is None:
                continue
            dur, d, o, w = best
            cur.append((c.cid, c.x, acc, dur, d, o, w))
            used |= 1 << c.cid
            acc += dur
            placed = True
            break
        if not placed:
            return None
    return None











