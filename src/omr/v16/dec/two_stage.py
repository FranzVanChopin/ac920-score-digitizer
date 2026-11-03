"""Two-stage decoder: beam search fixes the ORDER, exact DP solves the VALUES.

Stage A runs the existing beam search (which, given true attributes, is 100%
correct on dense measures).  Stage B freezes that partition and order, then
re-solves every event's (division, dots, warp) with the exact per-voice DP in
src/attr_dp.py -- no beam, no pruning, exact tiling by construction.

Because the policy's state features depend on the accumulated time, stage B is
run for a couple of iterations, re-estimating the trajectory each time.

The result is accepted only if its terminal loss beats stage A's, so this can
never be worse than the current pipeline on the same measure.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODELS = os.path.join(os.path.dirname(HERE), "results", "models")
OUT = os.path.join(os.path.dirname(HERE), "results")

import corpus as C  # noqa: E402
import metrics as M  # noqa: E402
import visual as V  # noqa: E402
from attr_dp import (build_options,  # noqa: E402
                     cells as cells_local, iso_penalty_vectors, voice_dp)
from joint_dp import joint_solve  # noqa: E402
from pickers import (CAND_DIM, STATE_DIM, STEM_FEAT_DIM, USE_STEM_FEATURES,
                     LearnedPicker, MLP, cand_features,
                     hint_indices, state_features)  # noqa: E402
from parallel_decode import decode_all
from pitch_repair import pitch_swap_repair  # noqa: E402
from column_joint import (column_joint_refine, column_violation,  # noqa: E402
                          spacing_residuals)
from repeat_consensus import repeat_consensus  # noqa: E402
from prod import PROD  # noqa: E402
from run_experiment import make_candidates, make_train_candidates  # noqa: E402
from search import _final_loss, solve  # noqa: E402

RD = collections.Counter()   # column-repair diagnostics
PB = collections.Counter()   # pitch-swap-repair diagnostics
BUCKETS = [(1, 4), (5, 8), (9, 12), (13, 99)]
LABELS = ["1-4", "5-8", "9-12", "13+"]


def canon_table(warp_vocab):
    from visual import canonical
    tbl, ok = {}, {}
    for d in range(7):
        for o in range(3):
            for w in warp_vocab:
                dur = V.duration_of(d, o, w)
                tbl[(d, o, tuple(w))] = dur
                ok[(d, o, tuple(w))] = (canonical(dur, warp_vocab) == (d, o, tuple(w)))
    return tbl, ok


def as_tuples(voices):
    """solve() returns lists of dicts; the loss wants (cid,x,onset,dur,d,o,w)."""
    return tuple(tuple((e["cid"], e["x"], e["onset"], e["dur"], e["d"], e["dots"],
                        tuple(e["warp"])) for e in v) for v in voices)


def dp_refine(cm, got, pk, warp_vocab, canon_ok, *, iters=2, k_attr=12,
              iso_w=0.0, exact_x=0.0, cx_bonus=0.0, xsp_w=0.0):
    """Re-solve attributes per voice, keeping stage A's partition and order.

    `exact_x` > 0 turns the EXACT engraving invariant into a hard constraint.
    Measured on all 37,884 candidate/truth pairs in the frozen corpus:
      * x_i < x_j  =>  onset_i <= onset_j          (0 violations) -- x order IS
        onset order, so the successor search must never go backwards in x;
      * |x_i - x_j| == 0  =>  |onset_i - onset_j| == 0  (11,178 pairs, 0
        exceptions) -- the same column means the same instant.
    The decoder previously used x only as a soft FEATURE, and the soft isochrony
    penalty was measured to do nothing.  Equality is a much stronger statement
    than proximity, so it is imposed on the DP state directly: an event whose x
    coincides with another voice's event may only take that exact onset.
    A large finite cost is used rather than infinity so that a wrong stage-A
    other-voice onset can still be overridden; B is only accepted when it beats
    A on the terminal loss, so this can never make the pipeline worse.
    """
    cid_of = {c.cid: c for c in cm.cands}
    cur = [[(e["cid"], e["x"], e["onset"], e["dur"], e["d"], e["dots"],
             tuple(e["warp"])) for e in v] for v in got]
    # column table for the x-spacing prior: every distinct x is one rhythmic column
    _xs = sorted({round(c.x, 9) for c in cm.cands})
    _col = {x: i for i, x in enumerate(_xs)}
    _ncol = max(1, len(_xs))
    if not cur:
        return got
    # reference columns from every OTHER voice: the isochrony signal the policy
    # cannot see but the DP can use
    all_ref = [(e["x"], e["onset"]) for v in got for e in v]
    for _ in range(max(1, iters)):
        new_voices = []
        for vi, v in enumerate(cur):
            seq = [cid_of[e[0]] for e in v]
            n = len(seq)
            # trajectory for the model features: from the current solution
            accs = [0.0] * n
            a = 0.0
            for i, e in enumerate(v):
                accs[i] = a
                a += e[3]
            opts = []
            for i, c in enumerate(seq):
                st = {"acc": accs[i], "last_x": (v[i - 1][1] if i else 0.0),
                      "remaining": seq[i:], "n_closed": 0,
                      "cur_voice": [z[0] for z in v[:i]], "all_cands": cm.cands, "n_beats": cm.gt.beats}
                F = cand_features(seq[i:], st, None, cm.length)
                S = state_features(st, cm.length, len(cm.cands), None)
                di, oi, wi = hint_indices(seq[i:], warp_vocab)
                D, O, W = pk.mlp.attr_logprobs_all(F, S, (di, oi, wi))
                # row 0 is the event itself (it is first in `remaining`)
                # X-SPACING DURATION PRIOR.  Engravers space notes roughly in proportion
                # to time, so the distance to the next event IN THIS VOICE predicts this
                # event's duration.  Measured on 37,498 ground-truth events: 84.62% exact,
                # 97.25% within one binary step, rank correlation 0.894.  The model only
                # uses x as a soft feature, so this geometric constraint is information it
                # cannot otherwise reach.
                _pred = 0.0
                if xsp_w and i + 1 < n:
                    _c0 = _col.get(round(v[i][1], 9))
                    _c1 = _col.get(round(v[i + 1][1], 9))
                    if _c0 is not None and _c1 is not None and _c1 > _c0:
                        _pred = cm.length * (_c1 - _c0) / max(1, _ncol)
                opts.append(build_options(D[0], O[0], W[0], warp_vocab, c,
                                          k_attr=k_attr, canon_ok=canon_ok,
                                          canon_pen=0.0, hint_w=PROD["hint_w"],
                                          cx_bonus=cx_bonus,
                                          dur_bias=xsp_w, dur_pred=_pred))
            pen = None
            hard = None
            if exact_x:
                # exact-column constraint: this voice's event at x may only take
                # the onset that other voices already placed at the same x
                others = [e for j, w2 in enumerate(cur) if j != vi for e in w2]
                L = cells_local(cm.length)
                hard = []
                for c in seq:
                    hit = [e for e in others if abs(e[1] - c.x) < 1e-9]
                    if not hit:
                        hard.append(None)
                        continue
                    arr = np.zeros(L + 1)
                    ok = set(cells_local(e[2]) for e in hit)
                    mask = np.ones(L + 1, dtype=bool)
                    for k in ok:
                        if 0 <= k <= L:
                            mask[k] = False
                    arr[mask] = exact_x
                    hard.append(arr)
            if iso_w:
                # REFERENCES MUST EXCLUDE THE VOICE BEING SOLVED.  Passing every
                # event (including this voice's own) put a reference exactly at the
                # current onset, so the penalty became a zero-cost anchor to stage
                # A's answer and the DP simply refused to move -- which is why
                # sweeping iso_w produced bit-identical output and the isochrony
                # signal looked worthless.  Only OTHER voices carry information.
                other_ref = [(e[1], e[2]) for j, w2 in enumerate(cur)
                             if j != vi for e in w2]
                if other_ref:
                    pen = iso_penalty_vectors([c.x for c in seq], other_ref,
                                              cm.length)
                    pen = [None if q is None else iso_w * q for q in pen]
            if hard is not None and pen is not None:
                merged = []
                for q, h in zip(pen, hard):
                    if q is None:
                        merged.append(h)
                    elif h is None:
                        merged.append(q)
                    else:
                        merged.append(q + h)
                pen = merged
            elif hard is not None:
                pen = hard
            assign, cost = voice_dp(n, cm.length, opts, state_pen=pen)
            if assign is None:
                return got                      # DP failed: keep stage A's voice
            new_voices.append([
                (c.cid, c.x, onset, dur, d, o, w)
                for c, (onset, dur, d, o, w) in zip(seq, assign)])
        cur = new_voices
    return tuple(tuple(v) for v in cur)


def column_repair(cm, voices, pk, warp_vocab, canon_ok, *, k_attr=40,
                  max_rounds=3):
    """Enforce the exact column invariant by local re-solve.

    Measured headroom (src/column_headroom.py, fold 0): on the dense bucket
    **72.4% of the WRONG measures break the invariant** that events sharing an x
    must share an onset (52.2% overall).  A per-voice HARD constraint cannot fix
    them -- round 28 showed it makes things worse, because it forces one voice to
    accept another voice's possibly-wrong onset.

    So instead of constraining, REPAIR: find a column whose events disagree, try
    each candidate onset for it, re-solve only the affected voices' DPs with that
    column pinned, and keep the variant with the lowest terminal loss.  Pinning
    reuses the existing state-penalty mechanism, and accept-only-if-better means
    this can never make a measure worse.
    """
    cid_of = {c.cid: c for c in cm.cands}
    work = [[(e[0], e[1], e[2], e[3], e[4], e[5], tuple(e[6])) for e in v]
            for v in voices]
    if not work:
        return tuple()

    def total_loss(vs):
        try:
            return _final_loss(tuple(tuple(e) for e in vs), cm.length, 1000.0,
                               0.0, 1, voice_w=PROD["voice_w"], dev_w=0.0,
                               tau_w=PROD["tau_w"], iso_w=PROD["iso_w"],
                               hint_w=PROD["hint_w"])
        except Exception:
            return float("inf")

    for _ in range(max_rounds):
        cols = {}
        for vi, v in enumerate(work):
            for k, e in enumerate(v):
                cols.setdefault(round(e[1], 9), []).append((vi, k, e[2]))
        bad = {x: lst for x, lst in cols.items()
               if len({round(t, 9) for _vi, _k, t in lst}) > 1}
        if not bad:
            break
        improved = False
        for x, lst in bad.items():
            best, best_loss = None, total_loss(work)
            for cand_t in sorted({t for _vi, _k, t in lst}):
                trial = [list(v) for v in work]
                ok = True
                for vi in sorted({vi for vi, _k, _t in lst}):
                    v = trial[vi]
                    seq = [cid_of[e[0]] for e in v]
                    n = len(seq)
                    accs, acc = [], 0.0
                    for e in v:
                        accs.append(acc)
                        acc += e[3]
                    opts = []
                    for i, c in enumerate(seq):
                        st = {"acc": accs[i],
                              "last_x": (v[i - 1][1] if i else 0.0),
                              "remaining": seq[i:], "n_closed": 0,
                              "cur_voice": [z[0] for z in v[:i]],
                              "all_cands": cm.cands, "n_beats": cm.gt.beats}
                        F = cand_features(seq[i:], st, None, cm.length)
                        S = state_features(st, cm.length, len(cm.cands), None)
                        di, oi, wi = hint_indices(seq[i:], warp_vocab)
                        D, O, W = pk.mlp.attr_logprobs_all(F, S, (di, oi, wi))
                        opts.append(build_options(D[0], O[0], W[0], warp_vocab, c,
                                                  k_attr=k_attr, canon_ok=canon_ok,
                                                  hint_w=PROD["hint_w"], cx_bonus=cx_bonus))
                    L = cells_local(cm.length)
                    pin = np.full(L + 1, 1e6)
                    tgt = cells_local(cand_t)
                    if 0 <= tgt <= L:
                        pin[tgt] = 0.0
                    pen = [pin if abs(e[1] - x) < 1e-9 else None for e in v]
                    asg, _cost = voice_dp(n, cm.length, opts, state_pen=pen)
                    if asg is None:
                        ok = False
                        break
                    trial[vi] = [(c.cid, c.x, onset, dur, d, o, w)
                                 for c, (onset, dur, d, o, w) in zip(seq, asg)]
                if not ok:
                    continue
                tl = total_loss(trial)
                if tl < best_loss - 1e-9:
                    best, best_loss = trial, tl
            if best is not None:
                work = best
                improved = True
        if not improved:
            break
    return tuple(tuple(v) for v in work)


def _loss_of(voices, cm):
    """Terminal loss of a solution, in the same units the pipeline compares.

    Accepts the tuple representation produced by as_tuples/column_repair.
    """
    vs = tuple(tuple(e) for e in voices)
    try:
        return _final_loss(vs, cm.length, 1000.0, 0.0, 1,
                           voice_w=PROD["voice_w"], dev_w=0.0,
                           tau_w=PROD["tau_w"], iso_w=PROD["iso_w"],
                           hint_w=PROD["hint_w"])
    except Exception:
        return float("inf")


def joint_refine(cm, got, pk, warp_vocab, canon_ok, *, max_states=60000):
    """Solve all voices at once subject to the exact column invariant.

    See src/joint_dp.py for why this is not the same as the per-voice DP: two
    voices can each tile [0,length) perfectly and still disagree on the columns,
    and 72.4% of the dense-bucket failures do exactly that.

    Cost maps use the FULL option set, not the top-k: in the joint formulation the
    duration is DETERMINED by the column structure, so a duration outside the
    model's top-k must still be priced rather than made unavailable.
    """
    cid_of = {c.cid: c for c in cm.cands}
    voices = [[cid_of[e[0]] for e in v] for v in got]
    if not voices:
        return got, None
    # stage-A trajectory supplies the state features, as in dp_refine
    accs_v = []
    for v in got:
        a, acc = 0.0, []
        for e in v:
            acc.append(a)
            a += e[3]
        accs_v.append(acc)
    cost_maps = []
    for vi, v in enumerate(got):
        seq = voices[vi]
        per = []
        for i, c in enumerate(seq):
            st = {"acc": accs_v[vi][i],
                  "last_x": (v[i - 1][1] if i else 0.0),
                  "remaining": seq[i:], "n_closed": 0,
                  "cur_voice": [z.cid for z in seq[:i]],
                  "all_cands": cm.cands, "n_beats": cm.gt.beats}
            F = cand_features(seq[i:], st, None, cm.length)
            S = state_features(st, cm.length, len(cm.cands), None)
            di, oi, wi = hint_indices(seq[i:], warp_vocab)
            D, O, W = pk.mlp.attr_logprobs_all(F, S, (di, oi, wi))
            m = {}
            for d in range(D.shape[1]):
                for o in range(O.shape[1]):
                    for k, w in enumerate(warp_vocab):
                        if canon_ok is not None and not canon_ok.get(
                                (d, o, tuple(w)), True):
                            continue
                        # INTEGER tick cells: float keys made the joint DP report
                        # "infeasible" on 719/721 measures through accumulated drift
                        dur = cells_local(V.duration_of(d, o, tuple(w)))
                        cost = -(float(D[0][d]) + float(O[0][o]) + float(W[0][k]))
                        cur = m.get(dur)
                        if cur is None or cost < cur[0]:
                            m[dur] = (cost, d, o, tuple(w))
            per.append(m)
        cost_maps.append(per)
    seqs = [(c.cid, c.x) for c in cm.cands]
    idx = {c.cid: i for i, c in enumerate(cm.cands)}
    packed = [[seqs[idx[c.cid]] for c in seq] for seq in voices]
    assign, cost, nst = joint_solve(packed, cm.length, cost_maps,
                                    max_states=max_states)
    if assign is None:
        return got, nst
    out = []
    for vi, v in enumerate(got):
        out.append([(e[0], e[1], onset / 12.0, dur / 12.0, d, o, w)
                    for e, (onset, dur, d, o, w) in zip(v, assign[vi])])
    return tuple(tuple(v) for v in out), nst


def to_eval(voices):
    return [[{"cid": e[0], "x": e[1], "onset": e[2], "dur": e[3], "d": e[4],
              "dots": e[5], "warp": e[6]} for e in v] for v in voices]


def fold_cache_key(fold, corpus_hash, hidden, hidden2, hidden3, epochs, lr, k,
                   label_smooth, state_noise=0.05, seed=7, synth_tag=""):
    """Single source of truth for the per-fold model cache key.

    Diagnostics must NOT guess a model file: several configs coexist per fold, so
    sorted(glob('ts_f0_*.npz'))[-1] can load a model from an unrelated experiment and
    report numbers that disagree with the headline (that happened once, a 58-measure
    discrepancy).  Use this function instead.

    `synth_tag` MUST be non-empty whenever synthetic data joins the training set.  Without
    it the key is identical to the real-data run, every fold silently reloads a model
    trained WITHOUT the synthetic data, and the experiment measures nothing -- the same
    failure mode as the `wp` parameter that was once skipped for exactly this reason.
    """
    import hashlib
    return hashlib.sha256(
        f"ts|{fold}|{corpus_hash}|{CAND_DIM}|{STATE_DIM}|{STEM_FEAT_DIM}|"
        f"{USE_STEM_FEATURES}|{hidden}|{hidden2}|{hidden3}|{epochs}|{lr}|{k}|"
        f"{label_smooth}|{state_noise}|{seed}|{synth_tag}".encode()).hexdigest()[:16]


def _fold_task(payload):
    """One fold, rebuilt from scratch inside the worker.

    Only (fold, argparse namespace) crosses the process boundary -- the corpus is
    reloaded here (about a second) rather than pickled (4,100 measures), and the
    trained model is never sent back, just the two metric dicts.
    """
    fold, ns = payload
    import corpus as C
    import visual as V
    from torch_train import train_torch
    frozen, _h = C.load_freeze()
    allm = frozen if frozen is not None else C.build_corpus()
    _t, _d, _e, (tr_p, dv_p, te_p) = C.split_by_piece(allm)
    wv = V.warp_vocab(allm)
    _, canon_ok = canon_table(wv)
    ma, mb = run_fold(ns, fold, allm, wv, canon_ok, train_torch, tr_p, dv_p, te_p)
    return fold, ma, mb


def _safe_log(p):
    import math as _m
    return _m.log(max(float(p), 1e-12))


def run_fold(a, fold, allm, wv, canon_ok, train_torch, tr_p, dv_p, te_p):
    pool = sorted(set(tr_p) | set(dv_p) | set(te_p))
    order = sorted(pool)
    rng = np.random.default_rng(4242)
    rng.shuffle(order)
    fo = {p: i % a.folds for i, p in enumerate(order)}
    by_piece = collections.defaultdict(list)
    for m in allm:
        by_piece[m.piece].append(m)
    train_ms = [m for p in pool if fo[p] != fold for m in by_piece[p]]
    val_ms = [m for p in pool if fo[p] == fold for m in by_piece[p]]
    # SYNTHETIC DATA IS TRAIN-ONLY.  It joins every fold's training set and never a
    # validation or test set, so the 50 real pieces stay an untouched holdout and the
    # corpus hash does not change.
    _stag = ""
    if getattr(a, "synth", ""):
        from synth_data import load_synth, select_synth, synth_tag
        _all = load_synth(a.synth)
        _sel = select_synth(_all, getattr(a, "synth_mix", 1.0), seed=a.seed)
        _rep = max(1, int(getattr(a, "synth_repeat", 1)))
        train_ms = list(train_ms) + _sel * _rep
        _stag = synth_tag(a.synth, getattr(a, "synth_mix", 1.0), _rep)
        print(f"  fold {fold}: +{len(_sel) * _rep} synthetic measures "
              f"(tag {_stag})", flush=True)
    # REAL FRONT-END HINT NOISE.  The iid hint noise loses a 3-note tuplet marking on all three
    # notes with p ~ 0.003; Codex's small symbol models lose it on 32% of tuplet groups
    # (lab/omr_proto/noise_stats.py, camera67).  With --real-noise RATES.json and a mix > 0, that
    # share of each measure's noise realisations uses real_noise.py's group-correlated noise with
    # the rates estimated WITHOUT this fold's pieces (key excl_fold_F).  Default: off, bit-identical.
    _rn_mix = float(getattr(a, "real_noise_mix", 0.0) or 0.0)
    _rn_rates = None
    if _rn_mix > 0:
        import hashlib
        from real_noise import DEFAULT_RATES
        _rnp = a.real_noise if os.path.isabs(a.real_noise) else os.path.join(os.path.dirname(HERE), a.real_noise)
        _rnd = json.load(io.open(_rnp, encoding="utf-8"))
        _rn_rates = dict(DEFAULT_RATES, **_rnd[f"excl_fold_{fold}"])
        with open(_rnp, "rb") as _fh:
            _stag += f"|rn{_rn_mix:g}{hashlib.sha256(_fh.read()).hexdigest()[:8]}"
        print(f"  fold {fold}: real front-end hint noise on {_rn_mix:g} of the realisations "
              f"(rates excl. fold {fold}: {_rn_rates})", flush=True)
    tr = make_train_candidates(train_ms, 0.85, k=a.k, real_rates=_rn_rates, real_mix=_rn_mix)
    _rx = float(getattr(a, "real_x_train", 0.0) or 0.0)
    if _rx > 0:
        # REAL-LAYOUT x IN TRAINING.  The model had only ever seen the simulator's synthetic
        # x (dt ** 0.6 with uniform jitter); on real PDMX layout the production decoder
        # loses 1.29 points even with column_snap grouping.  For every PDMX dose measure
        # with a default-x per event, the first round(_rx * k) of its k noise realisations
        # take the REAL x (column_snap.apply_real_x); the rest keep the simulator's.  Hint
        # noise is untouched, and the training set does not grow.
        if not getattr(a, "synth", "") or getattr(a, "synth_mix", 1.0) < 1.0 or _rep != 1:
            raise SystemExit("--real-x-train needs the full PDMX dose (synth_mix 1, repeat 1)")
        import hashlib
        from column_snap import apply_real_x
        _rxp = os.path.join(os.path.dirname(HERE), "pdmx_train_dose4000_realx.json")
        _rxd = json.load(io.open(_rxp, encoding="utf-8"))
        if len(_rxd["x_real"]) != len(_sel):
            raise SystemExit("pdmx_train_dose4000_realx.json does not match the dose")
        _base = len(train_ms) - len(_sel)
        _jn = int(round(_rx * a.k))
        _done = 0
        for _i, (_xr, _xn) in enumerate(zip(_rxd["x_real"], _rxd["x_notes"])):
            if _xr is None or any(_v is None for _v in _xr):
                continue
            for _j in range(_jn):
                apply_real_x(tr[_j * len(train_ms) + _base + _i].cands, _xr, _xn)
                _done += 1
        with open(_rxp, "rb") as _fh:
            _stag += f"|rx{_rx:g}{hashlib.sha256(_fh.read()).hexdigest()[:8]}"
        print(f"  fold {fold}: real-layout x on {_done} of {len(tr)} training candidate "
              f"measures (tag {_stag})", flush=True)
    # ---- time-warp prior debiasing (decode-time, default off) ----
    # The largest warp error is (3,2) -> (1,1): 185 of 302, and (3,2) has 14,240 training
    # events, so it is not sparsity.  (1,1) is 78% of all events, so its log-prob carries a
    # large PRIOR advantage that has nothing to do with this particular candidate.  This
    # subtracts A*log(prior) from the warp log-probs, centred to have zero mean, so only the
    # RELATIVE bias changes.  A = 0 keeps production bit-identical.
    _wpri = float(getattr(a, "warp_prior", 0.0) or 0.0)
    _wpair = float(getattr(a, "warp_pair_bias", 0.0) or 0.0)
    _wbias = None
    if _wpair:
        # TARGETED pair bias.  Both prior-correction forms failed: the unclipped one collapsed
        # to 58.38/36.29 because it boosted the five ultra-rare ratios (28-288 training events)
        # and pushed them into the top-3 candidates; centring then made every COMMON class
        # negative, so the clipped form still hurt (-0.97).  This variant touches only the
        # measured confusion -- (3,2) gets +A, (1,1) gets -A, everything else 0 -- which
        # isolates the (3,2)->(1,1) mechanism with no pollution from rare classes.
        _wbias = [0.0] * len(wv)
        for _i, _r in enumerate(wv):
            if tuple(_r) == (3, 2):
                _wbias[_i] = _wpair
            elif tuple(_r) == (1, 1):
                _wbias[_i] = -_wpair
        print("  fold {}: warp pair bias (3,2){:+.2f} (1,1){:+.2f}".format(
            fold, _wpair, -_wpair), flush=True)
    if _wpri:
        _cnt = {}
        _tot = 0
        for _m in train_ms:
            for _e in _m.events:
                _r = tuple(_e.warp)
                _cnt[_r] = _cnt.get(_r, 0) + 1
                _tot += 1
        _vals = []
        for _r in wv:
            _p = max(_cnt.get(tuple(_r), 0), 1) / max(1, _tot)
            _vals.append(-_wpri * _safe_log(_p))
        _mean = sum(_vals) / len(_vals)
        # CENTRE then CLIP.  Centring alone leaves the five ultra-rare ratios (28-288 training
        # events) with +1.2..+1.6 boosts, which pushes them into the top-3 warp candidates and
        # displaces the correct ones -- measured as a collapse to 58.38 (A=0.5) and 36.29
        # (A=1.0).  Clipping bounds how far any single class can move.
        _clip = float(getattr(a, "warp_clip", 0.5) or 0.5)
        _wbias = [max(-_clip, min(_clip, _v - _mean)) for _v in _vals]
        print("  fold {}: warp-prior debias A={} -> {}".format(
            fold, _wpri, " ".join("{}{:+.2f}".format(r, b) for r, b in zip(wv, _wbias))),
            flush=True)
    _de = max(0, int(getattr(a, "dense_extra", 0)))
    if _de:
        # DENSE-BAND-ONLY extra noise realisations.  The train/held-out test measured the
        # 16+ band as the only overfit one (+8.33 points), and overfitting is what to fix
        # there -- a margin loss would fit it even harder.  More realisations per dense
        # measure raises its share of the gradient without touching the sparse band, which
        # already generalises (gap +0.95).
        _dense = [m for m in train_ms if len(m.events) >= 16]
        for _j in range(_de):
            tr.extend(make_candidates(_dense, 900 + _j, 0.85))
        _stag = (_stag + "|" if _stag else "") + "de{}".format(_de)
        print("  fold {}: +{} dense-only realisations ({} dense measures x {})".format(
            fold, len(_dense) * _de, len(_dense), _de), flush=True)
    # PER-FOLD MODEL CACHE.  Without this every CV run retrains five models (~25 min)
    # even when nothing that affects training has changed, which makes the "modify ->
    # measure -> diagnose" loop painfully slow.  The key covers everything that
    # changes the weights: the fold, the corpus hash, the feature/state widths, and
    # the full trainer configuration.  `--no-cache` forces a retrain.
    _key = fold_cache_key(fold, C.corpus_hash(allm), a.hidden, a.hidden2,
                          a.hidden3, a.epochs, a.lr, a.k, a.label_smooth,
                          seed=a.seed, synth_tag=_stag)
    _cp = os.path.join(MODELS, f"ts_f{fold}_{_key}.npz")
    if a.cache and os.path.exists(_cp):
        mlp = MLP.load(_cp)
        if a.verbose:
            print(f"  fold {fold}: cached model {os.path.basename(_cp)}", flush=True)
    else:
        # Phase timing.  Measured CPU-side preprocessing at ~2 s for a whole fold, so any
        # wall-clock surprise has to be in training, pair-scoring or decoding -- these three
        # prints make that visible instead of guessed.
        _t_train = time.time()
        mlp = train_torch(tr, wv, epochs=a.epochs, hidden=a.hidden, lr=a.lr,
                          batch=int(getattr(a, "batch", 256)), seed=a.seed,
                          l2=1e-4, state_noise=0.05, verbose=False,
                          label_smooth=a.label_smooth, hidden2=a.hidden2,
                          hidden3=a.hidden3)
        print(f"  fold {fold}: TRAIN {time.time() - _t_train:.0f}s "
              f"({len(tr)} candidate measures, {a.epochs} epochs)", flush=True)
        if a.cache:
            try:
                os.makedirs(MODELS, exist_ok=True)
                mlp.save(_cp)
            except Exception:
                pass
    if getattr(a, "fp32", False):
        mlp.to_fp32()
    pk = LearnedPicker(mlp, wv)
    # pair scorer trained WITHOUT this fold, used only to re-rank beam successors
    _pm = None
    if getattr(a, "pair_w", 0.0) > 0:
        from pair_succ import PairSuccModel
        _t_pair = time.time()
        _pm = PairSuccModel().fit(allm, exclude_fold=fold, order_lookup=fo)
        print(f"  fold {fold}: PAIR-FIT {time.time() - _t_pair:.0f}s", flush=True)
    _t_valc = time.time()
    val = make_candidates(val_ms, 22, 0.85)
    print(f"  fold {fold}: VAL-CANDIDATES {time.time() - _t_valc:.0f}s "
          f"({len(val)} measures)", flush=True)

    A = []
    t0 = time.time()
    # CROSS-MEASURE CONTEXT (two passes).  Pass 1 decodes each measure independently and
    # records its per-voice median register; pass 2 re-decodes each measure with the
    # PREVIOUS measure's registers as context, so `_final_loss` can prefer partitions whose
    # voices continue across the barline.  Measured on the frozen corpus: 95.7% of voices
    # stay within one octave across a barline against a 71.2% chance baseline (5,750 shared
    # voices), and the decoder currently has no access to that at all.
    # The context is O(voices) floats per measure, so the per-measure working-set property
    # survives; what does NOT survive is strict measure independence, which is why this is
    # opt-in and validated by the full CV.
    _xmw = getattr(a, "xmeas_w", 0.0)
    # --stem-w / --staff-w override prod.EVAL for the beam's terminal loss only
    _prod = dict(PROD, stem_w=getattr(a, "stem_w", PROD.get("stem_w", 0.0)),
                 staff_w=getattr(a, "staff_w", PROD.get("staff_w", 0.0)))
    _prev = {}
    if _xmw:
        from search import _voice_median_pitch
        _by_piece = collections.defaultdict(list)
        for cm in val:
            _by_piece[cm.piece].append(cm)
        for _p, _cms in _by_piece.items():
            _reg = None
            for cm in _cms:
                _prev[id(cm)] = _reg
                _mid = {c.cid: c.midi for c in cm.cands}
                got, st = solve(cm.cands, cm.length, pk, wv, n_total=len(cm.cands),
                                n_beats=cm.gt.beats, pair_w=getattr(a, "pair_w", 0.0),
                                order_guard=a.order_guard, warp_bias=_wbias,
                                staff_guard=getattr(a, "staff_guard", False), **_prod)
                # `got` is the DICT form; _voice_median_pitch expects the tuple form
                _reg = [_voice_median_pitch(v, _mid) for v in as_tuples(got)]
    for cm in val:
        _scorer = None
        if _pm is not None:
            # NOTE the signature: search.solve passes the WHOLE current voice and the
            # accumulated ticks, because the contract was widened after the first
            # +0.49 run.  Passing the old (last_cid, rem) form here silently broke the
            # production default once already -- keep these in step.
            def _scorer(_cur, _rem, _acc, _cm=cm):
                return _pm.score_prob(_cm, _cur, _rem, _acc)
        got, st = solve(cm.cands, cm.length, pk, wv, n_total=len(cm.cands),
                        n_beats=cm.gt.beats, pair_w=getattr(a, "pair_w", 0.0),
                        pair_score=_scorer,
                        close_cost=getattr(a, "close_cost", 0.0),
                        xmeas_w=_xmw, warp_bias=_wbias,
                        prev_pitches=_prev.get(id(cm)) if _xmw else None,
                        midi_of=({c.cid: c.midi for c in cm.cands} if _xmw else None),
                        order_guard=a.order_guard,
                        staff_guard=getattr(a, "staff_guard", False), **_prod)
        A.append((got, st))
    ma = M.evaluate(val, A)
    # Sweep several isochrony weights off ONE trained model: training dominates
    # the cost, and the B arm's only varying input is `iso_w`.
    out = {}
    _cj = collections.Counter()      # column-joint diagnostics, this fold only
    _sp = collections.Counter()      # spacing-term diagnostics, this fold only
    for iw in a.iso_w:
        B = []
        for cm, (got, st) in zip(val, A):
            refined = dp_refine(cm, got, pk, wv, canon_ok, iters=a.iters,
                                k_attr=a.k_attr, iso_w=iw, exact_x=a.exact_x)
            try:
                la = _final_loss(as_tuples(got), cm.length, 1000.0, 0.0, 1,
                                 voice_w=PROD["voice_w"], dev_w=0.0,
                                 tau_w=PROD["tau_w"], iso_w=PROD["iso_w"],
                                 hint_w=PROD["hint_w"])
                lb = _final_loss(refined, cm.length, 1000.0, 0.0, 1,
                                 voice_w=PROD["voice_w"], dev_w=0.0,
                                 tau_w=PROD["tau_w"], iso_w=PROD["iso_w"],
                                 hint_w=PROD["hint_w"])
            except Exception:
                la, lb = 0.0, 1.0
            chosen = refined if lb < la else as_tuples(got)
            if a.joint:
                jr, _nst = joint_refine(cm, chosen, pk, wv, canon_ok)
                RD["joint_calls"] += 1
                if jr is not chosen and jr != chosen:
                    RD["joint_solved"] += 1
                    lj, lc = _loss_of(jr, cm), _loss_of(chosen, cm)
                    if lj < lc:
                        RD["joint_better"] += 1
                        chosen = jr
                    elif abs(lj - lc) < 1e-9:
                        RD["joint_tie"] += 1
                    else:
                        RD["joint_worse"] += 1
            if a.column_repair:
                # column_repair works on the tuple representation (same as
                # as_tuples), so pass `chosen` directly rather than to_eval(...)
                rep = column_repair(cm, chosen, pk, wv, canon_ok,
                                    k_attr=a.k_attr)
                if rep and _loss_of(rep, cm) < _loss_of(chosen, cm):
                    if a.repair_diag and rep != chosen:
                        ip = M.evaluate([cm], [(to_eval(chosen), st)])["perfect_pct"] > 50
                        iq = M.evaluate([cm], [(to_eval(rep), st)])["perfect_pct"] > 50
                        RD["changed"] += 1
                        RD["fixed"] += (iq and not ip)
                        RD["broke"] += (ip and not iq)
                        RD["same"] += (ip == iq)
                    chosen = rep
            if a.pitch_repair:
                # Undo same-onset same-duration voice swaps using pitch.  The repair
                # can only exchange notes that share an onset AND a duration, so it
                # cannot break exact tiling or change any onset/value -- the only
                # thing it can move is which voice an event belongs to.  Motivated by
                # the failure decomposition: 54% of remaining failures are
                # partition-only, and 6 of 35 are exactly this kind of swap.
                # Measured on fold 0: +0.14 perfect (2 fixed, 1 broken, 5 touched).
                pr = pitch_swap_repair(cm, chosen)
                if pr is not chosen and pr != chosen:
                    PB["touched"] += 1
                    ip = M.evaluate([cm], [(to_eval(chosen), st)])["perfect_pct"] > 50
                    iq = M.evaluate([cm], [(to_eval(pr), st)])["perfect_pct"] > 50
                    PB["fixed"] += (iq and not ip)
                    PB["broke"] += (ip and not iq)
                    chosen = pr
            if getattr(a, "column_joint", False):
                # The answer is provably wrong when two candidates in one x column get
                # different onsets (or two columns share one); re-solve every voice's
                # values jointly on the same partition.  See src/column_joint.py.
                why = column_violation(cm, chosen)
                if why is not None:
                    _cj["triggered"] += 1
                    _cj[why] += 1
                    ip = M.evaluate([cm], [(to_eval(chosen), st)])["perfect_pct"] > 50
                    cj = column_joint_refine(cm, chosen, pk, wv, canon_ok,
                                             k_attr=a.k_attr)
                    if cj is None:
                        _cj["no_solution"] += 1
                    else:
                        iq = M.evaluate([cm], [(to_eval(cj), st)])["perfect_pct"] > 50
                        _cj["fixed"] += (iq and not ip)
                        _cj["broke"] += (ip and not iq)
                        chosen = cj
            if getattr(a, "spacing_w", 0.0):
                # The answer's column gaps disagree with real engraving spacing: re-solve
                # its values jointly with the spacing cost.  See column_joint.py.
                from prod import SPACING_ALPHA, SPACING_C, SPACING_TRIGGER
                sr = spacing_residuals(cm, chosen, SPACING_ALPHA)
                if sr is not None and max(abs(e) for e in sr[0]) >= SPACING_TRIGGER:
                    _sp["triggered"] += 1
                    ip = M.evaluate([cm], [(to_eval(chosen), st)])["perfect_pct"] > 50
                    sp = column_joint_refine(cm, chosen, pk, wv, canon_ok, k_attr=a.k_attr,
                                             spacing=(SPACING_ALPHA, sr[1], SPACING_C,
                                                      a.spacing_w))
                    if sp is None:
                        _sp["no_solution"] += 1
                    else:
                        iq = M.evaluate([cm], [(to_eval(sp), st)])["perfect_pct"] > 50
                        _sp["changed"] += sp != tuple(tuple(v) for v in chosen)
                        _sp["fixed"] += (iq and not ip)
                        _sp["broke"] += (ip and not iq)
                        chosen = sp
            B.append((to_eval(chosen), st))
        if getattr(a, "repeat_consensus", False):
            # Needs the whole fold decoded first: twins are measures of one piece with the
            # same pitches and columns.  See src/repeat_consensus.py.
            _rc = collections.Counter()
            _new = repeat_consensus(val, [as_tuples(v) for v, _st in B], pk, wv, canon_ok,
                                    veto=a.repeat_veto, stats=_rc)
            B = [(to_eval(t), _st) for t, (_v, _st) in zip(_new, B)]
            print(f"  fold {fold}: repeat-consensus groups={_rc['groups']}  "
                  f"rewritten={_rc['rewritten']}  vetoed={_rc['vetoed']}", flush=True)
        out[iw] = M.evaluate(val, B)
    if getattr(a, "spacing_w", 0.0):
        print(f"  fold {fold}: spacing " + "  ".join(
            f"{k}={_sp[k]}" for k in ("triggered", "no_solution", "changed", "fixed",
                                      "broke")), flush=True)
    if getattr(a, "column_joint", False):
        print(f"  fold {fold}: column-joint " + "  ".join(
            f"{k}={_cj[k]}" for k in ("triggered", "column-split", "column-merge",
                                      "x-order", "no_solution", "fixed", "broke")),
              flush=True)
    # PARALLEL DECODE.  The decode is embarrassingly parallel over measures and is
    # pure numpy, so on a 32-core box it is the obvious place to spend cores.  It is
    # only used for the production configuration (no column_repair / joint /
    # repair_diag), because those paths also maintain RD/PB diagnostics that would
    # not aggregate across processes.  The numbers must be IDENTICAL to workers=1.
    _par_ok = (a.workers > 1 and a.cache and os.path.exists(_cp)
               and not a.joint and not a.column_repair and not a.repair_diag
               and not getattr(a, "repeat_consensus", False)
               and not getattr(a, "column_joint", False))
    if _par_ok:
        A, _Bs = decode_all(val, _cp, wv, canon_ok,
                            {"iso_w": list(a.iso_w), "iters": a.iters,
                             "k_attr": a.k_attr, "exact_x": a.exact_x,
                             "order_guard": a.order_guard,
                             "pitch_repair": a.pitch_repair},
                            workers=a.workers)
        ma = M.evaluate(val, A)
        out = {iw: M.evaluate(val, _Bs[iw]) for iw in a.iso_w}
        dt = time.time() - t0
        print(f"fold {fold}: n={len(val):4d} {dt:5.0f}s [{a.workers} workers] | "
              f"A perf={ma['perfect_pct']:6.2f} any={ma['any_field_error_pct']:5.2f}")
        for iw, mb in out.items():
            print(f"    iso_w={iw:<6g} B perf={mb['perfect_pct']:6.2f} "
                  f"any={mb['any_field_error_pct']:5.2f} "
                  f"voice={mb['voice_match_pct']:6.2f}")
        return ma, out
    dt = time.time() - t0
    print(f"fold {fold}: n={len(val):4d} {dt:5.0f}s | A perf={ma['perfect_pct']:6.2f} "
          f"any={ma['any_field_error_pct']:5.2f}")
    for iw, mb in out.items():
        print(f"    iso_w={iw:<6g} B perf={mb['perfect_pct']:6.2f} "
              f"any={mb['any_field_error_pct']:5.2f} voice={mb['voice_match_pct']:6.2f}")
    return ma, out



# ---------------------------------------------------------------------------
# PRODUCTION-CONFIGURATION TEST, DERIVED RATHER THAN HAND-MAINTAINED.
#
# Six separate bugs came from editing this test by hand every time a new flag appeared
# (--seed, --fp32, --pair-w, --close-cost, --synth, --dense-extra); each omission let an
# experiment overwrite the canonical artefact.  Deriving it removes the failure mode:
# argparse defaults ARE the production configuration, so "no flag differs from its default"
# IS "production", and any later flag is covered automatically.
#
# These flags only control HOW a run executes, never what it computes.
RUN_CONTROL = {
    "verbose", "workers", "fold_workers", "cache", "all_folds", "fold", "folds",
    "no_cache", "repair_diag", "limit", "limit_folds",
}


def production_deviations(args, parser):
    """Flags whose value differs from a NO-FLAG run of the same parser.

    Not `action.default`: argparse's declared default is not always the value a no-flag run
    ends up with (post-processing, `append`, None -> [0.0]).  `--iso-w` differs in shape and
    used to be reported as a deviation on EVERY run, which made `prod` false everywhere and
    stopped the canonical artefact from ever being written.  Parsing an empty argv gives the
    real no-flag baseline and is immune to any post-processing.
    """
    def norm(k, v):
        # --iso-w is DECLARED as the string "0" but main() rewrites it into [0.0] right
        # after parsing, so a raw comparison reported it as a deviation on every run and
        # prod was False everywhere -- which would have orphaned the canonical artefact
        # entirely.  Apply the same rewrite here.
        if k == "iso_w":
            if isinstance(v, str):
                return [float(x) for x in v.split(",") if x.strip()]
            return [float(x) for x in v]
        return v

    base = vars(parser.parse_args([]))
    out = []
    for k, v in vars(args).items():
        if k in RUN_CONTROL or k not in base:
            continue
        if norm(k, v) != norm(k, base[k]):
            out.append(k)
    return sorted(out)


def main():
    STEM_FEATURES_ON = bool(USE_STEM_FEATURES)
    ap = argparse.ArgumentParser()
    # DEFAULTS COME FROM THE PRODUCTION CONSTANTS, never from stale literals.
    # They used to hard-code hidden=192 / hidden2=0 / label_smooth=0.0 / k_attr=12,
    # so a user following the docs but omitting an explicit flag silently got the
    # OLD configuration with no warning -- the same failure class as the inverted
    # `--no-order-guard` flag that corrupted this artefact for seven rounds.
    # The imports must come FIRST: referencing `_DP_K_ATTR` before importing it
    # raised UnboundLocalError.
    from run_experiment import (MODEL_EPOCHS as _E, MODEL_HIDDEN as _H,
                                MODEL_HIDDEN2 as _H2, MODEL_HIDDEN3 as _H3,
                                MODEL_LABEL_SMOOTH as _LS)
    from prod import DP_K_ATTR as _DP_K_ATTR
    from prod import COLUMN_JOINT as _COLUMN_JOINT
    from prod import EVAL as _EVAL
    from prod import STAFF_GUARD as _STAFF_GUARD
    from prod import REPEAT_CONSENSUS as _REPEAT_CONSENSUS, REPEAT_VETO as _REPEAT_VETO
    from prod import SPACING_W as _SPACING_W
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    # ON by default: without it every CV run retrains five models (~25 min) even when
    # nothing that affects training changed. The key covers fold, corpus hash, the
    # feature/state widths and the whole trainer config, so a stale model is never
    # reused; --no-cache forces a retrain.
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes for the decode (0 = serial). "
                         "Must not change the numbers.")
    ap.add_argument("--iters", type=int, default=2)
    # DEFAULT ON: measured +0.63 CV perfect / +0.69 voice-match with any-field
    # UNCHANGED (the repair cannot alter a value, only which voice owns an event).
    # BooleanOptionalAction gives both --pitch-repair and --no-pitch-repair, so the
    # default is a positive flag and there is no `not a.no_...` inversion to get
    # wrong -- the exact bug that silently corrupted the artefact in Round 35.
    ap.add_argument("--pitch-repair", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="undo same-onset/same-duration voice swaps with pitch")
    ap.add_argument("--k-attr", type=int, default=_DP_K_ATTR,
                    help="options per event for the exact DP (production "
                         "DP_K_ATTR; the beam's own k_attr is separate, prod.K_ATTR)")
    ap.add_argument("--iso-w", default="0",
                    help="comma-separated list of DP isochrony weights to sweep")
    ap.add_argument("--label-smooth", type=float, default=_LS)
    ap.add_argument("--hidden", type=int, default=_H)
    ap.add_argument("--hidden2", type=int, default=_H2)
    ap.add_argument("--hidden3", type=int, default=_H3)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=_E)
    ap.add_argument("--batch", type=int, default=256,
                    help="GPU batch size.  Measured: a fold spends 94%% of its wall time in "
                         "TRAIN (1,109 s of 1,184 s at the 4,000-measure dose, 30 epochs), "
                         "at ~9.5 ms per update -- slow for a 1536-wide MLP, so the batch is "
                         "the first thing worth widening.")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--order-guard", action="store_true",
                    help="enforce x-order==onset-order inside the beam.  Measured "
                         "on the full 5-fold: costs 0.03 any-field and 0.03 voice "
                         "match with NO perfect gain, so it is OFF by default.")
    ap.add_argument("--column-repair", action="store_true")
    ap.add_argument("--repair-diag", action="store_true")
    ap.add_argument("--joint", action="store_true",
                    help="joint column DP over all voices (exact invariant)")
    # PRODUCTION (prod.COLUMN_JOINT).  Fires only when the final answer breaks the exact
    # x-column invariant (observable without ground truth), then re-solves all voices'
    # values jointly on the same partition.  See src/column_joint.py.
    # --no-column-joint reproduces the previous 95.81 headline bit for bit.
    ap.add_argument("--column-joint", action=argparse.BooleanOptionalAction,
                    default=_COLUMN_JOINT,
                    help="re-solve values jointly across voices when the answer "
                         "breaks the x-column invariant (src/column_joint.py)")
    ap.add_argument("--exact-x", type=float, default=0.0,
                    help="cost for violating the exact-column invariant (0=off)")
    # DEFAULT OFF, and this was measured BOTH ways.  Casting to float32 halves the
    # model (2392 -> 1196 KiB) and is numerically identical (per-fold equal to 4 dp),
    # but it made decoding 1.86x SLOWER (2.60 -> 4.83 s/page): the feature matrix is
    # float64, so mixing float32 weights with float64 features makes numpy upcast and
    # pay an extra conversion.  On a 4 GB PS DDR the memory saving is irrelevant, so
    # the floating-point default is float64.
    ap.add_argument("--fp32", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="cast the numpy model to float32 (halves memory, ~1.9x slower)")
    ap.add_argument("--pair-w", type=float, default=0.25,
                    help="weight of the pairwise voice-continuation "
                         "log-probability added to the successor "
                         "logits (0 = off; 0.25 measured best). "
                         "The scorer is trained per fold, excluding it.")
    ap.add_argument("--close-cost", type=float, default=0.5,
                    help="nats charged per voice closure in the beam; "
                         "closure is otherwise free, so the number of "
                         "voices was never a scored decision (0.5 "
                         "measured best on any-field)")
    ap.add_argument("--stem-w", type=float, default=_EVAL["stem_w"],
                    help="terminal-loss weight per voice that shares a staff yet mixes "
                         "up/down stems (search._stem_term); default prod.EVAL['stem_w']")
    ap.add_argument("--spacing-w", type=float, default=_SPACING_W,
                    help="weight of the real-engraving spacing cost in a column-joint "
                         "re-solve, run when an answer's column gaps break the spacing "
                         "model (column_joint.spacing_residuals); default prod.SPACING_W")
    ap.add_argument("--repeat-consensus", action=argparse.BooleanOptionalAction,
                    default=_REPEAT_CONSENSUS,
                    help="repeated measures of one piece share the best-supported answer "
                         "(src/repeat_consensus.py); default prod.REPEAT_CONSENSUS")
    ap.add_argument("--repeat-veto", type=float, default=_REPEAT_VETO,
                    help="nats by which a twin's own evidence must prefer its answer to "
                         "keep it; default prod.REPEAT_VETO")
    ap.add_argument("--staff-guard", action=argparse.BooleanOptionalAction,
                    default=_STAFF_GUARD,
                    help="a voice under construction only accepts candidates on its own "
                         "staff (exact for the ground truth); default prod.STAFF_GUARD")
    ap.add_argument("--staff-w", type=float, default=_EVAL["staff_w"],
                    help="terminal-loss weight per voice spread over more than one staff "
                         "(search._staff_term); default prod.EVAL['staff_w']")
    # PRODUCTION now ships a fixed extra-training set: 31 pieces of 测试集4 with every
    # piece that also appears in the frozen corpus removed (verified by exact AND
    # content-fingerprint dedup -- 6 duplicates existed because 测试集4 reuses frozen pieces
    # under different filenames).  Measured effect, leakage-free:
    #   frozen-corpus 5-fold 93.32 -> 94.58 (+1.26, 4/5 folds; t=1.9)
    #   clean same-distribution benchmark 89.54 -> 90.86 (+1.32, 5/5 folds; t=5.0)
    # It is a separate collection, so it never touches the frozen validation pieces.
    ap.add_argument("--warp-pair-bias", type=float, default=0.0,
                    help="bias (3,2) by +A and (1,1) by -A; others untouched")
    ap.add_argument("--warp-clip", type=float, default=0.5,
                    help="bound on the per-class prior bias (logits)")
    ap.add_argument("--warp-prior", type=float, default=0.0,
                    help="subtract A*log(prior) from the warp log-probs (0 = production)")
    # PRODUCTION extra training set, 2026-10-21.  Was `ts4_halfA_clean.json` (31 pieces of
    # 测试集4); now 4,000 dense+tuplet measures selected from PDMX under the cleanliness
    # filters (no lyrics, notes_per_bar<=40, no 'midi', n_tracks<=4).  Measured dose gradient
    # on the frozen corpus, 5 folds each, same holdout:
    #     735  -> 94.88 (+0.30, t=0.5)     2,000 -> 95.60 (+1.02, t=1.8)
    #     4,000 -> 95.81 (+1.23, t=2.3, 5/5 folds positive)   <-- adopted
    # 2,000 is the knee; 4,000 costs ~100 min instead of ~56 but is the only config with all
    # five folds positive and t>2.  The set is piece-disjoint from every evaluation target.
    ap.add_argument("--synth",
                    default=os.path.join(os.path.dirname(HERE),
                                         "pdmx_train_dose4000_clean.json"),
                    help="path to an adapted synthetic corpus (real conventions); "
                         "its measures join EVERY fold's training set and never any "
                         "validation set")
    ap.add_argument("--synth-mix", type=float, default=1.0,
                    help="fraction of the synthetic measures to use, stratified by "
                         "density band (1.0 = all)")
    ap.add_argument("--xmeas-w", type=float, default=0.0,
                    help="weight of the cross-measure register-continuity term; "
                         "enables two-pass decoding (measure t's voices inform t+1). "
                         "0 = off, which keeps per-measure decoding independent")
    ap.add_argument("--real-x-train", type=float, default=0.0,
                    help="fraction of each PDMX dose measure's noise realisations that use "
                         "its REAL engraving x (pdmx_train_dose4000_realx.json, grouped by "
                         "column_snap); 0 = simulator x only, as production")
    ap.add_argument("--real-noise", default="real_noise_rates_camera67.json",
                    help="per-fold rates of the real front end's group-correlated hint noise "
                         "(lab/omr_proto/noise_stats.py); used only when --real-noise-mix > 0")
    ap.add_argument("--real-noise-mix", type=float, default=0.0,
                    help="fraction of each measure's training noise realisations that use the "
                         "real front end's group-correlated hint noise; 0 = iid only, as production")
    ap.add_argument("--dense-extra", type=int, default=0,
                    help="extra noise realisations for measures with >=16 "
                         "events; the 16+ band is the only measured-overfit "
                         "one (+8.33 points train vs held-out)")
    ap.add_argument("--seed", type=int, default=7,
                    help="model init/training seed.  All headline numbers use "
                         "7; other seeds measure the run-to-run spread.")
    ap.add_argument("--fold-workers", type=int, default=0,
                    help="run folds concurrently (0 = sequential). "
                         "Lossless: folds are independent.")
    ap.add_argument("--all-folds", action="store_true")
    a = ap.parse_args()
    a.iso_w = [float(x) for x in str(a.iso_w).split(",") if x.strip()]

    from torch_train import train_torch
    frozen, h = C.load_freeze()
    allm = frozen if frozen is not None else C.build_corpus()
    tr_gt, dv_gt, te_gt, (tr_p, dv_p, te_p) = C.split_by_piece(allm)
    wv = V.warp_vocab(allm)
    _, canon_ok = canon_table(wv)
    folds = range(a.folds) if a.all_folds else [a.fold]
    if a.all_folds and a.fold_workers <= 1:
        print(f"  hint: --fold-workers {a.folds} runs the folds concurrently -- "
              f"measured 3.0x faster when models are cached (444.9 s -> 150.0 s) "
              f"with bit-identical numbers, because folds are independent.")
    per = []
    # FOLD-LEVEL PARALLELISM.  The five folds are completely independent: each trains
    # its own model from its own seed and scores its own pieces, so running them
    # concurrently cannot change a single number -- it only changes the wall clock
    # from "sum of five" to "slowest one".  The model is tiny (2.4 MB, batch 256), so
    # several folds coexist on the GPU comfortably.  This is the largest remaining
    # lossless speedup: the GPU training loop is 75% of a fold's cost.
    if a.fold_workers > 1 and len(list(folds)) > 1:
        import concurrent.futures as _cf
        import multiprocessing as _mp
        _ctx = _mp.get_context("spawn")
        _jobs = [(f, a) for f in folds]
        print(f"[fold-parallel] {len(_jobs)} folds on {a.fold_workers} processes")
        with _cf.ProcessPoolExecutor(max_workers=a.fold_workers,
                                     mp_context=_ctx) as _ex:
            _res = list(_ex.map(_fold_task, _jobs))
        _res.sort(key=lambda r: r[0])
        per = [{"fold": f, "A": ma, "B": mb} for f, ma, mb in _res]
    else:
        for f in folds:
            ma, mb = run_fold(a, f, allm, wv, canon_ok, train_torch, tr_p, dv_p, te_p)
            per.append({"fold": f, "A": ma, "B": mb})
    import statistics
    summary = {}
    iso_keys = list(per[0]["B"].keys()) if per else []
    if len(per) > 1:
        print("\n=== out-of-fold over all 50 pieces ===")
        for key in ("perfect_pct", "any_field_error_pct", "voice_match_pct"):
            va = [p["A"][key] for p in per]
            summary[key] = {"A_mean": statistics.mean(va),
                            "A_sd": statistics.stdev(va)}
            for iw in iso_keys:
                vb = [p["B"][iw][key] for p in per]
                summary[key][f"B_iso{iw}"] = {
                    "mean": statistics.mean(vb), "sd": statistics.stdev(vb),
                    "delta": statistics.mean(vb) - statistics.mean(va)}
            print(f"  {key:22s} A {statistics.mean(va):6.2f} +/- {statistics.stdev(va):5.2f}")
            for iw in iso_keys:
                vb = [p["B"][iw][key] for p in per]
                print(f"  {'':22s} B(iso={iw:<5g}) {statistics.mean(vb):6.2f} "
                      f"+/- {statistics.stdev(vb):5.2f}"
                      f"   delta {statistics.mean(vb)-statistics.mean(va):+.2f}")
        # AGGREGATION CONVENTION.  The headline above is a MACRO average: the mean of
        # the five per-fold percentages, so each fold (10 pieces) counts equally even
        # though the folds differ in size (600..964 measures).  A reader who instead
        # pools all measures gets a MICRO average that is ~0.7 points LOWER, which
        # looks like an inconsistency unless both are stated.  Record both.
        for key in ("perfect_pct", "any_field_error_pct", "voice_match_pct"):
            tot_m = sum(p["A"]["measures"] for p in per) or 1
            a_micro = sum(p["A"][key] * p["A"]["measures"] for p in per) / tot_m
            summary[key]["A_micro"] = a_micro
            for iw in iso_keys:
                b_micro = sum(p["B"][iw][key] * p["B"][iw]["measures"]
                              for p in per) / tot_m
                summary[key][f"B_iso{iw}"]["micro"] = b_micro
        _mac = statistics.mean([p["B"][iso_keys[0]]["perfect_pct"] for p in per])
        _mic = summary["perfect_pct"][f"B_iso{iso_keys[0]}"]["micro"]
        print(f"\n  aggregation: macro (per-fold mean) {_mac:.2f}%   "
              f"micro (pooled) {_mic:.2f}%   difference {_mac - _mic:+.2f}")
        print(f"  the HEADLINE is the MACRO average; both are recorded in the artefact")
    # PROVENANCE.  This file is the HEADLINE result, so it must state the exact
    # configuration that produced it.  An earlier version recorded only iters and
    # k_attr, which made the authoritative number impossible to reproduce from the
    # artefact alone -- and a restored copy had no config at all.
    from run_experiment import (MODEL_AUG, MODEL_EPOCHS, MODEL_HIDDEN,
                                MODEL_HIDDEN2, MODEL_HIDDEN3, MODEL_LABEL_SMOOTH,
                                MODEL_STATE_NOISE)
    from prod import PROD
    import corpus as _C
    _, chash = _C.load_freeze()
    cfg = {"hidden": a.hidden, "hidden2": a.hidden2, "hidden3": a.hidden3,
           "lr": a.lr, "epochs": a.epochs, "aug_k": a.k,
           "label_smooth": a.label_smooth, "state_noise": MODEL_STATE_NOISE,
           "iters": a.iters, "k_attr": a.k_attr, "iso_w": a.iso_w,
           "order_guard": a.order_guard, "exact_x": a.exact_x,
           "pitch_repair": a.pitch_repair,
           "column_repair": a.column_repair, "joint": a.joint,
           "column_joint": getattr(a, "column_joint", False),
           "column_joint_max_states": __import__("prod").COLUMN_JOINT_MAX_STATES,
           "stem_w": getattr(a, "stem_w", 0.0),
           "staff_w": getattr(a, "staff_w", 0.0),
           "staff_guard": getattr(a, "staff_guard", False),
           "repeat_consensus": getattr(a, "repeat_consensus", False),
           "repeat_veto": getattr(a, "repeat_veto", 4.0),
           "spacing_w": getattr(a, "spacing_w", 0.0),
           "real_x_train": getattr(a, "real_x_train", 0.0),
           "real_noise": getattr(a, "real_noise", "") or None,
           "real_noise_mix": getattr(a, "real_noise_mix", 0.0),
           "spacing_model": {k: getattr(__import__("prod"), k) for k in
                             ("SPACING_ALPHA", "SPACING_C", "SPACING_TRIGGER")},
           # These four decide the headline but were missing from the artefact, so a
           # reader could not tell which configuration produced a number.  The artefact
           # must be self-describing: anything that changes the output belongs here.
           "pair_w": getattr(a, "pair_w", 0.0),
           "close_cost": getattr(a, "close_cost", 0.0),
           "fp32": getattr(a, "fp32", False),
           "synth": getattr(a, "synth", "") or None,
           "synth_mix": (getattr(a, "synth_mix", 1.0)
                         if getattr(a, "synth", "") else None),
           "seed": a.seed,
           "beam": PROD["beam"], "eval_weights": PROD,
           "corpus_hash": chash, "folds": a.folds,
           "production_constants": {"MODEL_HIDDEN": MODEL_HIDDEN,
                                    "MODEL_HIDDEN2": MODEL_HIDDEN2,
                                    "MODEL_HIDDEN3": MODEL_HIDDEN3,
                                    "MODEL_EPOCHS": MODEL_EPOCHS,
                                    "MODEL_AUG": MODEL_AUG,
                                    "MODEL_LABEL_SMOOTH": MODEL_LABEL_SMOOTH}}
    # NEVER let a partial run clobber the headline artefact.  A single-fold sweep
    # used to fall through to this write and silently replaced the 5-fold
    # authoritative result with a fold-0-only one.
    if RD:
        print("diag: " + "  ".join(f"{k}={v}" for k, v in sorted(RD.items())))
    if not a.all_folds:
        print(f"\n(single-fold run: results/two_stage_cv.json NOT written)")
        return
    # ARTEFACT PROTECTION.  Every --all-folds run used to overwrite the single
    # authoritative file, so an experiment (a different seed, a disabled repair, a
    # feature toggle) silently replaced the headline result -- that happened once and
    # was only caught by the deliverable self-check comparing docs against the file.
    # Production configuration keeps the canonical name; anything else is written to a
    # suffixed file and says so.
    prod = (a.seed == 7 and a.pitch_repair and not a.order_guard
            and not a.joint and not a.column_repair
            and a.hidden == MODEL_HIDDEN and a.hidden2 == MODEL_HIDDEN2
            and a.hidden3 == MODEL_HIDDEN3 and a.epochs == MODEL_EPOCHS
            and abs(a.label_smooth - MODEL_LABEL_SMOOTH) < 1e-12
            and a.k == MODEL_AUG and a.k_attr == _DP_K_ATTR
            and len(a.iso_w) == 1 and abs(a.iso_w[0]) < 1e-12
            and not STEM_FEATURES_ON and not getattr(a, "fp32", False)
            and abs(getattr(a, "pair_w", 0.25) - 0.25) < 1e-12
            and abs(getattr(a, "close_cost", 0.5) - 0.5) < 1e-12
            and not getattr(a, "synth", "")
            and int(getattr(a, "dense_extra", 0)) == 0)
    # DERIVED guard overrides the hand-written one above; the list of
    # deviations also names the artefact suffix, so the two cannot drift.
    _dev = production_deviations(a, ap)
    prod = not _dev

    if prod:
        path = os.path.join(OUT, "two_stage_cv.json")
        tag = ""
    else:
        bits = []
        # the derived deviations are the authoritative suffix
        bits = [k.replace("_", "") for k in _dev]
        # a deviating BOOLEAN names only the flag, not its value; spell the OFF case so
        # --no-column-joint cannot reuse the file name of an earlier --column-joint run
        if "columnjoint" in bits and not getattr(a, "column_joint", False):
            bits[bits.index("columnjoint")] = "nocolumnjoint"
        # a deviating weight names its value too, so a sweep does not overwrite itself
        if "stemw" in bits:
            bits[bits.index("stemw")] = f"stemw{a.stem_w:g}"
        if "staffw" in bits:
            bits[bits.index("staffw")] = f"staffw{a.staff_w:g}"
        if "staffguard" in bits and not getattr(a, "staff_guard", False):
            bits[bits.index("staffguard")] = "nostaffguard"
        if "repeatconsensus" in bits and not getattr(a, "repeat_consensus", False):
            bits[bits.index("repeatconsensus")] = "norepeatconsensus"
        if "repeatveto" in bits:
            bits[bits.index("repeatveto")] = f"repeatveto{a.repeat_veto:g}"
        if "spacingw" in bits:
            bits[bits.index("spacingw")] = f"spacingw{a.spacing_w:g}"
        if "realxtrain" in bits:
            bits[bits.index("realxtrain")] = f"realxtrain{a.real_x_train:g}"
        if "realnoisemix" in bits:
            bits[bits.index("realnoisemix")] = f"realnoisemix{a.real_noise_mix:g}"
        if "realnoise" in bits:
            bits.remove("realnoise")                 # the mix names the run; the file is in the cfg
        if a.seed != 7:
            bits.append(f"seed{a.seed}")
        if not a.pitch_repair:
            bits.append("norepair")
        if a.order_guard:
            bits.append("orderguard")
        if a.hidden != MODEL_HIDDEN or a.hidden2 != MODEL_HIDDEN2:
            bits.append(f"h{a.hidden}x{a.hidden2}")
        if a.epochs != MODEL_EPOCHS:
            bits.append(f"ep{a.epochs}")
        if a.k != MODEL_AUG:
            bits.append(f"k{a.k}")
        if a.k_attr != _DP_K_ATTR:
            bits.append(f"kattr{a.k_attr}")
        if abs(a.label_smooth - MODEL_LABEL_SMOOTH) > 1e-12:
            bits.append(f"ls{a.label_smooth}")
        if STEM_FEATURES_ON:
            bits.append("stem")
        if getattr(a, "fp32", False):
            bits.append("fp32")
        if abs(getattr(a, "pair_w", 0.25) - 0.25) > 1e-12:
            bits.append(f"pairw{a.pair_w}")
        if abs(getattr(a, "close_cost", 0.5) - 0.5) > 1e-12:
            bits.append(f"cc{a.close_cost}")
        if getattr(a, "synth", ""):
            bits.append(f"synth{getattr(a, 'synth_mix', 1.0):g}")
        if int(getattr(a, "dense_extra", 0)):
            bits.append(f"de{a.dense_extra}")
        tag = "_" + ("_".join(bits) if bits else "custom")
        path = os.path.join(OUT, f"two_stage_cv{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"config": cfg, "summary": summary, "per_fold": per}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {os.path.basename(path)}  (corpus {chash[:12]}...)")
    if not prod:
        print("  NOTE: this is NOT the production configuration, so the canonical")
        print("  results/two_stage_cv.json was left untouched.")


if __name__ == "__main__":
    main()















