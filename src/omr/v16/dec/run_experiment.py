"""Main experiment: greedy vs rule-picker+search vs learned-picker+search.

Question under test (Starry Table 8, reproduced on THIS project's repertoire):
under an identical search framework, does a learned local policy beat a
hand-written rule policy at recovering measure-level voice + rhythm structure?
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import corpus as C  # noqa: E402
import metrics as M  # noqa: E402
import visual as V  # noqa: E402
from prod import PROD  # noqa: E402
from pickers import (CAND_DIM, STATE_DIM, MLP, HintRepairPicker,  # noqa: E402
                     HybridPicker,
                     LearnedPicker, RulePicker, cand_features, hint_indices,
                     state_features)
from search import greedy, solve, solve_best  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "results")
SRC = os.path.join(os.path.dirname(HERE), "src")


# --------------------------------------------------------------------------- #
def make_candidates(gt_measures, seed, hint_acc, drop=0.0, spur=0.0, real_rates=None):
    rng = random.Random(seed)
    return [V.synthesise(m, rng, hint_acc=hint_acc, drop_rate=drop, spur_rate=spur, real_rates=real_rates)
            for m in gt_measures]


def make_train_candidates(gt_measures, hint_acc, k=4, seed=11, real_rates=None, real_mix=0.0):
    """Several independent noise realisations per measure.

    The attribute heads overfit badly on a single realisation (warp accuracy fell
    21 points from train to test).  Seeing the same ground truth under different
    upstream hints forces them to learn when the hint can be trusted.

    With `real_rates`, the first round(real_mix * k) realisations use the group-correlated
    noise of a real front end (real_noise.py) instead of iid hint noise; real_mix 0 is the
    production path, bit-identical.
    """
    out = []
    n_real = int(round(real_mix * max(1, k))) if real_rates is not None else 0
    for j in range(max(1, k)):
        out.extend(make_candidates(gt_measures, seed + 1000 * j, hint_acc,
                                   real_rates=real_rates if j < n_real else None))
    return out


def build_training_samples(cand_measures, warp_vocab, state_noise=0.0, reps=1,
                           seed=7):
    """Teacher forcing along the TRUE trajectory, in the shuffled cid order.

    `state_noise` implements the DAgger-style fix for the measured failure mode.
    The model is only ever trained on states from the true trajectory, but at
    inference the search drifts: a wrong duration shifts every later onset in
    that voice (measured: 2.77% per-event attribute error becomes 18.86% event
    error when applied naively).  The policy therefore meets accumulated-time
    values it has never seen and degrades further.

    Injecting noise into the accumulated time AT TRAINING TIME -- while keeping
    the label as the true next action -- teaches exactly the robustness the
    search needs: "given that the clock has drifted by this much, which note
    comes next and what is its true time value".
    """
    widx = {w: i for i, w in enumerate(warp_vocab)}
    rng = random.Random(seed)
    samples = []
    for cm in cand_measures:
        eid2cid = {c.gt_eid: c.cid for c in cm.cands if c.gt_eid >= 0}
        xof = {c.cid: c.x for c in cm.cands}
        n_total = len(cm.cands)
        by_voice = {}
        for e in cm.gt.events:
            by_voice.setdefault(e.voice, []).append(e)
        for rep in range(max(1, reps)):
            # one drift offset per voice traversal, resampled per repetition
            drift = (rng.uniform(-state_noise, state_noise) * cm.length
                     if state_noise > 0 else 0.0)
            used = 0
            n_closed = 0
            for vi in sorted(by_voice):
                evs = sorted(by_voice[vi], key=lambda e: e.onset)
                acc, last_x, cur = 0, 0.0, []
                for e in evs:
                    cid = eid2cid[e.eid]
                    rem = [c for c in cm.cands if not (used >> c.cid) & 1]
                    tgt = next(i for i, c in enumerate(rem) if c.cid == cid)
                    seen_acc = acc + drift
                    seen_acc = min(max(seen_acc, 0.0), float(cm.length))
                    samples.append((
                        {"acc": seen_acc, "last_x": last_x, "remaining": rem,
                         "n_closed": n_closed, "cur_voice": cur,
                         "all_cands": cm.cands, "n_beats": cm.gt.beats},
                        rem, n_total, cm.length, tgt, e.d, e.dots,
                        widx.get(e.warp, 0)))
                    used |= 1 << cid
                    acc += e.dur
                    last_x = xof[cid]
                    cur = cur + [cid]
                rem = [c for c in cm.cands if not (used >> c.cid) & 1]
                samples.append((
                    {"acc": float(cm.length), "last_x": last_x,
                     "remaining": rem, "n_closed": n_closed, "cur_voice": cur,
                     "all_cands": cm.cands, "n_beats": cm.gt.beats},
                    rem, n_total, cm.length, len(rem), None, None, None))
                n_closed += 1
    return samples


def model_cache_path(cand_measures, warp_vocab, *, epochs, hidden, lr, seed, l2,
                     tag="", state_noise=0.0, reps=1, label_smooth=0.0,
                     hidden2=0, hidden3=0):
    """Canonical cache path for one training configuration.

    Shared by the numpy trainer and the GPU trainer (src/torch_train.py), so a
    model trained on CUDA is picked up transparently by the numpy pipeline.
    Hashing only a prefix of the data is NOT enough -- make_train_candidates
    builds k realisations by `for j in range(k): out.extend(...)`, so the first
    len(gt) entries are identical for every k and a8/a12 once collided here.
    """
    import hashlib
    h = hashlib.sha256(
        f"{len(cand_measures)}|{epochs}|{hidden}|{lr}|{seed}|{l2}|"
        f"{len(warp_vocab)}|{tag}|{state_noise}|{reps}|{CAND_DIM}|{STATE_DIM}|"
        f"{label_smooth}|{hidden2}|{hidden3}".encode())
    step = max(1, len(cand_measures) // 4000)
    for cm in cand_measures[::step]:
        h.update(f"{cm.length}|".encode())
        for c in cm.cands:
            h.update(f"{c.cid},{c.hint_d},{c.hint_dots},{c.hint_warp},{c.x:.4f};".encode())
    d = os.path.join(OUT, "models")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"mlp_{h.hexdigest()[:16]}.npz")


def get_model(cand_measures, warp_vocab, *, epochs=6, hidden=64, lr=2e-3,
              batch=32, seed=7, verbose=True, l2=1e-4, tag="", state_noise=0.0,
              reps=1, label_smooth=0.0, hidden2=0, hidden3=0):
    """Train once, cache to disk, reuse.

    Training is the dominant cost of an experiment, and every script used to
    retrain the identical model.  The cache is keyed on everything that changes
    the weights, and is shared with the GPU trainer.
    """
    path = model_cache_path(cand_measures, warp_vocab, epochs=epochs,
                            hidden=hidden, lr=lr, seed=seed, l2=l2, tag=tag,
                            state_noise=state_noise, reps=reps,
                            label_smooth=label_smooth, hidden2=hidden2)
    if os.path.exists(path):
        if verbose:
            print(f"  [cache] {os.path.basename(path)}")
        return MLP.load(path)
    if label_smooth:
        raise SystemExit(
            "label_smooth is implemented in the GPU trainer only; train via\n"
            "  python src/torch_train.py --label-smooth <eps>\n"
            "so the shared cache slot is filled with a matching model.")
    mlp = train(cand_measures, warp_vocab, epochs=epochs, hidden=hidden, lr=lr,
                batch=batch, seed=seed, verbose=verbose, l2=l2,
                state_noise=state_noise, reps=reps)
    mlp.save(path)
    return mlp


def production_model(train_gt, warp_vocab, hint_acc=0.85):
    """The ONE way to build the production model.

    Seven separate scripts had each grown their own training call, and several
    kept training an h64 model long after the production scale moved to h192 --
    silently reporting numbers from a different model than the headline results.
    Every script must come through here.
    """
    tr = make_train_candidates(train_gt, hint_acc, k=MODEL_AUG)
    mlp = get_model(tr, warp_vocab, epochs=MODEL_EPOCHS, hidden=MODEL_HIDDEN,
                    verbose=False, state_noise=MODEL_STATE_NOISE,
                    label_smooth=MODEL_LABEL_SMOOTH, hidden2=MODEL_HIDDEN2,
                    hidden3=MODEL_HIDDEN3)
    n_par = int(sum(getattr(mlp, k).size for k in mlp.params))
    want = expected_params(MODEL_HIDDEN, len(warp_vocab), MODEL_HIDDEN2,
                          MODEL_HIDDEN3)
    if n_par != want:
        raise SystemExit(f"model guard failed: {n_par} params, expected {want}")
    return mlp


def model_fingerprint(mlp):
    """Content hash of the weights.

    Two different h192 models appeared under one nominal configuration, so a
    reported number has to be traceable to the exact weights behind it.  Every
    metrics file records this fingerprint.
    """
    import hashlib
    hh = hashlib.sha256()
    for k in mlp.params:
        a = np.ascontiguousarray(getattr(mlp, k), dtype=np.float64)
        hh.update(k.encode())
        hh.update(a.tobytes())
    return hh.hexdigest()[:16]


def train(cand_measures, warp_vocab, *, epochs=6, hidden=64, lr=2e-3,
          batch=32, seed=7, verbose=True, l2=1e-4, state_noise=0.0, reps=1):
    mlp = MLP(seed=seed, hidden=hidden, n_warp=len(warp_vocab))
    samples = build_training_samples(cand_measures, warp_vocab,
                                      state_noise=state_noise, reps=reps,
                                      seed=seed)
    rng = random.Random(seed)
    weights = {"succ": 1.0, "d": 0.5, "dots": 0.3, "warp": 0.3}
    n = len(samples)
    if verbose:
        print(f"  training samples: {n}")
    for ep in range(epochs):
        rng.shuffle(samples)
        acc_g = None
        run_loss, steps, in_batch = 0.0, 0, 0
        for (state, rem, n_total, length, tgt, d, o, w) in samples:
            F = cand_features(rem, state, None, length)
            S = state_features(state, length, n_total, None)
            hi = hint_indices(rem, warp_vocab)
            # trust target: is the upstream hint for the consumed candidate right?
            tc = rem[tgt] if tgt < len(rem) else None
            if tc is not None and d is not None:
                tt = 1.0 if (min(max(tc.hint_d, 0), 6) == d
                             and min(max(tc.hint_dots, 0), 2) == o
                             and tuple(tc.hint_warp) == tuple(warp_vocab[w])) else 0.0
            else:
                tt = 1.0
            loss, g = mlp.sample_grad(F, S, hi, tgt, d, o, w, weights, tgt_trust=tt)
            run_loss += loss
            steps += 1
            in_batch += 1
            if acc_g is None:
                acc_g = {k: v.copy() for k, v in g.items()}
            else:
                for k in acc_g:
                    acc_g[k] += g[k]
            if in_batch == batch:
                for k in acc_g:
                    acc_g[k] /= in_batch
                for k in ("W3", "W4", "W5"):        # regularise the attribute heads
                    acc_g[k] = acc_g[k] + l2 * getattr(mlp, k)
                mlp.adam(acc_g, lr=lr)
                acc_g, in_batch = None, 0
        if acc_g is not None:
            for k in acc_g:
                acc_g[k] /= max(1, in_batch)
            mlp.adam(acc_g, lr=lr)
        if verbose:
            print(f"  epoch {ep+1}/{epochs}  mean loss {run_loss/max(1,n):.4f}")
    return mlp


# --------------------------------------------------------------------------- #
# Deterministic per-measure expansion cap.  The beam needs roughly beam*depth
# expansions before it reaches its FIRST complete solution, so a budget below
# that is a cliff rather than a graceful degradation (measured: 200 -> 4%
# perfect, 800 -> 78%).  800 keeps full quality while bounding the worst case,
# which is what the A53 budget needs.
BUDGET = 4000   # safety net only; cost is controlled by BEAM, not by truncation

# Model scale chosen by src/scale_sweep.py (dev) and confirmed by the piece-level
# cross-validation in src/cv_eval.py.  The lever is the number of noise
# realisations, NOT width: h128 a4 = 70.5% dev perfect (worse than the h64
# baseline at 71.5%), while h128 a8 = 78.0%.
#
# 5-fold piece-level CV, paired within fold against a8 (positive = better):
#     a12  +0.63 perfect, -0.19 any,  5 wins / 0 losses   <-- adopted
#     a8   (reference)
#     e60  +0.53 perfect,  3 wins / 2 losses
#     h256 +0.20 perfect,  2 wins / 3 losses
#     n0.00 -0.65 perfect, 1 win  / 4 losses  (state noise is a real gain)
# Capacity sweep on the FOLD-0 attribute diagnostic suggested 3072, but the
# honest 5-fold out-of-fold over all 50 pieces disagreed and preferred 1536:
#        13+ fold-0 attr acc     CV B perfect    CV any-field   params
#  1536        95.83%            91.55 +/- 5.76    4.34         87.5k
#  3072        96.27%            91.36 +/- 5.94    4.52        175k
# 6144        95.50%                -              -           350k
# A single fold is not a selection signal here (fold sd ~6), so the CV wins.
MODEL_HIDDEN = 1024
MODEL_HIDDEN2 = 256      # 0 = single hidden layer; >0 adds ReLU(H @ W2 + b2)
# A THIRD block was implemented and measured and is NOT used: on the fold-0 13+
# bucket it gave 96.19% (h3=64) and 94.53% (h3=128) against 96.63% for depth 2 --
# the extra ReLU shrinks the width reaching the heads from 256 to hidden3 and adds
# optimisation difficulty.  Depth is saturated at two layers.
MODEL_HIDDEN3 = 0
MODEL_EPOCHS = 30
MODEL_AUG = 12
# DAgger-style drift injection on the accumulated time.  Measured on the frozen
# test set: any-field 5.79 -> 4.53, voice match 93.23 -> 94.62 (paired McNemar
# p=0.377, direction consistent, not significant).  0.05 beats 0.02 and 0.10.
MODEL_STATE_NOISE = 0.05
# LABEL SMOOTHING on the attribute heads.  The availability/selection split showed
# the dense regime is NOT limited by whether the truth is offered (98.4-100% of
# measures have it in the option set) but by whether the exact DP PICKS it
# (87.8%).  Smoothing stops the head being over-confident in its top-1, so the
# exact-tiling constraint arbitrates near-ties.  Measured on the 5-fold
# out-of-fold over all 50 pieces: 89.21 -> 90.56 perfect, 5.91 -> 5.33 any-field,
# fold sd 7.73 -> 6.40.  Needs the GPU trainer (numpy path raises).
MODEL_LABEL_SMOOTH = 0.1
# The exact DP must be given more options once the distribution is flattened
# (flattening lowered top-12 availability from 98.4% to 95.3%); at 40 it is 100%.
DP_K_ATTR = 40
# Guard against silently running a different model than intended.  Several harness
# edits in this session failed to apply and produced numbers from the wrong
# configuration, so this is a hard error rather than a warning.
#
# The count is COMPUTED from the feature dimensions: hard-coding it just moves the
# drift one level down -- adding three input features moved 9628 -> 10204 and
# tripped a stale constant instead of being checked.
def expected_params(hidden, n_warp, hidden2=0, hidden3=0):
    from pickers import N_D, N_DOTS, STEM_FEAT_DIM
    h3 = hidden3 if hidden2 else 0
    hd = h3 if h3 else (hidden2 if hidden2 else hidden)
    return (CAND_DIM * hidden            # W1
            + STATE_DIM * hidden         # W1s
            + hidden                     # b1
            + (hidden * hidden2 + hidden2 if hidden2 else 0)   # W2, b2
            + (hidden2 * h3 + h3 if h3 else 0)                 # W2b, b2b
            + hd                         # w2
            + hd * N_D + N_D             # W3, b3
            + hd * N_DOTS + N_DOTS       # W4, b4
            + hd * n_warp + n_warp       # W5, b5
            + STATE_DIM + 1              # we, be
            + 3                          # beta
            + hd + 1                    # wt, bt
            + STEM_FEAT_DIM)            # wp (successor-only stem weights)

PASSES = [{"beam": 24, "k_attr": 5, "repair": False, "lam_iso": 0.0,
           "max_expansions": BUDGET},
          {"beam": 48, "k_attr": 14, "repair": True, "max_expansions": BUDGET},
          {"beam": 96, "k_attr": 24, "repair": True, "max_expansions": BUDGET}]


def run_arm(cand_measures, picker, warp_vocab, beam, k_attr=None, multipass=True,
            **overrides):
    """Every evaluator weight comes from src/prod.py; `overrides` is for sweeps.

    These parameters used to default to hard-coded literals, which silently
    overrode prod.PROD and made run_experiment disagree with every diagnostic
    that went through PROD (metrics_tuned2 came out bit-identical to
    metrics_final192 for exactly this reason).  Forwarding the whole dict also
    fixes the weights that were simply never passed at all, such as hint_w.
    """
    kw = dict(PROD)
    kw.update(overrides)
    if k_attr is not None:
        kw["k_attr"] = k_attr
    preds = []
    t0 = time.time()
    for cm in cand_measures:
        if multipass:
            voices, stats = solve_best(cm.cands, cm.length, picker, warp_vocab,
                                       PASSES, n_total=len(cm.cands), **kw, n_beats=cm.gt.beats)
        else:
            kw2 = dict(kw)
            kw2["beam"] = beam
            voices, stats = solve(cm.cands, cm.length, picker, warp_vocab,
                                  n_total=len(cm.cands), n_beats=cm.gt.beats, **kw2)
        preds.append((voices, stats))
    return preds, time.time() - t0


def run_greedy(cand_measures, warp_vocab):
    preds = []
    t0 = time.time()
    for cm in cand_measures:
        preds.append(greedy(cm.cands, cm.length, warp_vocab))
    return preds, time.time() - t0


def metamorphic(cand_measures, gt_measures, picker, warp_vocab, beam, seed0=999):
    """Same measures, candidates re-shuffled: is the SOLUTION order-invariant?"""
    alt = make_candidates(gt_measures, seed0, hint_acc=1.0)   # same geometry, new order
    base = make_candidates(gt_measures, seed0, hint_acc=1.0)
    differ = 0
    for cm_a, cm_b in zip(base, alt):
        va, _ = solve(cm_a.cands, cm_a.length, picker, warp_vocab, beam=beam,
                      n_total=len(cm_a.cands), n_beats=cm.gt.beats)
        vb, _ = solve(cm_b.cands, cm_b.length, picker, warp_vocab, beam=beam,
                      n_total=len(cm_b.cands), n_beats=cm.gt.beats)
        if M.structure_key(cm_a, va) != M.structure_key(cm_b, vb):
            differ += 1
    return differ, len(base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hint-acc", type=float, default=0.85)
    ap.add_argument("--epochs", type=int, default=MODEL_EPOCHS)
    ap.add_argument("--beam", type=int, default=BUDGET and PROD["beam"])
    ap.add_argument("--big-beam", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="cap measures per split (debug)")
    ap.add_argument("--voice-weight", type=float, default=0.02)
    ap.add_argument("--dev-weight", type=float, default=0.15)
    ap.add_argument("--augment", type=int, default=MODEL_AUG)
    ap.add_argument("--tau-weight", type=float, default=1.0)
    ap.add_argument("--iso-weight", type=float, default=1.0)
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("== building corpus ==")
    allm = (C.load_freeze()[0] or C.build_corpus())
    train_gt, dev_gt, test_gt, (tp, dp, ep) = C.split_by_piece(allm)
    if args.limit:
        train_gt, dev_gt, test_gt = train_gt[:args.limit], dev_gt[:args.limit], test_gt[:args.limit]
    print(f"train={len(train_gt)} dev={len(dev_gt)} test={len(test_gt)}")
    print(f"test pieces ({len(ep)}): {ep}")

    warp_vocab = V.warp_vocab(allm)
    print(f"warp vocab ({len(warp_vocab)}): {warp_vocab}")

    tr = make_train_candidates(train_gt, args.hint_acc, k=args.augment)
    dv = make_candidates(dev_gt, 22, args.hint_acc)
    te = make_candidates(test_gt, 33, args.hint_acc)

    print(f"== training learned picker (hint_acc={args.hint_acc}) ==")
    t0 = time.time()
    mlp = production_model(train_gt, warp_vocab, args.hint_acc)
    n_par = int(sum(getattr(mlp, k).size for k in mlp.params))
    print(f"  params {n_par} (guard ok)  fingerprint {model_fingerprint(mlp)}")
    print(f"  train+load time {time.time()-t0:.1f}s")

    rule = RulePicker(warp_vocab)
    learned = LearnedPicker(mlp, warp_vocab)

    arms = {}
    print("== greedy ==")
    p, dt = run_greedy(te, warp_vocab)
    arms["greedy (no search)"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    print(f"== rule picker + search (beam {args.beam}) ==")
    p, dt = run_rule = run_arm(te, rule, warp_vocab, args.beam)
    arms["rule + search"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    print(f"== learned picker + search (beam {args.beam}) ==")
    p, dt = run_arm(te, learned, warp_vocab, args.beam,
                            voice_w=args.voice_weight, dev_w=args.dev_weight)
    arms["learned + search"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    hybrid = HybridPicker(learned, warp_vocab)
    print(f"== hybrid (learned successor + hint attributes, beam {args.beam}) ==")
    p, dt = run_arm(te, hybrid, warp_vocab, args.beam,
                            voice_w=args.voice_weight, dev_w=args.dev_weight)
    arms["hybrid + search"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    print(f"== learned picker + search (big beam {args.big_beam}) ==")
    p, dt = run_arm(te, learned, warp_vocab, args.big_beam,
                            voice_w=args.voice_weight, dev_w=args.dev_weight)
    arms[f"learned + search (big beam)"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    hintrep = HintRepairPicker(learned, warp_vocab)
    print("== hint-repair (learned successor + hint time + targeted repair) ==")
    p, dt = run_arm(te, hintrep, warp_vocab, 32, multipass=False)
    arms["hint_repair + PROD"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    print("== PROD weights (single pass) ==")
    p, dt = run_arm(te, learned, warp_vocab, 16, multipass=False)
    arms["learned + PROD"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")
    p, dt = run_arm(te, learned, warp_vocab, 32, multipass=True)
    arms["learned + PROD multipass"] = (M.evaluate(te, p), dt, p)
    print(f"  {dt:.1f}s")

    rows = [(k, v[0]) for k, v in arms.items()]
    table = M.fmt(rows)
    print("\n" + table + "\n")

    # subset breakdown: monophonic vs polyphonic
    sub = {}
    for name, idx in (("mono", [i for i, c in enumerate(te) if c.gt.n_voices == 1]),
                      ("poly", [i for i, c in enumerate(te) if c.gt.n_voices > 1])):
        if not idx:
            continue
        cm_sub = [te[i] for i in idx]
        srows = []
        for k, (_m, _dt, preds) in arms.items():
            srows.append((k, M.evaluate(cm_sub, [preds[i] for i in idx])))
        sub[name] = srows
        print(f"--- {name} ({len(idx)} measures) ---")
        print(M.fmt(srows) + "\n")

    # fairness subset: only measures where EVERY arm produced a valid score
    # (a score that exactly tiles each voice).  Without this, a chain that never
    # tries to fill the measure looks better than one that is held to validity.
    allnames = list(arms)
    valid_idx = [i for i in range(len(te))
                 if all(arms[a][2][i][1].get("valid") for a in allnames)]
    if valid_idx:
        cm_sub = [te[i] for i in valid_idx]
        srows = [(k, M.evaluate(cm_sub, [arms[k][2][i] for i in valid_idx]))
                 for k in allnames]
        sub["valid_only"] = srows
        print(f"--- valid_only ({len(valid_idx)} measures where ALL arms are valid) ---")
        print(M.fmt(srows) + "\n")

    print("== metamorphic: candidate-order invariance ==")
    meta = {}
    for name, picker in (("rule", rule), ("learned", learned)):
        d, n = metamorphic(te[:120], test_gt[:120], picker, warp_vocab, args.beam)
        meta[name] = {"differ": d, "of": n, "pct": 100.0 * d / max(1, n)}
        print(f"  {name}: {d}/{n} measures change when input order changes ({meta[name]['pct']:.1f}%)")

    res = {
        "config": vars(args),
        "split": {"train": len(train_gt), "dev": len(dev_gt), "test": len(test_gt),
                  "test_pieces": ep},
        "warp_vocab": [list(w) for w in warp_vocab],
        "model_fingerprint": model_fingerprint(mlp),
        "model_params": int(sum(getattr(mlp, k).size for k in mlp.params)),
        "main": {k: v[0] for k, v in arms.items()},
        "seconds": {k: v[1] for k, v in arms.items()},
        "subsets": {k: {n: m for n, m in v} for k, v in sub.items()},
        "metamorphic": meta,
    }
    with open(os.path.join(OUT, f"metrics_{args.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, f"table_{args.tag}.txt"), "w", encoding="utf-8") as fh:
        fh.write(table + "\n\n")
        for k, v in sub.items():
            fh.write(f"--- {k} ---\n" + M.fmt(v) + "\n\n")
        fh.write(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\nmodel fingerprint: {}  ({} params)".format(
        res["model_fingerprint"], res["model_params"]))
    print(f"wrote {OUT}/metrics_{args.tag}.json and table_{args.tag}.txt")


if __name__ == "__main__":
    main()




































