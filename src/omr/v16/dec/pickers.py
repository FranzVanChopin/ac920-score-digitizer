"""Three decoding policies behind one interface.

  GreedyDecoder  - no search, x-order single chain      (the "do nothing" floor)
  RulePicker     - hand-written local rules              (baseline, same search)
  LearnedPicker  - small numpy MLP, teacher-forced       (the proposal)

Every picker answers, for the remaining candidates of a measure:
    succ          : log-probabilities over [cand_0 .. cand_{n-1}, EOS]
    d, dots, warp : log-probabilities for each candidate

Design note: the attribute heads carry a LEARNABLE bias toward the upstream
hint.  Measurement showed the raw head was worse than simply copying the hint,
so the model must be able to start from "trust the hint" and learn only when to
deviate.
"""
from __future__ import annotations

import math

import numpy as np

from visual import duration_of, hint_duration

STEM_FEAT_DIM = 3
# OFF by default.  Three attempts to give the model a partition feature all
# REGRESSED the 5-fold CV:
#     pitch in the shared trunk ... -1.65 (Round 41)
#     stem  in the shared trunk ... -1.38 (Round 46)
#     stem  on the successor-only path ... -0.89 (Round 48)
# and every one of them also worsened any_field_error_pct, so this is not a matter
# of where the feature is attached -- the 31-dimensional feature set is at a local
# optimum and extra per-candidate inputs cost more than they give.  The machinery is
# kept, switched off, so the experiment can be re-run cheaply.
USE_STEM_FEATURES = False
CAND_DIM = 31   # 24 + 4 leave-one-out residuals + 3 counterfactuals
STATE_DIM = 6
N_D = 7
N_DOTS = 3
BETA_MAX = 12.0
# TRUST FLOOR (inference-time, no retraining).  Measured: on the worst piece the
# trust head reports 74.4% while the hints there are actually 85.4% correct, so the
# hint anchor is weakened exactly where the model is least able to replace it --
# `keep` (P(model right | hint right)) collapses to 75.6% against 98.5% elsewhere.
# max(trust, TRUST_FLOOR) floors the anchor without touching the weights.
TRUST_FLOOR = 0.0


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def cand_features(cands, state, warp_index, length):
    n = len(cands)
    last_x = state["last_x"]
    acc = state["acc"]
    # METRE (tested, reverted).  The time signature is exact for every measure in
    # the corpus (length == beats * unit, checked on all 4100) and is the
    # strongest music-theoretic constraint available, so three metre features were
    # added and the model retrained.  Result: per-event accuracy unchanged
    # (13+ bucket 95.99% -> 95.86%) and the 5-fold out-of-fold total moved
    # 89.21 -> 88.97, i.e. indistinguishable within a fold sd of ~8.  The simpler
    # model is kept.  The n_beats plumbing stays because it is correct and will be
    # needed if metre ever enters the DP cost instead of the features.
    _ = state["n_beats"]
    rem_ticks = length - acc
    xs = [c.x for c in cands]
    order = np.argsort(np.argsort(xs)) if n else np.zeros(0)
    # global tiling context: how well do the remaining hints add up?
    sum_hint = sum(hint_duration(c) for c in cands)
    misfit = (sum_hint - rem_ticks) / max(1.0, length)
    # x-ordered neighbourhood: a tuplet is defined by its beam group, so the
    # model must be able to see the events immediately around it.
    xs_sorted = sorted(range(n), key=lambda j: cands[j].x)
    rank_of = {j: r for r, j in enumerate(xs_sorted)}
    # LEAVE-ONE-OUT tiling residual.  `misfit` above includes the candidate
    # itself, so it cannot say WHICH event is the wrong one -- it only says the
    # measure is off.  Subtracting the candidate's own hint answers the question
    # that actually matters: "if everything else already fits, this is the
    # culprit".  Dense measures are exactly where the model is weakest
    # (96.4% vs 99.6% division accuracy), so a signal that localises the error is
    # worth more there than anywhere else.
    hint_of = [hint_duration(c) for c in cands]
    staff_sum = {}
    for c, hv in zip(cands, hint_of):
        staff_sum[c.staff] = staff_sum.get(c.staff, 0.0) + hv
    # NEIGHBOUR-CLASS COUNTERFACTUAL.  The upstream error is always the same
    # shape: the division class is reported one step off (`hd = e.d +- 1`), which
    # is exactly what makes dense measures fail -- with ~19 events the measure is
    # perfect only if EVERY event is right, so per-event accuracy must approach
    # 99.7% there.  Rather than hope the model infers this, evaluate the
    # hypothesis directly: substitute each neighbouring class and see whether the
    # tiling residual gets better.
    ones = []
    for c in cands:
        hdd = min(max(c.hint_d, 0), 6)
        alts = []
        for delta in (-1, 1):
            dd = hdd + delta
            if 0 <= dd <= 6:
                alts.append((delta, duration_of(dd, min(max(c.hint_dots, 0), 2),
                                                c.hint_warp)))
        ones.append(alts)
    F = np.zeros((n, CAND_DIM), dtype=np.float64)
    for i, c in enumerate(cands):
        hd = hint_duration(c)
        dx = c.x - last_x
        frac = hd / max(rem_ticks, 1.0)
        r = rank_of[i]
        pv = cands[xs_sorted[r - 1]] if r > 0 else None
        nx = cands[xs_sorted[r + 1]] if r + 1 < n else None
        # A tuplet is a property of a BEAM GROUP, not of one event, so look at
        # the whole measure's x-neighbourhood rather than only what is left.
        allc = state.get("all_cands") or cands
        win = []
        if allc:
            idxs = sorted(range(len(allc)), key=lambda j: allc[j].x)
            pos = next((q for q, j in enumerate(idxs) if allc[j].cid == c.cid), None)
            if pos is not None:
                win = [allc[idxs[q]] for q in range(max(0, pos - 2), min(len(idxs), pos + 3))
                       if q != pos]
        grp_n = len(win)
        grp_w = sum(1 for z in win if z.hint_warp != (1, 1))
        grp_r = (sum(z.hint_warp[1] / max(1, z.hint_warp[0]) for z in win) / grp_n
                 if grp_n else 0.0)
        grp_s = (sum(1 for z in win if z.staff == c.staff) / grp_n) if grp_n else 0.0

        def ratio(z):
            return 0.0 if z is None else z.hint_warp[1] / max(1, z.hint_warp[0])

        # how close does each neighbour class come to closing the tiling gap?
        base = abs(sum_hint - rem_ticks)
        best_gap, best_delta, n_better = abs(sum_hint - rem_ticks), 0, 0
        for delta, alt in ones[i]:
            gap = abs(sum_hint - hd + alt - rem_ticks)
            if gap < best_gap - 1e-9:
                best_gap, best_delta = gap, delta
        for delta, alt in ones[i]:
            if abs(sum_hint - hd + alt - rem_ticks) < base - 1e-9:
                n_better += 1

        F[i] = (
            c.x / 1000.0,
            c.staff / 4.0,
            1.0 if c.is_rest else 0.0,
            c.hint_d / 6.0,
            c.hint_dots / 2.0,
            c.hint_warp[1] / max(1, c.hint_warp[0]),
            dx / 1000.0,
            abs(dx) / 1000.0,
            hd / length,
            (rem_ticks - hd) / length,
            (order[i] / max(1, n - 1)) if n else 0.0,
            sum_hint / max(1.0, length),
            misfit,
            min(frac, 4.0) / 4.0,
            (c.x - pv.x) / 1000.0 if pv is not None else 1.0,
            (nx.x - c.x) / 1000.0 if nx is not None else 1.0,
            1.0 if (pv is not None and pv.staff == c.staff) else 0.0,
            1.0 if (nx is not None and nx.staff == c.staff) else 0.0,
            ratio(pv),
            ratio(nx),
            grp_w / 4.0,
            grp_r,
            grp_s,
            1.0 if c.hint_warp != (1, 1) else 0.0,
            (sum_hint - hd) / max(1.0, length),
            (sum_hint - hd - rem_ticks) / max(1.0, length),
            (staff_sum.get(c.staff, 0.0) - hd) / max(1.0, length),
            (staff_sum.get(c.staff, 0.0) - hd - rem_ticks) / max(1.0, length),
            best_gap / max(1.0, length),
            best_delta / 2.0,
            n_better / 2.0,
        )
    return F


def stem_features(cands, state):
    """Per-candidate STEM features used ONLY by the successor head.

    Kept out of cand_features on purpose.  Two experiments added a partition
    feature to the shared trunk and both REGRESSED the 5-fold CV even though the
    feature itself was strong:
        pitch in cand_features  -> -1.65  (Round 41)
        stem  in cand_features  -> -1.38  (Round 46)
    In both cases ny_field_error_pct jumped (3.99 -> 5.47 / 6.21), i.e. the value
    heads were damaged.  Measured strength of stem: exact within-staff partition
    95.53% vs 8.03% for chance, so it is worth wiring in properly -- through its own
    additive term on the successor logits, leaving every value head untouched.
    """
    n = len(cands)
    last = 0
    cur = state.get("cur_voice") or []
    if cur:
        want = cur[-1]
        for z in (state.get("all_cands") or cands):
            if z.cid == want:
                last = int(getattr(z, "stem", 0))
                break
    P = np.zeros((n, STEM_FEAT_DIM), dtype=np.float64)
    for i, c in enumerate(cands):
        st = int(getattr(c, "stem", 0))
        P[i, 0] = float(st)
        P[i, 1] = 1.0 if st != 0 else 0.0
        P[i, 2] = (1.0 if st == last else -1.0) if (st != 0 and last != 0) else 0.0
    return P


def state_features(state, length, n_total, warp_index):
    rem = length - state["acc"]
    n_rem = len(state["remaining"])
    return np.array([
        state["acc"] / length,
        rem / length,
        n_rem / max(1, n_total),
        state["n_closed"] / 4.0,
        len(state["cur_voice"]) / 8.0,
        1.0,
    ], dtype=np.float64)


def hint_indices(cands, warp_vocab):
    """Index of the upstream hint inside each head's class space."""
    n = len(cands)
    di = np.zeros(n, dtype=np.int64)
    oi = np.zeros(n, dtype=np.int64)
    wi = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(cands):
        di[i] = min(max(c.hint_d, 0), N_D - 1)
        oi[i] = min(max(c.hint_dots, 0), N_DOTS - 1)
        try:
            wi[i] = warp_vocab.index(c.hint_warp)
        except ValueError:
            wi[i] = 0
    return di, oi, wi


# --------------------------------------------------------------------------- #
# tiny MLP with Adam (numpy only; no torch)
# --------------------------------------------------------------------------- #
class MLP:
    def __init__(self, seed=7, hidden=64, n_warp=3, hidden2=0, hidden3=0):
        rng = np.random.default_rng(seed)
        h = hidden
        s = lambda a, b: rng.normal(0.0, math.sqrt(2.0 / (a + b)), size=(a, b))
        self.H = h
        # DEPTH.  Width was swept 192 -> 6144 and peaks at 1536; a second ReLU
        # block then gave +0.65 CV points, largest on the hardest fold, so a third
        # is worth testing -- it costs only (hidden2 x hidden3) MACs, an order of
        # magnitude less than the second block.  Every layer id 0 keeps the
        # original single-hidden-layer net so all cached models still load.
        self.hidden2 = int(hidden2)
        self.hidden3 = int(hidden3) if self.hidden2 else 0
        self.n_warp = n_warp
        self.W1 = s(CAND_DIM, h)
        self.W1s = s(STATE_DIM, h)
        self.b1 = np.zeros(h)
        hd = self.hidden2 if self.hidden2 else h
        if self.hidden2:
            self.W2 = s(h, self.hidden2)
            self.b2 = np.zeros(self.hidden2)
        if self.hidden3:
            # `W3`/`b3` are the division head, so the third block uses `W2b`.
            self.W2b = s(self.hidden2, self.hidden3)
            self.b2b = np.zeros(self.hidden3)
            hd = self.hidden3
        self.w2 = s(hd, 1).ravel()
        self.W3 = s(hd, N_D)
        self.b3 = np.zeros(N_D)
        self.W4 = s(hd, N_DOTS)
        self.b4 = np.zeros(N_DOTS)
        self.W5 = s(hd, n_warp)
        self.b5 = np.zeros(n_warp)
        self.we = s(STATE_DIM, 1).ravel()
        self.be = np.zeros(1)
        # learnable anchor toward the upstream hint, one per attribute head
        # start close to "copy the upstream hint"; the data must argue its way out
        self.beta = np.array([2.5, 2.5, 2.5])
        # additive successor-only weights for the stem features.  Starts at
        # ZERO, so with no training signal the model is bit-identical to the
        # one that produced 92.83 / 3.99 / 96.22.
        self.wp = np.zeros(STEM_FEAT_DIM)
        # False reproduces the pre-stem model bit-for-bit
        self.use_stem = True
        # TRUST head: per-candidate probability that the upstream hint is right.
        # A constant anchor protects the 85% clean events but stays too timid on
        # the corrupted 15%; predicting trust per event lets the model override
        # exactly where the hint is wrong without damaging the clean majority.
        self.wt = s(hd, 1).ravel()
        self.bt = np.zeros(1)
        self.params = ["W1", "W1s", "b1"]
        if self.hidden2:
            self.params += ["W2", "b2"]
        if self.hidden3:
            self.params += ["W2b", "b2b"]
        self.params += ["w2", "W3", "b3", "W4", "b4",
                        "W5", "b5", "we", "be", "beta", "wt", "bt", "wp"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    # ---- forward ----
    def forward(self, F, S, hi=None, Fp=None):
        Z = F @ self.W1 + S @ self.W1s + self.b1
        Hh = np.maximum(Z, 0.0)
        if self.hidden2:
            Hh = np.maximum(Hh @ self.W2 + self.b2, 0.0)
        if self.hidden3:
            Hh = np.maximum(Hh @ self.W2b + self.b2b, 0.0)
        # STEM goes ONLY here: the successor logits decide the voice
        # partition, and dlog/olog/wlog below never see it.
        succ = Hh @ self.w2
        if Fp is not None:
            succ = succ + Fp @ self.wp
        eos = float(S @ self.we + self.be[0])
        dlog = Hh @ self.W3 + self.b3
        olog = Hh @ self.W4 + self.b4
        wlog = Hh @ self.W5 + self.b5
        trust = None
        if hi is not None:
            di, oi, wi = hi
            if len(di):
                r = np.arange(len(di))
                # per-candidate trust gates the hint anchor on every head
                _z = Hh @ self.wt + self.bt[0]
                # numerically stable logistic; np.exp overflows for large |z|
                trust = np.where(_z >= 0, 1.0 / (1.0 + np.exp(-np.abs(_z))),
                                 np.exp(-np.abs(_z)) / (1.0 + np.exp(-np.abs(_z))))
                # TRUST FLOOR (inference-time, no retraining).  Measured: on the
                # worst piece the trust head reports 74.4% while the hints there are
                # actually 85.4% correct, so the anchor is weakened exactly where
                # the model cannot replace it -- `keep` collapses to 75.6% against
                # 98.5% elsewhere.  Floors the anchor without touching weights.
                _t = trust if TRUST_FLOOR <= 0.0 else np.maximum(trust, TRUST_FLOOR)
                dlog[r, di] += self.beta[0] * _t
                olog[r, oi] += self.beta[1] * _t
                wlog[r, wi] += self.beta[2] * _t
        return Z, Hh, succ, eos, dlog, olog, wlog, trust

    @staticmethod
    def _logsoftmax(x):
        m = np.max(x)
        z = x - m
        return z - math.log(float(np.sum(np.exp(z))) + 1e-300)

    @staticmethod
    def _rowls(x):
        m = x.max(axis=1, keepdims=True)
        z = x - m
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True) + 1e-300)

    # ---- one sample: loss and gradients ----
    def sample_grad(self, F, S, hi, tgt_succ, tgt_d, tgt_dots, tgt_warp, weights,
                    tgt_trust=None, trust_w=0.3):
        if self.hidden2:
            raise SystemExit(
                "the numpy backward pass does not implement the second hidden "
                "layer; train depth-2 models with src/torch_train.py, whose "
                "forward_torch mirrors MLP.forward and is checked by --verify.")
        Z, Hh, succ, eos, dlog, olog, wlog, trust = self.forward(F, S, hi)
        n = F.shape[0]
        allsucc = np.concatenate([succ, np.array([eos])])
        ls = self._logsoftmax(allsucc)
        loss = -weights["succ"] * ls[tgt_succ]

        g = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        dall = np.exp(ls)
        dall[tgt_succ] -= 1.0
        dall *= weights["succ"]
        g["w2"] += Hh.T @ dall[:n]
        g["we"] += S * dall[n]
        g["be"] += np.array([dall[n]])
        dH = np.outer(dall[:n], self.w2)

        di, oi, wi = hi
        k = tgt_succ
        dtrust = 0.0
        for wn, logits, tgt, W, hidx, bi in (
                ("d", dlog, tgt_d, self.W3, di, 0),
                ("dots", olog, tgt_dots, self.W4, oi, 1),
                ("warp", wlog, tgt_warp, self.W5, wi, 2)):
            if tgt is None:
                continue
            ll = self._logsoftmax(logits[k])
            loss += -weights[wn] * float(ll[tgt])
            ga = np.zeros_like(logits)
            row = np.exp(ll)
            row[tgt] -= 1.0
            ga[k] = row * weights[wn]
            g["W3" if wn == "d" else ("W4" if wn == "dots" else "W5")] += Hh.T @ ga
            g["b3" if wn == "d" else ("b4" if wn == "dots" else "b5")] += ga.sum(axis=0)
            # d(loss)/d(beta) picks up the trust gate of the consumed candidate
            tv = float(trust[k]) if trust is not None else 1.0
            g["beta"][bi] += weights[wn] * float(row[hidx[k]]) * tv
            # ...and the same term flows back into the trust head
            dtrust += weights[wn] * float(row[hidx[k]]) * self.beta[bi]
            dH += ga @ W.T

        # trust is defined over candidates only; there is no trust for the EOS
        # action, so the block is skipped when the target is EOS (k == n).
        if trust is not None and tgt_trust is not None and k < len(trust):
            tv = float(trust[k])
            tv = min(max(tv, 1e-6), 1 - 1e-6)
            loss += -trust_w * (tgt_trust * math.log(tv)
                                + (1 - tgt_trust) * math.log(1 - tv))
            # BCE gradient plus the path through the gated hint anchor
            dt = (tv - tgt_trust) * trust_w + dtrust * tv * (1.0 - tv)
            vec = np.zeros(n)
            vec[k] = dt
            g["wt"] += Hh.T @ vec
            g["bt"] += np.array([vec.sum()])
            dH += np.outer(vec, self.wt)

        dZ = dH * (Z > 0)
        g["W1"] += F.T @ dZ
        g["W1s"] += S[:, None] * dZ.sum(axis=0)[None, :]
        g["b1"] += dZ.sum(axis=0)
        return loss, g

    def adam(self, grads, lr=2e-3):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            gk = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * (gk * gk)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))
        self.beta = np.clip(self.beta, 0.0, BETA_MAX)

    # ---- persistence: training is the other half of the iteration cost ----
    def save(self, path):
        np.savez(path, H=np.array([self.H]), n_warp=np.array([self.n_warp]),
                 hidden2=np.array([self.hidden2]),
                 hidden3=np.array([self.hidden3]),
                 **{k: getattr(self, k) for k in self.params})

    def to_fp32(self):
        """Cast every parameter to float32, in place.

        Training runs in float32 (torch); the numpy model was float64 purely because
        numpy defaults to it, which doubles the shipped footprint (2392 -> 1196 KiB)
        for no accuracy reason.  Inference-only: the values are the same weights.
        """
        for k in self.params:
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float32))
        return self

    @staticmethod
    def load(path):
        z = np.load(path)
        h2 = int(z["hidden2"][0]) if "hidden2" in z.files else 0
        h3 = int(z["hidden3"][0]) if "hidden3" in z.files else 0
        m = MLP(hidden=int(z["H"][0]), n_warp=int(z["n_warp"][0]), hidden2=h2,
                hidden3=h3)
        for k in m.params:
            # BACKWARD COMPATIBILITY.  A model saved before a new parameter existed
            # must still load, with the parameter at its constructor value (zeros for
            # wp), which is exactly the older model.  Without this an improved
            # architecture cannot read any existing cache.
            setattr(m, k, z[k] if k in z.files else getattr(m, k))
        return m

    # ---- inference ----
    def succ_logprobs(self, F, S, Fp=None):
        _Z, _Hh, succ, eos, _d, _o, _w, _t = self.forward(F, S, Fp=Fp)
        return self._logsoftmax(np.concatenate([succ, np.array([eos])]))

    def attr_logprobs_all(self, F, S, hi):
        _Z, _Hh, _s, _e, dlog, olog, wlog, _t = self.forward(F, S, hi)
        return self._rowls(dlog), self._rowls(olog), self._rowls(wlog)

    def attr_logprobs_trust(self, F, S, hi):
        """Attribute distributions plus the per-candidate hint-trust probability."""
        _Z, _Hh, _s, _e, dlog, olog, wlog, t = self.forward(F, S, hi)
        return self._rowls(dlog), self._rowls(olog), self._rowls(wlog), t


# --------------------------------------------------------------------------- #
# rule policy
# --------------------------------------------------------------------------- #
class RulePicker:
    name = "rule"

    def __init__(self, warp_vocab):
        self.warp_vocab = warp_vocab

    def succ_logprobs(self, cands, state, length, n_total):
        ln = np.full(len(cands) + 1, -20.0)
        if not cands:
            ln[-1] = 0.0
            return ln
        last_x = state["last_x"]
        rem = length - state["acc"]
        for i, c in enumerate(cands):
            dx = c.x - last_x
            score = -abs(dx) / 100.0
            if dx < -1e-9:
                score -= 4.0
            hd = hint_duration(c)
            score -= abs(rem - hd) / max(1.0, length) * 3.0
            ln[i] = score
        ln[-1] = 0.0 if abs(rem) < 1e-6 else -6.0
        m = ln.max()
        z = np.exp(ln - m)
        return (ln - m) - math.log(float(z.sum()))

    def attr_logprobs(self, cands, state, length, n_total):
        """Soft distributions centred on the upstream hint, one row per candidate."""
        n = len(cands)
        p_hint = 0.70
        nw = len(self.warp_vocab)
        D = np.full((n, N_D), np.log((1.0 - p_hint) / (N_D - 1)))
        O = np.full((n, N_DOTS), np.log((1.0 - p_hint) / (N_DOTS - 1)))
        W = np.full((n, nw), np.log((1.0 - p_hint) / max(1, nw - 1)))
        for i, c in enumerate(cands):
            D[i, min(max(c.hint_d, 0), N_D - 1)] = np.log(p_hint)
            O[i, min(max(c.hint_dots, 0), N_DOTS - 1)] = np.log(p_hint)
            try:
                W[i, self.warp_vocab.index(c.hint_warp)] = np.log(p_hint)
            except ValueError:
                W[i, 0] = np.log(p_hint)
        return D, O, W


# --------------------------------------------------------------------------- #
# learned policy
# --------------------------------------------------------------------------- #
class EnsemblePicker:
    """Average the log-probabilities of several independently seeded models.

    Round 17 localised the whole dense-regime loss to SELECTION: the true value is
    among the offered options in 100% of measures, but the exact DP -- which is
    optimal given the costs -- picks it only ~90% of the time.  So the costs are
    the error term, and averaging independent models is the standard way to reduce
    exactly that variance.  Inference cost scales linearly in the ensemble size,
    which matters for the FPGA budget, so the gain is measured before it is
    adopted.
    """

    name = "ensemble"

    def __init__(self, mlps, warp_vocab):
        self.mlps = list(mlps)
        self.mlp = self.mlps[0]          # some diagnostics read .mlp
        self.warp_vocab = warp_vocab

    def _stack(self, cands, state, length, n_total):
        F = cand_features(cands, state, None, length)
        S = state_features(state, length, n_total, None)
        hi = hint_indices(cands, self.warp_vocab)
        return F, S, hi

    def succ_logprobs(self, cands, state, length, n_total):
        if not cands:
            return np.array([0.0])
        F, S, _hi = self._stack(cands, state, length, n_total)
        return np.mean([m.succ_logprobs(F, S) for m in self.mlps], axis=0)

    def attr_logprobs(self, cands, state, length, n_total):
        F, S, hi = self._stack(cands, state, length, n_total)
        Ds, Os, Ws = [], [], []
        for m in self.mlps:
            D, O, W, _t = m.attr_logprobs_trust(F, S, hi)
            Ds.append(D); Os.append(O); Ws.append(W)
        return np.mean(Ds, axis=0), np.mean(Os, axis=0), np.mean(Ws, axis=0)

    def logprobs_both(self, cands, state, length, n_total):
        F, S, hi = self._stack(cands, state, length, n_total)
        succ = np.mean([m.succ_logprobs(F, S) for m in self.mlps], axis=0)
        Ds, Os, Ws, Ts = [], [], [], []
        for m in self.mlps:
            D, O, W, t = m.attr_logprobs_trust(F, S, hi)
            Ds.append(D); Os.append(O); Ws.append(W); Ts.append(t)
        return (succ, np.mean(Ds, axis=0), np.mean(Os, axis=0),
                np.mean(Ws, axis=0), np.mean(Ts, axis=0))


class LearnedPicker:
    name = "learned"

    def __init__(self, mlp: MLP, warp_vocab):
        self.mlp = mlp
        self.warp_vocab = warp_vocab

    def succ_logprobs(self, cands, state, length, n_total):
        if not cands:
            return np.array([0.0])
        F = cand_features(cands, state, None, length)
        S = state_features(state, length, n_total, None)
        # stem enters through its own additive term on the successor logits only
        P = stem_features(cands, state) if USE_STEM_FEATURES else None
        return self.mlp.succ_logprobs(F, S, Fp=P)

    def attr_logprobs(self, cands, state, length, n_total):
        F = cand_features(cands, state, None, length)
        S = state_features(state, length, n_total, None)
        hi = hint_indices(cands, self.warp_vocab)
        return self.mlp.attr_logprobs_all(F, S, hi)

    def logprobs_both(self, cands, state, length, n_total):
        """Successor and attribute distributions from ONE feature pass.

        The search asked for both at every expansion, so the (non-trivial)
        feature construction ran twice on identical inputs.
        """
        F = cand_features(cands, state, None, length)
        S = state_features(state, length, n_total, None)
        hi = hint_indices(cands, self.warp_vocab)
        succ = self.mlp.succ_logprobs(F, S)
        D, O, W, trust = self.mlp.attr_logprobs_trust(F, S, hi)
        return succ, D, O, W, trust




class HybridPicker:
    """Learned successor policy + upstream-hint attributes.

    Measurement showed the learned attribute head generalises worse than simply
    copying the hint (test config accuracy 76% vs hint 85%), while the successor
    head is essentially perfect (top-1 99.4%).  This arm keeps the part that
    works and lets the search's structural constraints -- exact tiling and the
    reachability prune -- do the attribute repair.
    """
    name = "hybrid"

    def __init__(self, learned, warp_vocab, p_hint=0.70):
        self.learned = learned
        self.warp_vocab = warp_vocab
        self._rule = RulePicker(warp_vocab)
        self._rule_p = p_hint

    def succ_logprobs(self, cands, state, length, n_total):
        return self.learned.succ_logprobs(cands, state, length, n_total)

    def attr_logprobs(self, cands, state, length, n_total):
        return self._rule.attr_logprobs(cands, state, length, n_total)



class HintRepairPicker:
    """Learned successor policy + upstream hints as the PRIMARY time value.

    Why not the learned attribute head: the upstream corruption is one-shot, so
    the hint's three heads are wrong together and its joint accuracy (84.9%) is
    higher than any per-head combination of the model (~79.6%).  The model's job
    is therefore the voice/order decision (top-1 99.4%), while attributes start
    from the hint and are repaired only by structure: the exact-tiling
    requirement plus a musically targeted neighbourhood of the hinted duration
    (double / half / dotted / triplet ratio), which is where the upstream errors
    actually live.
    """
    name = "hint_repair"
    always_wide = True

    def __init__(self, learned, warp_vocab):
        self.learned = learned
        self.warp_vocab = warp_vocab

    def succ_logprobs(self, cands, state, length, n_total):
        return self.learned.succ_logprobs(cands, state, length, n_total)

    def attr_logprobs(self, cands, state, length, n_total):
        n = len(cands)
        nw = len(self.warp_vocab)
        D = np.full((n, N_D), np.log(0.5 / (N_D - 1)))
        O = np.full((n, N_DOTS), np.log(0.5 / (N_DOTS - 1)))
        W = np.full((n, nw), np.log(0.5 / max(1, nw - 1)))
        for i, c in enumerate(cands):
            D[i, min(max(c.hint_d, 0), N_D - 1)] = np.log(0.5)
            O[i, min(max(c.hint_dots, 0), N_DOTS - 1)] = np.log(0.5)
            try:
                W[i, self.warp_vocab.index(c.hint_warp)] = np.log(0.5)
            except ValueError:
                W[i, 0] = np.log(0.5)
        return D, O, W









