"""Pairwise voice-continuation scorer, trained per fold, used to re-rank beam successors.

The relation "are these two notes in the same voice" is measured to be highly
informative (AUC 0.961; whole-partition recovery 83.54% against the shipped MLP's
75.21%), but it must not be bolted on as per-candidate features -- that was tried three
times and always regressed (pitch -1.65, stem -1.38, stem-signature-only -0.89), because
the shared trunk also feeds the value heads.

So it enters the search as a RE-RANKING term on the successor logits, which leaves the
value heads and the exact-tiling mask untouched.

Training uses piece-level folds: to decode fold f, the scorer is fitted on the pieces of
the other folds only.
"""
from __future__ import annotations

import collections

import numpy as np

FEATS = 10


def _pair_feat(a, b):
    return np.array([
        (b.x - a.x) / 100.0,
        (b.midi - a.midi) / 12.0 if (a.midi > 0 and b.midi > 0) else 0.0,
        1.0 if b.staff == a.staff else 0.0,
        float(a.stem),
        float(b.stem),
        float(a.stem) * float(b.stem),
        1.0 if (a.stem != 0 and a.stem == b.stem) else 0.0,
        1.0 if (a.stem != 0 and b.stem != 0 and a.stem != b.stem) else 0.0,
        1.0 if b.x >= a.x else 0.0,
        1.0 if (a.is_rest or b.is_rest) else 0.0,
    ], dtype=np.float64)


class PairSuccModel:
    """Logistic regression on ordered candidate pairs; scores voice continuation."""

    def __init__(self):
        self.mu = None
        self.sd = None
        self.w = None
        self.b = 0.0
        self.by_cid = {}

    def fit(self, frozen, *, exclude_fold, order_lookup):
        X, y = [], []
        for m in frozen:
            if order_lookup[m.piece] == exclude_fold:
                continue
            voice_of = {e.eid: e.voice for e in m.events}
            # synthesise candidate geometry deterministically for the pair features
            import visual as V
            import random as _r
            rng = _r.Random(22)
            cm = V.synthesise(m, rng, hint_acc=0.85)
            keep = [c for c in cm.cands if c.gt_eid >= 0 and c.gt_eid in voice_of]
            by_staff = collections.defaultdict(list)
            for c in keep:
                by_staff[c.staff].append(c)
            for st, g in by_staff.items():
                if len({voice_of[c.gt_eid] for c in g}) < 2:
                    continue
                for a in g:
                    for b in g:
                        if a.cid == b.cid:
                            continue
                        X.append(_pair_feat(a, b))
                        y.append(1.0 if voice_of[a.gt_eid] == voice_of[b.gt_eid] else 0.0)
        X = np.array(X)
        y = np.array(y)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.zeros(Z.shape[1])
        b = 0.0
        for _ in range(400):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            w -= 0.5 * (Z.T @ (p - y) / len(y) + 1e-3 * w)
            b -= 0.5 * float((p - y).mean())
        self.w, self.b = w, b
        return self

    def score_prob(self, cm, cur, rem, acc=0.0):
        """cur is the current voice as tuples (cid, x, onset, ...)."""
        last_cid = cur[-1][0] if cur else -1
        """Same signal, but as a LOG-PROBABILITY in [-inf, 0].

        The raw score is a log-odds, whose scale (+-10) is far larger than the policy's
        successor log-probabilities (about -5..0), so an additive term with weight ~1
        swamps the model's own preference.  Mapping through log-sigmoid puts the pair
        signal on the same footing as succ_lp, so the weight becomes comparable.
        """
        z = self.score_vec(cm, cur, rem, acc)
        return -np.log1p(np.exp(-np.clip(z, -30, 30)))

    def score_vec(self, cm, cur, rem, acc=0.0):
        last_cid = cur[-1][0] if cur else -1
        """Log-odds that each remaining candidate continues the voice of `last_cid`."""
        a = self.by_cid.get(id(cm))
        if a is None:
            a = {c.cid: c for c in cm.cands}
            self.by_cid[id(cm)] = a
        la = a.get(last_cid)
        if la is None:
            return np.zeros(len(rem))
        F = np.array([_pair_feat(la, c) for c in rem])
        Z = (F - self.mu) / self.sd
        return Z @ self.w + self.b
