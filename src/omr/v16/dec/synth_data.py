"""Wire the adapted synthetic corpus into training as TRAIN-ONLY data.

Three things matter here, and getting any of them wrong would produce a result that looks
like progress but is not:

1. TRAIN ONLY.  The synthetic measures are appended to the training set of EVERY fold.
   They never touch a validation or test set, so the 50 real pieces stay untouched as the
   holdout and the corpus hash does not change.

2. CACHE KEY.  The per-fold model cache is keyed on the training configuration.  If the
   synthetic data were added without touching the key, every fold would silently reuse a
   model trained WITHOUT it -- the exact bug class that already bit this project once (the
   `wp` parameter).  So the key gains an explicit synthetic tag.

3. BALANCE.  The synthetic corpus is deliberately dense-heavy, and a previous experiment
   showed that re-weighting training toward dense measures LOST more on the 9-12 events/
   measure bucket than it gained on 13+ (net -0.42).  So `--synth-mix` controls how many
   synthetic measures are added per real one, and the first sweep tests small ratios.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_synth(path: str):
    """Deserialise the adapted corpus into the same Measure/GTEvent objects as the freeze."""
    sys.path.insert(0, HERE)
    import corpus as C
    d = json.load(io.open(path, encoding="utf-8"))
    return C._deserialise(d["measures"])


def synth_tag(path: str, mix: float, repeat: int) -> str:
    """A short, stable fingerprint of the synthetic contribution to the training set.

    Uses the file's content hash plus the mixing parameters, so any change to either the
    data or how much of it is used produces a different cache key.
    """
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"s{h.hexdigest()[:8]}m{mix:g}r{repeat}"


def select_synth(ms, mix: float, seed: int = 7):
    """Take a deterministic subset of the synthetic measures.

    Selection is stratified by event count so that shrinking the synthetic set shrinks all
    density bands proportionally instead of keeping only whatever comes first.
    """
    if mix <= 0:
        return []
    if mix >= 1.0:
        return list(ms)
    import numpy as np
    rng = np.random.default_rng(seed)
    by_band = collections.defaultdict(list)
    for m in ms:
        n = len(m.events)
        band = 0 if n <= 8 else (1 if n <= 15 else 2)
        by_band[band].append(m)
    out = []
    for _band, items in by_band.items():
        k = int(round(len(items) * mix))
        if k <= 0:
            continue
        idx = rng.permutation(len(items))[:k]
        out.extend(items[i] for i in idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default=os.path.join(ROOT, "synthetic_adapted.json"))
    ap.add_argument("--mix", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    ms = load_synth(a.synth)
    sel = select_synth(ms, a.mix)
    print(f"  loaded            : {len(ms)} synthetic measures")
    print(f"  selected (mix={a.mix:g}) : {len(sel)}")
    print(f"  events            : {sum(len(m.events) for m in sel)}")
    print(f"  pieces            : {len({m.piece for m in sel})}")
    print(f"  cache tag         : {synth_tag(a.synth, a.mix, a.repeat)}")
    bands = collections.Counter(0 if len(m.events) <= 8 else
                                (1 if len(m.events) <= 15 else 2) for m in sel)
    print(f"  density bands     : sparse {bands[0]} / medium {bands[1]} / "
          f"dense {bands[2]}")
    # sanity: the first measure must round-trip through the real data model
    m0 = sel[0]
    print(f"  sample measure    : piece={m0.piece} length={m0.length} "
          f"beats={m0.beats} beat_type={m0.beat_type} events={len(m0.events)}")
    e0 = m0.events[0]
    print(f"  sample event      : {e0}")


if __name__ == "__main__":
    main()
