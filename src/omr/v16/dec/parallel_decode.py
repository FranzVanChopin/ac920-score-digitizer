"""Parallel per-measure decoding.

The decode loop is embarrassingly parallel over measures and is pure numpy, so it is
the natural place to use the machine's cores: 32 logical cores sit idle while one
thread runs the beam search and the exact DP for ~700-960 measures per fold (~65 s of
the ~370 s cold fold, and essentially all of the ~65 s warm fold).

Design notes that matter for correctness:

* CANDIDATES ARE BUILT IN THE PARENT.  `make_candidates` draws from ONE `random.Random`
  across the whole measure list, so rebuilding them inside workers (or chunking the
  list differently) would change the injected hint noise and silently change the
  numbers.  The parent builds them once and ships the (picklable, ~1.2 KB) measures.
* THE MODEL IS LOADED ONCE PER WORKER from its cache file, not per task, and not
  pickled through the queue.
* DIAGNOSTIC COUNTERS (RD/PB) are only aggregated in serial mode; with workers > 1
  they are unavailable, which is stated rather than silently zeroed.

Everything else mirrors the serial loop exactly, so workers > 1 must reproduce the
same numbers as --workers 1.
"""
from __future__ import annotations

import os
import sys

_CTX = {}


def _init(model_path, warp_vocab, canon_ok, params):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pickers import LearnedPicker, MLP
    _CTX["pk"] = LearnedPicker(MLP.load(model_path), warp_vocab)
    _CTX["wv"] = warp_vocab
    _CTX["canon_ok"] = canon_ok
    _CTX["p"] = params


def _work(cm):
    """One measure -> (A_prediction, {iso_w: B_prediction})."""
    from prod import PROD
    from search import solve
    from two_stage import as_tuples, dp_refine, _final_loss, _loss_of, to_eval
    from pitch_repair import pitch_swap_repair

    pk, wv, canon_ok, p = _CTX["pk"], _CTX["wv"], _CTX["canon_ok"], _CTX["p"]
    got, st = solve(cm.cands, cm.length, pk, wv, n_total=len(cm.cands),
                    n_beats=cm.gt.beats, order_guard=p["order_guard"], **PROD)
    la = _final_loss(as_tuples(got), cm.length, 1000.0, 0.0, 1,
                     voice_w=PROD["voice_w"], dev_w=0.0, tau_w=PROD["tau_w"],
                     iso_w=PROD["iso_w"], hint_w=PROD["hint_w"])
    out = {}
    for iw in p["iso_w"]:
        refined = dp_refine(cm, got, pk, wv, canon_ok, iters=p["iters"],
                            k_attr=p["k_attr"], iso_w=iw, exact_x=p["exact_x"])
        try:
            lb = _final_loss(refined, cm.length, 1000.0, 0.0, 1,
                             voice_w=PROD["voice_w"], dev_w=0.0, tau_w=PROD["tau_w"],
                             iso_w=PROD["iso_w"], hint_w=PROD["hint_w"])
        except Exception:
            la, lb = 0.0, 1.0
        chosen = refined if lb < la else as_tuples(got)
        if p["pitch_repair"]:
            pr = pitch_swap_repair(cm, chosen)
            if pr is not chosen and pr != chosen:
                chosen = pr
        out[iw] = (to_eval(chosen), st)
    return (got, st), out


def decode_all(cm_list, model_path, warp_vocab, canon_ok, params, workers=1):
    """Return (A, {iso_w: B}) for the whole list; identical to the serial loop."""
    iso_ws = params["iso_w"]
    if workers is None or workers <= 1 or len(cm_list) < 4:
        A, Bs = [], {iw: [] for iw in iso_ws}
        for cm in cm_list:
            a, d = _work_init_local(model_path, warp_vocab, canon_ok, params, cm)
            A.append(a)
            for iw in iso_ws:
                Bs[iw].append(d[iw])
        return A, Bs

    import concurrent.futures as cf
    A, Bs = [], {iw: [] for iw in iso_ws}
    ctx = __import__("multiprocessing").get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                               initializer=_init,
                               initargs=(model_path, warp_vocab, canon_ok,
                                         params)) as ex:
        for a, d in ex.map(_work, cm_list, chunksize=8):
            A.append(a)
            for iw in iso_ws:
                Bs[iw].append(d[iw])
    return A, Bs


def _work_init_local(model_path, warp_vocab, canon_ok, params, cm):
    """Serial fallback: initialise the worker context in-process once."""
    if "pk" not in _CTX:
        _init(model_path, warp_vocab, canon_ok, params)
    return _work(cm)
