"""Board-side decode: extraction + production measure decoder without the training corpus.

Same steps as lab/omr_proto/proto.py (Decoder.measure, repeat consensus, MusicXML writer), but the
warp vocabulary, the fold MLP and the pair scorer come from files in weights/ instead of the
frozen corpus, so nothing but numpy and the decoder sources is needed on the board.
"""
import collections
import json
from pathlib import Path

import numpy as np

import extract as X
import pitch as PT
from column_joint import column_joint_refine, column_violation
from pair_succ import PairSuccModel
from pickers import MLP, LearnedPicker
from pitch_repair import pitch_swap_repair
from prod import COLUMN_JOINT_MAX_STATES, DP_K_ATTR, PROD, REPEAT_VETO
from repeat_consensus import repeat_consensus
from search import _final_loss, solve
from two_stage import as_tuples, canon_table, dp_refine

import proto_xml as PX


class BoardDecoder:
    def __init__(self, wdir, clib=None):
        wdir = Path(wdir)
        self.wv = [tuple(w) for w in json.loads((wdir / "warp_vocab.json").read_text())]
        _t, self.canon_ok = canon_table(self.wv)
        self.pk = LearnedPicker(MLP.load(str(wdir / "decoder_fold0.npz")), self.wv)
        z = np.load(wdir / "pair_fold0.npz")
        self.pm = PairSuccModel()
        self.pm.mu, self.pm.sd, self.pm.w, self.pm.b = z["mu"], z["sd"], z["w"], float(z["b"])
        self.cs = None                       # stage A in C++ (cdec/libdecoder.so), same answers as solve()
        if clib:
            from cdecoder import CSolver, params_from_prod
            self.cs = CSolver(self.pk.mlp, self.pm, self.wv, lib=str(clib), params=params_from_prod(PROD))

    def measure(self, cm, n_beats):
        cs = self.cs
        if cs is not None:
            got, _st = cs.solve(cm, n_beats)
        else:
            sc = (lambda cur, rem, acc, _cm=cm: self.pm.score_prob(_cm, cur, rem, acc))
            got, _st = solve(cm.cands, cm.length, self.pk, self.wv, n_total=len(cm.cands), n_beats=n_beats,
                             pair_w=0.25, pair_score=sc, close_cost=0.5, order_guard=False, **PROD)
        if cs is not None:                       # C++ dp_refine, identical to two_stage.dp_refine
            refined = cs.refine(cm, got, n_beats, iters=2, k_attr=DP_K_ATTR, hint_w=PROD["hint_w"])
        else:
            refined = dp_refine(cm, got, self.pk, self.wv, self.canon_ok, iters=2, k_attr=DP_K_ATTR, iso_w=0.0, exact_x=0.0)
        kw = dict(voice_w=PROD["voice_w"], dev_w=0.0, tau_w=PROD["tau_w"], iso_w=PROD["iso_w"], hint_w=PROD["hint_w"])
        try:
            la = _final_loss(as_tuples(got), cm.length, 1000.0, 0.0, 1, **kw)
            lb = _final_loss(refined, cm.length, 1000.0, 0.0, 1, **kw)
        except Exception:  # noqa: BLE001
            la, lb = 0.0, 1.0
        chosen = refined if lb < la else as_tuples(got)
        pr = pitch_swap_repair(cm, chosen)
        if pr is not chosen and pr != chosen:
            chosen = pr
        if column_violation(cm, chosen) is not None:
            if cs is not None:                   # C++ column joint DP, identical result
                cj = cs.column_joint(cm, chosen, n_beats, k_attr=DP_K_ATTR, hint_w=PROD["hint_w"],
                                     max_states=COLUMN_JOINT_MAX_STATES)
            else:
                cj = column_joint_refine(cm, chosen, self.pk, self.wv, self.canon_ok, k_attr=DP_K_ATTR)
            if cj is not None:
                chosen = cj
        return chosen


_CTX = {}
_DECODERS = {}


def decoder_for(wdir, clib=None):
    """One BoardDecoder per (weights dir, library) for the process (run_page --serve decodes many pages): its weights
    and tables are only read while decoding; the pair model's per-measure lookup (keyed by id(cm), and ids are
    reused across pages) is emptied for every page, as a new decoder starts with it empty."""
    key = (str(Path(wdir).resolve()), str(clib) if clib else None)
    dec = _DECODERS.get(key)
    if dec is None:
        dec = _DECODERS[key] = BoardDecoder(wdir, clib)
    dec.pm.by_cid = {}
    return dec


def _decode_one(j):
    dec, cms, ms = _CTX["dec"], _CTX["cms"], _CTX["ms"]
    if cms[j] is None:
        return j, None, None
    try:
        return j, dec.measure(cms[j], ms[j].beats), None
    except Exception as e:  # noqa: BLE001
        return j, None, type(e).__name__


def decode_page(graph, staves, headers, wdir, meter=None, workers=None, clib=None):
    """Measures are decoded independently, so on Linux they are spread over `workers` forked
    processes (default: all cores) that share the loaded model; the result is identical to the
    sequential loop.  Repeat consensus runs afterwards on the whole page, as before."""
    stats = collections.Counter()
    stats["repitched_heads"] = PT.repitch_graph(graph, staves, headers)[0]
    meter = meter or (4, 4)
    ms, meter = X.page_measures(graph, meter)
    if not X.has_meter_glyph(graph):
        inf = X.infer_meter(ms)
        if inf:
            ms, _ = X.page_measures(graph, inf)
            stats["meter_from_content"] += 1
    dec = decoder_for(wdir, clib)
    cms = [X.to_cand_measure(m, "photo", seed=1000 * m.index + 7) for m in ms]
    import multiprocessing as mp
    import os
    if workers is None:
        workers = os.cpu_count() if hasattr(os, "fork") else 1
    _CTX.update(dec=dec, cms=cms, ms=ms)
    order = sorted(range(len(cms)), key=lambda j: -(len(cms[j].cands) if cms[j] else 0))   # big first
    if workers > 1 and hasattr(os, "fork"):
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_decode_one, order, chunksize=1)
    else:
        res = [_decode_one(j) for j in order]
    answers = [None] * len(cms)
    for j, ans, err in res:
        answers[j] = ans
        if err:
            stats[f"err:{err}"] += 1
    stats["workers"] = workers
    flat = [j for j in range(len(cms)) if cms[j] is not None and answers[j] is not None]
    if flat:
        new = repeat_consensus([cms[j] for j in flat], [answers[j] for j in flat], dec.pk, dec.wv, dec.canon_ok,
                               veto=REPEAT_VETO)
        for j, a in zip(flat, new):
            stats["consensus_rewrites"] += a != answers[j]
            answers[j] = a
    stats["measures"] = len(ms)
    return ms, cms, answers, dict(stats)


def write_xml(path, measures, cms, answers):
    PX.write_xml(path, measures, [PX.ours_streams(m, cm, a) for m, cm, a in zip(measures, cms, answers)])
