"""Starry-style evaluation: per-field event errors + measure-level exactness."""
from __future__ import annotations

import math


def _partition(voices):
    """list of voices -> canonical set of frozensets of cids."""
    return frozenset(frozenset(e["cid"] for e in v) for v in voices if v)


def evaluate(cand_measures, predictions):
    """cand_measures: list[CandMeasure]; predictions: list[(voices, stats)]."""
    n_ev = 0
    tick_err = div_err = dot_err = warp_err = any_err = 0
    sq = 0.0
    n_meas = 0
    perfect = vmatch = texact = 0
    solved = 0
    valid_n = 0
    expanded_total = 0
    cov_total = 0.0

    for cm, (voices, stats) in zip(cand_measures, predictions):
        n_meas += 1
        expanded_total += stats.get("expanded", 0)
        cov_total += stats.get("coverage", 0.0)
        solved += 1 if stats.get("complete") else 0
        valid_n += 1 if stats.get("valid") else 0

        pred = {}
        for vi, v in enumerate(voices):
            for e in v:
                pred[e["cid"]] = (vi, e["onset"], e["d"], e["dots"], e["warp"])

        cid_of_eid = {c.gt_eid: c.cid for c in cm.cands if c.gt_eid >= 0}
        gt_voices = {}
        for e in cm.gt.events:
            gt_voices.setdefault(e.voice, []).append(e)

        # ---- event level ----
        e_ok_all, e_v_ok = True, True
        for e in cm.gt.events:
            cid = cid_of_eid.get(e.eid)
            p = pred.get(cid) if cid is not None else None
            n_ev += 1
            if p is None:
                tick_err += 1
                div_err += 1
                dot_err += 1
                warp_err += 1
                any_err += 1
                sq += float(cm.length * cm.length)
                e_ok_all = False
                continue
            _vi, onset, d, dots, warp = p
            t_bad = abs(onset - e.onset) > 1
            d_bad = d != e.d
            o_bad = dots != e.dots
            w_bad = tuple(warp) != tuple(e.warp)
            tick_err += t_bad
            div_err += d_bad
            dot_err += o_bad
            warp_err += w_bad
            if t_bad or d_bad or o_bad or w_bad:
                any_err += 1
                e_ok_all = False
            sq += float((onset - e.onset) ** 2)

        # ---- measure level ----
        # A ground-truth event may have NO candidate at all when the upstream
        # detector dropped it.  That is a real failure mode of a real front end --
        # it must count against the measure, not crash the scorer.
        missing_cid = object()
        gt_part = frozenset(
            frozenset(cid_of_eid.get(e.eid, missing_cid) for e in evs)
            for evs in gt_voices.values())
        pr_part = _partition(voices)
        vm = (gt_part == pr_part) and missing_cid not in gt_part
        vmatch += 1 if vm else 0
        if vm:
            ok_t = all(abs(pred[cid_of_eid[e.eid]][1] - e.onset) <= 1 for e in cm.gt.events)
            texact += 1 if ok_t else 0
            if e_ok_all:
                perfect += 1

    def pct(x):
        return 100.0 * x / max(1, n_ev)

    return {
        "measures": n_meas,
        "events": n_ev,
        "solved_pct": 100.0 * solved / max(1, n_meas),
        "valid_pct": 100.0 * valid_n / max(1, n_meas),
        "any_field_error_pct": pct(any_err),
        "tick_error_pct": pct(tick_err),
        "division_error_pct": pct(div_err),
        "dots_error_pct": pct(dot_err),
        "warp_error_pct": pct(warp_err),
        "tick_rmse": math.sqrt(sq / max(1, n_ev)),
        "perfect_pct": 100.0 * perfect / max(1, n_meas),
        "voice_match_pct": 100.0 * vmatch / max(1, n_meas),
        "tick_exact_pct": 100.0 * texact / max(1, n_meas),
        "mean_expansions": expanded_total / max(1, n_meas),
        "mean_coverage_pct": 100.0 * cov_total / max(1, n_meas),
    }


def fmt(rows, cols=None):
    cols = cols or ["any_field_error_pct", "tick_rmse", "tick_error_pct",
                    "division_error_pct", "dots_error_pct", "warp_error_pct",
                    "perfect_pct", "voice_match_pct", "tick_exact_pct", "valid_pct", "mean_coverage_pct"]
    head = f"{'arm':<22}" + "".join(f"{c:>12}" for c in cols)
    out = [head, "-" * len(head)]
    for name, m in rows:
        out.append(f"{name:<22}" + "".join(f"{m[c]:>12.2f}" for c in cols))
    return "\n".join(out)


def structure_key(cm, voices):
    """Order-invariant signature of a solution: gt_eid -> (voice#, onset, dur)."""
    cid2eid = {c.cid: c.gt_eid for c in cm.cands}
    items = []
    for vi, v in enumerate(sorted(voices, key=lambda v: min(e["x"] for e in v) if v else 0.0)):
        for e in sorted(v, key=lambda e: e["onset"]):
            items.append((cid2eid.get(e["cid"], -1), vi, e["onset"], e["dur"]))
    return tuple(sorted(items))




