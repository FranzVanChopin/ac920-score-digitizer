"""Ground-truth corpus: real measures from the test set, split by piece.

Only measures where EVERY voice contiguously tiles [0, length) are kept, so the
decoding problem is well posed (each voice must exactly fill the measure).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from musicxml_measures import Measure as RawMeasure, parse_piece  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "测试集")


@dataclass
class GTEvent:
    eid: int
    voice: int          # canonical voice index within the measure (0..k-1)
    staff: int
    onset: int
    dur: int
    d: int
    dots: int
    warp: tuple
    is_rest: bool
    midi: int = -1      # pitch; -1 when unavailable (rests / unpitched)
    stem: int = 0       # 1 up / -1 down / 0 unknown -- the engraver's own
                        # voice marker (opposite stems for co-staff voices)


@dataclass
class GTMeasure:
    piece: str
    length: int
    beats: int
    beat_type: int
    events: list

    @property
    def n_voices(self) -> int:
        return max((e.voice for e in self.events), default=-1) + 1


def _gcd_warp(a: int, n: int) -> tuple:
    g = math.gcd(a, n)
    return (a // g, n // g)


def _contiguous(events, length) -> bool:
    """Every voice must start at 0 and tile the measure with no gaps/overlaps."""
    by_voice = {}
    for e in events:
        by_voice.setdefault(e.voice, []).append(e)
    for _v, evs in by_voice.items():
        evs = sorted(evs, key=lambda e: e.onset)
        cur = 0
        for e in evs:
            if e.onset != cur:
                return False
            cur = e.onset + e.dur
        if cur != length:
            return False
    return True


def _piece_signature(measures) -> str:
    """Content hash of one piece: bar structure plus every event, order-independent.

    Ignores the filename and the piece name on purpose -- the duplicates that matter are the
    same music saved under a different title.  `build_corpus` already canonicalises the
    voice indices by first appearance, so equal music hashes equal here.
    """
    h = hashlib.sha256()
    for m in measures:
        h.update(("|%d,%d,%d" % (m.length, m.beats, m.beat_type)).encode())
        for e in sorted(m.events, key=lambda x: (x.voice, x.onset, x.dur)):
            h.update((";%d,%d,%d,%d,%d,%d,%d,%d,%d,%d" % (
                e.voice, e.staff, e.onset, e.dur, e.d, e.dots,
                e.warp[0], e.warp[1], 1 if e.is_rest else 0, e.midi)).encode())
    return h.hexdigest()


def build_corpus(max_events: int = 32, max_voices: int = 4, root: str = None) -> list:
    """Parse a directory tree into the canonical corpus.

    `root` defaults to the shared 测试集 directory.  It exists so a DIFFERENT population
    (e.g. 测试集4) can be parsed with exactly the same normalisation -- canonical voice
    numbering, gcd-reduced warps, the exact-tiling check -- instead of a second
    implementation drifting away from this one.

    NOTE the default walk is RECURSIVE, so `build_corpus()` already includes 测试集2/3/4.
    The frozen corpus predates those folders, which is why results must be compared against
    the freeze rather than rebuilt.
    """
    files = []
    walk_root = root or ROOT
    if not os.path.isdir(walk_root):
        # Silent emptiness is the dangerous failure here: a typo'd or stale ROOT would make
        # build_corpus() return [] and downstream code would happily train on nothing.
        raise FileNotFoundError(
            "corpus root does not exist: %r\n"
            "  Pass root=<path to the MusicXML tree> explicitly.  The default ROOT is "
            "derived from this file's location and only resolves when the 测试集 folder "
            "sits next to the package." % walk_root)
    for dirpath, _dirs, names in os.walk(walk_root):
        for n in names:
            if n.lower().endswith((".musicxml", ".mxl", ".xml")):
                files.append(os.path.join(dirpath, n))
    # Shallowest path first, so the original 测试集/ copy of a piece wins over the copies
    # that later sibling folders (测试集2/3/4) introduce under a different filename.
    files.sort(key=lambda p: (p.count(os.sep), p))

    out = []
    seen_sigs = set()
    for f in files:
        name = os.path.basename(f)
        try:
            raw = parse_piece(f, name)
        except Exception:  # noqa: BLE001
            continue
        piece_out = []
        for m in raw:
            if not (1 <= len(m.events) <= max_events):
                continue
            if not (1 <= len(m.voices) <= max_voices):
                continue
            # canonical voice index, ordered by first appearance
            order, idx = [], {}
            for e in m.events:
                if e.voice not in idx:
                    idx[e.voice] = len(order)
                    order.append(e.voice)
            evs = [
                GTEvent(eid=i, voice=idx[e.voice], staff=e.staff, onset=e.onset,
                        dur=e.dur, d=e.d, dots=e.dots,
                        warp=_gcd_warp(*e.warp), is_rest=e.is_rest,
                        midi=int(getattr(e, "midi", -1)),
                        stem=int(getattr(e, "stem", 0)))
                for i, e in enumerate(m.events)
            ]
            if not _contiguous(evs, m.length):
                continue
            piece_out.append(GTMeasure(piece=name, length=m.length, beats=m.beats,
                                       beat_type=m.beat_type, events=evs))
        # ---- drop copies of a piece we have already taken ----
        # The same music appears in several folders under different filenames, and filename
        # matching is NOT sufficient: `Clair_de_Lune__Debussy.mxl` and
        # `clair-de-lune-claude-debussy.mxl` normalise differently yet are the same piece.
        # A content hash is the only reliable identity here.
        if not piece_out:
            continue
        sig = _piece_signature(piece_out)
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        out.extend(piece_out)
    return out


# --------------------------------------------------------------------------- #
# FREEZE.  The corpus directory is shared and was observed to change mid-session
# (seven .mxl files were added, moving the test split from 461 to 650 measures),
# so every number before and after that change was measured on a different set.
# A frozen snapshot with a content hash makes results comparable and reproducible.
# --------------------------------------------------------------------------- #
FREEZE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results", "corpus_freeze.json")


def _serialise(measures):
    # `midi` IS serialised again (Round 42).  Round 41 removed it after the pitch
    # FEATURES regressed the CV by 1.65 points; but the failure-mechanism analysis
    # then showed 54% of the remaining failures are partition-only, and 6 of 35 are
    # swaps of two notes at the SAME onset and duration on one staff -- exactly the
    # case where pitch is the only disambiguator.  Pitch is therefore kept as DATA
    # (consumed by a post-hoc repair pass) while `cand_features` stays at 31 dims, so
    # the learned model is completely untouched by this change.
    out = []
    for m in measures:
        out.append({
            "piece": m.piece, "length": m.length, "beats": m.beats,
            "beat_type": m.beat_type,
            "events": [[e.eid, e.voice, e.staff, e.onset, e.dur, e.d, e.dots,
                        list(e.warp), int(e.is_rest), int(getattr(e, "midi", -1)),
                        int(getattr(e, "stem", 0))]
                       for e in m.events],
        })
    return out


def _deserialise(rows):
    out = []
    for r in rows:
        evs = []
        for row in r["events"]:
            row = list(row) + [0] * (11 - len(row))
            evs.append(GTEvent(eid=row[0], voice=row[1], staff=row[2], onset=row[3],
                               dur=row[4], d=row[5], dots=row[6],
                               warp=tuple(row[7]), is_rest=bool(row[8]),
                               midi=row[9], stem=row[10]))
        out.append(GTMeasure(piece=r["piece"], length=r["length"],
                             beats=r["beats"], beat_type=r["beat_type"], events=evs))
    return out


def corpus_hash(measures) -> str:
    import hashlib
    blob = json.dumps(_serialise(measures), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def save_freeze(measures, note=""):
    os.makedirs(os.path.dirname(FREEZE), exist_ok=True)
    payload = {"hash": corpus_hash(measures), "note": note,
               "count": len(measures), "measures": _serialise(measures)}
    with open(FREEZE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return payload["hash"]


def load_freeze():
    if not os.path.exists(FREEZE):
        return None, None
    with open(FREEZE, encoding="utf-8") as fh:
        payload = json.load(fh)
    ms = _deserialise(payload["measures"])
    return ms, payload["hash"]


def split_by_piece(measures, seed: int = 20261014, frac=(0.70, 0.15, 0.15)):
    """Split by PIECE (never by measure), stratified by polyphonic density.

    Pieces are ordered by how many polyphonic measures they contribute and dealt
    round-robin, so every split gets both sparse and dense material.  This is
    stratification, not cherry-picking: no measure is inspected while choosing.
    """
    import random

    by_piece = {}
    for m in measures:
        by_piece.setdefault(m.piece, []).append(m)
    poly = {p: sum(1 for m in ms if m.n_voices > 1) for p, ms in by_piece.items()}
    pieces = sorted(by_piece, key=lambda p: (-poly[p], p))
    rng = random.Random(seed)
    rng.shuffle(pieces)                      # break ties/piece-name order
    pieces.sort(key=lambda p: -poly[p])      # dense first, then deal round-robin

    buckets = [[], [], []]
    quota = [frac[0], frac[1], frac[2]]
    target = [len(measures) * q for q in quota]
    filled = [0, 0, 0]
    for p in pieces:
        k = min(range(3), key=lambda i: (filled[i] / max(target[i], 1.0), i))
        buckets[k].append(p)
        filled[k] += len(by_piece[p])

    tr, dv, te = [set(b) for b in buckets]
    train = [m for m in measures if m.piece in tr]
    dev = [m for m in measures if m.piece in dv]
    test = [m for m in measures if m.piece in te]
    return train, dev, test, (sorted(tr), sorted(dv), sorted(te))


if __name__ == "__main__":
    import sys as _s
    if "--freeze" in _s.argv:
        ms = build_corpus()
        h = save_freeze(ms, note="frozen during goal round 2")
        print(f"frozen {len(ms)} measures -> {FREEZE}")
        print(f"corpus hash: {h}")
        _s.exit(0)
    frozen, fh = load_freeze()
    if frozen is not None and "--live" not in _s.argv:
        print(f"USING FROZEN CORPUS ({len(frozen)} measures, hash {fh[:16]}...)")
        ms = frozen
        tr, dv, te, (a, b, c) = split_by_piece(ms)
        print(f"train={len(tr)} ({len(a)} pieces)  dev={len(dv)} ({len(b)})  test={len(te)} ({len(c)})")
        poly = sum(1 for m in ms if m.n_voices > 1)
        print(f"polyphonic: {poly} ({100.0*poly/max(1,len(ms)):.1f}%)")
        print(f"test pieces ({len(c)}): {c}")
        _s.exit(0)
    ms = build_corpus()
    print(f"contiguous-tiling measures: {len(ms)}")
    tr, dv, te, (a, b, c) = split_by_piece(ms)
    print(f"train={len(tr)} ({len(a)} pieces)  dev={len(dv)} ({len(b)})  test={len(te)} ({len(c)})")
    import collections

    poly = sum(1 for m in ms if m.n_voices > 1)
    print(f"polyphonic: {poly} ({100.0*poly/max(1,len(ms)):.1f}%)")
    print(f"voices dist: {collections.Counter(m.n_voices for m in ms).most_common()}")
    ev = [len(m.events) for m in ms]
    print(f"events/measure: min={min(ev)} med={sorted(ev)[len(ev)//2]} max={max(ev)} mean={sum(ev)/len(ev):.1f}")
    print(f"\ntest pieces: {c}")

