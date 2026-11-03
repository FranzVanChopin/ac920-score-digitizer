"""Check that engraving adds notation only: merge the given page.musicxml files plainly and engraved, and compare the
sounding events of every measure (staff, voice, onset, duration, pitch / rest, chord).  Prints what was added and
checks every beam group opens and closes.

  python check_engrave.py PAGE.musicxml [PAGE.musicxml ...]      (each with headers.json next to it)
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_pages  # noqa: E402


def events(part):
    out = []
    for m in part.findall("measure"):
        pos, last, ev = 0, 0, []
        for c in m:
            if c.tag == "backup":
                pos -= int(c.findtext("duration"))
            elif c.tag == "forward":
                pos += int(c.findtext("duration"))
            elif c.tag == "note":
                d = int(c.findtext("duration", "0"))
                chord = c.find("chord") is not None
                at = last if chord else pos
                p = c.find("pitch")
                what = ("rest",) if p is None else (p.findtext("step"), p.findtext("alter", "0"), p.findtext("octave"))
                ev.append((c.findtext("staff"), c.findtext("voice"), at, d, chord, what))
                if not chord:
                    last, pos = pos, pos + d
        out.append(sorted(ev))
    return out


def beams_ok(part):
    """every voice's beam number n goes begin (continue)* end, hooks alone"""
    bad = 0
    for m in part.findall("measure"):
        open_ = {}
        for n in m.findall("note"):
            v = (n.findtext("staff"), n.findtext("voice"))
            for b in n.findall("beam"):
                k = (v, b.get("number"))
                if b.text == "begin":
                    bad += k in open_
                    open_[k] = True
                elif b.text in ("continue", "end"):
                    bad += k not in open_
                    if b.text == "end":
                        open_.pop(k, None)
        bad += len(open_)
    return bad


def main(pages):
    plain, _ = merge_pages.merge(pages, engrave=False)
    eng, counts = merge_pages.merge(pages, engrave=True)
    a, b = events(plain.getroot().find("part")), events(eng.getroot().find("part"))
    same = sum(x == y for x, y in zip(a, b))
    part = eng.getroot().find("part")
    added = Counter()
    for n in part.iter("note"):
        for t in ("type", "dot", "accidental", "time-modification", "beam"):
            added[t] += len(n.findall(t))
        if n.find("rest") is not None and n.find("rest").get("measure") == "yes":
            added["measure_rests"] += 1
    keys = [k.findtext("fifths") for k in part.iter("key")]
    clefs = [c.findtext("sign") + c.findtext("line") for c in part.iter("clef")]
    print(f"{len(pages)} pages, {sum(counts)} measures: sounding events identical in {same}/{len(a)} measures"
          f"{'' if len(a) == len(b) else ' (MEASURE COUNT DIFFERS)'}")
    print(f"   keys {keys}, clefs {clefs}, added {dict(added)}, broken beam groups {beams_ok(part)}")
    return same == len(a) == len(b) and beams_ok(part) == 0


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1:]) else 1)
