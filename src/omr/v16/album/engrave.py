"""Notation for display (MuseScore, Verovio) added to a recognized page's measures.

run_page.py writes sounding pitch and duration only.  This adds what a notation program needs to draw it well:
* clefs and the key signature: the majority of the page's systems in headers.json (the clefs / key the decoder used
  for pitch; a system that disagrees is rare and only affects how a note is drawn, not which pitch it is),
* written accidentals: per measure and staff against the key signature, in onset order (Verovio draws an accidental
  only from an <accidental> element),
* note types, dots and triplet <time-modification> from duration / divisions,
* beams by beat (quarter in x/2 and x/4, dotted quarter in compound x/8) per voice, secondary beams for 16ths and
  shorter, a whole-measure rest as <rest measure="yes"/>.
Sounding pitch (<pitch>, <alter>), durations, voices, staves, backup / forward and the measures themselves are not
changed (check_engrave.py compares them).
"""
import math
from collections import Counter, defaultdict
from fractions import Fraction
import xml.etree.ElementTree as ET

TYPES = [("whole", Fraction(4)), ("half", Fraction(2)), ("quarter", Fraction(1)), ("eighth", Fraction(1, 2)),
         ("16th", Fraction(1, 4)), ("32nd", Fraction(1, 8)), ("64th", Fraction(1, 16)), ("128th", Fraction(1, 32))]
SHARPS = "FCGDAEB"                     # order in which a key signature adds sharps (flats: reversed)
ACCIDENTAL = {2: "double-sharp", 1: "sharp", 0: "natural", -1: "flat", -2: "flat-flat"}
CLEF = {"treble": ("G", "2"), "bass": ("F", "4"), "alto": ("C", "3"), "tenor": ("C", "4")}
NOTE_ORDER = ["grace", "cue", "chord", "pitch", "unpitched", "rest", "duration", "tie", "instrument", "voice",
              "type", "dot", "accidental", "time-modification", "stem", "notehead", "staff", "beam", "notations",
              "lyric"]


def key_alter(fifths, step):
    if fifths > 0:
        return 1 if step in SHARPS[:fifths] else 0
    if fifths < 0:
        return -1 if step in SHARPS[::-1][:-fifths] else 0
    return 0


def page_header(headers):
    """(key fifths, {staff: clef name}) by majority over the page's systems; clefs default to treble / bass"""
    keys = Counter(s.get("key", 0) for s in headers)
    key = keys.most_common(1)[0][0] if keys else 0
    clefs = {}
    staves = sorted({int(n) for s in headers for n in s.get("clefs", {})}) or [1, 2]
    for n in staves:
        found = Counter(s["clefs"].get(str(n)) for s in headers if s.get("clefs", {}).get(str(n)))
        clefs[n] = found.most_common(1)[0][0] if found else ("bass" if n == 2 and len(staves) == 2 else "treble")
    return key, clefs


def note_type(value):
    """(type, dots, tuplet (actual, normal) or None) of a duration in quarters"""
    for name, base in TYPES:
        for dots in range(4):
            if base * (2 - Fraction(1, 2 ** dots)) == value:
                return name, dots, None
    for name, base in TYPES:                        # triplets: two thirds of a plain value
        for dots in range(2):
            if base * (2 - Fraction(1, 2 ** dots)) * Fraction(2, 3) == value:
                return name, dots, (3, 2)
    name, base = min(TYPES, key=lambda t: abs(math.log2(float(value / t[1]))))
    ratio = base / value
    return name, 0, (ratio.numerator, ratio.denominator)


def _reorder(note):
    children = list(note)
    for c in children:
        note.remove(c)
    rank = {t: i for i, t in enumerate(NOTE_ORDER)}
    for c in sorted(children, key=lambda c: rank.get(c.tag, len(rank))):    # stable within a tag
        note.append(c)


def _attributes(measure, divisions, key, beats, beat_type, staves, clefs):
    """the first measure's <attributes> in schema order: divisions, key, time, staves, clefs"""
    old = measure.find("attributes")
    att = ET.Element("attributes")
    ET.SubElement(att, "divisions").text = str(divisions)
    k = ET.SubElement(att, "key")
    ET.SubElement(k, "fifths").text = str(key)
    t = ET.SubElement(att, "time")
    ET.SubElement(t, "beats").text = str(beats)
    ET.SubElement(t, "beat-type").text = str(beat_type)
    ET.SubElement(att, "staves").text = str(staves)
    for n in range(1, staves + 1):
        c = ET.SubElement(att, "clef", {"number": str(n)})
        sign, line = CLEF.get(clefs.get(n, "treble"), CLEF["treble"])
        ET.SubElement(c, "sign").text = sign
        ET.SubElement(c, "line").text = line
    if old is not None:
        i = list(measure).index(old)
        measure.remove(old)
        measure.insert(i, att)
    else:
        measure.insert(0, att)


def _events(measure):
    """[(onset ticks, note element, is chord member)] in document order"""
    out, pos, last = [], 0, 0
    for c in measure:
        if c.tag == "backup":
            pos -= int(c.findtext("duration"))
        elif c.tag == "forward":
            pos += int(c.findtext("duration"))
        elif c.tag == "note":
            d = int(c.findtext("duration", "0"))
            if c.find("chord") is not None:
                out.append((last, c, True))
            else:
                out.append((pos, c, False))
                last = pos
                pos += d
    return out


BEAMS = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4, "128th": 5}    # beams a note of this type carries


def _beam_groups(heads, beat):
    """heads: [(onset, duration, note)] of one voice in onset order -> list of groups (lists) to beam"""
    groups, cur, cur_beat = [], [], None
    for onset, dur, note in heads:
        beamable = note.find("rest") is None and dur > 0 and note.findtext("type") in BEAMS
        b = onset // beat
        fits = beamable and (onset + dur) <= (b + 1) * beat
        if fits and cur and cur_beat == b and cur[-1][0] + cur[-1][1] == onset:
            cur.append((onset, dur, note))
            continue
        if len(cur) >= 2:
            groups.append(cur)
        cur, cur_beat = ([(onset, dur, note)], b) if fits else ([], None)
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def _beam(group):
    """beam number 1 joins the whole group; number n joins neighbours that both carry n beams (16ths for 2, ...),
    a lone one gets a hook"""
    counts = [BEAMS[n.findtext("type")] for _, _, n in group]
    for level in range(1, max(counts) + 1):
        flags = [c >= level for c in counts]
        for i, (_, _, note) in enumerate(group):
            if not flags[i]:
                continue
            prev = i > 0 and flags[i - 1]
            nxt = i + 1 < len(group) and flags[i + 1]
            if prev and nxt:
                kind = "continue"
            elif nxt:
                kind = "begin"
            elif prev:
                kind = "end"
            else:
                kind = "forward hook" if i == 0 else "backward hook"
            ET.SubElement(note, "beam", {"number": str(level)}).text = kind


def engrave_page(part, headers):
    """add notation to the measures of one page's <part> in place; returns counts of what was added"""
    key, clefs = page_header(headers)
    stats = Counter()
    divisions, beats, beat_type, staves = 480, 4, 4, max(clefs) if clefs else 2
    measures = part.findall("measure")
    for j, m in enumerate(measures):
        att = m.find("attributes")
        if att is not None:
            divisions = int(att.findtext("divisions", divisions))
            staves = int(att.findtext("staves", staves))
            t = att.find("time")
            if t is not None:
                beats, beat_type = int(t.findtext("beats")), int(t.findtext("beat-type"))
        if j == 0:
            _attributes(m, divisions, key, beats, beat_type, staves, clefs)
            stats["key"] = key
        length = divisions * 4 * beats // beat_type
        compound = beat_type == 8 and beats % 3 == 0 and beats > 3
        beat = divisions * 3 // 2 if compound else divisions * 4 // beat_type if beat_type >= 4 else divisions
        events = _events(m)
        # types, dots, triplets, whole-measure rests
        for onset, note, chord in events:
            d = int(note.findtext("duration", "0"))
            if d <= 0 or note.find("type") is not None:
                continue
            rest = note.find("rest")
            if rest is not None and d == length:
                rest.set("measure", "yes")
                continue
            name, dots, tup = note_type(Fraction(d, divisions))
            ET.SubElement(note, "type").text = name
            for _ in range(dots):
                ET.SubElement(note, "dot")
            if tup and note.find("time-modification") is None:
                tm = ET.SubElement(note, "time-modification")
                ET.SubElement(tm, "actual-notes").text = str(tup[0])
                ET.SubElement(tm, "normal-notes").text = str(tup[1])
                stats["tuplet_notes"] += 1
            stats["types"] += 1
        # written accidentals: per staff, in onset order, against the key and earlier notes of the measure
        state = {}
        for onset, note, chord in sorted(events, key=lambda e: e[0]):
            p = note.find("pitch")
            if p is None or note.find("accidental") is not None:
                continue
            staff = note.findtext("staff", "1")
            step, octave = p.findtext("step"), p.findtext("octave")
            alter = int(round(float(p.findtext("alter", "0"))))
            current = state.get((staff, step, octave), key_alter(key, step))
            if alter != current and alter in ACCIDENTAL:
                ET.SubElement(note, "accidental").text = ACCIDENTAL[alter]
                state[(staff, step, octave)] = alter
                stats["accidentals"] += 1
        # beams, per voice and staff, chord heads only
        voices = defaultdict(list)
        for onset, note, chord in events:
            if not chord:
                voices[(note.findtext("staff", "1"), note.findtext("voice", "1"))].append(
                    (onset, int(note.findtext("duration", "0")), note))
        for heads in voices.values():
            for g in _beam_groups(sorted(heads, key=lambda h: h[0]), beat):
                _beam(g)
                stats["beam_groups"] += 1
        for _, note, _ in events:
            _reorder(note)
    return dict(stats)


def name_part(root):
    """'Music' (the recognizer's placeholder) -> 'Piano' for a two-staff part"""
    for sp in root.iter("score-part"):
        n = sp.find("part-name")
        if n is not None and n.text == "Music":
            n.text = "Piano"
