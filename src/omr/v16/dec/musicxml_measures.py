"""Parse real MusicXML into measure-level ground-truth events.

Ground truth per event (a chord counts as ONE event, since our problem is
rhythm/voice, not pitch):

    voice   : canonical (staff, raw voice) identity, used only for scoring
    staff   : 1-based staff number
    onset   : ticks from measure start (1920 ticks == whole note)
    dur     : actual sounding duration in ticks
    d       : division class, base = 1920 * 2**-d   (whole=0, half=1, quarter=2 ...)
    dots    : number of dots
    warp    : tuple (actual, normal) from <time-modification>, or (1, 1)

Written duration = 1920 * 2**-d * (2 - 2**-dots);  actual = written * normal/actual_notes.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

TICKS_PER_QUARTER = 480
WHOLE = 4 * TICKS_PER_QUARTER  # 1920

TYPE_TO_D = {
    "whole": 0, "half": 1, "quarter": 2, "eighth": 3,
    "16th": 4, "32nd": 5, "64th": 6, "128th": 7, "breve": -1, "long": -2,
}


def written_ticks(d: int, dots: int) -> float:
    """1920 * 2**-d * (2 - 2**-dots), the Starry duration formula."""
    return WHOLE * (2.0 ** -d) * (2.0 - 2.0 ** -dots)


_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass
class Event:
    eid: int
    voice: tuple           # (staff, raw voice string) - identity, for scoring only
    staff: int
    onset: int             # ticks
    dur: int               # ticks (actual, warp applied)
    d: int
    dots: int
    warp: tuple            # (actual_notes, normal_notes); (1,1) == none
    is_rest: bool
    n_noteheads: int = 1
    x_px: float = 0.0
    hint_d: int = -1       # noisy upstream soft hint (filled by the visual layer)
    hint_dots: int = -1
    hint_warp: tuple = (1, 1)
    # PITCH.  Deliberately out of scope at first ("rhythm/voice, not pitch"), but the
    # per-bucket failure decomposition showed that 50% (5-8 events) and 65% (13+
    # events) of the remaining failures involve a WRONG VOICE PARTITION, and 369
    # measures carry three voices on two staves.  Two voices sharing a staff are
    # indistinguishable from (x, staff) alone -- only the vertical position tells
    # them apart.  -1 means "not available" (rests, unpitched).
    midi: int = -1         # 12*(octave+1) + semitone, matching MusicXML <pitch>
    # STEM DIRECTION is the engraver's own voice marker: a second voice on one staff is
    # conventionally written with the opposite stem.  Measured on the source: present
    # on 99.8% of real notes, constant inside a voice 97.4% of the time, and distinct
    # between co-staff voices in 81.3% of cases -- against 20.9% for pitch ordering,
    # which is indistinguishable from chance.  1 = up, -1 = down, 0 = unknown.
    stem: int = 0

    @property
    def written(self) -> float:
        return written_ticks(self.d, self.dots)


@dataclass
class Measure:
    piece: str
    index: int
    beats: int
    beat_type: int
    length: int            # ticks expected by the time signature
    events: list = field(default_factory=list)

    @property
    def voices(self) -> list:
        seen, out = set(), []
        for e in self.events:
            if e.voice not in seen:
                seen.add(e.voice)
                out.append(e.voice)
        return out


def _txt(node, tag, default=None):
    c = node.find(tag)
    return c.text.strip() if c is not None and c.text else default


def _int(node, tag, default=0):
    c = node.find(tag)
    try:
        return int(c.text.strip())
    except (AttributeError, ValueError, TypeError):
        return default


def load_root(path: str):
    """Open a .musicxml/.xml or a compressed .mxl and return its root element."""
    if path.lower().endswith(".mxl"):
        import io
        import zipfile

        with zipfile.ZipFile(path) as z:
            target = None
            try:
                cont = ET.fromstring(z.read("META-INF/container.xml"))
                rf = cont.find(".//rootfile")
                if rf is not None:
                    target = rf.get("full-path")
            except KeyError:
                pass
            if target is None or target not in z.namelist():
                cands = [n for n in z.namelist()
                         if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
                if not cands:
                    raise ValueError("no score file inside mxl")
                target = cands[0]
            return ET.fromstring(z.read(target))
    with open(path, "rb") as fh:
        return ET.fromstring(fh.read())


def parse_piece(path: str, piece_name: str) -> list:
    """Return a list of Measure objects with ground-truth events."""
    root = load_root(path)
    out = []
    for part in root.findall("part"):
        divisions = 1
        beats, beat_type = 4, 4
        staves = 1
        for mi, m in enumerate(part.findall("measure")):
            # ---- attributes are cumulative across measures ----
            attrs = m.find("attributes")
            if attrs is not None:
                d = _int(attrs, "divisions", 0)
                if d:
                    divisions = d
                t = attrs.find("time")
                if t is not None:
                    b, bt = _txt(t, "beats"), _txt(t, "beat-type")
                    if b and bt:
                        try:
                            beats, beat_type = int(b), int(bt)
                        except ValueError:
                            pass
                s = _int(attrs, "staves", 0)
                if s:
                    staves = s

            scale = TICKS_PER_QUARTER / float(divisions)
            length = int(round(beats * (4.0 / beat_type) * TICKS_PER_QUARTER))

            # ---- walk the measure with ONE shared cursor (MusicXML semantics:
            #      <backup>/<forward> move a per-part position, not a per-staff one) ----
            cursor = 0
            events, prev = [], None
            skip = False
            for n in m:
                tag = n.tag
                if tag == "backup":
                    cursor -= int(round(_int(n, "duration") * scale))
                elif tag == "forward":
                    cursor += int(round(_int(n, "duration") * scale))
                elif tag == "note":
                    if n.find("grace") is not None:
                        continue                 # graces are out of scope here
                    dur = _int(n, "duration")
                    st = _int(n, "staff", 1)
                    is_chord = n.find("chord") is not None
                    is_rest = n.find("rest") is not None
                    voice = _txt(n, "voice", "1")
                    if is_chord and prev is not None:
                        prev.n_noteheads += 1
                        continue
                    t = _txt(n, "type")
                    dcls = TYPE_TO_D.get(t, None)
                    if dcls is None:
                        # no <type> (e.g. a whole-measure rest): derive from the duration
                        q = (dur * scale) / WHOLE
                        dcls = None
                        for k in range(-2, 8):
                            if abs(2.0 ** -k - q) < 1e-6:
                                dcls = k
                                break
                        if dcls is None:
                            skip = True
                            break
                    dots = len(n.findall("dot"))
                    tm = n.find("time-modification")
                    warp = (1, 1)
                    if tm is not None:
                        an, nn = _int(tm, "actual-notes", 1), _int(tm, "normal-notes", 1)
                        if an and nn:
                            warp = (an, nn)
                    pts = n.findall("pitch")
                    midi = -1
                    if pts:
                        pn = pts[0]
                        step = _txt(pn, "step", "")
                        octv = _int(pn, "octave", 0)
                        alter = _int(pn, "alter", 0)
                        semi = _STEP.get(step)
                        if semi is not None and octv:
                            midi = 12 * (octv + 1) + semi + alter
                    _st = _txt(n, "stem", "").lower()
                    stem = 1 if _st == "up" else (-1 if _st == "down" else 0)
                    ev = Event(
                        eid=len(events), voice=(st, voice), staff=st,
                        onset=cursor, dur=int(round(dur * scale)),
                        d=dcls, dots=dots, warp=warp, is_rest=is_rest,
                        midi=midi, stem=stem,
                    )
                    events.append(ev)
                    prev = ev
                    cursor = ev.onset + ev.dur
            if skip or not events:
                continue

            # ---- keep only well-formed, non-pickup measures ----
            if length <= 0 or len(events) > MAX_EVENTS_PER_MEASURE:
                continue
            per_voice_end = {}
            ok = True
            for e in events:
                per_voice_end[e.voice] = max(per_voice_end.get(e.voice, 0), e.onset + e.dur)
                if e.onset < 0:
                    ok = False
            if not ok or any(v != length for v in per_voice_end.values()):
                continue                      # drops pickups and broken cursor maths
            if len(per_voice_end) > MAX_VOICES_PER_MEASURE:
                continue

            out.append(Measure(piece=piece_name, index=mi + 1, beats=beats,
                               beat_type=beat_type, length=length, events=events))
    return out


MAX_EVENTS_PER_MEASURE = 32
MAX_VOICES_PER_MEASURE = 4

