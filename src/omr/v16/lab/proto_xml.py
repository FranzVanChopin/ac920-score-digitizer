"""MusicXML writing for the prototype (shared by proto.py and the board run).

Streams are (staff, [(onset, dur, event)]) in time order; durations in ticks of 480 per quarter,
exactly the decoder's grid.  Measure rests (extract.measure_rests) are written as whole bars.
"""
import collections
import xml.etree.ElementTree as ET

import extract as X

def _note(parent, staff, voice, dur, pitch=None, chord=False):
    n = ET.SubElement(parent, "note")
    if chord:
        ET.SubElement(n, "chord")
    if pitch is None:
        ET.SubElement(n, "rest")
    else:
        p = ET.SubElement(n, "pitch")
        ET.SubElement(p, "step").text = pitch[0]
        if int(pitch[2] or 0):
            ET.SubElement(p, "alter").text = str(int(pitch[2]))
        ET.SubElement(p, "octave").text = str(int(pitch[1]))
    ET.SubElement(n, "duration").text = str(int(dur))
    ET.SubElement(n, "voice").text = str(voice)
    ET.SubElement(n, "staff").text = str(staff)


def write_xml(path, measures, streams_per_measure):
    """streams_per_measure[i] = list of streams; a stream = (staff, [(onset, dur, event)]) in time
    order.  Durations in ticks of 480 per quarter, which is exactly the decoder's grid."""
    root = ET.Element("score-partwise", version="3.1")
    pl = ET.SubElement(root, "part-list")
    sp = ET.SubElement(pl, "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Music"
    part = ET.SubElement(root, "part", id="P1")
    n_staves = max((m.n_staves for m in measures), default=1)
    prev_meter = None
    for m, streams in zip(measures, streams_per_measure):
        me = ET.SubElement(part, "measure", number=str(m.index + 1))
        if m.index == 0 or (m.beats, m.beat_type) != prev_meter:
            at = ET.SubElement(me, "attributes")
            if m.index == 0:
                ET.SubElement(at, "divisions").text = "480"
                ET.SubElement(at, "staves").text = str(n_staves)
            t = ET.SubElement(at, "time")
            ET.SubElement(t, "beats").text = str(m.beats)
            ET.SubElement(t, "beat-type").text = str(m.beat_type)
            prev_meter = (m.beats, m.beat_type)
        cursor = 0
        for vi, (staff, evs) in enumerate(streams, start=1):
            if cursor:
                b = ET.SubElement(me, "backup")
                ET.SubElement(b, "duration").text = str(int(cursor))
                cursor = 0
            for onset, dur, e in evs:
                onset, dur = int(round(onset)), int(round(dur))
                if dur <= 0:
                    continue
                if onset > cursor:
                    f = ET.SubElement(me, "forward")
                    ET.SubElement(f, "duration").text = str(onset - cursor)
                    cursor = onset
                elif onset < cursor:
                    b = ET.SubElement(me, "backup")
                    ET.SubElement(b, "duration").text = str(cursor - onset)
                    cursor = onset
                if e.is_rest:
                    _note(me, e.staff, vi, dur)
                else:
                    for k, p in enumerate(e.pitches):
                        _note(me, e.staff, vi, dur, p, chord=k > 0)
                cursor += dur
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def ours_streams(m, cm, answer):
    by_cid = {c.cid: c for c in cm.cands} if cm else {}
    streams = []
    for voice in answer or []:
        evs = []
        for cid, _x, onset, dur, *_rest in voice:
            evs.append((onset, dur, m.events[by_cid[cid].gt_eid]))
        if evs:
            streams.append((evs[0][2].staff, sorted(evs, key=lambda t: t[0])))
    for i in sorted(X.measure_rests(m)):
        streams.append((m.events[i].staff, [(0, m.length, m.events[i])]))
    return streams


def codex_streams(m):
    groups = collections.defaultdict(list)
    for e in m.events:
        c = e.codex
        if c["onset"] is None or not c["duration"]:
            continue
        groups[(e.staff, c["voice"])].append((c["onset"], c["duration"], e))
    return [(k[0], sorted(v, key=lambda t: t[0])) for k, v in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))]


