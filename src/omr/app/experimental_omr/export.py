"""
输出：MusicXML 与 Standard MIDI File。

为什么不用 music21 / mido 这些现成库？
------------------------------------
1. **P3 要把这条链搬到 Zynq 的 Ubuntu rootfs 上。** 少一个重依赖就少一堆麻烦，
   单声部的 MusicXML 和 SMF 都简单到不值得引依赖。
2. 数据模型是我们自己的（model.py），来回转换反而增加出错面。
3. 我们对"一个音符该写成什么"有完全控制权（比如 alter 要不要含调号），
   用库反而要跟它的语义打架。

music21 仍然有用 —— 作为**独立的第三方校验器**：把它读我们的 MusicXML 能不能
成功解析、小节时值对不对，当作一道额外的关卡（见 cli.py 的 `validate`）。
"""

from __future__ import annotations

import datetime as _dt
import struct
from xml.sax.saxutils import escape

from . import theory
from .model import PPQ, Event, Measure, Score

# ---------------------------------------------------------------------------
# 公共小工具
# ---------------------------------------------------------------------------


def _split_type_and_dots(ticks: int) -> tuple[str, int]:
    """tick -> (MusicXML <type> 名称, 附点个数)。"""
    if ticks == PPQ // 3:
        return "eighth", 0
    name = theory.ticks_to_name(ticks)
    if name.startswith("dotted_"):
        name, dots = name[len("dotted_"):], 1
    else:
        dots = 0
    # The internal duration names are descriptive; MusicXML's <type>
    # enumeration uses numeric values for short notes.
    return {"sixteenth": "16th", "thirty_second": "32nd"}.get(name, name), dots


CLEF_XML = {
    "treble": ("G", 2),
    "bass": ("F", 4),
    "alto": ("C", 3),
    "tenor": ("C", 4),
}

ACCIDENTAL_XML = {
    "sharp": "sharp",
    "flat": "flat",
    "natural": "natural",
    "double_sharp": "double-sharp",
    "double_flat": "flat-flat",
}


def _beam_marks(events: list[Event]) -> dict[int, list[tuple[int, str]]]:
    """Group short notes within each quarter-note beat for MusicXML engraving.

    The score model stores durations and onsets, not beam geometry.  Keep this
    notation-only decision in the exporter; rests and time gaps break a group.
    """
    roots: list[Event] = []
    previous: Event | None = None
    for event in sorted(events, key=lambda item: item.onset_ticks):
        chord_tone = (previous is not None and event.chord_id is not None
                      and event.chord_id == previous.chord_id
                      and event.onset_ticks == previous.onset_ticks
                      and event.duration_ticks == previous.duration_ticks)
        if not chord_tone:
            roots.append(event)
        previous = event

    marks: dict[int, list[tuple[int, str]]] = {}

    def commit(group: list[Event]) -> None:
        if len(group) < 2:
            return
        for level in (1, 2, 3):
            eligible = ([True] * len(group) if level == 1 else
                        [e.duration_ticks <= PPQ // (2 ** level)
                         for e in group])
            for index, event in enumerate(group):
                if not eligible[index]:
                    continue
                before = index > 0 and eligible[index - 1]
                after = index + 1 < len(group) and eligible[index + 1]
                if before and after:
                    value = "continue"
                elif after:
                    value = "begin"
                elif before:
                    value = "end"
                elif level > 1:
                    value = "backward hook" if index == len(group) - 1 else "forward hook"
                else:
                    continue
                marks.setdefault(id(event), []).append((level, value))

    group: list[Event] = []
    for event in roots:
        beat = event.onset_ticks // PPQ
        fits = (event.type == "note" and event.duration_ticks < PPQ
                and event.onset_ticks + event.duration_ticks <= (beat + 1) * PPQ)
        if not fits or (group and (group[-1].onset_ticks + group[-1].duration_ticks
                                   != event.onset_ticks
                                   or group[-1].onset_ticks // PPQ != beat)):
            commit(group)
            group = []
        if fits:
            group.append(event)
    commit(group)
    return marks


# ---------------------------------------------------------------------------
# MusicXML
# ---------------------------------------------------------------------------


def to_musicxml(score: Score, software: str = "FPGA-OMR/0.1") -> str:
    """生成 MusicXML 4.0 partwise。单声部单谱表，但结构上支持多 part / 多 staff。"""
    L: list[str] = []
    a = L.append
    a('<?xml version="1.0" encoding="UTF-8"?>')
    a('<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
      '"http://www.musicxml.org/dtds/partwise.dtd">')
    a('<score-partwise version="4.0">')
    if score.title:
        a("  <work><work-title>%s</work-title></work>" % escape(score.title))
    a("  <identification>")
    if score.composer:
        a("    <creator type=\"composer\">%s</creator>" % escape(score.composer))
    a("    <encoding>")
    a("      <software>%s</software>" % escape(software))
    a("      <encoding-date>%s</encoding-date>"
      % _dt.date.today().isoformat())
    a("    </encoding>")
    a("  </identification>")

    # part-list
    a("  <part-list>")
    for pi, part in enumerate(score.parts, start=1):
        a('    <score-part id="P%d">' % pi)
        a("      <part-name>%s</part-name>" % escape(part.name))
        a("    </score-part>")
    a("  </part-list>")

    for pi, part in enumerate(score.parts, start=1):
        a('  <part id="P%d">' % pi)
        staff_list = part.staves or []
        # 各谱表的小节数可能不同，取最大者
        n_measures = max((len(s.measures) for s in staff_list), default=0)
        previous_key = staff_list[0].key_fifths if staff_list else 0
        previous_clefs = [staff.clef for staff in staff_list]
        for mi in range(n_measures):
            measures = [s.measures[mi] if mi < len(s.measures) else None
                        for s in staff_list]
            ref = next((m for m in measures if m is not None), None)
            if ref is None:
                continue
            current_key = ref.key_fifths if ref.key_fifths is not None else previous_key
            current_clefs = [m.clef if m is not None and m.clef is not None
                             else previous_clefs[si]
                             for si, m in enumerate(measures)]
            a('    <measure number="%d">' % ref.number)
            if mi == 0:
                a("      <attributes>")
                a("        <divisions>%d</divisions>" % PPQ)
                a("        <key><fifths>%d</fifths></key>" % current_key)
                beats, beat_type = ref.time_signature
                a("        <time><beats>%d</beats><beat-type>%d</beat-type></time>"
                  % (beats, beat_type))
                if len(staff_list) > 1:
                    a("        <staves>%d</staves>" % len(staff_list))
                for staff, current_clef in zip(staff_list, current_clefs):
                    sign, line = CLEF_XML.get(current_clef, ("G", 2))
                    number = ' number="%d"' % staff.number if len(staff_list) > 1 else ""
                    a("        <clef%s><sign>%s</sign><line>%d</line></clef>"
                      % (number, sign, line))
                a("      </attributes>")
            elif current_key != previous_key or current_clefs != previous_clefs:
                a("      <attributes>")
                if current_key != previous_key:
                    a("        <key><fifths>%d</fifths></key>" % current_key)
                for staff, current_clef, previous_clef in zip(
                        staff_list, current_clefs, previous_clefs):
                    if current_clef != previous_clef:
                        sign, line = CLEF_XML.get(current_clef, ("G", 2))
                        number = (' number="%d"' % staff.number
                                  if len(staff_list) > 1 else "")
                        a("        <clef%s><sign>%s</sign><line>%d</line></clef>"
                          % (number, sign, line))
                a("      </attributes>")
            previous_key = current_key
            previous_clefs = current_clefs
            cursor = 0
            for si, m in enumerate(measures):
                if m is None:
                    continue
                staff_no = staff_list[si].number
                for v in m.voices:
                    # A MusicXML backup rewinds the actual stream position. The
                    # recognized staff may be incomplete, so the meter length is
                    # not a valid backup amount here.
                    if cursor:
                        a("      <backup><duration>%d</duration></backup>" % cursor)
                        cursor = 0
                    previous = None
                    beam_marks = _beam_marks(v.events)
                    for event_index, ev in enumerate(sorted(
                            v.events, key=lambda event: event.onset_ticks)):
                        if ev.duration_ticks <= 0 or ev.onset_ticks < 0:
                            raise ValueError("MusicXML event has an invalid onset or duration")
                        chord_tone = (
                            previous is not None
                            and ev.chord_id is not None
                            and ev.chord_id == previous.chord_id
                            and ev.onset_ticks == previous.onset_ticks
                            and ev.duration_ticks == previous.duration_ticks
                        )
                        if not chord_tone:
                            distance = ev.onset_ticks - cursor
                            if distance > 0:
                                a("      <forward><duration>%d</duration></forward>" % distance)
                            elif distance < 0:
                                a("      <backup><duration>%d</duration></backup>" % -distance)
                            cursor = ev.onset_ticks
                        a(_event_xml(ev, staff_no, v.id, len(staff_list), chord_tone,
                                     beam_marks.get(id(ev), []),
                                     f"nP{pi}M{mi + 1}S{staff_no}V{v.id}E{event_index}"))
                        if not chord_tone:
                            cursor += ev.duration_ticks
                        previous = ev
            a("    </measure>")
        a("  </part>")
    a("</score-partwise>")
    return "\n".join(L) + "\n"


def _event_xml(ev: Event, staff_no: int, voice_id: int, n_staves: int,
               chord_tone: bool = False,
               beam_marks: list[tuple[int, str]] | None = None,
               xml_id: str | None = None) -> str:
    L: list[str] = []
    a = L.append
    a('      <note id="%s">' % xml_id if xml_id else "      <note>")
    if chord_tone:
        a("        <chord/>")
    if ev.type == "rest":
        a("        <rest/>")
    else:
        letter, octave, offset = _parse_name(ev.sounding_name or ev.written_name)
        a("        <pitch>")
        a("          <step>%s</step>" % letter)
        if offset:
            a("          <alter>%d</alter>" % offset)
        a("          <octave>%d</octave>" % octave)
        a("        </pitch>")
    a("        <duration>%d</duration>" % ev.duration_ticks)
    if ev.tie in ("start", "continue"):
        a('        <tie type="start"/>')
    elif ev.tie == "stop":
        a('        <tie type="stop"/>')
    a("        <voice>%d</voice>" % voice_id)
    type_name, dots = _split_type_and_dots(ev.duration_ticks)
    a("        <type>%s</type>" % type_name)
    for _ in range(dots):
        a("        <dot/>")
    if ev.accidental:
        a("        <accidental>%s</accidental>"
          % ACCIDENTAL_XML.get(ev.accidental, ev.accidental))
    if ev.duration_ticks == PPQ // 3:
        a("        <time-modification><actual-notes>3</actual-notes>"
          "<normal-notes>2</normal-notes><normal-type>eighth</normal-type>"
          "</time-modification>")
    if n_staves > 1:
        a("        <staff>%d</staff>" % staff_no)
    for number, value in beam_marks or []:
        a('        <beam number="%d">%s</beam>' % (number, value))
    if (ev.tie in ("start", "continue", "stop")
            or ev.slur in ("start", "stop")
            or ev.tuplet in ("start", "stop")):
        a("        <notations>")
        if ev.tie in ("start", "continue"):
            a('          <tied type="start"/>')
        if ev.tie == "stop":
            a('          <tied type="stop"/>')
        if ev.slur in ("start", "stop"):
            a('          <slur type="%s" number="1"/>' % ev.slur)
        if ev.tuplet in ("start", "stop"):
            a('          <tuplet type="%s" number="1"/>' % ev.tuplet)
        a("        </notations>")
    a("      </note>")
    return "\n".join(L)


def _parse_name(name: str) -> tuple[str, int, int]:
    """'F#4' -> ('F', 4, 1)；空串或休止符返回 ('C', 4, 0)。"""
    if not name:
        return "C", 4, 0
    try:
        return theory.parse_pitch_name(name)
    except Exception:
        return "C", 4, 0


def write_musicxml(score: Score, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_musicxml(score))


# ---------------------------------------------------------------------------
# Standard MIDI File (format 1)
# ---------------------------------------------------------------------------


def _vlq(n: int) -> bytes:
    """MIDI 变长量（variable-length quantity）。"""
    if n < 0:
        raise ValueError("delta time 不能为负")
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _meta(kind: int, data: bytes) -> bytes:
    return bytes([0xFF, kind]) + _vlq(len(data)) + data


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">I", len(payload)) + payload


def to_midi(score: Score, tempo_bpm: float = 100.0, program: int = 0,
            channel: int = 0) -> bytes:
    """生成 SMF format 1：track 0 放速度/拍号/调号，track 1 放音符。

    format 1 的好处：所有 track 共享同一条时间轴，后面扩展多声部/多乐器时
    每个声部一条 track 即可，不需要改结构。
    """
    staves = [s for p in score.parts for s in p.staves]
    if not staves:
        raise ValueError("乐谱为空，无法导出 MIDI")
    ref_measure = next((m for s in staves for m in s.measures), None)
    ts = ref_measure.time_signature if ref_measure else (4, 4)
    fifths = staves[0].key_fifths

    # --- track 0：tempo track -------------------------------------------
    us_per_quarter = int(round(60_000_000.0 / max(1.0, tempo_bpm)))
    t0 = bytearray()
    t0 += _vlq(0) + _meta(0x51, struct.pack(">I", us_per_quarter)[1:])   # tempo
    t0 += _vlq(0) + _meta(0x58, bytes([ts[0], ts[1].bit_length() - 1, 24, 8]))  # 拍号
    sf = max(-7, min(7, fifths))
    t0 += _vlq(0) + _meta(0x59, struct.pack(">bB", sf, 0))               # 调号
    t0 += _vlq(0) + _meta(0x03, "OMR".encode("ascii"))                   # track name
    t0 += _vlq(0) + _meta(0x2F, b"")                                     # end of track

    # --- track 1：音符 ---------------------------------------------------
    events: list[tuple[int, int, bytes]] = []
    events.append((0, 0, bytes([0xC0 | channel, program & 0x7F])))
    if score.title:
        events.append((0, 0, _meta(0x03, score.title.encode("utf-8", "replace"))))

    any_note = False
    for part, st, m in _iter_with_measure_offset(score):
        for v in m.voices:
            for ev in v.events:
                if ev.type != "note" or ev.pitch < 0:
                    continue
                any_note = True
                t_on = ev.onset_ticks
                t_off = ev.onset_ticks + max(1, ev.duration_ticks)
                events.append((t_on, 2, bytes([0x90 | channel, ev.pitch & 0x7F, 96])))
                # 同一 tick 上先 note-off 再 note-on，避免同音高被吞掉
                events.append((t_off, 1, bytes([0x80 | channel, ev.pitch & 0x7F, 64])))
    if not any_note:
        events.append((0, 1, bytes([0x80 | channel, 60, 0])))

    events.sort(key=lambda e: (e[0], e[1]))
    t1 = bytearray()
    prev = 0
    for tick, _order, data in events:
        t1 += _vlq(tick - prev) + data
        prev = tick
    t1 += _vlq(0) + _meta(0x2F, b"")

    header = _chunk(b"MThd", struct.pack(">HHH", 1, 2, PPQ))
    return bytes(header + _chunk(b"MTrk", bytes(t0)) + _chunk(b"MTrk", bytes(t1)))


def _iter_with_measure_offset(score: Score):
    """遍历所有 (part, staff, measure)。tick 已经是小节内相对值。

    注意：本项目的 onset_ticks 是**小节内相对 tick**（每个小节从 0 开始），
    因为真实 OMR 里跨小节的绝对时间要先确定拍号和小节划分才敢算。
    导出 MIDI 时需要累加成全局时间轴。
    """
    for p in score.parts:
        for s in p.staves:
            base = 0
            for m in s.measures:
                yield p, s, m
                base += m.expected_ticks


def to_absolute_events(score: Score) -> list[tuple[int, Event]]:
    """把"小节内相对 onset"转成全局绝对 tick 的事件流。

    ⚠️ 实现已统一到 `model.iter_absolute_events` —— 播放、MIDI 导出、
    放音高亮三者必须共用同一份换算，否则会出现"音频和屏幕高亮对不上"。
    """
    from .model import iter_absolute_events
    return [(t, ev) for t, _m, ev in iter_absolute_events(score)]


def to_midi_absolute(score: Score, tempo_bpm: float = 100.0, program: int = 0,
                     channel: int = 0) -> bytes:
    """和 `to_midi` 相同，但使用 `to_absolute_events` 的实际小节长度推进。"""
    staves = [s for p in score.parts for s in p.staves]
    if not staves:
        raise ValueError("乐谱为空")
    ref = next((m for s in staves for m in s.measures), None)
    ts = ref.time_signature if ref else (4, 4)
    us_per_quarter = int(round(60_000_000.0 / max(1.0, tempo_bpm)))

    t0 = bytearray()
    t0 += _vlq(0) + _meta(0x51, struct.pack(">I", us_per_quarter)[1:])
    t0 += _vlq(0) + _meta(0x58, bytes([ts[0], ts[1].bit_length() - 1, 24, 8]))
    t0 += _vlq(0) + _meta(0x59, struct.pack(">bB", max(-7, min(7, staves[0].key_fifths)), 0))
    t0 += _vlq(0) + _meta(0x03, "OMR".encode("ascii"))
    t0 += _vlq(0) + _meta(0x2F, b"")

    events: list[tuple[int, int, bytes]] = [(0, 0, bytes([0xC0 | channel, program & 0x7F]))]
    for t, ev in to_absolute_events(score):
        if ev.type != "note" or ev.pitch < 0:
            continue
        events.append((t, 2, bytes([0x90 | channel, ev.pitch & 0x7F, 96])))
        events.append((t + max(1, ev.duration_ticks), 1,
                       bytes([0x80 | channel, ev.pitch & 0x7F, 64])))
    events.sort(key=lambda e: (e[0], e[1]))
    t1 = bytearray()
    prev = 0
    for tick, _o, data in events:
        t1 += _vlq(tick - prev) + data
        prev = tick
    t1 += _vlq(0) + _meta(0x2F, b"")

    return bytes(_chunk(b"MThd", struct.pack(">HHH", 1, 2, PPQ))
                 + _chunk(b"MTrk", bytes(t0)) + _chunk(b"MTrk", bytes(t1)))


def write_midi(score: Score, path: str, **kw) -> None:
    with open(path, "wb") as f:
        f.write(to_midi_absolute(score, **kw))
