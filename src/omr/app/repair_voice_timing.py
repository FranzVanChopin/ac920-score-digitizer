"""Conservative MusicXML timing repair using symbol reading order and noteheads.

Homr usually emits note elements in approximate left-to-right order even when
its voice labels and backups are wrong. Search for two complete, monotone voice
streams, but change a measure only when every minimum-cost assignment implies
the same note onsets. No source score or image ground truth is used.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

def _backup(duration: int) -> ET.Element:
    element = ET.Element("backup")
    ET.SubElement(element, "duration").text = str(duration)
    return element


def _forward(duration: int) -> ET.Element:
    element = ET.Element("forward")
    ET.SubElement(element, "duration").text = str(duration)
    return element


def _search(notes: list[ET.Element], target: int,
            note_x: dict[ET.Element, float] | None = None) -> tuple[int, tuple] | None:
    events = []
    for note in notes:
        duration = note.findtext("duration")
        if duration is None or int(duration) <= 0:
            return None
        events.append((int(duration), note.findtext("voice"),
                       note.find("chord") is not None))
    xs = [note_x.get(note) for note in notes] if note_x is not None else []
    use_geometry = (len(xs) == len(notes) and all(x is not None for x in xs)
                    and all(xs[index] >= xs[index - 1] - 3
                            for index in range(1, len(xs))))

    @lru_cache(maxsize=50_000)
    def solve(index: int, first: int, second: int, last: int,
              previous_voice: int, previous_onset: int) -> tuple | None:
        if index == len(events):
            return (0, (), frozenset({()})) if first == second == target else None
        duration, original_voice, chord = events[index]
        best = None
        for voice in (1, 2):
            cursor = first if voice == 1 else second
            merge = chord and voice == previous_voice
            onset = previous_onset if merge else cursor
            end = cursor if merge else cursor + duration
            if onset < last or end > target:
                continue
            if (use_geometry and index and
                    abs(xs[index] - xs[index - 1]) <= 3 and onset != last):
                continue
            next_first, next_second = ((end, second) if voice == 1
                                       else (first, end))
            following = solve(index + 1, next_first, next_second, onset,
                              voice, onset)
            if following is None:
                continue
            following_cost, following_path, following_signatures = following
            cost = following_cost + (0 if original_voice == str(voice) else 10)
            cost += 0 if not chord or merge else 1
            path = ((voice, onset, merge),) + following_path
            signatures = frozenset((onset,) + suffix
                                   for suffix in following_signatures)
            if best is None or cost < best[0]:
                best = (cost, path, signatures)
            elif cost == best[0]:
                # Two distinct timing signatures suffice to reject a repair.
                union = best[2] | signatures
                best = (cost, best[1], frozenset(sorted(union)[:2]))
        return best

    result = solve(0, 0, 0, 0, 0, 0)
    if result is None or len(result[2]) != 1:
        return None
    return result[0], result[1]


def repair_single_stream_measure(measure: ET.Element, target: int,
                                 note_x: dict[ET.Element, float] | None) -> dict | None:
    """Correct one stray voice label when the image proves a single upper stream.

    The score must already be overfull. Both staves must independently fill the
    bar, and every upper notehead must match a strictly ordered image column.
    This avoids guessing from the meter alone when genuine voices overlap.
    """
    if note_x is None or target <= 0 or target > 256:
        return None
    elements = list(measure)
    indices = [index for index, element in enumerate(elements)
               if element.tag == "note"]
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    if (any(element.tag not in {"note", "backup"}
            for element in elements[first:last + 1]) or
            any(element.tag != "barline" for element in elements[last + 1:]) or
            any(element.tag in {"backup", "forward"}
                for element in elements[:first])):
        return None
    notes = [element for element in elements[first:last + 1]
             if element.tag == "note"]
    if any(note.find("grace") is not None or note.find("rest") is not None
           for note in notes):
        return None
    top = [note for note in notes if note.findtext("staff", "1") == "1"]
    bass = [note for note in notes if note.findtext("staff", "1") == "2"]
    stray = [index for index, note in enumerate(top)
             if note.findtext("voice") == "2"]
    if (len(top) < 4 or len(top) > 24 or not bass or
            len(top) + len(bass) != len(notes) or
            len(stray) != 1 or stray[0] in (0, len(top) - 1) or
            top[stray[0]].find("chord") is not None or
            any(note.findtext("voice") not in {"1", "2"} for note in top) or
            len({note.findtext("voice") for note in bass}) != 1 or
            any(note not in note_x for note in top)):
        return None
    desired: dict[ET.Element, int] = {}
    for staff_notes in (top, bass):
        cursor = 0
        previous = None
        for note in staff_notes:
            duration_text = note.findtext("duration")
            if duration_text is None or not duration_text.isdigit():
                return None
            duration = int(duration_text)
            if duration <= 0:
                return None
            if note.find("chord") is not None:
                # Stacked chords have one head followed by any number of
                # chord heads; all share the same duration and image column.
                if previous is None or duration != int(previous.findtext("duration")):
                    return None
                desired[note] = desired[previous]
                if staff_notes is top and abs(note_x[note] - note_x[previous]) > 3:
                    return None
            else:
                desired[note] = cursor
                if staff_notes is top and previous is not None:
                    previous_x = note_x[previous]
                    if note_x[note] - previous_x <= 3:
                        return None
                cursor += duration
            previous = note
        if cursor != target:
            return None
    # A MusicXML chord refers to the immediately previous note element, even
    # across backups. Preserve that adjacency before rewriting any element.
    previous = None
    for note in notes:
        if note.find("chord") is not None and (
                previous is None or
                note.findtext("staff", "1") != previous.findtext("staff", "1") or
                desired[note] != desired[previous]):
            return None
        previous = note
    for note in top:
        note.find("voice").text = "1"
    rebuilt = elements[:first]
    cursor = 0
    previous_onset = None
    for note in notes:
        onset = desired[note]
        if note.find("chord") is not None:
            if onset != previous_onset:
                return None
        else:
            if onset < cursor:
                rebuilt.append(_backup(cursor - onset))
            elif onset > cursor:
                rebuilt.append(_forward(onset - cursor))
            cursor = onset + int(note.findtext("duration"))
        rebuilt.append(note)
        previous_onset = onset
    measure[:] = rebuilt + elements[last + 1:]
    return {"assignment_cost": 10, "notes": len(top),
            "geometry_constrained": True, "method": "single_stream"}


def repair_measure(measure: ET.Element, target: int,
                   note_x: dict[ET.Element, float] | None = None) -> dict | None:
    if target <= 0 or target > 256:
        return None
    elements = list(measure)
    indices = [index for index, element in enumerate(elements)
               if element.tag == "note"]
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    if any(element.tag not in {"note", "backup"}
           for element in elements[first:last + 1]):
        return None
    if any(element.tag != "barline" for element in elements[last + 1:]):
        return None
    if any(element.tag in {"backup", "forward"} for element in elements[:first]):
        return None
    notes = [element for element in elements[first:last + 1]
             if element.tag == "note"]
    if any(note.find("grace") is not None for note in notes):
        return None
    top = [note for note in notes if note.findtext("staff", "1") == "1"]
    bass = [note for note in notes if note.findtext("staff", "1") == "2"]
    if len(top) < 4 or len(top) > 24 or not bass or len(top) + len(bass) != len(notes):
        return None
    if any(note.find("rest") is not None for note in top):
        return None
    if {note.findtext("voice") for note in top} != {"1", "2"}:
        return None
    if len({note.findtext("voice") for note in bass}) != 1:
        return None
    if any(note.find("chord") is not None for note in bass):
        return None
    bass_duration = sum(int(note.findtext("duration", "0")) for note in bass
                        if note.find("chord") is None)
    if bass_duration != target:
        return None
    solution = _search(top, target, note_x)
    if solution is None:
        return None
    cost, assignments = solution
    # A chord marker references the immediately preceding note in document
    # order. Reject an interleaved chord instead of changing its semantics.
    assignment_by_note = dict(zip(top, assignments))
    top_previous = None
    all_previous = None
    for note in notes:
        if note in assignment_by_note:
            assignment = assignment_by_note[note]
            if assignment[2] and all_previous is not top_previous:
                return None
            top_previous = note
        all_previous = note

    desired: dict[ET.Element, int] = {}
    for note, (voice, _onset, merge) in zip(top, assignments):
        desired[note] = _onset
    bass_cursor = 0
    for note in bass:
        desired[note] = bass_cursor
        bass_cursor += int(note.findtext("duration", "0"))

    previous_onset = None
    for note in notes:
        effective_chord = (note.find("chord") is not None and
                           assignment_by_note.get(note, (None, None, False))[2])
        if effective_chord and desired[note] != previous_onset:
            return None
        previous_onset = desired[note]
    for note, (voice, _onset, merge) in zip(top, assignments):
        note.find("voice").text = str(voice)
        if note.find("chord") is not None and not merge:
            note.remove(note.find("chord"))

    # Preserve the original note order, including slur start/stop ordering.
    # MusicXML backups/forwards place each independent note on its chosen beat.
    rebuilt = elements[:first]
    cursor = 0
    previous_onset = None
    for note in notes:
        onset = desired[note]
        if note.find("chord") is not None:
            assert onset == previous_onset
        else:
            if onset < cursor:
                rebuilt.append(_backup(cursor - onset))
            elif onset > cursor:
                rebuilt.append(_forward(onset - cursor))
            cursor = onset + int(note.findtext("duration", "0"))
        rebuilt.append(note)
        previous_onset = onset
    measure[:] = rebuilt + elements[last + 1:]
    geometry_used = (note_x is not None and all(note in note_x for note in top)
                     and all(note_x[top[index]] >= note_x[top[index - 1]] - 3
                             for index in range(1, len(top))))
    return {"assignment_cost": cost, "notes": len(top),
            "geometry_constrained": geometry_used}


def repair_score(root: ET.Element,
                 initial_meter: tuple[int, int] | None = None,
                 *, audit_timing_fn=None, parse_meter_fn=None,
                 note_x: dict[ET.Element, float] | None = None) -> list[dict]:
    if audit_timing_fn is None or parse_meter_fn is None:
        from board_homr import audit_timing, parse_meter
        audit_timing_fn = audit_timing
        parse_meter_fn = parse_meter
    overfull = {(row["part"], row["measure"])
                for row in audit_timing_fn(root)["overfull_measures"]}
    repaired = []
    for part_index, part in enumerate(root.findall("part"), 1):
        divisions = 1
        meter = initial_meter
        for measure_index, measure in enumerate(part.findall("measure"), 1):
            for attributes in measure.findall("attributes"):
                value = attributes.findtext("divisions")
                if value is not None:
                    divisions = int(value)
                time = attributes.find("time")
                if time is not None:
                    meter = parse_meter_fn(f"{time.findtext('beats')}/"
                                           f"{time.findtext('beat-type')}")
            if (part_index, measure_index) not in overfull or meter is None:
                continue
            target = 4 * meter[0] * divisions // meter[1]
            if target * meter[1] != 4 * meter[0] * divisions:
                continue
            outcome = repair_single_stream_measure(measure, target, note_x)
            if outcome is None:
                outcome = repair_measure(measure, target, note_x)
            if outcome is not None:
                repaired.append({"part": part_index, "measure": measure_index,
                                 **outcome})
    return repaired


def main() -> None:
    from board_homr import audit_timing, parse_meter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-meter", type=parse_meter)
    args = parser.parse_args()
    tree = ET.parse(args.score)
    repaired = repair_score(tree.getroot(), args.initial_meter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    print(json.dumps({"repaired": repaired,
                      "remaining_timing_review": audit_timing(tree.getroot())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
