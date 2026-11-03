"""Remove decoder-only upper-staff notes when segmentation proves the sequence.

This is deliberately narrow: every detected upper notehead in a system must
match a MusicXML diatonic position, skipped XML notes must be duplicate heads
in overfull bars, and the remaining image columns must fill the meter exactly.
No reference score or second inference pass is used.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import xml.etree.ElementTree as ET

from align_homr_geometry import _position


def _systems(root: ET.Element) -> list[list[tuple[int, ET.Element]]]:
    parts = root.findall("part")
    if len(parts) != 1:
        return []
    systems: list[list[tuple[int, ET.Element]]] = [[]]
    for measure_index, measure in enumerate(parts[0].findall("measure"), 1):
        printed = measure.find("print")
        if (measure_index > 1 and printed is not None and
                printed.get("new-system") == "yes"):
            systems.append([])
        systems[-1].extend((measure_index, note) for note in measure.findall("note")
                           if note.find("pitch") is not None and
                           note.findtext("staff", "1") == "1" and
                           note.find("grace") is None)
    return systems


def _top_heads(system: dict) -> list[dict]:
    staffs = system.get("staffs", [])
    if len(staffs) != 1:
        return []
    staff = staffs[0]
    midpoint = sum(staff["y_range"]) / 2
    raw = sorted((note for note in staff["notes"] if note["y"] < midpoint),
                 key=lambda note: note["x"])
    groups: list[list[dict]] = []
    for note in raw:
        if not groups or note["x"] - groups[-1][-1]["x"] > 3:
            groups.append([])
        groups[-1].append(note)
    return [note for group in groups
            for note in sorted(group, key=lambda item: item["position"])]


def _pitch_key(note: ET.Element) -> tuple[str, str, str]:
    pitch = note.find("pitch")
    return (pitch.findtext("step"), pitch.findtext("alter", "0"),
            pitch.findtext("octave"))


def _align(xml_notes: list[tuple[int, ET.Element]], heads: list[dict]
           ) -> tuple[dict[ET.Element, dict], list[ET.Element]] | None:
    """Match all image heads, allowing only excess decoder notes."""
    n, m = len(xml_notes), len(heads)
    # Bound the quadratic table on the board even for an unusually dense page.
    if not heads or n > 128 or m > 128 or n < m or n - m > 4:
        return None
    positioned = [_position(note) for _, note in xml_notes]
    if any(value is None for value in positioned):
        return None
    positions = [value[1] for value in positioned]
    frequencies = Counter((measure, positions[index], _pitch_key(note))
                          for index, (measure, note) in enumerate(xml_notes))
    # Each cell is (number of deleted XML notes, frequency cost). A rare
    # accidental reading loses a tie against a consistent repeated pitch.
    huge = (n + m + 1, n * n + 1)
    dp = [[huge] * (m + 1) for _ in range(n + 1)]
    dp[n][m] = (0, 0)
    for i in range(n - 1, -1, -1):
        measure, note = xml_notes[i]
        frequency = frequencies[(measure, positions[i], _pitch_key(note))]
        if dp[i + 1][m][0] <= n:
            dp[i][m] = (dp[i + 1][m][0] + 1,
                        dp[i + 1][m][1] + frequency)
        for j in range(m - 1, -1, -1):
            delete = (dp[i + 1][j][0] + 1,
                      dp[i + 1][j][1] + frequency)
            match = (dp[i + 1][j + 1] if
                     positions[i] == int(heads[j]["position"]) else huge)
            dp[i][j] = min(match, delete)
    if dp[0][0][0] != n - m:
        return None
    mapped: dict[ET.Element, dict] = {}
    skipped: list[ET.Element] = []
    i = j = 0
    while i < n:
        measure, note = xml_notes[i]
        frequency = frequencies[(measure, positions[i], _pitch_key(note))]
        delete = (dp[i + 1][j][0] + 1,
                  dp[i + 1][j][1] + frequency)
        if (j < m and positions[i] == int(heads[j]["position"]) and
                dp[i][j] == dp[i + 1][j + 1] and
                (dp[i][j] != delete or frequency > 1)):
            mapped[note] = heads[j]
            i += 1
            j += 1
        elif dp[i][j] == delete:
            skipped.append(note)
            i += 1
        else:
            return None
    return (mapped, skipped) if j == m else None


def _note_onsets(measure: ET.Element) -> dict[ET.Element, int] | None:
    cursor = 0
    previous = 0
    result = {}
    for element in measure:
        if element.tag in {"backup", "forward"}:
            duration = element.findtext("duration")
            if duration is None:
                return None
            cursor += int(duration) * (1 if element.tag == "forward" else -1)
        elif element.tag == "note":
            onset = previous if element.find("chord") is not None else cursor
            result[element] = onset
            if element.find("chord") is None:
                previous = onset
                if element.find("grace") is None:
                    cursor += int(element.findtext("duration", "0"))
    return result


def _target_by_measure(root: ET.Element,
                       initial_meter: tuple[int, int] | None) -> dict[int, int]:
    result = {}
    divisions = 1
    meter = initial_meter
    for index, measure in enumerate(root.find("part").findall("measure"), 1):
        for attributes in measure.findall("attributes"):
            if attributes.findtext("divisions") is not None:
                divisions = int(attributes.findtext("divisions"))
            time = attributes.find("time")
            if time is not None:
                try:
                    meter = (sum(int(piece) for piece in
                                 time.findtext("beats", "").split("+")),
                             int(time.findtext("beat-type", "")))
                except ValueError:
                    meter = None
        if meter is not None:
            length = Fraction(4 * meter[0] * divisions, meter[1])
            if length.denominator == 1:
                result[index] = int(length)
    return result


def _repair_measure(measure: ET.Element, matched: dict[ET.Element, dict],
                    skipped: set[ET.Element], target: int) -> dict | None:
    elements = list(measure)
    indices = [index for index, element in enumerate(elements)
               if element.tag == "note"]
    if not indices:
        return None
    first, last = indices[0], indices[-1]
    if (any(element.tag not in {"note", "backup"}
            for element in elements[first:last + 1]) or
            any(element.tag != "barline" for element in elements[last + 1:])):
        return None
    notes = [element for element in elements[first:last + 1]
             if element.tag == "note"]
    upper = [note for note in notes if note.findtext("staff", "1") == "1"]
    if (not skipped or any(note not in matched and note not in skipped
                           for note in upper)):
        return None
    if any(note.find("grace") is not None or note.find("rest") is not None
           for note in upper):
        return None
    safe_children = {"pitch", "duration", "voice", "type", "staff", "notations"}
    if any(note.find("chord") is not None or
           any(child.tag not in safe_children for child in note) or
           len(note.find("notations") or []) > 0
           for note in skipped):
        return None
    active = [note for note in upper if note not in skipped]
    if not active:
        return None
    duration_set = {int(note.findtext("duration", "0")) for note in active}
    if len(duration_set) != 1 or 0 in duration_set:
        return None
    duration = next(iter(duration_set))
    if any(int(note.findtext("duration", "0")) != duration for note in skipped):
        return None
    heads = [matched[note] for note in active]
    columns: list[list[ET.Element]] = []
    for note, head in zip(active, heads):
        if not columns or head["x"] - matched[columns[-1][-1]]["x"] > 3:
            columns.append([])
        columns[-1].append(note)
    if len(columns) * duration != target:
        return None
    desired = _note_onsets(measure)
    if desired is None:
        return None
    for column_index, column in enumerate(columns):
        for note_index, note in enumerate(column):
            if (note.find("chord") is not None) != (note_index > 0):
                return None
            desired[note] = column_index * duration
    # A decoder duplicate is normally beside another head at the same staff
    # position. An isolated minority accidental is also allowed when repeated
    # matched heads overwhelmingly agree on the ordinary pitch for that line.
    pitch_counts = Counter((_position(note), _pitch_key(note)) for note in active)
    for note in skipped:
        index = upper.index(note)
        adjacent = upper[max(0, index - 1):index] + upper[index + 1:index + 2]
        same_position = any(other in matched and
                            _position(other) == _position(note)
                            for other in adjacent)
        minority = (note.find("accidental") is None and
                    max((count for (position, _pitch), count in
                         pitch_counts.items() if position == _position(note)),
                        default=0) >= 3 *
                    (pitch_counts[(_position(note), _pitch_key(note))] + 1))
        if not (same_position or minority):
            return None
    kept = [note for note in notes if note not in skipped]
    previous = None
    for note in kept:
        if (note.find("chord") is not None and
                (previous is None or desired[note] != desired[previous] or
                 note.findtext("staff", "1") != previous.findtext("staff", "1"))):
            return None
        previous = note
    rebuilt = elements[:first]
    cursor = 0
    for note in kept:
        onset = desired[note]
        if note.find("chord") is None:
            if onset < cursor:
                backup = ET.Element("backup")
                ET.SubElement(backup, "duration").text = str(cursor - onset)
                rebuilt.append(backup)
            elif onset > cursor:
                forward = ET.Element("forward")
                ET.SubElement(forward, "duration").text = str(onset - cursor)
                rebuilt.append(forward)
            cursor = onset + (0 if note.find("grace") is not None
                              else int(note.findtext("duration", "0")))
        if note in matched:
            note.find("voice").text = "1"
        rebuilt.append(note)
    measure[:] = rebuilt + elements[last + 1:]
    return {"removed_decoder_notes": len(skipped), "upper_columns": len(columns),
            "matched_upper_notes": len(active)}


def repair_score_from_noteheads(root: ET.Element, geometry: dict,
                                initial_meter: tuple[int, int] | None,
                                audit_timing_fn) -> list[dict]:
    xml_systems = _systems(root)
    detected_systems = geometry.get("systems", [])
    if not xml_systems or len(xml_systems) != len(detected_systems):
        return []
    overfull = {row["measure"] for row in
                audit_timing_fn(root)["overfull_measures"] if row["part"] == 1}
    targets = _target_by_measure(root, initial_meter)
    outcomes = []
    for xml_notes, system in zip(xml_systems, detected_systems):
        heads = _top_heads(system)
        alignment = _align(xml_notes, heads)
        if alignment is None:
            continue
        matched, skipped = alignment
        skip_by_measure = Counter(measure for measure, note in xml_notes
                                  if note in skipped)
        if (not skip_by_measure or any(number not in overfull or
                                      number not in targets or count > 2
                                      for number, count in skip_by_measure.items())):
            continue
        notes_by_measure: dict[int, dict[ET.Element, dict]] = {}
        for number, note in xml_notes:
            if note in matched:
                notes_by_measure.setdefault(number, {})[note] = matched[note]
        measures = root.find("part").findall("measure")
        for number in skip_by_measure:
            measure = measures[number - 1]
            outcome = _repair_measure(measure, notes_by_measure.get(number, {}),
                                      set(skipped) & set(measure.findall("note")),
                                      targets[number])
            if outcome is not None:
                outcomes.append({"part": 1, "measure": number, **outcome})
    return outcomes
