"""Repair extreme upper octave chords from newly retained image noteheads.

No reference score is read. A unique sequence alignment, a visible lower
octave partner in the same column, and an extended-ledger segmentation head
are all required. Search tables are bounded to 256 notes per staff/system.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from statistics import median

from align_homr_geometry import _position, _xml_systems
from head_alignment_dp import try_align


def _groups(notes):
    groups = []
    for note in notes:
        if (note.find("chord") is None or not groups or
                note.findtext("staff", "1") != groups[-1][0].findtext("staff", "1")):
            groups.append([])
        groups[-1].append(note)
    return groups


def _onset_groups(root):
    result = {}
    for measure in root.findall("part/measure"):
        cursor = previous = 0
        for element in measure:
            if element.tag in {"backup", "forward"}:
                cursor += int(element.findtext("duration", "0")) * (1 if element.tag == "forward" else -1)
            elif element.tag == "note":
                onset = previous if element.find("chord") is not None else cursor
                if element.find("chord") is None:
                    previous = onset
                    if element.find("grace") is None:
                        cursor += int(element.findtext("duration", "0"))
                if element.find("pitch") is not None:
                    result.setdefault((measure, element.findtext("staff", "1"), onset), []).append(element)
    return result


def _align(notes, heads):
    n, m = len(notes), len(heads)
    if not n or not m or max(n, m) > 256:
        return {}
    fast = try_align(notes, heads, _position)
    if fast is not None:
        return fast
    positions = [_position(note)[1] for note in notes]
    inf = 3 * (n + m) + 1

    def cost(i, j):
        delta = abs(positions[i] - int(heads[j]["position"]))
        if not delta:
            return 0
        if heads[j].get("extended_ledger") and positions[i] >= 16 and delta <= 2:
            return delta
        return inf

    forward = [[inf] * (m + 1) for _ in range(n + 1)]
    forward[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            value = forward[i][j]
            if i < n:
                forward[i+1][j] = min(forward[i+1][j], value + 3)
            if j < m:
                forward[i][j+1] = min(forward[i][j+1], value + 3)
            if i < n and j < m:
                forward[i+1][j+1] = min(forward[i+1][j+1], value + cost(i, j))
    backward = [[inf] * (m + 1) for _ in range(n + 1)]
    backward[n][m] = 0
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            value = backward[i][j]
            if i:
                backward[i-1][j] = min(backward[i-1][j], value + 3)
            if j:
                backward[i][j-1] = min(backward[i][j-1], value + 3)
            if i and j:
                backward[i-1][j-1] = min(backward[i-1][j-1], value + cost(i-1, j-1))
    best = forward[n][m]
    mapped = {}
    for i, note in enumerate(notes):
        can_skip = any(forward[i][j] + 3 + backward[i+1][j] == best
                       for j in range(m + 1))
        possible = [j for j in range(m)
                    if forward[i][j] + cost(i, j) + backward[i+1][j+1] == best]
        if not can_skip and len(possible) == 1:
            mapped[note] = heads[possible[0]]
    return mapped


def _shared_stem(lower, upper):
    """Accept one stem or two collinear segments split by a ledger line."""
    a, b = lower.get("stem"), upper.get("stem")
    space = upper.get("staff_space")
    if not a or not b or not space:
        return False
    if a == b:
        return True
    if abs(a[0] - b[0]) > 1.5 or max(a[2], b[2]) > space / 4:
        return False
    # Both segments must be on the same side of the aligned heads.
    side_a, side_b = a[0] - lower["x"], b[0] - upper["x"]
    if side_a * side_b <= 0:
        return False
    intervals = sorted([(a[1] - a[3]/2, a[1] + a[3]/2),
                        (b[1] - b[3]/2, b[1] + b[3]/2)])
    gap = max(0, intervals[1][0] - intervals[0][1])
    return gap <= space * 0.6


def _anchored_position(head, mapped):
    """Calibrate an extreme head with normal notes on both sides of it."""
    space = head.get("staff_space")
    if not space:
        return None
    sides = [[], []]
    for note, anchor in mapped.items():
        position = _position(note)[1]
        if (anchor.get("extended_ledger") or not 9 <= position <= 16 or
                position != int(anchor["position"])):
            continue
        dx = anchor["x"] - head["x"]
        if not 3 < abs(dx) <= space * 12:
            continue
        estimate = position + 2 * (anchor["y"] - head["y"]) / space
        sides[0 if dx < 0 else 1].append(estimate)
    if any(not side for side in sides) or sum(map(len, sides)) < 3:
        return None
    estimates = [median(side) for side in sides]
    target = round(sum(estimates)/2)
    if any(abs(value - target) > 0.25 for value in estimates):
        return None
    if abs(target - int(head["position"])) > 1:
        return None
    return target


def repair(root: ET.Element, geometry: dict) -> list[dict]:
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    measure_of = {note: (index, measure)
                  for index, measure in enumerate(root.findall("part/measure"), 1)
                  for note in measure.findall("note")}
    changed = []
    added_count = 0
    cohorts = _onset_groups(root)
    for notes, detected in zip(systems, geometry["systems"]):
        staffs = detected.get("staffs", [])
        if len(staffs) != 1:
            continue
        staff = staffs[0]
        midpoint = sum(staff["y_range"]) / 2
        groups = _groups([note for note in notes if note.findtext("staff", "1") == "1"])
        ordered = [note for group in groups for note in sorted(group, key=lambda n: _position(n)[1])]
        raw = sorted((head for head in staff["notes"] if head["y"] < midpoint),
                     key=lambda h: h["x"])
        head_groups = []
        for head in raw:
            if not head_groups or head["x"] - head_groups[-1][-1]["x"] > 3:
                head_groups.append([])
            head_groups[-1].append(head)
        heads = [head for group in head_groups for head in sorted(group, key=lambda h: h["position"])]
        mapped = _align(ordered, heads)
        used = set()
        for group in [group for group in cohorts.values() if group[0] in notes]:
            if (group[0].findtext("staff", "1") != "1" or
                    any(note.find("grace") is not None for note in group)):
                continue
            for lower in list(group):
                head = mapped.get(lower)
                if head is None or _position(lower)[1] != int(head["position"]):
                    continue
                target = _position(lower)[1] + 7
                visible = [h for h in heads if h.get("extended_ledger") and
                           abs(int(h["position"]) - target) <= 1 and
                           abs(h["x"] - head["x"]) <= 3 and
                           h.get("staff_space") and
                           abs(2 * (head["y"] - h["y"]) / h["staff_space"] - 7) <= 0.3]
                if target < 17 or len(visible) != 1 or id(visible[0]) in used:
                    continue
                if any(_position(note)[1] == target for note in group):
                    continue
                errors = [note for note in group if note is not lower and
                          note in mapped and mapped[note] is visible[0] and
                          abs(_position(note)[1] - target) <= 2]
                index, measure = measure_of[lower]
                if len(errors) == 1:
                    note = errors[0]
                    old = ET.tostring(note.find("pitch"), encoding="unicode")
                    pitch = note.find("pitch")
                    pitch.find("step").text = lower.findtext("pitch/step")
                    pitch.find("octave").text = str(int(lower.findtext("pitch/octave")) + 1)
                    changed.append({"measure": index, "method": "ledger_octave_pitch",
                                    "old_pitch": old, "new_step": pitch.findtext("step"),
                                    "new_octave": pitch.findtext("octave")})
                elif not errors and added_count < 4:
                    # A candidate already assigned to another XML note cannot
                    # be inserted as an additional chord member.
                    if any(h is visible[0] for h in mapped.values()):
                        continue
                    if not _shared_stem(head, visible[0]):
                        continue
                    note = copy.deepcopy(lower)
                    for child in list(note):
                        if child.tag not in {"pitch", "duration", "voice", "type", "dot", "time-modification", "staff"}:
                            note.remove(child)
                    note.insert(0, ET.Element("chord"))
                    note.find("pitch/octave").text = str(int(lower.findtext("pitch/octave")) + 1)
                    # A chord attaches to the immediately preceding note.
                    # Append next to the lower partner's original chord group.
                    elements = list(measure)
                    location = elements.index(lower) + 1
                    while location < len(elements) and elements[location].tag == "note" and elements[location].find("chord") is not None:
                        location += 1
                    measure.insert(location, note)
                    group.append(note)
                    added_count += 1
                    changed.append({"measure": index, "method": "visible_ledger_octave_added",
                                    "step": note.findtext("pitch/step"),
                                    "octave": note.findtext("pitch/octave")})
                else:
                    continue
                used.add(id(visible[0]))
        # Solitary high notes have no octave partner. Correct them only when
        # independent ordinary notes on both sides agree on the staff grid.
        for note, head in mapped.items():
            if not head.get("extended_ledger") or id(head) in used:
                continue
            old_position = _position(note)[1]
            target = _anchored_position(head, mapped)
            if (target is None or target < 17 or target == old_position or
                    old_position < 16 or abs(target - old_position) > 2):
                continue
            pitch = note.find("pitch")
            # Accidentals are an independent recognition problem. Do not
            # transfer an accidental to a different letter without evidence.
            if int(pitch.findtext("alter", "0")) or note.find("accidental") is not None:
                continue
            diatonic = int(pitch.findtext('octave'))*7 + 'CDEFGAB'.index(pitch.findtext('step')) + target-old_position
            step, octave = "CDEFGAB"[diatonic % 7], diatonic // 7
            index, _ = measure_of[note]
            old = ET.tostring(pitch, encoding="unicode")
            pitch.find("step").text = step
            pitch.find("octave").text = str(octave)
            changed.append({"measure": index, "method": "ledger_anchored_pitch",
                            "old_pitch": old, "new_step": step,
                            "new_octave": str(octave)})
    return changed
