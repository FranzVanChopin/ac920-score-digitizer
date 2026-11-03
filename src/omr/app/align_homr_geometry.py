"""Conservatively align Homr MusicXML notes with its segmentation noteheads.

The decoder normally emits symbols left to right. Notes in each detected x
cluster are aligned only if staff and diatonic-position multisets agree. Any
uncertain cluster is left unmapped. This code uses no reference score.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from weakref import WeakKeyDictionary


STEP = {letter: index for index, letter in enumerate("CDEFGAB")}
_CLEF_BASES = WeakKeyDictionary()


def _base(note):
    return _CLEF_BASES.get(note, 30 if note.findtext('staff','1') == '1' else 18)


def _head_staff(staff, head):
    grid = staff.get('grid', [])
    if grid and len(grid[0]['y']) == 5:
        return 1
    return 1 if head['y'] < sum(staff['y_range'])/2 else 2


def _position(note: ET.Element) -> tuple[int, int] | None:
    pitch = note.find("pitch")
    if pitch is None:
        return None
    step = pitch.findtext("step")
    octave = pitch.findtext("octave")
    if step not in STEP or octave is None:
        return None
    staff = int(note.findtext("staff", "1"))
    if staff not in (1, 2):
        return None
    # Treble bottom line E4 and bass bottom line G2 both have position 1.
    base = _base(note)
    if base is None:
        return None
    return staff, int(octave) * 7 + STEP[step] - base + 1


def _xml_systems(root: ET.Element) -> list[list[ET.Element]]:
    parts = root.findall("part")
    if len(parts) != 1:
        return []
    systems: list[list[ET.Element]] = [[]]
    bases = {1: 30, 2: 18}
    for index, measure in enumerate(parts[0].findall("measure")):
        print_element = measure.find("print")
        if (index and print_element is not None and
                print_element.get("new-system") == "yes"):
            systems.append([])
        for element in measure:
            if element.tag == 'attributes':
                for clef in element.findall('clef'):
                    number = int(clef.get('number','1'))
                    reference = {'G':32, 'F':24, 'C':28}.get(clef.findtext('sign'))
                    line = int(clef.findtext('line','0'))
                    bases[number] = (reference-2*(line-1)+7*int(clef.findtext('clef-octave-change','0'))
                                     if reference is not None and 1 <= line <= 5 else None)
            elif element.tag == 'note' and element.find('pitch') is not None:
                _CLEF_BASES[element] = bases.get(int(element.findtext('staff','1')))
                systems[-1].append(element)
    return systems


def align_score_geometry(root: ET.Element, geometry: dict,
                         x_tolerance: float = 3.0) -> tuple[dict[ET.Element, float], dict]:
    xml_systems = _xml_systems(root)
    detected_systems = geometry.get("systems", [])
    if len(xml_systems) != len(detected_systems):
        return {}, {"error": "system count differs", "matched_notes": 0,
                    "xml_systems": len(xml_systems),
                    "geometry_systems": len(detected_systems)}
    aligned: dict[ET.Element, float] = {}
    exact_groups = all_groups = 0
    for xml_notes, system in zip(xml_systems, detected_systems):
        staffs = system.get("staffs", [])
        if len(staffs) != 1:
            continue
        staff = staffs[0]
        midpoint = sum(staff["y_range"]) / 2
        detected = sorted(staff["notes"], key=lambda note: note["x"])
        groups: list[list[dict]] = []
        for note in detected:
            if not groups or note["x"] - groups[-1][-1]["x"] > x_tolerance:
                groups.append([note])
            else:
                groups[-1].append(note)
        offset = 0
        for group in groups:
            all_groups += 1
            chunk = xml_notes[offset:offset + len(group)]
            if len(chunk) != len(group):
                break
            expected = Counter(_position(note) for note in chunk)
            observed = Counter((_head_staff(staff, note),
                                int(note["position"])) for note in group)
            if expected == observed and None not in expected:
                x = sum(float(note["x"]) for note in group) / len(group)
                aligned.update({note: x for note in chunk})
                exact_groups += 1
            offset += len(group)
    return aligned, {"matched_notes": len(aligned),
                     "xml_notes": sum(len(system) for system in xml_systems),
                     "exact_groups": exact_groups, "groups": all_groups,
                     "xml_systems": len(xml_systems)}


def align_score_geometry_with_gaps(
        root: ET.Element, geometry: dict, x_tolerance: float = 3.0
) -> tuple[dict[ET.Element, float], dict]:
    """Match only notes whose x group is identical on every optimal edit path.

    An isolated omitted/duplicated note can shift the strict sequential
    alignment for the rest of a system. This bounded dynamic program permits
    skipped XML notes or image x groups, while refusing ambiguous repeats.
    It does not change ``align_score_geometry`` used by timing repair.
    """
    xml_systems = _xml_systems(root)
    detected_systems = geometry.get("systems", [])
    if len(xml_systems) != len(detected_systems):
        return {}, {"error": "system count differs", "matched_notes": 0}
    aligned: dict[ET.Element, float] = {}
    total_cost = 0
    ambiguous = 0
    skipped = 0
    for xml_notes, system in zip(xml_systems, detected_systems):
        staffs = system.get("staffs", [])
        if len(staffs) != 1:
            continue
        staff = staffs[0]
        midpoint = sum(staff["y_range"]) / 2
        detected = sorted(staff["notes"], key=lambda note: note["x"])
        groups: list[list[dict]] = []
        for note in detected:
            if not groups or note["x"] - groups[-1][-1]["x"] > x_tolerance:
                groups.append([note])
            else:
                groups[-1].append(note)
        n, m = len(xml_notes), len(groups)
        if max(n, m) > 512:
            skipped += n
            continue
        positions = [_position(note) for note in xml_notes]
        signatures = [Counter((_head_staff(staff, note),
                               int(note["position"])) for note in group)
                      for group in groups]
        matches = [[False] * m for _ in range(n + 1)]
        for j, group in enumerate(groups):
            length = len(group)
            for i in range(n - length + 1):
                chunk = positions[i:i + length]
                matches[i][j] = None not in chunk and Counter(chunk) == signatures[j]
        inf = n + sum(len(group) for group in groups) + 1
        forward = [[inf] * (m + 1) for _ in range(n + 1)]
        forward[0][0] = 0
        for i in range(n + 1):
            for j in range(m + 1):
                value = forward[i][j]
                if value == inf:
                    continue
                if i < n:
                    forward[i + 1][j] = min(forward[i + 1][j], value + 1)
                if j < m:
                    forward[i][j + 1] = min(forward[i][j + 1], value + len(groups[j]))
                    length = len(groups[j])
                    if i + length <= n and matches[i][j]:
                        forward[i + length][j + 1] = min(
                            forward[i + length][j + 1], value)
        backward = [[inf] * (m + 1) for _ in range(n + 1)]
        backward[n][m] = 0
        for i in range(n, -1, -1):
            for j in range(m, -1, -1):
                value = backward[i][j]
                if value == inf:
                    continue
                if i:
                    backward[i - 1][j] = min(backward[i - 1][j], value + 1)
                if j:
                    backward[i][j - 1] = min(backward[i][j - 1],
                                             value + len(groups[j - 1]))
                    length = len(groups[j - 1])
                    if i >= length and matches[i - length][j - 1]:
                        backward[i - length][j - 1] = min(
                            backward[i - length][j - 1], value)
        best = forward[n][m]
        total_cost += best
        candidates: list[set[int]] = [set() for _ in range(n)]
        can_skip = [False] * n
        for i in range(n + 1):
            for j in range(m + 1):
                if i < n and forward[i][j] + 1 + backward[i + 1][j] == best:
                    can_skip[i] = True
                if j < m:
                    length = len(groups[j])
                    if i + length <= n and matches[i][j] and \
                            forward[i][j] + backward[i + length][j + 1] == best:
                        for index in range(i, i + length):
                            candidates[index].add(j)
        for index, note in enumerate(xml_notes):
            possible = candidates[index]
            if can_skip[index]:
                skipped += 1
            elif len(possible) == 1:
                group = groups[next(iter(possible))]
                aligned[note] = sum(float(item["x"]) for item in group) / len(group)
            elif len(possible) > 1:
                ambiguous += 1
    return aligned, {"matched_notes": len(aligned),
                     "xml_notes": sum(len(system) for system in xml_systems),
                     "edit_cost": total_cost, "ambiguous_notes": ambiguous,
                     "skippable_notes": skipped, "xml_systems": len(xml_systems)}
