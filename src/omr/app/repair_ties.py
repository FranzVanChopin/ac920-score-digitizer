"""Experimental conservative MusicXML tie recovery from Homr slur candidates.

Requires an image in the coordinate frame of the geometry sidecar. The
--resize-input option is only for the clean synthetic benchmark pages; a real
camera image may have undergone rotation or other Homr preprocessing.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from align_homr_geometry import (_position, _xml_systems, align_score_geometry,
                                 align_score_geometry_with_gaps)


PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def pitch_key(note: ET.Element) -> tuple[int, int] | None:
    pitch = note.find("pitch")
    if pitch is None or pitch.findtext("step") not in PITCH_CLASS:
        return None
    return (int(note.findtext("staff", "1")),
            (int(pitch.findtext("octave")) + 1) * 12 +
            PITCH_CLASS[pitch.findtext("step")] +
            int(pitch.findtext("alter", "0")))


def candidate_pairs(root: ET.Element):
    active = {}
    for part in root.findall("part"):
        active.clear()
        for measure in part.findall("measure"):
            for note in measure.findall("note"):
                key_pitch = pitch_key(note)
                if key_pitch is None:
                    continue
                for slur in note.findall("notations/slur"):
                    key = key_pitch[0], slur.get("number", "1")
                    if slur.get("type") == "start":
                        active[key] = (note, slur)
                    elif slur.get("type") == "stop" and key in active:
                        first, first_slur = active.pop(key)
                        if pitch_key(first) == key_pitch:
                            yield first, note, first_slur, slur


def localized_notes(root: ET.Element, geometry: dict, mapping: dict):
    result = {}
    for system_index, (xml_notes, system) in enumerate(zip(
            _xml_systems(root), geometry["systems"])):
        staff = system["staffs"][0]
        midpoint = sum(staff["y_range"]) / 2
        for note in xml_notes:
            if note not in mapping:
                continue
            position = _position(note)
            if position is None:
                continue
            candidates = [found for found in staff["notes"]
                          if abs(found["x"] - mapping[note]) < 3.5
                          and int(found["position"]) == position[1]
                          and (1 if found["y"] < midpoint else 2) == position[0]]
            if candidates:
                found = min(candidates,
                            key=lambda item: abs(item["x"] - mapping[note]))
                result[note] = (system_index, float(found["x"]), float(found["y"]))
    return result


def parenthesis_ink(residual: np.ndarray, x: float, y: float) -> list[list[int]]:
    cx, cy = round(x), round(y)
    if cx < 20 or cy < 20 or cx + 20 >= residual.shape[1] or cy + 20 >= residual.shape[0]:
        return [[0, 0], [0, 0]]
    values = []
    for dx0, dx1 in ((-17, -9), (9, 17)):
        values.append([int(residual[cy + dy0:cy + dy1,
                                    cx + dx0:cx + dx1].sum())
                       for dy0, dy1 in ((-19, -5), (5, 19))])
    return values


def has_parentheses(note: ET.Element, residual: np.ndarray,
                    location: tuple[int, float, float]) -> bool:
    head = note.find("notehead")
    if head is not None and head.get("parentheses") == "yes":
        return True
    _, x, y = location
    return all(value >= 10 for side in parenthesis_ink(residual, x, y)
               for value in side)


def curve_below_heads(ink: np.ndarray,
                      first: tuple[int, float, float],
                      last: tuple[int, float, float]) -> bool:
    """Require a continuous shallow downward arc between matched noteheads."""
    if first[0] != last[0] or not 55 <= last[1] - first[1] <= 700:
        return False
    x0, y0 = first[1:]
    x1, y1 = last[1:]
    fractions = (0.17, 0.33, 0.50, 0.67, 0.83)
    for height in range(10, 27, 2):
        offsets = []
        for fraction in fractions:
            x = round(x0 + (x1 - x0) * fraction)
            baseline = y0 + (y1 - y0) * fraction
            y = round(baseline + 4 * height * fraction * (1 - fraction))
            ys = np.where(ink[y - 3:y + 4, x - 2:x + 3])[0]
            if not len(ys):
                break
            observed = y - 3 + int(ys[np.argmin(abs(ys - 3))])
            offsets.append(observed - baseline)
        if len(offsets) == len(fractions) and offsets[2] >= 8 and \
                offsets[2] - (offsets[0] + offsets[4]) / 2 >= 3:
            return True
    return False


def curve_across_system_break(ink: np.ndarray,
                              first: tuple[int, float, float],
                              last: tuple[int, float, float]) -> bool:
    """Look for both visible halves of a downward tie split by a line break."""
    if last[0] != first[0] + 1 or last[1] >= first[1]:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    short_ink = ink - cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    x0, y0 = round(first[1]), round(first[2])
    x1, y1 = round(last[1]), round(last[2])
    left = short_ink[y0 + 3:y0 + 19, x0 + 10:x0 + 150]
    right = short_ink[y1 + 3:y1 + 19, x1 - 45:x1 - 5]
    if left.shape != (16, 140) or right.shape != (16, 40):
        return False

    def has_fragment(crop: np.ndarray, minimum_width: int) -> bool:
        _, _, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
        return any(width >= minimum_width and height >= 5
                   for _, _, width, height, _ in stats[1:])

    return has_fragment(left, 60) and has_fragment(right, 10)


def short_arc_above_heads(ink: np.ndarray,
                          first: tuple[int, float, float],
                          last: tuple[int, float, float]) -> bool:
    """Check for a connected short arch over two adjacent repeated heads."""
    if first[0] != last[0] or not 18 <= last[1] - first[1] <= 100 or \
            abs(last[2] - first[2]) > 4:
        return False
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))
    short_ink = ink - cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    x0, x1 = round(first[1]), round(last[1])
    y = round(min(first[2], last[2]))
    crop = short_ink[y - 20:y - 2, x0 + 8:x1 - 8]
    if crop.size == 0:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    return any(width >= max(10, 0.65 * crop.shape[1]) and height >= 5
               for _, _, width, height, _ in stats[1:])


def add_tie(note: ET.Element, kind: str) -> None:
    tie = ET.Element("tie", {"type": kind})
    duration = note.find("duration")
    note.insert(list(note).index(duration) + 1, tie)
    notations = note.find("notations")
    if notations is None:
        notations = ET.SubElement(note, "notations")
    ET.SubElement(notations, "tied", {"type": kind})


def ordered_endpoints(first: tuple[int, float, float],
                      last: tuple[int, float, float]) -> bool:
    return ((first[0] == last[0] and last[1] - first[1] >= 18) or
            (last[0] == first[0] + 1 and last[1] < first[1]))


def recover_crossline_chord_starts(root: ET.Element, geometry: dict,
                                   located: dict, pairs: list) -> int:
    """Localize a final chord when notehead positions drift by one staff step."""
    systems = _xml_systems(root)
    indexes = {note: (system_index, index)
               for system_index, notes in enumerate(systems)
               for index, note in enumerate(notes)}
    recovered = 0
    for first, last, _, _ in pairs:
        if first in located or last not in located:
            continue
        system_index, note_index = indexes[first]
        if indexes[last][0] != system_index + 1:
            continue
        xml_notes = systems[system_index]
        companions = []
        for note in xml_notes[note_index + 1:]:
            if note.find("chord") is None:
                break
            if note.findtext("voice") == first.findtext("voice") and \
                    note.findtext("staff", "1") == first.findtext("staff", "1"):
                companions.append(note)
        if len(companions) != 1 or _position(first) is None or \
                _position(companions[0]) is None:
            continue
        if any(note.find("chord") is None and
               note.findtext("voice") == first.findtext("voice") and
               note.findtext("staff", "1") == first.findtext("staff", "1")
               for note in xml_notes[note_index + 1:]):
            continue
        staff = geometry["systems"][system_index]["staffs"][0]
        midpoint = sum(staff["y_range"]) / 2
        lower_x = staff["x_range"][1] - 550

        def possible_heads(note: ET.Element) -> list[dict]:
            expected = _position(note)
            return [head for head in staff["notes"]
                    if head["x"] >= lower_x and
                    abs(int(head["position"]) - expected[1]) <= 1 and
                    (1 if head["y"] < midpoint else 2) == expected[0]]

        pairs_of_heads = [(upper, lower)
                          for upper in possible_heads(first)
                          for lower in possible_heads(companions[0])
                          if abs(upper["x"] - lower["x"]) <= 3.5 and
                          lower["y"] > upper["y"] + 12]
        if len(pairs_of_heads) != 1:
            continue
        upper, lower = pairs_of_heads[0]
        if located[last][0] != system_index + 1 or \
                located[last][1] >= upper["x"]:
            continue
        located[first] = (system_index, float(upper["x"]), float(upper["y"]))
        located[companions[0]] = (system_index, float(lower["x"]), float(lower["y"]))
        recovered += 1
    return recovered


def repair(root: ET.Element, geometry: dict, image: np.ndarray,
           allow_ambiguous_same_measure: bool = False) -> dict:
    mapped, alignment = align_score_geometry(root, geometry)
    gap_mapping, gap_alignment = align_score_geometry_with_gaps(root, geometry)
    strict_mapping = mapped
    mapped = {**gap_mapping, **strict_mapping}
    gap_located = localized_notes(root, geometry, gap_mapping)
    pairs = list(candidate_pairs(root))
    for first, last, _, _ in pairs:
        if first in gap_located and last in gap_located and \
                ordered_endpoints(gap_located[first], gap_located[last]):
            mapped[first] = gap_mapping[first]
            mapped[last] = gap_mapping[last]
    located = localized_notes(root, geometry, mapped)
    recovered_crossline_chords = recover_crossline_chord_starts(
        root, geometry, located, pairs)
    binary = (image < 140).astype(np.uint8)
    stafflines = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
    residual = binary - stafflines
    all_notes = root.findall("part/measure/note")
    indexes = {note: i for i, note in enumerate(all_notes)}
    measure_of = {note: index for index, measure in
                  enumerate(root.findall("part/measure"), 1)
                  for note in measure.findall("note")}
    accepted = []
    review = []
    accepted_notes = []
    unresolved_short = []
    for first, last, start_slur, stop_slur in pairs:
        between = all_notes[indexes[first] + 1:indexes[last]]
        intervening = any(note.findtext("voice") == first.findtext("voice")
                          and note.findtext("staff", "1") == first.findtext("staff", "1")
                          and note.find("chord") is None for note in between)
        reason = None
        if not allow_ambiguous_same_measure and \
                measure_of[first] == measure_of[last]:
            reason = "same_measure_tie_slur_ambiguous"
        elif first.findtext("voice") != last.findtext("voice") or intervening:
            reason = "not_adjacent_in_voice"
        elif first not in located or last not in located:
            reason = "unmapped_notehead"
        elif not ordered_endpoints(located[first], located[last]):
            reason = "invalid_endpoint_order"
        elif has_parentheses(first, residual, located[first]) or \
                has_parentheses(last, residual, located[last]):
            reason = "parenthesized_notehead"
        summary = {"staff_midi": pitch_key(first),
                   "x": [located[n][1] if n in located else None
                         for n in (first, last)],
                   "reason": reason}
        if reason is not None:
            review.append(summary)
            if reason == "unmapped_notehead":
                unresolved_short.append((first, last, start_slur, stop_slur,
                                         summary))
            continue
        first.find("notations").remove(start_slur)
        last.find("notations").remove(stop_slur)
        add_tie(first, "start")
        add_tie(last, "stop")
        accepted.append(summary)
        accepted_notes.append((first, last))
    systems = _xml_systems(root)
    system_index = {note: index for index, notes in enumerate(systems)
                    for note in notes}
    used_arcs = {(round(located[a][1]), round(located[b][1]), located[a][0])
                 for a, b in accepted_notes if a in located and b in located}
    for first, last, start_slur, stop_slur, old_summary in unresolved_short:
        if system_index[first] != system_index[last] or _position(first) is None:
            continue
        index = system_index[first]
        staff = geometry["systems"][index]["staffs"][0]
        midpoint = sum(staff["y_range"]) / 2
        pitch_position = _position(first)
        heads = sorted((head for head in staff["notes"]
                        if int(head["position"]) == pitch_position[1] and
                        (1 if head["y"] < midpoint else 2) == pitch_position[0]),
                       key=lambda head: head["x"])
        matches = []
        for a, b in zip(heads, heads[1:]):
            first_loc = (index, float(a["x"]), float(a["y"]))
            last_loc = (index, float(b["x"]), float(b["y"]))
            key = round(a["x"]), round(b["x"]), index
            if key not in used_arcs and short_arc_above_heads(
                    binary, first_loc, last_loc):
                matches.append((first_loc, last_loc))
        if len(matches) != 1:
            continue
        first_loc, last_loc = matches[0]
        if has_parentheses(first, residual, first_loc) or \
                has_parentheses(last, residual, last_loc):
            continue
        located[first], located[last] = first_loc, last_loc
        first.find("notations").remove(start_slur)
        last.find("notations").remove(stop_slur)
        add_tie(first, "start")
        add_tie(last, "stop")
        old_summary["x"] = [first_loc[1], last_loc[1]]
        old_summary["reason"] = "unique_visible_short_arc"
        review.remove(old_summary)
        accepted.append(old_summary)
        accepted_notes.append((first, last))
        used_arcs.add((round(first_loc[1]), round(last_loc[1]), index))
    companions = []
    used = {note for pair in accepted_notes for note in pair}
    for first, last in accepted_notes:
        for other_first in all_notes:
            if other_first in used or other_first not in located:
                continue
            if other_first.findtext("voice") != first.findtext("voice") or \
                    pitch_key(other_first) is None or \
                    not (located[other_first][0] == located[first][0] and
                         abs(located[other_first][1] - located[first][1]) <= 3.5 and
                         located[other_first][2] > located[first][2] + 12):
                continue
            for other_last in all_notes:
                if other_last in used or other_last not in located or \
                        pitch_key(other_last) != pitch_key(other_first) or \
                        other_last.findtext("voice") != last.findtext("voice") or \
                        not (located[other_last][0] == located[last][0] and
                             abs(located[other_last][1] - located[last][1]) <= 3.5 and
                             located[other_last][2] > located[last][2] + 12):
                    continue
                if has_parentheses(other_first, residual, located[other_first]) or \
                        has_parentheses(other_last, residual, located[other_last]):
                    continue
                if not (curve_below_heads(binary, located[other_first],
                                          located[other_last]) or
                        curve_across_system_break(binary, located[other_first],
                                                  located[other_last])):
                    continue
                add_tie(other_first, "start")
                add_tie(other_last, "stop")
                used.update((other_first, other_last))
                companions.append({"staff_midi": pitch_key(other_first),
                                   "x": [located[other_first][1],
                                         located[other_last][1]],
                                   "reason": "visible_chord_companion_arc"})
                break
    return {"alignment": alignment, "gap_alignment": gap_alignment,
            "recovered_crossline_chords": recovered_crossline_chords,
            "converted": accepted,
            "companion_ties": companions, "review": review}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resize-input", action="store_true")
    parser.add_argument("--aggressive-same-measure", action="store_true",
                        help="research only: auto-convert ambiguous same-measure arcs")
    args = parser.parse_args()
    root = ET.parse(args.score).getroot()
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))
    image = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image}")
    width, height = geometry["preprocessed_size"]
    if image.shape != (height, width):
        if not args.resize_input:
            raise ValueError("Image/geometry size mismatch; supply Homr's preprocessed image")
        image = cv2.resize(image, (width, height))
    report = repair(root, geometry, image, args.aggressive_same_measure)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(args.output, encoding="utf-8", xml_declaration=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
