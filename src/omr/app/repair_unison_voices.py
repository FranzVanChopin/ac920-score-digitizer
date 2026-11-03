"""Recover an omitted opening quarter beside a same-pitch half note.

Two separated heads must show solid/hollow interiors and opposing stems.
The following quarter and closing half-note column must also agree with the
image. This narrow pattern resolves a five-beat decoder stream in 4/4.
"""

import copy
import xml.etree.ElementTree as ET

from align_homr_geometry import _position, _xml_systems
from repair_duplicate_noteheads import _note_onsets, _target_by_measure
from repair_grace_timing import _rebuild
from repair_ledger_chords import _groups
from repair_pitch_context import _ink
from repair_augmentation_dots import _head_map


def _stem_end(head, ink, up):
    """Find the far end of a continuous upright attached to this head."""
    s, x, y = head["staff_space"], head["x"], head["y"]
    nominal = round(x+(0.6*s if up else -0.6*s))
    ends = []
    for col in range(nominal-2, nominal+3):
        start = round(y+(-0.4*s if up else 0.4*s))
        sign = -1 if up else 1
        last, blanks = start, 0
        for distance in range(round(s*7)):
            row = start+sign*distance
            if not 0 <= row < ink.shape[0] or not 0 <= col < ink.shape[1]:
                break
            if ink[row, col]:
                last, blanks = row, 0
            else:
                blanks += 1
                if blanks >= 2:
                    break
        length = abs(last-start)
        if length >= 1.4*s:
            ends.append((length, col, last))
    return max(ends)[1:] if ends else None


def _single_beam(a, b, ink, up):
    s = a["staff_space"]
    ends = [_stem_end(h, ink, up) for h in (a, b)]
    if any(e is None for e in ends):
        return False
    (xa, ya), (xb, yb) = ends
    if not 0.7*s <= xb-xa <= 4*s or abs(ya-yb) > s:
        return False
    for fraction in (0.25, 0.5, 0.75):
        x = round(xa+(xb-xa)*fraction)
        y = ya+(yb-ya)*fraction
        t, end = max(0, round(y-0.6*s)), min(ink.shape[0], round(y+0.6*s))
        spans, start = [], None
        for row in range(t, end+1):
            black = row < end and bool(ink[row, x])
            if black and start is None:
                start = row
            if not black and start is not None:
                if 0.2*s <= row-start <= round(0.6*s):
                    spans.append((start+row-1)/2)
                start = None
        if len(spans) != 1 or abs(spans[0]-y) > 0.55*s:
            return False
    return True


def repair_shared_heads(root, geometry, image):
    """Recover a second eighth voice from two stems on one filled head."""
    mapped, ink = _head_map(root, geometry), _ink(image)
    changes = []
    divisions = 1
    for number, measure in enumerate(root.findall("part/measure"), 1):
        for attrs in measure.findall("attributes"):
            divisions = int(attrs.findtext("divisions", str(divisions)))
        if divisions % 2:
            continue
        duration = divisions//2
        onsets = _note_onsets(measure)
        notes = measure.findall("note")
        if len(notes) > 128:
            continue
        by_staff = {sn: [p for p in notes if p in mapped and p.findtext("staff", "1") == sn]
                    for sn in ("1", "2")}
        updates = []
        for n in notes:
            if (n not in mapped or n.find("grace") is not None or n.find("time-modification") is not None
                    or n.find("dot") is not None or n.findtext("type") not in {"quarter", "eighth"}):
                continue
            head = mapped[n]
            if (not head.get("stem") or _interior(head, ink) < 0.7
                    or not _stem_end(head, ink, True) or not _stem_end(head, ink, False)):
                continue
            sn = n.findtext("staff", "1")
            # An already exported second pitch event must not be duplicated.
            same = [p for p in notes if p.find("pitch") is not None and p.find("grace") is None
                    and p.findtext("staff", "1") == sn and onsets[p] == onsets[n]
                    and all(p.findtext("pitch/"+tag, "0") == n.findtext("pitch/"+tag, "0")
                            for tag in ("step", "octave", "alter"))]
            if len(same) != 1:
                continue
            future = [p for p in by_staff[sn] if 0.7*head["staff_space"] < mapped[p]["x"]-head["x"] < 4*head["staff_space"]
                      and onsets[p] == onsets[n]+duration and p.findtext("type") == "eighth"
                      and p.find("dot") is None and p.find("time-modification") is None]
            # Both orientations are checked. One must be an independently
            # visible single beam whose matching voice leaves this interval free.
            choices = []
            original_up = head["stem"][0] > head["x"]
            for up in (not original_up,):
                connected = [p for p in future if _single_beam(head, mapped[p], ink, up)]
                if len(connected) != 1:
                    continue
                voices = {p.findtext("voice") for p in by_staff[sn]
                          if p.findtext("voice") != n.findtext("voice")
                          and _stem_end(mapped[p], ink, up)}
                free = []
                for voice in voices:
                    occupied = [p for p in notes if p.findtext("voice") == voice
                                and p.findtext("staff", "1") == sn and p.find("grace") is None
                                and onsets[p] < onsets[n]+duration
                                and onsets[p]+int(p.findtext("duration", "0")) > onsets[n]]
                    if not occupied:
                        free.append(voice)
                if len(free) == 1:
                    choices.append((free[0], up))
                elif not free and connected[0].findtext("voice") == n.findtext("voice"):
                    used = {p.findtext("voice", "1") for p in notes}
                    used.update(v for _, v, _ in updates)
                    voice = next(str(v) for v in range(1, 33) if str(v) not in used)
                    choices.append((voice, up))
            if len(choices) == 1:
                updates.append((n, *choices[0]))
        if not updates or len(updates) > 4:
            continue
        saved = copy.deepcopy(list(measure))
        for n, voice, up in updates:
            clone = ET.Element("note")
            clone.append(copy.deepcopy(n.find("pitch")))
            for tag, value in (("duration", duration), ("voice", voice), ("type", "eighth"),
                               ("stem", "up" if up else "down"), ("staff", n.findtext("staff", "1"))):
                ET.SubElement(clone, tag).text = str(value)
            measure.insert(list(measure).index(n), clone)
            onsets[clone] = onsets[n]
        if _rebuild(measure, onsets):
            changes.append({"measure": number, "method": "shared_filled_head_two_stems_single_beam",
                            "added_notes": len(updates), "head_x": [mapped[n]["x"] for n, _, _ in updates]})
        else:
            measure[:] = saved
    return changes


def repair_overlapping_ending(root, geometry, image, initial_meter=None):
    """Place a beamed ending under a sustained chord using printed spacing.

    Both staves must already contain a quarter plus a dotted half in 4/4.
    The extra pair must have a separate opposing stem and single beam. Two
    correctly timed bass columns independently corroborate its half-beat grid.
    """
    mapped, ink = _head_map(root, geometry), _ink(image)
    targets = _target_by_measure(root, initial_meter)
    changes = []
    for number, measure in enumerate(root.findall("part/measure"), 1):
        notes = measure.findall("note")
        if not notes or len(notes) > 16 or any(n.find("grace") is not None for n in notes):
            continue
        top = _groups([n for n in notes if n.findtext("staff", "1") == "1"])
        bass = _groups([n for n in notes if n.findtext("staff", "1") == "2"])
        if len(top) != 4 or len(bass) != 2 or any(len(g) != 1 for g in [top[0], top[2], top[3], *bass]):
            continue
        rest, sustained, late = top[0][0], top[1], [top[2][0], top[3][0]]
        quarter = int(rest.findtext("duration", "0"))
        if (rest.find("rest") is None or rest.findtext("type") != "quarter"
                or targets.get(number) != 4*quarter or quarter <= 0 or quarter % 2
                or any(n.findtext("type") != "half" or len(n.findall("dot")) != 1
                       or int(n.findtext("duration", "0")) != 3*quarter for n in [*sustained, bass[1][0]])
                or bass[0][0].findtext("type") != "quarter"
                or int(bass[0][0].findtext("duration", "0")) != quarter
                or any(n not in mapped or n.findtext("type") != "eighth" or n.find("dot") is not None
                       or n.find("time-modification") is not None
                       or int(n.findtext("duration", "0")) != quarter//2 for n in late)
                or any(n not in mapped for g in bass for n in g)):
            continue
        desired = _note_onsets(measure)
        if (desired[rest] != 0 or any(desired[n] != quarter for n in sustained)
                or [desired[g[0]] for g in bass] != [0, quarter]
                or [desired[n] for n in late] != [4*quarter, 4*quarter+quarter//2]):
            continue
        a, b = [mapped[g[0]] for g in bass]
        unit = b["x"]-a["x"]
        if unit < 2*a["staff_space"]:
            continue
        if any(abs((mapped[n]["x"]-a["x"])/unit-t) > 0.2
               for n, t in zip(late, (3, 3.5))):
            continue
        if (not _single_beam(mapped[late[0]], mapped[late[1]], ink, False)
                or not any(n in mapped and abs(mapped[n]["x"]-b["x"]) < 3
                           and _stem_end(mapped[n], ink, True) for n in sustained)):
            continue
        saved = copy.deepcopy(list(measure))
        used = {n.findtext("voice", "1") for n in notes}
        voice = next(str(v) for v in range(1, 33) if str(v) not in used)
        for n, t in zip(late, (3*quarter, 3*quarter+quarter//2)):
            desired[n] = t
            n.find("voice").text = voice
        if _rebuild(measure, desired):
            changes.append({"measure": number, "method": "opposing_beamed_ending_with_bass_spacing",
                            "changed_onsets": 2, "new_voice": voice,
                            "bass_anchor_x": [a["x"], b["x"]]})
        else:
            measure[:] = saved
    return changes


def _stem(head, ink, up):
    import cv2
    s = head["staff_space"]
    x = head["x"]+(0.6*s if up else -0.6*s)
    a, b = ((head["y"]-2.8*s, head["y"]-0.4*s) if up
            else (head["y"]+0.4*s, head["y"]+2.2*s))
    top, bottom = round(a), round(b)
    if top < 0 or bottom > ink.shape[0]:
        return False
    return any(cv2.countNonZero(ink[top:bottom, col:col+1]) >= 0.85*(bottom-top)
               for col in range(round(x)-2, round(x)+3))


def _interior(head, ink):
    import cv2
    x, y, s = head["x"], head["y"], head["staff_space"]
    l, r = round(x-0.2*s), round(x+0.2*s)
    t, b = round(y-0.12*s), round(y+0.12*s)
    crop = ink[t:b, l:r]
    return cv2.countNonZero(crop)/crop.size if crop.size else None


def repair(root, geometry, image, initial_meter=None):
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    ink, targets = _ink(image), _target_by_measure(root, initial_meter)
    system_of = {n: i for i, notes in enumerate(systems) for n in notes}
    changes = []
    for number, measure in enumerate(root.findall("part/measure"), 1):
        notes = measure.findall("note")
        if (not notes or len(notes) > 12 or any(n.find("pitch") is None or n.find("grace") is not None
                       or n.find("dot") is not None or n.find("time-modification") is not None for n in notes)):
            continue
        groups = _groups(notes)
        if len(groups) != 4 or len(groups[0]) != 1 or len(groups[1]) != 1 or len(groups[3]) != 1:
            continue
        first, second, closing, bass = groups[0][0], groups[1][0], groups[2], groups[3][0]
        staff_no, position = _position(first)
        if (staff_no != 1 or first.findtext("type") != "half" or second.findtext("type") != "quarter"
                or second.findtext("staff", "1") != "1" or bass.findtext("staff", "1") != "2"
                or any(n.findtext("type") != "half" or n.findtext("staff", "1") != "1" for n in closing)):
            continue
        quarter = int(second.findtext("duration", "0"))
        if (targets.get(number) != 4*quarter or quarter <= 0
                or any(n.findtext("voice") != first.findtext("voice") for n in [second, *closing])
                or any(int(n.findtext("duration", "0")) != 2*quarter for n in [first, *closing, bass])):
            continue
        staffs = geometry["systems"][system_of[first]].get("staffs", [])
        if len(staffs) != 1:
            continue
        staff = staffs[0]
        midpoint = sum(staff["y_range"])/2
        heads = sorted([h for h in staff["notes"] if h["y"] < midpoint], key=lambda h: h["x"])
        same = [h for h in heads if h["position"] == position and h.get("staff_space")]
        if len(same) != 2:
            continue
        solid, hollow = same
        s = solid["staff_space"]
        if (not 0.8*s <= hollow["x"]-solid["x"] <= 1.6*s or abs(solid["y"]-hollow["y"]) > 0.25*s
                or _interior(solid, ink) < 0.7 or _interior(hollow, ink) > 0.25
                or not _stem(solid, ink, False) or not _stem(hollow, ink, True)):
            continue
        # A low treble ledger note can cross the grand-staff midpoint. Its
        # negative original staff position still identifies this nearby head.
        next_heads = [h for h in staff["notes"] if h["position"] == _position(second)[1]
                      and h["x"] > hollow["x"]+s]
        if len(next_heads) != 1:
            continue
        column_heads = [[h for h in staff["notes"]
                         if (h["y"] < midpoint) == (n.findtext("staff", "1") == "1")
                         and h["position"] == _position(n)[1] and h["x"] > next_heads[0]["x"]+s]
                        for n in [*closing, bass]]
        columns = [h["x"] for h in column_heads[0]
                   if all(len([k for k in choices if abs(k["x"]-h["x"]) <= 3]) == 1
                          for choices in column_heads)]
        if len(columns) != 1:
            continue
        saved = copy.deepcopy(list(measure))
        desired = _note_onsets(measure)
        if desired[first] != 0:
            continue
        clone = ET.Element("note")
        clone.append(copy.deepcopy(first.find("pitch")))
        for tag, value in (("duration", quarter), ("voice", first.findtext("voice", "1")),
                           ("type", "quarter"), ("stem", "down"), ("staff", "1")):
            ET.SubElement(clone, tag).text = str(value)
        used = {n.findtext("voice", "1") for n in notes}
        voice = next(str(v) for v in range(1, 33) if str(v) not in used)
        first.find("voice").text = voice
        stem = first.find("stem")
        if stem is None:
            stem = ET.Element("stem")
            first.insert(list(first).index(first.find("staff")), stem)
        stem.text = "up"
        measure.insert(list(measure).index(first), clone)
        desired[clone] = 0
        desired[second] = quarter
        for n in [*closing, bass]:
            desired[n] = 2*quarter
        if _rebuild(measure, desired):
            changes.append({"measure": number, "method": "solid_hollow_unison_with_opposing_stems",
                            "added_notes": 1, "changed_onsets": len(closing)+2,
                            "head_x": [solid["x"], hollow["x"]]})
        else:
            measure[:] = saved
    return changes
