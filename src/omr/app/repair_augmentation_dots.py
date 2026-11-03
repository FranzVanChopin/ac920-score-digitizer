"""Recover a second augmentation dot with matched image heads and bar guards.

Both round blobs must be visible at each chord member's expected dot row.
Adding the dot must make the existing voice fill the meter exactly. Other
voices keep their old onsets when MusicXML backups are rebuilt.
"""

from fractions import Fraction
import xml.etree.ElementTree as ET

from align_homr_geometry import _position, _xml_systems, _head_staff
from repair_ledger_chords import _align, _groups
from repair_duplicate_noteheads import _note_onsets, _target_by_measure


def _raster_hollow_map(root, geometry, image):
    """Locate unmapped line-crossed hollow heads inside verified bar regions.

    A staff line divides a hollow head into two small enclosed white holes.
    Both holes, the pitch row and the exact visible bar count must agree.
    This bypasses segmentation clumps without changing the neural decoder.
    """
    import cv2
    from statistics import median
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)[1]
    systems = [[]]
    for i, measure in enumerate(root.findall("part/measure")):
        p = measure.find("print")
        if i and p is not None and p.get("new-system") == "yes":
            systems.append([])
        systems[-1].append(measure)
    if len(systems) != len(geometry.get("systems", [])):
        return {}
    result = {}
    for measures, system in zip(systems, geometry["systems"]):
        if len(system.get("staffs", [])) != 1:
            continue
        staff = system["staffs"][0]
        grid = staff.get("grid", [])
        if not grid or len(grid[0]["y"]) != 10:
            continue
        top, bottom = round(staff["y_range"][0]), round(staff["y_range"][1])
        left, right = map(round, staff["x_range"])
        margin = round(median(b-a for a, b in zip(grid[0]["y"][:4], grid[0]["y"][1:5])))
        left, right = max(0, left-margin), min(ink.shape[1]-1, right+margin)
        columns = [x for x in range(left, right+1)
                   if cv2.countNonZero(ink[top:bottom, x:x+1]) >= 0.92*(bottom-top)]
        bars = []
        for x in columns:
            if not bars or x-bars[-1][-1] > 3:
                bars.append([])
            bars[-1].append(x)
        if len(bars) != len(measures)+1 or any(len(b) > 5 for b in bars):
            continue
        bounds = [sum(b)/len(b) for b in bars]
        for measure, start, end in zip(measures, bounds, bounds[1:]):
            for group in _groups(measure.findall("note")):
                if (len(group) != 1 or group[0].find("pitch") is None
                        or group[0].findtext("type") != "half" or len(group[0].findall("dot")) != 1):
                    continue
                n = group[0]
                staff_no, position = _position(n)
                if position % 2 != 1:
                    continue
                point = min(grid, key=lambda p: abs(p["x"]-(start+end)/2))
                lines = point["y"][(staff_no-1)*5:staff_no*5]
                space = median(b-a for a, b in zip(lines, lines[1:]))
                y = lines[-1]-(position-1)*space/2
                l, r = round(start+0.5*space), round(end-0.5*space)
                t, b = round(y-space), round(y+space)
                if min(l, t) < 0 or b > ink.shape[0] or r-l > 1920:
                    continue
                contours, hierarchy = cv2.findContours(ink[t:b, l:r], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if hierarchy is None:
                    continue
                holes = []
                for i, c in enumerate(contours):
                    x0, y0, w, h = cv2.boundingRect(c)
                    if (hierarchy[0][i][3] >= 0 and 0.35*space <= w <= 1.1*space
                            and 0.15*space <= h <= 0.65*space and cv2.contourArea(c) >= 3):
                        holes.append((x0, y0, w, h))
                candidates = []
                for a in holes:
                    for d in holes:
                        if (d[1] <= a[1] or abs((a[0]+a[2]/2)-(d[0]+d[2]/2)) > 0.4*space
                                or not 0.2*space < d[1]-a[1] < 0.8*space):
                            continue
                        x0, y0 = min(a[0], d[0]), a[1]
                        w = max(a[0]+a[2], d[0]+d[2])-x0
                        h = max(a[1]+a[3], d[1]+d[3])-y0
                        cx, cy = l+x0+w/2, t+y0+h/2
                        local = min(grid, key=lambda p: abs(p["x"]-cx))["y"][staff_no*5-1]
                        if (0.65*space <= w <= 1.35*space and 0.55*space <= h <= 1.3*space
                                and abs(1+2*(local-cy)/space-position) < 0.4):
                            candidates.append({"x": cx, "y": cy, "position": position,
                                               "staff_space": space, "size": [w, h],
                                               "method": "paired_white_holes_in_verified_bar"})
                if len(candidates) == 1:
                    result[n] = candidates[0]
    return result


def _head_map(root, geometry):
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return {}
    result = {}
    for notes, system in zip(systems, geometry["systems"]):
        if len(system.get("staffs", [])) != 1:
            continue
        staff = system["staffs"][0]
        midpoint = sum(staff["y_range"])/2
        for number in (1, 2):
            selected = [n for n in notes if n.findtext("staff", "1") == str(number)]
            ordered = [n for g in _groups(selected) for n in sorted(g, key=lambda n: _position(n)[1])]
            raw = sorted([h for h in staff["notes"] if _head_staff(staff, h) == number],
                         key=lambda h: h["x"])
            groups = []
            for h in raw:
                if not groups or h["x"]-groups[-1][-1]["x"] > 3:
                    groups.append([])
                groups[-1].append(h)
            heads = [h for g in groups for h in sorted(g, key=lambda h: h["position"])]
            result.update(_align(ordered, heads))
    return result


def _two_dots(heads, image):
    import cv2
    spaces = [h.get("staff_space") for h in heads]
    if any(not s for s in spaces):
        return False
    space = sum(spaces)/len(spaces)
    x = max(h["x"] for h in heads)
    left, right = round(x+space*0.5), round(x+space*2.5)
    top = round(min(h["y"] for h in heads)-space)
    bottom = round(max(h["y"] for h in heads)+space*0.6)
    if top < 0 or left < 0 or bottom > image.shape[0] or right > image.shape[1]:
        return False
    crop = image[top:bottom, left:right]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(crop, 140, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for contour in contours:
        bx, by, w, h = cv2.boundingRect(contour)
        if (0.2*space <= min(w, h) <= max(w, h) <= 0.6*space and
                min(w, h)/max(w, h) >= 0.7 and
                cv2.countNonZero(ink[by:by+h, bx:bx+w])/(w*h) >= 0.5):
            dots.append((left+bx+(w-1)/2, top+by+(h-1)/2))
    for head in heads:
        expected_y = head["y"] - (space/2 if head["position"] % 2 else 0)
        row = sorted([d for d in dots if abs(d[1]-expected_y) <= space*0.25])
        if (len(row) != 2 or not 0.4*space <= row[1][0]-row[0][0] <= space or
                abs(row[1][1]-row[0][1]) > space*0.2):
            return False
    return True


def repair(root, geometry, image, initial_meter=None):
    mapped = _head_map(root, geometry)
    for n, head in _raster_hollow_map(root, geometry, image).items():
        mapped.setdefault(n, head)
    targets = _target_by_measure(root, initial_meter)
    changes = []
    for number, measure in enumerate(root.findall("part/measure"), 1):
        target = targets.get(number)
        if target is None or target > 256 or measure.get("implicit") == "yes":
            continue
        elements = list(measure)
        indices = [i for i, e in enumerate(elements) if e.tag == "note"]
        if not indices or len(indices) > 128:
            continue
        first, last = indices[0], indices[-1]
        if (any(e.tag not in {"note", "backup"} for e in elements[first:last+1]) or
                any(e.tag != "barline" for e in elements[last+1:])):
            continue
        notes = measure.findall("note")
        if any(n.find("grace") is not None for n in notes):
            continue
        groups = _groups(notes)
        totals = {}
        for n in notes:
            key = (n.findtext("staff", "1"), n.findtext("voice"))
            totals[key] = totals.get(key, 0)+(int(n.findtext("duration", "0"))
                                             if n.find("chord") is None else 0)
        updates = []
        for group in groups:
            if any(n not in mapped or n.findtext("type") != "half" or
                   len(n.findall("dot")) != 1 or n.find("time-modification") is not None
                   for n in group):
                continue
            old = int(group[0].findtext("duration", "0"))
            new = Fraction(old*7, 6)
            key = (group[0].findtext("staff", "1"), group[0].findtext("voice"))
            if (new.denominator != 1 or new <= old or
                    totals[key]+new-old != target or
                    any(int(n.findtext("duration")) != old for n in group)):
                continue
            if _two_dots([mapped[n] for n in group], image):
                updates.append((group, key, new.numerator))
        if not updates or len({key for _, key, _ in updates}) != len(updates):
            continue
        onsets = _note_onsets(measure)
        for group, key, new in updates:
            for n in group:
                n.find("duration").text = str(new)
                n.insert(list(n).index(n.find("dot"))+1, ET.Element("dot"))
            cursor = previous = 0
            for n in notes:
                if (n.findtext("staff", "1"), n.findtext("voice")) != key:
                    continue
                if n.find("chord") is None:
                    previous = cursor
                    cursor += int(n.findtext("duration"))
                onsets[n] = previous
        rebuilt = elements[:first]
        cursor = previous = 0
        for n in notes:
            onset = onsets[n]
            if n.find("chord") is None:
                if onset != cursor:
                    shift = ET.Element("forward" if onset > cursor else "backup")
                    ET.SubElement(shift, "duration").text = str(abs(onset-cursor))
                    rebuilt.append(shift)
                previous = onset
                cursor = onset+int(n.findtext("duration"))
            rebuilt.append(n)
        measure[:] = rebuilt+elements[last+1:]
        changes.append({"measure": number, "method": "visible_second_dot",
                        "changed_durations": sum(len(g) for g, _, _ in updates)})
    return changes
