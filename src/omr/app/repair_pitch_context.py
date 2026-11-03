"""Restore sharp context only when printed geometry rules out cancellation.

This uses the same preprocessed image, bounded local crops and staff grid.
No score reference, second model or music-language guess is involved.
"""

import xml.etree.ElementTree as ET
from statistics import median

from align_homr_geometry import _xml_systems
from repair_augmentation_dots import _head_map


def _ink(image):
    import cv2
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(image, 140, 255, cv2.THRESH_BINARY_INV)[1]


def _sharp_pairs(crop, space):
    import cv2
    vertical = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, round(1.3*space))))
    count, _, stats, centers = cv2.connectedComponentsWithStats(vertical)
    bars = [(centers[i][0], centers[i][1], stats[i, cv2.CC_STAT_HEIGHT])
            for i in range(1, count)
            if 1.4*space <= stats[i, cv2.CC_STAT_HEIGHT] <= 2.7*space
            and stats[i, cv2.CC_STAT_WIDTH] <= 0.3*space]
    return [((a[0]+b[0])/2, (a[1]+b[1])/2)
            for a in bars for b in bars
            if 0.25*space <= b[0]-a[0] <= 0.65*space
            and abs(a[1]-b[1]) <= 0.25*space and abs(a[2]-b[2]) <= 0.3*space]


def printed_sharps(staff, ink):
    """Require the ordered signature independently on both piano staves."""
    grid = staff.get("grid", [])
    if not grid or len(grid[0]["y"]) != 10 or not staff.get("notes"):
        return None
    first = min(h["x"] for h in staff["notes"])
    point = min(grid, key=lambda p: abs(p["x"]-(staff["x_range"][0]+first)/2))
    counts = []
    for number in (1, 2):
        lines = point["y"][(number-1)*5:number*5]
        space = median(b-a for a, b in zip(lines, lines[1:]))
        left, right = round(staff["x_range"][0]+3.3*space), round(first-0.65*space)
        top, bottom = round(lines[0]-2*space), round(lines[-1]+space)
        if right-left < 1.1*space or min(left, top) < 0:
            return None
        pairs = [(left+x, top+y) for x, y in _sharp_pairs(ink[top:bottom, left:right], space)]
        expected = (9, 6, 10, 7, 4, 8, 5) if number == 1 else (7, 4, 8, 5, 2, 6, 3)
        previous, count = None, 0
        for position in expected:
            y = lines[-1]-(position-1)*space/2
            choices = [x for x, cy in pairs if abs(cy-y) <= 0.3*space
                       and (previous is None or 0.6*space < x-previous < 1.5*space)]
            if len(choices) != 1:
                break
            previous = choices[0]
            count += 1
        counts.append(count)
    return counts[0] if counts[0] == counts[1] and counts[0] else None


def no_accidental(head, all_heads, ink):
    """A sufficiently wide, otherwise empty strip must have no upright ink.

    Horizontal staff/ledger lines are removed, but short natural stems remain.
    Nearby noteheads and stems cause abstention rather than being erased.
    """
    import cv2
    space = head.get("staff_space")
    if not space:
        return False
    x, y = head["x"], head["y"]
    left, right = round(x-2.3*space), round(x-0.65*space)
    previous = [h for h in all_heads if h is not head and h["x"] < x-3
                and abs(h["y"]-y) < 1.7*space]
    if previous:
        left = max(left, round(max(h["x"]+max(h.get("size", [space])[0]/2,
                                                space*0.6) for h in previous)+1))
    top, bottom = round(y-1.3*space), round(y+1.3*space)
    if (min(left, top) < 0 or bottom > ink.shape[0] or right > ink.shape[1]
            or right-left < 0.75*space):
        return False
    if any(left-space*0.5 < h["x"] < right and abs(h["y"]-y) < 1.7*space
           for h in all_heads if h is not head):
        return False
    crop = ink[top:bottom, left:right]
    horizontal = cv2.morphologyEx(crop, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (round(space*1.2), 1)))
    remaining = cv2.subtract(crop, horizontal)
    vertical = cv2.morphologyEx(remaining, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, round(space*0.45)))))
    return cv2.countNonZero(vertical) == 0 and cv2.countNonZero(remaining) < space*2


def repair(root, geometry, image):
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    ink = _ink(image)
    mapped = _head_map(root, geometry)
    measures = root.findall("part/measure")
    system_by_note = {n: i for i, notes in enumerate(systems) for n in notes}
    active_key, changes = 0, []
    for number, measure in enumerate(measures, 1):
        pitched = [n for n in measure.findall("note") if n.find("pitch") is not None]
        if not pitched:
            continue
        si = system_by_note[pitched[0]]
        staffs = geometry["systems"][si].get("staffs", [])
        if len(staffs) != 1:
            continue
        staff = staffs[0]
        # Cancellation detection was validated for piano grand staves only.
        # A one-staff crop can have a dense displaced accidental column.
        if staff.get('grid') and len(staff['grid'][0]['y']) != 10:
            continue
        fifths = measure.find("attributes/key/fifths")
        if fifths is not None:
            new_key = int(fifths.text)
            printed = measure.find("print")
            if (new_key == 0 and active_key > 0 and printed is not None
                    and printed.get("new-system") == "yes"
                    and printed_sharps(staff, ink) == active_key):
                fifths.text = str(active_key)
                changes.append({"measure": number, "method": "printed_sharp_signature",
                                "fifths": active_key})
            else:
                active_key = new_key
        key = {step: 1 for step in "FCGDAEB"[:max(0, active_key)]}
        context = {}
        # XML is interleaved by image column; keep grace and ordinary notes.
        for n in pitched:
            p = n.find("pitch")
            step, octave = p.findtext("step"), p.findtext("octave")
            identity = (n.findtext("staff", "1"), step, octave)
            actual = int(p.findtext("alter", "0"))
            expected = context.get(identity, key.get(step, 0))
            h = mapped.get(n)
            absent = h is not None and no_accidental(h, staff["notes"], ink)
            if (actual == 0 and expected == 1 and n.find("accidental") is None and absent):
                alter = p.find("alter")
                if alter is None:
                    alter = ET.Element("alter")
                    p.insert(list(p).index(p.find("step"))+1, alter)
                alter.text = "1"
                changes.append({"measure": number, "method": "sharp_context_without_cancellation",
                                "staff": identity[0], "pitch": step+octave,
                                "x": h["x"], "y": h["y"]})
                actual = 1
            # A decoded alteration without a glyph is not a new accidental.
            if n.find("accidental") is not None or h is not None and not absent:
                context[identity] = actual
    return changes
