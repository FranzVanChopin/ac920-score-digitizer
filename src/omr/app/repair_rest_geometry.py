"""Assign a half rest to the staff where its isolated rectangle is printed."""

from align_homr_geometry import _xml_systems
from repair_augmentation_dots import _head_map
from repair_duplicate_noteheads import _note_onsets
from repair_pitch_context import _ink
from repair_unison_voices import _stem


def repair(root, geometry, image):
    import cv2
    from statistics import median
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    system_of = {n: i for i, notes in enumerate(systems) for n in notes}
    mapped, ink, changes = _head_map(root, geometry), _ink(image), []
    for number, measure in enumerate(root.findall("part/measure"), 1):
        onsets = _note_onsets(measure)
        for n in measure.findall("note"):
            if (n.find("rest") is None or n.findtext("type") != "half" or n.find("dot") is not None
                    or n.find("chord") is not None or n.find("grace") is not None):
                continue
            peers = [p for p in measure.findall("note") if p in mapped and onsets[p] == onsets[n]]
            if not peers:
                continue
            xs = [mapped[p]["x"] for p in peers]
            if max(xs)-min(xs) > 3:
                continue
            staff_list = geometry["systems"][system_of[peers[0]]].get("staffs", [])
            if len(staff_list) != 1 or not staff_list[0].get("grid"):
                continue
            staff, x = staff_list[0], sum(xs)/len(xs)
            point = min(staff["grid"], key=lambda p: abs(p["x"]-x))
            if len(point["y"]) != 10:
                continue
            ys = point["y"]
            space = median([ys[i+1]-ys[i] for i in range(9) if i != 4])
            l, r = round(x-space), round(x+space)
            t, b = round(ys[0]-space), round(ys[-1]+space)
            if min(l, t) < 0 or r > ink.shape[1] or b > ink.shape[0]:
                continue
            crop = ink[t:b, l:r]
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rectangles = []
            for c in contours:
                bx, by, w, h = cv2.boundingRect(c)
                body = crop[by:by+max(0, h-2), bx:bx+w]
                rows = [cv2.countNonZero(row) for row in body]
                if (0.9*space <= w <= 2*space and 0.25*space <= h <= 0.7*space
                        and bx > 0 and by > 0 and bx+w < r-l and by+h < b-t
                        and rows and min(rows) >= 0.9*space and max(rows) <= 1.6*space
                        and max(rows)-min(rows) <= 2
                        and abs(l+bx+w/2-x) < 0.3*space):
                    rectangles.append((l+bx+w/2, t+by+h/2))
            if len(rectangles) != 1:
                continue
            cy = rectangles[0][1]
            gap_midpoint = (ys[4]+ys[5])/2
            if abs(cy-gap_midpoint) < 0.6*space:
                # A rest on an added ledger line can lie exactly between
                # staves. Two preceding opposing-stem quarter columns identify
                # its otherwise merged second voice; proximity is insufficient.
                duration = int(n.findtext("duration", "0"))
                if onsets[n] != duration or duration % 2:
                    continue
                possible = []
                for sn in ("1", "2"):
                    proven = True
                    for onset in (0, duration//2):
                        heads = [mapped[p] for p in measure.findall("note") if p in mapped
                                 and p.findtext("staff", "1") == sn and onsets[p] == onset
                                 and p.findtext("type") == "quarter"
                                 and int(p.findtext("duration", "0")) == duration//2]
                        if (len(heads) != 2 or abs(heads[0]["x"]-heads[1]["x"]) > 3
                                or not any(_stem(h, ink, True) for h in heads)
                                or not any(_stem(h, ink, False) for h in heads)):
                            proven = False
                    if proven:
                        possible.append(sn)
                if len(possible) != 1:
                    continue
                found = possible[0]
            else:
                found = "1" if cy < gap_midpoint else "2"
            staff_element = n.find("staff")
            if staff_element is not None and staff_element.text != found:
                old = staff_element.text
                staff_element.text = found
                changes.append({"measure": number, "method": "isolated_half_rest_rectangle",
                                "old_staff": old, "staff": found, "x": rectangles[0][0], "y": cy})
    return changes
