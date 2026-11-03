"""Recognize a small opening grace run and anchor slow notes to fast voices.

Only a completely mapped, full-bar fast voice and a pair of image-aligned
half notes on the other staff are eligible. Rests retain their old onsets.
"""

import xml.etree.ElementTree as ET
import copy

from repair_augmentation_dots import _head_map
from repair_duplicate_noteheads import _note_onsets, _target_by_measure


def _wedge_above(head, image):
    import cv2
    s, x, y = head["staff_space"], head["x"], head["y"]
    l, r = round(x-1.2*s), round(x+1.2*s)
    t, b = round(y-2.2*s), round(y-0.65*s)
    if min(l, t) < 0 or r > image.shape[1] or b > image.shape[0]:
        return False
    crop = image[t:b, l:r]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(crop, 140, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (round(s*1.6), 1)))
    ink = cv2.subtract(ink, horizontal)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, round(s*0.2)))))
    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in contours:
        bx, by, w, h = cv2.boundingRect(c)
        if (not 0.3*s <= w <= 0.7*s or not 0.55*s <= h <= 1.05*s
                or bx == 0 or by == 0 or bx+w == r-l or by+h == b-t
                or abs(l+bx+w/2-x) > 0.25*s):
            continue
        mask = ink[by:by+h, bx:bx+w]
        height = max(2, h//3)
        upper = cv2.countNonZero(mask[:height])
        lower = cv2.countNonZero(mask[-height:])
        if upper > 1.6*lower and lower > 0 and 0.15 < cv2.contourArea(c)/(w*h) < 0.6:
            found.append(c)
    return len(found) == 1


def _rebuild(measure, desired):
    elements = list(measure)
    indices = [i for i, e in enumerate(elements) if e.tag == "note"]
    first, last = indices[0], indices[-1]
    if (any(e.tag not in {"note", "backup", "forward"} for e in elements[first:last+1])
            or any(e.tag not in {"barline"} for e in elements[last+1:])):
        return False
    rebuilt = elements[:first]
    cursor = previous = 0
    for n in measure.findall("note"):
        onset = desired[n]
        if n.find("chord") is not None:
            if onset != previous:
                return False
        else:
            if onset != cursor:
                shift = ET.Element("forward" if onset > cursor else "backup")
                ET.SubElement(shift, "duration").text = str(abs(onset-cursor))
                rebuilt.append(shift)
            previous = onset
            cursor = onset + (0 if n.find("grace") is not None else int(n.findtext("duration", "0")))
        rebuilt.append(n)
    measure[:] = rebuilt+elements[last+1:]
    return True


def repair(root, geometry, image=None, initial_meter=None):
    mapped = _head_map(root, geometry)
    targets = _target_by_measure(root, initial_meter)
    changes = []
    for number, measure in enumerate(root.findall("part/measure"), 1):
        target = targets.get(number)
        notes = measure.findall("note")
        if not target or target > 256 or not notes or len(notes) > 128:
            continue
        ordinary = [n for n in notes if n.find("pitch") is not None and n.find("grace") is None]
        # A single dense voice supplies an independent printed time axis.
        anchors = []
        for key in {(n.findtext("staff", "1"), n.findtext("voice")) for n in ordinary}:
            stream = [n for n in notes if (n.findtext("staff", "1"), n.findtext("voice")) == key]
            leads = [n for n in stream if n.find("chord") is None and n.find("grace") is None]
            if (len(leads) < 8 or any(n.find("rest") is not None or n not in mapped
                                       or n.find("grace") is not None for n in stream)
                    or sum(int(n.findtext("duration", "0")) for n in leads) != target):
                continue
            cursor, xs, axis = 0, [], []
            for n in leads:
                x = mapped[n]["x"]
                if xs and x <= xs[-1]+3:
                    break
                xs.append(x)
                axis.append((x, cursor))
                cursor += int(n.findtext("duration"))
            else:
                anchors.append((key, stream, axis))
        if len(anchors) != 1:
            continue
        key, stream, axis = anchors[0]
        original = _note_onsets(measure)
        if (any(original[n] != t for n, (_, t) in zip(
                [n for n in stream if n.find("chord") is None], axis))
                or any(n.findtext("staff", "1") == key[0] and n not in stream for n in ordinary)):
            continue
        other = [n for n in notes if n.findtext("staff", "1") != key[0] and n.find("pitch") is not None]
        small = [n for n in other if n.find("grace") is None and n.findtext("type") in {"16th", "32nd"}
                 and n.find("chord") is None and n in mapped and mapped[n].get("staff_space")
                 and mapped[n]["size"][0] <= 1.05*mapped[n]["staff_space"]
                 and mapped[n]["size"][1] <= 0.8*mapped[n]["staff_space"]
                 and mapped[n]["x"] < axis[0][0]-3]
        slow = [n for n in other if n.find("grace") is None and n not in small]
        if (len(slow) != 2 or any(n not in mapped or n.findtext("type") != "half"
                                   or n.find("chord") is not None for n in slow)
                or sum(int(n.findtext("duration")) for n in slow) != target):
            continue
        times = [{t for x, t in axis if abs(x-mapped[n]["x"]) <= 3} for n in slow]
        if any(len(t) != 1 for t in times) or sorted(next(iter(t)) for t in times) != [0, target//2]:
            continue
        if small:
            opening = [n for n in other if n in mapped and mapped[n]["x"] < mapped[slow[0]]["x"]-3]
            opening.sort(key=lambda n: mapped[n]["x"])
            if (len(opening) < 3 or not any(n.find("grace") is not None for n in opening)
                    or any(n.find("grace") is None and n not in small for n in opening)
                    or any(not 0 < mapped[b]["x"]-mapped[a]["x"] < 1.8*mapped[a]["staff_space"]
                           for a, b in zip(opening, opening[1:]))):
                continue
        desired = original
        old_slow = [desired[n] for n in slow]
        saved = copy.deepcopy(list(measure))
        # The fast voice already gives the musical onsets correctly. Only the
        # slower voice affected by decoder cursor interleaving is adjusted.
        for n, t in zip(slow, times):
            desired[n] = next(iter(t))
        for n in small:
            n.insert(0, ET.Element("grace"))
            n.remove(n.find("duration"))
            desired[n] = 0
        removed = []
        rests = [n for n in notes if n.findtext("staff", "1") != key[0] and n.find("rest") is not None]
        # The decoder has mistaken two identical tiny staccatissimo wedges
        # for rests. Each rest must occupy its paired half note's old column.
        if (image is not None and len(rests) == 2 and len(set(old_slow)) == 2
                and sorted(desired[n] for n in rests) == sorted(old_slow)
                and all(_wedge_above(mapped[n], image) for n in slow)):
            for n in rests:
                measure.remove(n)
                removed.append(n)
        if _rebuild(measure, desired):
            changed = sum(a != desired[n] for a, n in zip(old_slow, slow))
            if changed or small or removed:
                changes.append({"measure": number, "method": "opening_grace_and_fast_voice_columns",
                                "converted_graces": len(small), "changed_onsets": changed,
                                "removed_wedge_rests": len(removed)})
        else:
            measure[:] = saved
    return changes
