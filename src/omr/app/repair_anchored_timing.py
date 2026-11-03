"""Reflow complete voices and locate partial voices from image columns.

Only overfull bars with unchanged pitches, durations and voice labels are
eligible. Complete voices supply timing anchors; every partial pitched event
must coincide with an unambiguous anchor. Ambiguous bars are left untouched.
"""

from collections import OrderedDict
from fractions import Fraction
import xml.etree.ElementTree as ET


def repair(root, note_x, audit_timing_fn, parse_meter_fn):
    overfull = {(row["part"], row["measure"])
                for row in audit_timing_fn(root)["overfull_measures"]}
    changes = []
    for part_index, part in enumerate(root.findall("part"), 1):
        divisions, meter = 1, None
        for index, measure in enumerate(part.findall("measure"), 1):
            for attributes in measure.findall("attributes"):
                value = attributes.findtext("divisions")
                if value is not None:
                    divisions = int(value)
                time = attributes.find("time")
                if time is not None:
                    meter = parse_meter_fn(f"{time.findtext('beats')}/{time.findtext('beat-type')}")
            if ((part_index, index) not in overfull or meter is None or
                    measure.get("implicit") == "yes"):
                continue
            target = Fraction(meter[0] * 4 * divisions, meter[1])
            if target.denominator != 1 or not 0 < target <= 256:
                continue
            result = _measure(measure, target.numerator, note_x)
            if result:
                changes.append({"part": part_index, "measure": index, **result})
    return changes


def _measure(measure, target, note_x):
    elements = list(measure)
    indices = [i for i, element in enumerate(elements) if element.tag == "note"]
    if not indices or len(indices) > 128:
        return None
    first, last = indices[0], indices[-1]
    if (any(e.tag not in {"note", "backup"} for e in elements[first:last+1]) or
            any(e.tag != "barline" for e in elements[last+1:]) or
            any(e.tag in {"backup", "forward"} for e in elements[:first])):
        return None
    streams = OrderedDict()
    previous = None
    original = {}
    cursor = previous_onset = 0
    for e in elements[first:last+1]:
        if e.tag == "backup":
            cursor -= int(e.findtext("duration", "0"))
            continue
        duration = int(e.findtext("duration", "0"))
        voice = e.findtext("voice")
        if e.find("grace") is not None or duration <= 0 or voice is None:
            return None
        key = (e.findtext("staff", "1"), voice)
        groups = streams.setdefault(key, [])
        if e.find("chord") is not None:
            if previous != key or not groups or duration != groups[-1][0]:
                return None
            groups[-1][1].append(e)
            original[e] = previous_onset
        else:
            groups.append((duration, [e]))
            previous_onset = cursor
            original[e] = cursor
            cursor += duration
        previous = key
    totals = {key: sum(duration for duration, _ in groups)
              for key, groups in streams.items()}
    complete = [key for key in streams if totals[key] == target]
    partial = [key for key in streams if totals[key] < target]
    if len(complete) < 2 or not partial or len(complete)+len(partial) != len(streams):
        return None
    desired, anchors = {}, []
    for key in complete:
        onset, seen_x, mapped_groups = 0, [], 0
        for duration, notes in streams[key]:
            xs = [note_x[n] for n in notes if n in note_x]
            if xs:
                if max(xs)-min(xs) > 3:
                    return None
                x = sum(xs)/len(xs)
                if seen_x and x <= seen_x[-1] + 3:
                    return None
                seen_x.append(x)
                anchors.append((x, onset))
                mapped_groups += 1
            for n in notes:
                desired[n] = onset
            onset += duration
        if mapped_groups < 3 or mapped_groups / len(streams[key]) < 0.6:
            return None
    # Two complete voices at the same printed column must agree in time.
    if any(abs(x1-x2) <= 3 and t1 != t2
           for x1, t1 in anchors for x2, t2 in anchors):
        return None
    for key in partial:
        end = 0
        for duration, notes in streams[key]:
            if any(n.find("pitch") is None or n not in note_x for n in notes):
                return None
            xs = [note_x[n] for n in notes]
            if max(xs)-min(xs) > 3:
                return None
            times = {t for x, t in anchors if abs(x - sum(xs)/len(xs)) <= 3}
            if len(times) != 1:
                return None
            onset = next(iter(times))
            if onset < end or onset+duration > target:
                return None
            for n in notes:
                desired[n] = onset
            end = onset + duration
    changed = sum(original[n] != t for n, t in desired.items())
    if not changed:
        return None
    rebuilt = elements[:first]
    cursor = previous_onset = 0
    for e in elements[first:last+1]:
        if e.tag != "note":
            continue
        onset = desired[e]
        if e.find("chord") is not None:
            if onset != previous_onset:
                return None
        else:
            if onset != cursor:
                shift = ET.Element("forward" if onset > cursor else "backup")
                ET.SubElement(shift, "duration").text = str(abs(onset-cursor))
                rebuilt.append(shift)
            cursor = onset + int(e.findtext("duration"))
            previous_onset = onset
        rebuilt.append(e)
    rebuilt.extend(elements[last+1:])
    measure[:] = rebuilt
    return {"method": "complete_voice_image_anchors", "changed_onsets": changed,
            "complete_voices": len(complete), "partial_voices": len(partial)}
