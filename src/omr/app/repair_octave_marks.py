"""Recover upper ottava spans from an 8 glyph, line and downward end hook.

Two closed counters identify the numeral candidate. Its attached horizontal
line and end hook distinguish it from ordinary text. A bounded, unique image
sequence alignment must explain every head in a span before changing notes.
No reference score, font template or additional inference model is used.
"""

from itertools import combinations
from statistics import median
import xml.etree.ElementTree as ET

from align_homr_geometry import _position, _xml_systems
from repair_ledger_chords import _groups


def detect(geometry, image):
    import cv2
    import numpy as np

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (image < 140).astype(np.uint8)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    spans = []
    for index, contour in enumerate(contours):
        if hierarchy[0, index, 3] != -1:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if not 5 <= width <= 40 or not 10 <= height <= 50:
            continue
        child, holes, hole_heights = hierarchy[0, index, 2], [], []
        while child >= 0:
            area = cv2.contourArea(contours[child])
            if area >= 2:
                holes.append(area)
                hole_heights.append(cv2.boundingRect(contours[child])[3])
            child = hierarchy[0, child, 0]
        if (len(holes) != 2 or min(holes)/max(holes) < 0.4 or
                min(hole_heights)/max(hole_heights) < 0.75):
            continue
        eligible = []
        for system_index, system in enumerate(geometry.get("systems", [])):
            if len(system.get("staffs", [])) != 1:
                continue
            staff = system["staffs"][0]
            spaces = [n["staff_space"] for n in staff["notes"] if n.get("staff_space")]
            if not spaces:
                continue
            space = median(spaces)
            top = staff["y_range"][0]
            if (staff["x_range"][0] <= x <= staff["x_range"][1] and
                    0.6*space < top-y-height < 7*space and
                    0.6*space <= width <= 1.6*space and
                    space <= height <= 2*space):
                eligible.append((system_index, space))
        if len(eligible) != 1:
            continue
        system_index, space = eligible[0]
        start_x = x + width - 2
        max_gap = max(2, round(space*0.45))
        endpoints = []
        for line_y in range(max(0, y-3), min(ink.shape[0], y+4)):
            row = ink[line_y, start_x:min(ink.shape[1], start_x+1024)]
            black = np.flatnonzero(row)
            if not len(black) or black[0] > max_gap:
                continue
            end = int(black[0])
            for value in black[1:]:
                if value-end > max_gap+1:
                    break
                end = int(value)
            if not space <= end <= 64*space:
                continue
            coverage = float(row[:end+1].mean())
            if not 0.3 <= coverage <= 0.8:
                continue
            hook_x = start_x+end
            hook = ink[line_y:line_y+round(space),
                       max(0, hook_x-1):hook_x+2].max(axis=1)
            if len(hook) != round(space) or float(hook.mean()) < 0.7:
                continue
            endpoints.append((line_y, hook_x))
        if not endpoints:
            continue
        # Neighboring rows must agree on the end of the same bracket.
        ends = [p[1] for p in endpoints]
        if max(ends)-min(ends) > 3:
            continue
        spans.append({"system": system_index, "left": x-space*0.3,
                      "right": median(ends)+space*0.3, "y": y,
                      "method": "two_counter_8_line_hook"})
    return spans


def _assignment(notes, heads, octave):
    if not 0 < len(heads) <= len(notes) <= 4:
        return None
    if not octave:
        if sorted(_position(n)[1] for n in notes) != sorted(h["position"] for h in heads):
            return None
        return (0, tuple(range(len(notes)))) if len(notes) == len(heads) else None
    if any(n.find("grace") is not None for n in notes):
        return None
    if len({(n.findtext("voice"), n.findtext("duration")) for n in notes}) != 1:
        return None
    positions = [_position(n)[1] for n in notes]
    best, paths = None, []
    # Vertical order is fixed. Reassigning octave partners in the opposite
    # order creates tied costs even though their image order is unambiguous.
    for indices in combinations(range(len(notes)), len(heads)):
        targets = [int(h["position"])+7 for h in heads]
        deltas = [targets[j]-positions[i] for j, i in enumerate(indices)]
        if any(d % 7 or abs(d) > 14 for d in deltas):
            continue
        omitted = set(range(len(notes)))-set(indices)
        if any(not any(positions[k] % 7 == p % 7 and positions[k] != p
                       for p in (positions[i] for i in indices)) for k in omitted):
            continue
        cost = sum(abs(d)//7 for d in deltas)+len(omitted)
        if best is None or cost < best:
            best, paths = cost, [indices]
        elif cost == best:
            paths.append(indices)
    return (best, paths[0]) if len(paths) == 1 else None


def _align(groups, head_groups, spans):
    n, m = len(groups), len(head_groups)
    if not n or not m or max(n, m) > 128:
        return {}
    active = [next((i for i, span in enumerate(spans)
                   if span["left"] <= group[0]["x"] <= span["right"]), None)
              for group in head_groups]
    assignments = [[_assignment(notes, heads, active[j] is not None)
                    for j, heads in enumerate(head_groups)] for notes in groups]
    inf = 4*sum(map(len, groups))+4*sum(map(len, head_groups))+1
    f = [[inf]*(m+1) for _ in range(n+1)]
    b = [[inf]*(m+1) for _ in range(n+1)]
    f[0][0] = b[n][m] = 0
    for i in range(n+1):
        for j in range(m+1):
            v = f[i][j]
            if i < n:
                f[i+1][j] = min(f[i+1][j], v+4*len(groups[i]))
            if j < m:
                f[i][j+1] = min(f[i][j+1], v+4*len(head_groups[j]))
            if i < n and j < m and assignments[i][j] is not None:
                f[i+1][j+1] = min(f[i+1][j+1], v+assignments[i][j][0])
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            v = b[i][j]
            if i:
                b[i-1][j] = min(b[i-1][j], v+4*len(groups[i-1]))
            if j:
                b[i][j-1] = min(b[i][j-1], v+4*len(head_groups[j-1]))
            if i and j and assignments[i-1][j-1] is not None:
                b[i-1][j-1] = min(b[i-1][j-1], v+assignments[i-1][j-1][0])
    mapped = {}
    for j in range(m):
        if active[j] is None:
            continue
        skip = any(f[i][j]+4*len(head_groups[j])+b[i][j+1] == f[n][m]
                   for i in range(n+1))
        choices = [i for i in range(n) if assignments[i][j] is not None and
                   f[i][j]+assignments[i][j][0]+b[i+1][j+1] == f[n][m]]
        if not skip and len(choices) == 1:
            i = choices[0]
            mapped[j] = (groups[i], assignments[i][j][1], active[j])
    return mapped


def repair(root, geometry, image):
    if root.find(".//octave-shift") is not None:
        return []
    systems = _xml_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    spans = detect(geometry, image)
    parents = {n: m for m in root.findall("part/measure") for n in m.findall("note")}
    changes = []
    for index, (notes, system) in enumerate(zip(systems, geometry["systems"])):
        selected = [s for s in spans if s["system"] == index]
        if not selected:
            continue
        staff = system["staffs"][0]
        midpoint = sum(staff["y_range"])/2
        groups = _groups([n for n in notes if n.findtext("staff", "1") == "1"])
        groups = [sorted(g, key=lambda n: _position(n)[1]) for g in groups]
        raw = sorted([h for h in staff["notes"] if h["y"] < midpoint], key=lambda h: h["x"])
        head_groups = []
        for head in raw:
            same_stem = (head_groups and head.get("stem") and
                         head.get("staff_space") and
                         head["x"]-head_groups[-1][0]["x"] <= 1.5*head["staff_space"] and
                         any(h.get("stem") and abs(h["stem"][0]-head["stem"][0]) <= 1.5
                             for h in head_groups[-1]))
            if not head_groups or (head["x"]-head_groups[-1][-1]["x"] > 3 and not same_stem):
                head_groups.append([])
            head_groups[-1].append(head)
        head_groups = [sorted(g, key=lambda h: h["position"]) for g in head_groups]
        mapped = _align(groups, head_groups, selected)
        for span_index, span in enumerate(selected):
            indices = [j for j, g in enumerate(head_groups)
                       if span["left"] <= g[0]["x"] <= span["right"]]
            if not 1 <= len(indices) <= 16 or any(j not in mapped for j in indices):
                continue
            kept, changed_pitches, removed = [], 0, 0
            for j in indices:
                group, assignment, _ = mapped[j]
                retained = [group[i] for i in assignment]
                # An octave chord may be reordered by the decoder. Retain
                # its original leading element's cursor advance.
                original_order = [n for n in parents[group[0]].findall("note") if n in group]
                surviving = [n for n in original_order if n in retained]
                for n in group:
                    if n not in retained:
                        parents[n].remove(n)
                        removed += 1
                for k, n in enumerate(surviving):
                    chord = n.find("chord")
                    if k == 0 and chord is not None:
                        n.remove(chord)
                    elif k and chord is None:
                        n.insert(0, ET.Element("chord"))
                for head, n in zip(head_groups[j], retained):
                    delta = (int(head["position"])+7-_position(n)[1])//7
                    if delta:
                        n.find("pitch/octave").text = str(int(n.findtext("pitch/octave"))+delta)
                        changed_pitches += 1
                kept.extend(surviving)
            if kept and (changed_pitches or removed):
                # Export actual pitches with ledger lines. In interleaved
                # Homr voices Verovio 6.3 attaches octave-shift directions to
                # the wrong staff despite staff=1. Pitch-only notation is
                # equivalent musically and does not transpose the bass.
                changes.append({**span, "changed_pitches": changed_pitches,
                                "removed_decoder_octave_duplicates": removed,
                                "export_notation": "actual_pitch_ledger_lines"})
    return changes
