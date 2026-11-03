"""Bounded single-staff retry for overfull dense bass octave passages.

Reuses Homr's already loaded Transformer, sequentially. Only complete,
image-verified runs of beamed sixteenth octave chords can replace a voice.
The input image and model output are the only evidence used at runtime.
"""

import copy
from fractions import Fraction
from statistics import median
import xml.etree.ElementTree as ET

from align_homr_geometry import _position
from omr_bar_geometry import bar_bounds, measure_systems
from repair_duplicate_noteheads import _note_onsets
from repair_grace_timing import _rebuild
from repair_ledger_chords import _groups, _shared_stem
from repair_pitch_context import _ink
from repair_unison_voices import _interior, _stem_end


def head_groups(staff, start, end):
    mid = sum(staff["y_range"])/2
    heads = sorted((h for h in staff["notes"] if h["y"] > mid and start < h["x"] < end),
                   key=lambda h: h["x"])
    groups = []
    for head in heads:
        if not groups or head["x"]-groups[-1][-1]["x"] > 3:
            groups.append([])
        groups[-1].append(head)
    return groups


def _double_beam(a, b, ink):
    space = a.get("staff_space")
    if not space or not 0.7*space < b["x"]-a["x"] < 5*space:
        return False
    sa, sb = _stem_end(a, ink, True), _stem_end(b, ink, True)
    if not sa or not sb or abs(sa[1]-sb[1]) > space:
        return False
    for ratio in (0.4, 0.5, 0.6):
        x, y = (round(sa[j]*(1-ratio)+sb[j]*ratio) for j in (0, 1))
        top, bottom = y-round(0.5*space), y+round(1.8*space)
        if min(top, x) < 0 or bottom > ink.shape[0] or x >= ink.shape[1]:
            return False
        runs = []
        for row in range(top, bottom):
            if ink[row, x]:
                if not runs or row-runs[-1][-1] > 1:
                    runs.append([])
                runs[-1].append(row)
        if (len(runs) != 2 or any(not 0.25*space <= len(r) <= 0.65*space for r in runs)
                or not 0.5*space <= median(runs[1])-median(runs[0]) <= space):
            return False
    return True


def verify_candidate(measure, groups, ink, divisions, target):
    """Validate all chord columns; allow one independently proven ledger fix."""
    notes = measure.findall("note")
    if (not 8 <= len(groups) <= 64 or any(len(g) != 2 for g in groups)
            or len(notes) != 2*len(groups)
            or any(n.find("pitch") is None or n.find("grace") is not None
                   or n.find("tie") is not None or n.find("dot") is not None
                   or n.find("time-modification") is not None
                   or n.findtext("type") != "16th" or n.find("accidental") is not None
                   or int(n.findtext("pitch/alter", "0")) != 0
                   or Fraction(int(n.findtext("duration", "0")), divisions) != Fraction(1, 4)
                   for n in notes)):
        return None
    xml_groups = _groups(notes)
    if (len(xml_groups) != len(groups) or any(len(g) != 2 for g in xml_groups)
            or len(groups)*Fraction(1, 4) != target):
        return None
    # Every decoded column must have the same positive printed double beam.
    upper_heads = [min(g, key=lambda h: h["y"]) for g in groups]
    connected = set()
    for i, (a, b) in enumerate(zip(upper_heads, upper_heads[1:])):
        if _double_beam(a, b, ink):
            connected.update((i, i+1))
    if len(connected) != len(groups):
        return None
    fixes = []
    for xml_group, heads in zip(xml_groups, groups):
        low_head, high_head = sorted(heads, key=lambda h: h["y"], reverse=True)
        space = low_head.get("staff_space")
        if (not space or abs(2*(low_head["y"]-high_head["y"])/space-7) > 0.3
                or any((_interior(h, ink) or 0) < 0.7 for h in heads)):
            return None
        low, high = sorted(xml_group, key=lambda n: _position(n)[1])
        # Notes are normalized to staff 2 before this test.
        position = _position(low)[1]
        if position != int(low_head["position"]) or low_head.get("extended_ledger"):
            return None
        if _position(high)[1] == position+7:
            continue
        if (fixes or not high_head.get("extended_ledger")
                or abs(_position(high)[1]-position-7) > 2
                or not _shared_stem(low_head, high_head)):
            return None
        pitch = high.find("pitch")
        old = pitch.findtext("step")+pitch.findtext("octave")
        pitch.find("step").text = low.findtext("pitch/step")
        pitch.find("octave").text = str(int(low.findtext("pitch/octave"))+1)
        fixes.append({"old": old, "new": pitch.findtext("step")+pitch.findtext("octave"),
                      "x": high_head["x"]})
    return fixes


def replace_bass(measure, candidate, candidate_divisions, old_divisions):
    old = [n for n in measure.findall("note") if n.findtext("staff", "1") == "2"]
    if not old or len({n.findtext("voice") for n in old}) != 1:
        return False
    saved = copy.deepcopy(list(measure))
    desired = _note_onsets(measure)
    voice = old[0].findtext("voice")
    replacements = copy.deepcopy(candidate.findall("note"))
    onset = previous = 0
    for n in replacements:
        duration = Fraction(int(n.findtext("duration"))*old_divisions, candidate_divisions)
        if duration.denominator != 1:
            return False
        n.find("duration").text = str(duration.numerator)
        n.find("voice").text = voice
        n.find("staff").text = "2"
        if n.find("chord") is None:
            previous = onset
            onset += duration.numerator
        desired[n] = previous
    location = list(measure).index(old[0])
    for n in old:
        measure.remove(n)
        desired.pop(n)
    for n in reversed(replacements):
        measure.insert(location, n)
    # All onsets are already captured. Old interleaved shifts may now sit
    # after the final retained note; rebuild them from the desired times.
    for e in list(measure):
        if e.tag in {"backup", "forward"}:
            measure.remove(e)
    if not _rebuild(measure, desired):
        measure[:] = saved
        return False
    return True


def retry_dense_bass(root, geometry, image, config, initial_meter=None):
    from homr.staff_parsing import add_image_into_tr_omr_canvas
    from homr.staff_parsing_tromr import predict_best
    from homr.music_xml_generator import generate_xml, XmlGeneratorArguments
    from homr.model import Staff, StaffPoint
    systems = measure_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return []
    import cv2
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = _ink(image)
    divisions, meter, key, attempts = 1, initial_meter, 0, 0
    reports = []
    for si, (measures, system) in enumerate(zip(systems, geometry["systems"]), 1):
        staff = system["staffs"][0] if len(system.get("staffs", [])) == 1 else None
        bounds = bar_bounds(staff, image) if staff else None
        jobs = []
        for i, m in enumerate(measures):
            for attributes in m.findall("attributes"):
                divisions = int(attributes.findtext("divisions", str(divisions)))
                time = attributes.find("time")
                if time is not None:
                    meter = (int(time.findtext("beats")), int(time.findtext("beat-type")))
                key = int(attributes.findtext("key/fifths", str(key)))
            if bounds is None or len(bounds)-1 != len(measures) or meter is None or key != 0:
                continue
            target = Fraction(meter[0]*4, meter[1])
            old = [n for n in m.findall("note") if n.findtext("staff", "1") == "2"]
            if (not old or m.get("implicit") == "yes"
                    or len({n.findtext("voice") for n in old}) != 1
                    or any(n.find("pitch") is None or n.find("grace") is not None
                           or n.find("tie") is not None or n.find("accidental") is not None
                           or int(n.findtext("pitch/alter", "0")) for n in old)):
                continue
            groups = head_groups(staff, bounds[i], bounds[i+1])
            total = sum(Fraction(int(n.findtext("duration", "0")), divisions)
                        for n in old if n.find("chord") is None)
            if (total <= target or abs(len(old)-sum(map(len, groups))) < 2
                    or not 8 <= len(groups) <= 64 or any(len(g) != 2 for g in groups)):
                continue
            jobs.append((i, m, groups, divisions, target))
        if not jobs or attempts >= 4:
            continue
        grid = staff["grid"]
        space = median(b-a for p in grid for a, b in zip(p["y"][5:9], p["y"][6:10]))
        top = max(round((max(p["y"][4] for p in grid)+min(p["y"][5] for p in grid))/2),
                  round(min(p["y"][5] for p in grid)-10*space), 0)
        bottom = min(round(max(p["y"][9] for p in grid)+4*space), image.shape[0])
        left = max(0, round(staff["x_range"][0]-2*space))
        right = min(image.shape[1], round(staff["x_range"][1]+space))
        canvas = add_image_into_tr_omr_canvas(image[top:bottom, left:right])
        dummy = Staff([StaffPoint(p["x"], p["y"][5:], 0) for p in grid])
        # predict_best otherwise drops tokens labelled lower before we can
        # normalize them. Keep every detected token from this physical crop.
        dummy.is_grandstaff = True
        # A single physical staff must not keep hallucinated upper/lower labels.
        tokens = [s.to_upper_position() for s in predict_best(canvas, dummy, config)]
        attempts += 1
        generated = generate_xml(XmlGeneratorArguments(), [tokens], "staff retry")
        candidate = ET.fromstring(generated.to_string())
        cm = candidate.findall("part/measure")
        cd = int(candidate.findtext("part/measure/attributes/divisions", "1"))
        clefs = candidate.findall("part/measure/attributes/clef")
        if len(cm) != len(measures) or not clefs or any(c.findtext("sign") != "F" or c.findtext("line") != "4" for c in clefs):
            reports.append({"system": si, "applied": False, "reason": "retry_bar_or_clef_disagrees"})
            continue
        for i, m, groups, old_div, target in jobs:
            c = cm[i]
            for n in c.findall("note"):
                n.find("staff").text = "2"
            fixes = verify_candidate(c, groups, ink, cd, target)
            applied = fixes is not None and replace_bass(m, c, cd, old_div)
            reports.append({"system": si, "measure": m.get("number"), "applied": applied,
                            "method": "single_staff_verified_sixteenth_octaves",
                            "candidate_notes": len(c.findall("note")),
                            "pitch_fixes": fixes if applied else [],
                            "reason": "image_and_complete_voice_agree" if applied else "retry_image_check_failed"})
    return reports
