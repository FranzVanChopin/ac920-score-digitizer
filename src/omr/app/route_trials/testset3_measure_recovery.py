"""Image-certified prefix-bar recovery from an alternate prediction.

The primary score is never completed from an expected count.  A clipped
grand-staff, ten continuously visible raster lines and full-height barlines
propose an uncertain prefix.  A separately predicted alternate may supply
events only when every new pitched event has an image head and all retained
clocks agree.  Existing note/event XML stays intact.  No model is imported.
"""
from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction
from statistics import median
import time
import xml.etree.ElementTree as E

import cv2
import numpy as np

from homr_staff_bidirectional import extend_grid
from omr_bar_geometry import measure_systems
from omr_event_records import records
from repair_augmentation_dots import _head_map

MAX_PIXELS = 16_000_000
MAX_DIMENSION = 8192
MAX_SYSTEMS = 64
MAX_GRID_POINTS = 1024
MAX_HEADS = 2048
MAX_BARS = 128
MAX_EVENTS = 128
MAX_NEW_BARS = 4
MAX_ROI_PIXELS = 1_000_000


def _image_valid(geometry, image):
    return (isinstance(image, np.ndarray) and image.ndim == 2 and image.dtype == np.uint8
            and 0 < image.size <= MAX_PIXELS and max(image.shape) <= MAX_DIMENSION
            and geometry.get('preprocessed_size') == [image.shape[1], image.shape[0]])


def _staff_valid(staff, image):
    grid = staff.get('grid', [])
    if not 2 <= len(grid) <= MAX_GRID_POINTS or len(staff.get('notes', [])) > MAX_HEADS:
        return False
    try:
        if any(len(point['y']) != 10 for point in grid):
            return False
        xs = [float(point['x']) for point in grid]
        if any(b <= a for a, b in zip(xs, xs[1:])):
            return False
        values = [value for point in grid for value in (point['x'], *point['y'])]
        values += list(staff['x_range']) + list(staff['y_range'])
        if not all(np.isfinite(value) for value in values):
            return False
        if any(not 0 <= point['x'] < image.shape[1]
               or any(not 0 <= value < image.shape[0] for value in point['y']) for point in grid):
            return False
        return all(5 <= b-a <= 40 for point in grid
                   for offset in (0, 5) for a, b in zip(point['y'][offset:offset+4], point['y'][offset+1:offset+5]))
    except (KeyError, TypeError, ValueError):
        return False


def _space(staff):
    return median(b-a for point in staff['grid'] for offset in (0, 5)
                  for a, b in zip(point['y'][offset:offset+4], point['y'][offset+1:offset+5]))


def _line_ys(staff, x):
    xs = [point['x'] for point in staff['grid']]
    return [float(np.interp(x, xs, [point['y'][index] for point in staff['grid']])) for index in range(10)]


def _line_support(staff, image, start, end):
    """Actual new samples, not endpoint extrapolation or staff-count evidence."""
    space = _space(staff)
    xs = np.arange(max(0, round(start)), min(image.shape[1], round(end)+1), dtype=np.int32)
    if len(xs) < 4*space:
        return None
    minimum = 1.
    for index in range(10):
        ys = np.rint(np.interp(xs, [point['x'] for point in staff['grid']],
                               [point['y'][index] for point in staff['grid']])).astype(np.int32)
        radius = max(1, round(.12*space))
        dark = np.zeros(len(xs), dtype=bool)
        for dy in range(-radius, radius+1):
            yy = np.clip(ys+dy, 0, image.shape[0]-1)
            dark |= image[yy, xs] < 140
        # Require every local section; a long ledger run cannot bridge a gap.
        step = max(4, round(3*space))
        scores = [float(dark[left:left+step].mean()) for left in range(0, len(xs), step)
                  if len(dark[left:left+step]) >= max(4, step//2)]
        minimum = min(minimum, min(scores, default=0.))
    return minimum


def _bar_bounds(staff, image):
    """Full grand-staff crossings, traced in the supplied curved grid frame."""
    space = _space(staff)
    left, right = staff['x_range']
    xs = np.arange(max(0, round(left-space)), min(image.shape[1], round(right+space)+1), dtype=np.int32)
    if len(xs) > MAX_DIMENSION:
        return None
    gx = [point['x'] for point in staff['grid']]
    top = np.interp(xs, gx, [point['y'][0] for point in staff['grid']])
    bottom = np.interp(xs, gx, [point['y'][-1] for point in staff['grid']])
    if np.any(bottom-top < 8*space):
        return None
    yy = np.rint(top[:, None]+(bottom-top)[:, None]*np.linspace(.015, .985, 96)).astype(np.int32)
    if yy.min() < 0 or yy.max() >= image.shape[0]:
        return None
    coverage = np.zeros((len(xs), 96), dtype=bool)
    for dx in (-1, 0, 1):
        xx = np.clip(xs+dx, 0, image.shape[1]-1)
        coverage |= image[yy, xx[:, None]] < 140
    selected = xs[coverage.mean(axis=1) >= .96]
    groups = []
    for value in selected:
        if not groups or value-groups[-1][-1] > max(3, .65*space):
            groups.append([])
        groups[-1].append(int(value))
    if not 2 <= len(groups) <= MAX_BARS+1 or any(group[-1]-group[0] > space for group in groups):
        return None
    bounds = [sum(group)/len(group) for group in groups]
    if (abs(bounds[0]-left) > 1.5*space or abs(bounds[-1]-right) > 1.5*space
            or any(b-a < 3*space for a, b in zip(bounds, bounds[1:]))):
        return None
    return bounds


def _roi_tiles(staff, bounds, missing, image):
    space = _space(staff)
    left = max(0, round(bounds[0]-space))
    right = min(image.shape[1], round(bounds[missing]+space))
    top = max(0, round(min(point['y'][0] for point in staff['grid'])-6*space))
    bottom = min(image.shape[0], round(max(point['y'][-1] for point in staff['grid'])+6*space))
    width = max(1, MAX_ROI_PIXELS//max(1, bottom-top))
    result = []
    while left < right:
        stop = min(right, left+width)
        result.append([left, top, stop, bottom])
        left = stop
    return result


def prepare(root, geometry, image):
    """Return input-only positive evidence and bounded uncertain prefix ROIs.

    ``image`` must be the recognizer's saved preprocessing raster, with exactly
    the geometry's frame.  This function changes neither argument.
    """
    start = time.perf_counter()
    report = dict(enabled=True, method='ten_lines_and_grandstaff_crossings_before_local_prediction',
                  reference_supplied=False, additional_model_calls=0, changes=[], candidates=[],
                  coordinate_frame='homr_preprocessed_pixels', maximum_native_pixels=MAX_PIXELS,
                  maximum_roi_pixels=MAX_ROI_PIXELS, maximum_new_bars=MAX_NEW_BARS)
    systems = measure_systems(root)
    detected = geometry.get('systems', [])
    if not _image_valid(geometry, image):
        report['abstained'] = 'preprocessed_frame_or_input_budget_mismatch'
    elif not 2 <= len(systems) == len(detected) <= MAX_SYSTEMS:
        report['abstained'] = 'ambiguous_part_or_system_inventory'
    else:
        starts = {float(system['staffs'][0]['x_range'][0]) for system in detected
                  if len(system.get('staffs', [])) == 1 and _staff_valid(system['staffs'][0], image)}
        for index, (measures, system) in enumerate(zip(systems, detected)):
            if len(system.get('staffs', [])) != 1 or not 1 <= len(measures) <= MAX_BARS:
                continue
            staff = system['staffs'][0]
            if not _staff_valid(staff, image):
                continue
            space = _space(staff)
            extended, continuations = extend_grid(staff['grid'], starts, (), image)
            left = [item for item in continuations if item['direction'] == 'left'
                    and item['from_x']-item['to_x'] >= 4*space]
            if not left:
                continue
            proposed = copy.deepcopy(staff)
            proposed['grid'] = extended
            proposed['x_range'] = [extended[0]['x'], extended[-1]['x']]
            proposed['y_range'] = [min(point['y'][0] for point in extended), max(point['y'][-1] for point in extended)]
            new_points = [point for point in extended if point['x'] < staff['x_range'][0]]
            # The retained detector endpoint may be noisy.  Certify the new
            # raster-fitted points themselves, rather than interpolating their
            # final few pixels toward that unchanged, noisy original endpoint.
            support_end = new_points[-1]['x'] if new_points else proposed['x_range'][0]
            support = _line_support(proposed, image, proposed['x_range'][0], support_end)
            bounds = _bar_bounds(proposed, image)
            if support is None or support < .95 or bounds is None:
                continue
            missing = len(bounds)-1-len(measures)
            if not 1 <= missing <= MAX_NEW_BARS:
                continue
            # The old start can lie inside a missing first bar.  Requiring its
            # first boundary to precede that start incorrectly loses this case.
            if not bounds[0]+4*space < staff['x_range'][0] < bounds[missing+1]-.5*space:
                continue
            report['candidates'].append(dict(system=index+1, decoded_measures=len(measures),
                visible_intervals=len(bounds)-1, missing_prefix_intervals=missing,
                old_x_range=list(staff['x_range']), proposed_x_range=proposed['x_range'],
                bar_x=bounds, staff_space=space, minimum_local_line_support=support,
                continuation=left, certified_new_line_sample_range=[proposed['x_range'][0], support_end],
                uncertain_symbol_rois=_roi_tiles(proposed, bounds, missing, image),
                proposed_grid=extended, proposed_y_range=proposed['y_range'],
                requires_alternate_prediction=True, no_empty_measures_inserted=True))
    report['elapsed_s'] = time.perf_counter()-start
    return report


def _clocks(rows):
    return Counter((row['staff'], row['voice'], row['onset'], row['duration'], row['chord'], row['grace']) for row in rows)


def _meters(root):
    value = None; result = []
    for measure in root.findall('part/measure'):
        for attributes in measure.findall('attributes'):
            for printed in attributes.findall('time'):
                beats, units = printed.findall('beats'), printed.findall('beat-type')
                if len(beats) != 1 or len(units) != 1:
                    value = None; continue
                pieces = (beats[0].text or '').split('+')
                text = (units[0].text or '').strip()
                if not all(piece.strip().isdigit() for piece in pieces) or not text.isdigit():
                    value = None; continue
                count, unit = sum(int(piece) for piece in pieces), int(text)
                value = Fraction(4*count, unit) if 1 <= count <= 16 and unit in (1,2,4,8,16,32) else None
        result.append(value)
    return result


def _complete_lanes(rows, meter):
    if meter is None:
        return False
    lanes = {}
    for row in rows:
        if row['grace'] or row['staff'] not in (1, 2):
            return False
        if not row['chord']:
            lanes.setdefault((row['staff'], row['voice']), []).append((row['onset'], row['duration']))
    if not lanes or len(lanes) > 4:
        return False
    for events in lanes.values():
        end = Fraction(0)
        for onset, duration in sorted(events):
            if onset != end or duration <= 0:
                return False
            end = onset+duration
        if end != meter:
            return False
    return True


def _head_ink(head, image, hollow=False):
    space = float(head['staff_space']); x, y = float(head['x']), float(head['y'])
    left, right = round(x-.65*space), round(x+.65*space)+1
    top, bottom = round(y-.48*space), round(y+.48*space)+1
    if min(left, top) < 0 or right > image.shape[1] or bottom > image.shape[0]:
        return False
    yy, xx = np.mgrid[top:bottom, left:right]
    radius = ((xx-x)/(.6*space))**2+((yy-y)/(.42*space))**2
    ink = image[top:bottom, left:right] < 140
    if hollow:
        core, ring = radius < .22, (radius > .52) & (radius < 1.15)
        return bool(core.any() and ring.any() and ink[core].mean() < .65 and ink[ring].mean() >= .5)
    mask = radius < .72
    return bool(mask.any() and ink[mask].mean() >= .82)


def _quarter_rest(row, rows, mapped, staff, image, lo, hi):
    """A compact unsupported zigzag at a peer-staff witnessed onset.

    No semantic certainty is assigned to arbitrary residual components.
    Other rest kinds and absent peer columns abstain for local recognition.
    """
    if row['node'].findtext('type') != 'quarter' or row['duration'] != 1:
        return False
    peers = [mapped[item['node']] for item in rows if item['midi'] is not None
             and item['onset'] == row['onset'] and item['node'] in mapped]
    if not peers:
        return False
    space = _space(staff); x = median(head['x'] for head in peers)
    if max(head['x'] for head in peers)-min(head['x'] for head in peers) > .45*space:
        return False
    lines = _line_ys(staff, x)[(row['staff']-1)*5:row['staff']*5]
    if len(lines) != 5:
        return False
    left, right = max(round(lo+.5*space), round(x-.65*space)), min(round(hi-.5*space), round(x+.65*space))
    top, bottom = round(lines[0]-.3*space), round(lines[-1]+.3*space)
    if min(left, top) < 0 or right >= image.shape[1] or bottom >= image.shape[0] or right <= left:
        return False
    crop = (image[top:bottom+1, left:right+1] < 140).astype(np.uint8)
    for line in lines:
        cy = round(line)-top
        crop[max(0, cy-1):min(crop.shape[0], cy+2)] = 0
    # Three proven staff-line rows were removed above.  A five-row closing
    # bridges that exact damage without joining glyphs in horizontal space.
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(crop)
    if count > 32:
        return False
    candidates = []
    for index in range(1, count):
        bx, by, width, height, area = stats[index]
        if not (.3*space <= width <= 1.2*space and 1.3*space <= height <= 3.8*space
                and .35*space*space <= area <= 2.8*space*space):
            continue
        # The quarter-rest bends in both horizontal directions.  A narrow
        # upright, natural pair or rectangular beam is insufficient evidence.
        centers = []
        edges = np.linspace(by, by+height, 9).astype(int)
        for a, b in zip(edges[:-1], edges[1:]):
            yy, xx = np.nonzero(labels[a:b] == index)
            if len(xx):
                centers.append(float(np.mean(xx)))
        deltas = [b-a for a,b in zip(centers,centers[1:]) if abs(b-a) >= .12*space]
        if (len(centers) == 8 and max(centers)-min(centers) >= .25*space
                and any(a*b < 0 for a,b in zip(deltas,deltas[1:]))):
            candidates.append(index)
    return len(candidates) == 1


def _division_before(root, stop):
    divisions = 1
    for index, measure in enumerate(root.findall('part/measure')):
        if index > stop:
            break
        for attributes in measure.findall('attributes'):
            divisions = int(attributes.findtext('divisions', str(divisions)))
    return divisions


def repair(root, geometry, image, *, alternate_root=None, alternate_geometry=None):
    """Return diagnostics; transactionally insert certified alternate prefixes.

    Without an alternate, return conventional evidence/ROIs without mutation.
    With an alternate, only new prefix bars and the affected geometry change;
    existing notes, backups, forwards, attributes and clocks are preserved.
    """
    report = prepare(root, geometry, image)
    if alternate_root is None or alternate_geometry is None or not report['candidates']:
        return report
    start = time.perf_counter()
    systems, others = measure_systems(root), measure_systems(alternate_root)
    if (not _image_valid(alternate_geometry, image) or len(others) != len(systems)
            or len(alternate_geometry.get('systems', [])) != len(systems)):
        report['fusion_abstained'] = 'alternate_frame_or_system_inventory_mismatch'
        return report
    original_rows, alternate_rows = records(root), records(alternate_root)
    primary_meters, alternate_meters = _meters(root), _meters(alternate_root)
    mapped = _head_map(alternate_root, alternate_geometry)
    original_offsets = [sum(map(len, systems[:index])) for index in range(len(systems))]
    alternate_offsets = [sum(map(len, others[:index])) for index in range(len(others))]
    jobs = []
    for proof in report['candidates']:
        index = proof['system']-1; missing = proof['missing_prefix_intervals']; bounds = proof['bar_x']
        old, candidate = systems[index], others[index]
        base_offset, candidate_offset = original_offsets[index], alternate_offsets[index]
        outcome = dict(system=index+1, accepted=False)
        report.setdefault('fusion_checks', []).append(outcome)
        if len(candidate) != len(bounds)-1 or len(candidate)-len(old) != missing:
            outcome['abstained'] = 'alternate_does_not_match_visible_intervals'; continue
        # Repeated rhythms alone cannot supply notes.  Every retained clock
        # must nevertheless agree before any alternate prefix may be used.
        if any(_clocks(original_rows[base_offset+j]) != _clocks(alternate_rows[candidate_offset+missing+j]) for j in range(len(old))):
            outcome['abstained'] = 'retained_clocks_disagree'; continue
        other_system = alternate_geometry['systems'][index]
        if len(other_system.get('staffs', [])) != 1:
            outcome['abstained'] = 'alternate_staff_inventory_ambiguous'; continue
        staff = other_system['staffs'][0]
        if not _staff_valid(staff, image):
            outcome['abstained'] = 'alternate_staff_geometry_invalid'; continue
        alternate_bounds = _bar_bounds(staff, image); space = proof['staff_space']
        if (alternate_bounds is None or len(alternate_bounds) != len(bounds)
                or max(abs(a-b) for a,b in zip(alternate_bounds,bounds)) > .5*space):
            outcome['abstained'] = 'alternate_physical_boundaries_disagree'; continue
        valid = True; supported = 0; rest_supported = 0
        for bar_index in range(missing):
            rows = alternate_rows[candidate_offset+bar_index]; lo, hi = bounds[bar_index:bar_index+2]
            if not 2 <= len(rows) <= MAX_EVENTS or any(row['grace'] or row['duration'] <= 0 or row['onset'] < 0 or row['onset']+row['duration'] > 16 for row in rows):
                valid = False; outcome['abstained'] = 'new_events_over_budget_or_unsupported_grace'; break
            if (alternate_meters[candidate_offset+bar_index] != primary_meters[base_offset]
                    or not _complete_lanes(rows, alternate_meters[candidate_offset+bar_index])):
                valid = False; outcome['abstained'] = 'new_lanes_do_not_fill_predicted_meter'; break
            for row in rows:
                if row['midi'] is None:
                    if row['node'].find('rest') is None or not _quarter_rest(row, rows, mapped, staff, image, lo, hi):
                        valid = False; outcome['abstained'] = 'new_rest_requires_local_symbol_confirmation'; break
                    rest_supported += 1; continue
                head = mapped.get(row['node'])
                if head is None or not lo+.45*space < head['x'] < hi-.45*space:
                    valid = False; outcome['abstained'] = 'new_pitch_not_uniquely_mapped_inside_bar'; break
                hollow = row['node'].findtext('type') in ('half','whole')
                if not _head_ink(head, image, hollow):
                    valid = False; outcome['abstained'] = 'new_head_has_no_raster_shape_support'; break
                supported += 1
            if not valid:
                break
        outcome.update(image_supported_pitched_notes=supported, image_supported_quarter_rests=rest_supported)
        if not valid:
            continue
        division = _division_before(root, max(0, base_offset-1))
        if not 1 <= division <= 9600:
            outcome['abstained'] = 'primary_divisions_out_of_budget'; continue
        inserted = []
        for local_index, measure in enumerate(candidate[:missing]):
            copied = copy.deepcopy(measure)
            candidate_division = _division_before(alternate_root, candidate_offset+local_index)
            if not 1 <= candidate_division <= 9600:
                valid = False; break
            for duration in copied.findall('.//duration'):
                number = Fraction(int(duration.text)*division, candidate_division)
                if number.denominator != 1:
                    valid = False; break
                duration.text = str(number.numerator)
            for attributes in copied.findall('attributes'):
                for value in attributes.findall('divisions'):
                    value.text = str(division)
            for layout in copied.findall('print'):
                copied.remove(layout)
            inserted.append(copied)
        if not valid:
            outcome['abstained'] = 'new_durations_not_integral_in_primary_divisions'; continue
        outcome['accepted'] = True
        jobs.append((index, old[0], inserted, staff, proof, outcome))
    if jobs:
        part = root.find('part')
        protected = {id(measure): E.tostring(measure) for measure in part.findall('measure')}
        protected_clocks = {id(measure): _clocks(rows) for measure, rows in zip(part.findall('measure'), records(root))}
        for index, anchor, inserted, staff, proof, outcome in jobs:
            layouts = list(anchor.findall('print'))
            for layout in layouts:
                anchor.remove(layout)
            for layout in layouts:
                inserted[0].insert(0, layout)
            position = list(part).index(anchor)
            for measure in inserted:
                part.insert(position, measure); position += 1
            geometry['systems'][index] = copy.deepcopy(alternate_geometry['systems'][index])
            report['changes'].append(dict(method='raster_certified_prefix_from_alternate_prediction',
                system=index+1, inserted_measures=len(inserted), inserted_notes=sum(len(measure.findall('note')) for measure in inserted),
                bar_x=proof['bar_x'], existing_event_xml_preserved=True, existing_clocks_preserved=True,
                additional_local_model_required=True))
        for measure, rows in zip(part.findall('measure'), records(root)):
            if id(measure) in protected_clocks:
                assert _clocks(rows) == protected_clocks[id(measure)], 'Inserted context changed an old event clock'
        # Layout markers and measure numbers are the only old XML edits.
        for number, measure in enumerate(part.findall('measure'), 1):
            if id(measure) in protected:
                before = E.fromstring(protected[id(measure)])
                after = copy.deepcopy(measure)
                for node in (before, after):
                    node.attrib.pop('number', None)
                    for layout in node.findall('print'):
                        node.remove(layout)
                assert E.tostring(before) == E.tostring(after), 'An existing event or attribute subtree changed'
            measure.set('number', str(number))
    report['fusion_elapsed_s'] = time.perf_counter()-start
    return report


def fuse(base, candidate, base_geometry, candidate_geometry, image):
    """Compatibility with the existing sequential structure-retry callback."""
    # The caller owns the transaction which commits affected geometry only
    # after every score/sidecar check.  Its live original must not change here.
    trial_geometry = copy.deepcopy(base_geometry)
    return repair(base, trial_geometry, image, alternate_root=candidate,
                  alternate_geometry=candidate_geometry)['changes']
