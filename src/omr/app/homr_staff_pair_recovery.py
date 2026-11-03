"""Certify a cropped grand staff from geometry and the original raster.

This detects an optional local retry; it does not modify a score or introduce
expected bar counts. Two adjacent five-line systems must fit a grand-staff
layout already observed on the page, have a raster connector at their common
left edge, and admit five-line-certified continuation to common endpoints.
Independent single-staff pages and ambiguous three-staff groups abstain.
"""
from __future__ import annotations

import numpy as np

from homr_staff_bidirectional import extend_grid


def _sole(system):
    staffs = system.get('staffs', [])
    return staffs[0] if len(staffs) == 1 else None


def _grid(staff, count):
    if staff is None:
        return None
    grid = staff.get('grid', [])
    if not 12 <= len(grid) <= 1024:
        return None
    try:
        xs = np.asarray([p['x'] for p in grid], float)
        ys = np.asarray([p['y'] for p in grid], float)
    except (KeyError, ValueError, TypeError):
        return None
    if ys.shape != (len(grid), count) or not np.isfinite(xs).all() or not np.isfinite(ys).all() or np.any(np.diff(xs) < 0) or len(np.unique(xs)) < 12:
        return None
    gaps = np.concatenate([np.diff(ys[:, i:i+5], axis=1).ravel() for i in range(0, count, 5)])
    space = float(np.median(gaps))
    if not 5 <= space <= 40 or np.quantile(abs(gaps-space), .95) > .35*space or np.max(abs(gaps-space)) > .6*space:
        return None
    return xs, ys, space


def _connector(image, upper, lower, left, space):
    # A thin vertical line beside the initial clefs is a stronger physical
    # connection than matching gaps alone. Never use a closing operation that
    # might manufacture a connection between separate systems.
    xmin = max(1, int(round(left-space*.3)))
    xmax = min(image.shape[1]-2, int(round(left+space*.8)))
    top = int(round(upper[0]))
    bottom = int(round(lower[-1]))
    if top < 0 or bottom >= image.shape[0] or not 8*space <= bottom-top <= 24*space:
        return None
    radius = max(1, round(.08*space))
    block = max(4, round(.5*space))
    accepted = []
    for x in range(xmin, xmax+1):
        hits = (image[top:bottom+1, max(0,x-radius):x+radius+1] < 150).any(axis=1)
        support = float(hits.mean())
        minimum = min(float(hits[k:k+block].mean()) for k in range(0, len(hits), block))
        if support >= .95 and minimum >= .8:
            accepted.append((support, minimum, x))
    if not accepted:
        return None
    support, minimum, x = max(accepted)
    return dict(x=x, support=support, minimum_block_support=minimum)


def find_pair_candidates(geometry, image):
    """Return bounded image-certified local pair candidates without mutation."""
    if image.ndim != 2 or image.size > 16000000 or max(image.shape) > 8192:
        return []
    if geometry.get('preprocessed_size') != [image.shape[1], image.shape[0]]:
        return []
    systems = geometry.get('systems', [])
    if not 3 <= len(systems) <= 64:
        return []
    templates = []
    for system in systems:
        parsed = _grid(_sole(system), 10)
        if parsed is not None:
            xs, ys, space = parsed
            templates.append((space, float(np.median(ys[:, 5]-ys[:, 4])) / space))
    if len(templates) < 2:
        return []
    expected_space, expected_gap = np.median(templates, axis=0)
    possible = []
    for index in range(len(systems)-1):
        upper = _sole(systems[index]); lower = _sole(systems[index+1])
        a = _grid(upper, 5); b = _grid(lower, 5)
        if a is None or b is None:
            continue
        ax, ay, asp = a; bx, by, bsp = b
        space = (asp+bsp)/2
        if abs(asp-bsp) > .12*space or abs(space-expected_space) > .2*expected_space:
            continue
        gap = float(np.median(by[:, 0])-np.median(ay[:, -1]))/space
        if not 4 <= gap <= 14 or abs(gap-expected_gap) > max(1.5, .2*expected_gap):
            continue
        left = min(ax[0], bx[0]); right = max(ax[-1], bx[-1])
        if min(ax[-1], bx[-1])-max(ax[0], bx[0]) < 12*space or right-left > 160*space:
            continue
        grids = []
        changes = []
        for staff in (upper, lower):
            result, records = extend_grid(staff['grid'], [left], [right], image)
            grids.append(result); changes.append(records)
        if not changes[0] and not changes[1]:
            continue
        if any(abs(g[0]['x']-left) > space or abs(g[-1]['x']-right) > space for g in grids):
            continue
        connector = _connector(image, grids[0][0]['y'], grids[1][0]['y'], left, space)
        if connector is None:
            continue
        possible.append(dict(old_system_indices=[index,index+1], staff_space=space,
                             normalized_inner_gap=gap, template_inner_gap=float(expected_gap),
                             common_x_range=[float(left),float(right)],
                             extensions=changes, connector=connector,
                             method='adjacent_five_line_grids_with_raster_connector_and_page_layout'))
    # Three consecutive five-line systems can create competing adjacent pairs;
    # no greedy choice is justified without stronger instrument evidence.
    counts = {}
    for candidate in possible:
        for i in candidate['old_system_indices']:
            counts[i] = counts.get(i,0)+1
    return [c for c in possible if all(counts[i] == 1 for i in c['old_system_indices'])]


def certify_pair_mapping(base_geometry, candidate_geometry, image):
    """Map a second view only when every unpaired system is byte-identical.

    The result gives system indices, not MusicXML measure indices; callers must
    preserve the original measures in all other systems and independently
    validate the notes exported by the recovered system.
    """
    pairs = find_pair_candidates(base_geometry, image)
    if not pairs:
        return []
    old = base_geometry['systems']; new = candidate_geometry.get('systems', [])
    if candidate_geometry.get('preprocessed_size') != base_geometry.get('preprocessed_size') or len(new) != len(old)-len(pairs):
        return []
    by_index = {c['old_system_indices'][0]: c for c in pairs}
    mappings = []
    i = j = 0
    while i < len(old):
        if i not in by_index:
            if old[i] != new[j]:
                return []
            i += 1; j += 1
            continue
        c = by_index[i]; merged = _sole(new[j]); parsed = _grid(merged,10)
        if parsed is None:
            return []
        xs, ys, space = parsed
        left,right = c['common_x_range']
        if abs(xs[0]-left) > space or abs(xs[-1]-right) > space:
            return []
        # Compare original overlap line-by-line; do not accept a nearby but
        # different staff when the retry reordered several systems.
        for half,original in enumerate((old[i],old[i+1])):
            staff = _sole(original)
            for point in staff['grid']:
                at = int(np.argmin(abs(xs-point['x'])))
                if abs(xs[at]-point['x']) > space or max(abs(ys[at,half*5:half*5+5]-point['y'])) > .6*space:
                    return []
        mappings.append(dict(c, candidate_system_index=j))
        i += 2; j += 1
    return mappings
