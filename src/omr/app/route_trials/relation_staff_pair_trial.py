"""Fuse a raster-certified staff pair while retaining every other measure.

The first supported case is an independently decoded upper staff plus an
empty lower staff. The retry supplies only their common physical bars, with
all pitch heads verified; a retry never replaces unrelated page content.
"""
import copy
from collections import Counter
import xml.etree.ElementTree as E
import cv2
import numpy as np

from homr_staff_pair_recovery import certify_pair_mapping
from omr_bar_geometry import measure_systems, bar_bounds
from omr_event_records import records
from repair_augmentation_dots import _head_map
from align_homr_geometry import _position
from .relation_tail_splice_experiment import contexts
from .relation_coverage import core, hollow_core
from .relation_hollow_evidence import enclosed_hollow


def _subsequence(needle, haystack):
    index = 0
    for value in haystack:
        if index < len(needle) and value == needle[index]:
            index += 1
    return index == len(needle)


def _whole_rest_body(staff, image, lo, hi, space):
    """Prove one compact filled rectangle hanging below lower line two.

    The raster is sampled below the line, without removing or drawing pixels.
    Its centre must lie in the middle portion of the physical bar. A half-rest
    above line three, a long beam, and curved noteheads do not satisfy the
    vertical position, aspect and rectangular fill requirements.
    """
    if not 8*space <= hi-lo <= 160*space:
        return None
    left = max(0,int(round(lo+.25*(hi-lo))))
    right = min(image.shape[1],int(round(lo+.75*(hi-lo))))
    if right-left < 2*space:
        return None
    grid = staff['grid']; xs = np.asarray([p['x'] for p in grid],float)
    line = np.asarray([p['y'][6] for p in grid],float)
    xx = np.arange(left,right)
    y = np.rint(np.interp(xx,xs,line)).astype(int)
    offsets = np.arange(max(2,round(.14*space)), max(3,round(.75*space))+1)
    if len(offsets) < 4 or y.min()+offsets.min() < 0 or y.max()+offsets.max() >= image.shape[0]:
        return None
    black = (image[y[None,:]+offsets[:,None],xx[None,:]] < 150).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(black,8)
    proven = []
    for x,top,w,h,area in stats[1:]:
        bottom = offsets[int(top+h-1)]
        if x == 0 or x+w == black.shape[1] or top == 0 and offsets[0] > .2*space or top+h == black.shape[0]:
            continue
        if .65*space <= w <= 2*space and .15*space <= h <= .5*space and area/(w*h) >= .85 and bottom <= .6*space:
            proven.append(dict(x=float(left+x+w/2),y=float(np.interp(left+x+w/2,xs,line)+(offsets[top]+bottom)/2),
                               width=int(w),height=int(h),rectangular_fill=float(area/(w*h)),
                               method='whole_rest_rectangle_below_lower_second_line'))
    return proven[0] if len(proven) == 1 else None


def fuse(base, candidate, base_geometry, candidate_geometry, image):
    if len(base.findall('part')) != 1 or len(candidate.findall('part')) != 1:
        return []
    mapping = certify_pair_mapping(base_geometry, candidate_geometry, image)
    if not mapping:
        return []
    old_systems = measure_systems(base); new_systems = measure_systems(candidate)
    if len(old_systems) != len(base_geometry['systems']) or len(new_systems) != len(candidate_geometry['systems']):
        return []
    old_records = records(base); new_records = records(candidate)
    if len(old_records) > 1024 or len(new_records) > 1024:
        return []
    old_start = []; new_start = []; at = 0
    for group in old_systems:
        old_start.append(at); at += len(group)
    at = 0
    for group in new_systems:
        new_start.append(at); at += len(group)
    old_context = contexts(base); new_context = contexts(candidate)
    heads = _head_map(candidate, candidate_geometry)
    jobs = []
    for item in mapping:
        oi,li = item['old_system_indices']; ni = item['candidate_system_index']
        upper = old_systems[oi]; lower = old_systems[li]; merged = new_systems[ni]
        if not 1 <= len(upper) == len(lower) == len(merged) <= 8:
            continue
        # The rest-only lower staff offers no contradictory pitched evidence.
        # Recovering two dense voices requires a separate joint decoder and is
        # outside this conservative transaction.
        og = base_geometry['systems'][li]['staffs'][0]
        if og.get('notes'):
            continue
        lower_rows = old_records[old_start[li]:old_start[li]+len(lower)]
        if any(len(rs) != 1 or rs[0]['midi'] is not None or rs[0]['grace'] or rs[0]['node'].findtext('type') != 'whole' for rs in lower_rows):
            continue
        before = old_start[oi]-1; after = new_start[ni]-1
        if before < 0 or after < 0 or old_context[before] != new_context[after]:
            continue
        ng = candidate_geometry['systems'][ni]['staffs'][0]
        bounds = bar_bounds(ng, image)
        if bounds is None or len(bounds) != len(merged)+1:
            continue
        space = item['staff_space']; accepted = True; note_count = 0; rest_proofs = []
        for number,m in enumerate(merged):
            rs = new_records[new_start[ni]+number]
            if not 2 <= len(rs) <= 128 or m.findall('attributes') or m.get('implicit') == 'yes':
                accepted = False; break
            if number and m.find('print') is not None:
                accepted = False; break
            pitched = [r for r in rs if r['midi'] is not None]
            rests = [r for r in rs if r['midi'] is None]
            if len(rests) != 1 or rests[0]['staff'] != 2 or rests[0]['grace'] or rests[0]['onset'] != 0 or rests[0]['node'].findtext('type') != 'whole':
                accepted = False; break
            if any(r['staff'] != 1 or r['grace'] or r['duration'] <= 0 or not 0 <= r['onset'] <= 16 or r['onset']+r['duration'] > 16 for r in pitched):
                accepted = False; break
            lo,hi = bounds[number:number+2]; hs = [heads.get(r['node']) for r in pitched]
            rest_proof = _whole_rest_body(ng,image,lo,hi,space)
            if rest_proof is None:
                accepted = False; break
            rest_proofs.append(rest_proof)
            if any(h is None or not lo+space/2 < h['x'] < hi-space/2 or _position(r['node'])[1] != h['position'] for r,h in zip(pitched,hs)):
                accepted = False; break
            observed = [h for h in ng['notes'] if lo+space/2 < h['x'] < hi-space/2]
            if Counter(id(h) for h in hs) != Counter(id(h) for h in observed):
                accepted = False; break
            for r,h in zip(pitched,hs):
                xy = dict(xy=[h['x'],h['y']])
                proof = (hollow_core(xy,image,h['staff_space']) or enclosed_hollow(xy,image,h['staff_space'])) if r['node'].findtext('type') in ('whole','half') else core(xy,image,h['staff_space'])
                if not proof:
                    accepted = False; break
            if not accepted:
                break
            previous = old_records[old_start[oi]+number]
            if any(r['grace'] or r['staff'] != 1 or r['midi'] is None for r in previous) or not _subsequence([r['midi'] for r in previous], [r['midi'] for r in pitched]):
                accepted = False; break
            note_count += len(pitched)
        if accepted:
            jobs.append((item, upper+lower, merged, note_count, rest_proofs))
    if not jobs:
        return []
    part = base.find('part'); original = {id(m):E.tostring(m) for m in part.findall('measure')}
    changes = []
    for item,deleted,inserted,note_count,rest_proofs in reversed(jobs):
        position = list(part).index(deleted[0])
        for m in deleted:
            part.remove(m)
        for offset,m in enumerate(inserted):
            part.insert(position+offset,copy.deepcopy(m))
        changes.append(dict(old_system_indices=item['old_system_indices'], candidate_system_index=item['candidate_system_index'],
                            old_measures=len(deleted),new_measures=len(inserted),verified_pitch_heads=note_count,
                            whole_rest_raster_evidence=rest_proofs,
                            unrelated_measure_bytes_preserved=True,
                            method='local_image_certified_grandstaff_replacement'))
    assert all(E.tostring(m) == original[id(m)] for m in part.findall('measure') if id(m) in original)
    return changes[::-1]
