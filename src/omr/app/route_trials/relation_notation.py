"""Image-certified upper ottava correction, with bounded row continuation.

Only the photograph, predicted five-line tracks and output events are used.
Pitch edits retain the number, onset, duration and voice of every event.
"""
import hashlib
import json
from statistics import median
import xml.etree.ElementTree as E
import cv2
import numpy as np
from align_homr_geometry import _position, _xml_systems

FACTOR = 16 / 19.8


def spans(geometry, image):
    """An 8 with a thin dashed bracket must terminate or reach the next row.

    A narrow band follows small camera skew. An open right edge is accepted
    only through consecutive, left-prefix 8 brackets ending in a real hook.
    Unlabelled or unfinished continuations are deliberately left uncertain.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = (image < 140).astype('uint8')
    cs, hierarchy = cv2.findContours(ink, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(geometry.get('systems', [])) > 64:
        return []
    candidates = []
    for i, c in enumerate(cs):
        if hierarchy[0, i, 3] != -1:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not 5 <= w <= 64 or not 10 <= h <= 80:
            continue
        child, holes = hierarchy[0, i, 2], []
        while child >= 0:
            area = cv2.contourArea(cs[child])
            if area >= 2:
                holes.append((area, cv2.boundingRect(cs[child])[3]))
            child = hierarchy[0, child, 0]
        if (len(holes) != 2 or min(a for a, _ in holes) / max(a for a, _ in holes) < .4
                or min(b for _, b in holes) / max(b for _, b in holes) < .75):
            continue
        eligible = []
        for si, sys in enumerate(geometry['systems']):
            if len(sys.get('staffs', [])) != 1:
                continue
            st = sys['staffs'][0]
            values = [n['staff_space'] for n in st['notes'] if n.get('staff_space')]
            if not values:
                continue
            space = median(values)
            if (st['x_range'][0] <= x <= st['x_range'][1]
                    and .6 * space < st['y_range'][0] - y - h < 7 * space
                    and .6 * space <= w <= 1.6 * space and space <= h <= 2 * space):
                eligible.append((si, st, space))
        if len(eligible) != 1:
            continue
        si, st, space = eligible[0]
        start, stop = x + w + 1, min(image.shape[1], round(st['x_range'][1] + space))
        top, bottom = max(0, round(y - .6 * space)), min(image.shape[0], round(y + .6 * space))
        band = ink[top:bottom, start:stop]
        if not band.size or stop - start > 4096:
            continue
        xs = np.flatnonzero(band.max(axis=0))
        if not len(xs) or xs[0] > space:
            continue
        end = int(xs[0])
        for v in xs[1:]:
            if v - end > max(3, round(.6 * space)) + 1:
                break
            end = int(v)
        if not 2 * space < end <= 160 * space:
            continue
        line = band[:, :end + 1]
        coverage = float(line.max(axis=0).mean())
        if not .3 <= coverage <= .8:
            continue
        # Every dash is thin; fit a bounded camera slope to the line centers.
        count, _, stats, centers = cv2.connectedComponentsWithStats(line)
        dashes = [(centers[j][0], centers[j][1]) for j in range(1, count)
                  if .1 * space <= stats[j, cv2.CC_STAT_WIDTH] <= 1.2 * space
                  and stats[j, cv2.CC_STAT_HEIGHT] <= .35 * space]
        if len(dashes) < 4:
            continue
        points = np.array(dashes)
        slope, intercept = np.polyfit(*points.T, 1)
        for _ in range(3):
            good = np.abs(points[:, 1] - (slope * points[:, 0] + intercept)) <= 2
            if good.sum() < 4:
                break
            slope, intercept = np.polyfit(*points[good].T, 1)
        good = np.abs(points[:, 1] - (slope * points[:, 0] + intercept)) <= 2
        if abs(slope) > .025 or good.sum() < max(4, .85 * len(dashes)):
            continue
        ex, ey = start + end, round(top + slope * end + intercept)
        hook = ink[max(0, ey):ey + round(space), max(0, ex - 2):ex + 3].max(axis=1)
        closed = len(hook) == round(space) and float(hook.mean()) >= .7
        edge = abs(ex - st['x_range'][1]) <= space
        if not closed and not edge:
            continue
        candidates.append({'system': si, 'left': x - .3 * space, 'right': ex + .3 * space,
                           'y': y, 'closed': closed, 'space': space,
                           'prefix': x <= min(n['x'] for n in st['notes']) + space,
                           'method': 'two_counter_8_dashed_bracket'})
        if len(candidates) > 64:
            return []
    candidates.sort(key=lambda s: (s['system'], s['left']))
    accepted = []
    for candidate in candidates:
        current, visited = candidate, [candidate]
        while not current['closed']:
            following = [s for s in candidates if s['system'] == current['system'] + 1 and s['prefix']]
            if len(following) != 1:
                break
            current = following[0]
            visited.append(current)
        if current['closed']:
            candidate = dict(candidate, continuation_rows=len(visited) - 1)
            accepted.append(candidate)
    return accepted


def geometry_from_graph(graph, staff_geometry):
    """Convert image tracks and head centers to the existing position grid."""
    from pathlib import Path
    import hashlib
    result = {'systems': []}
    if len(staff_geometry['staves']) > 128 or len(graph['systems']) > 64:
        return {}
    source = None
    source_path = Path(staff_geometry.get('source', ''))
    if source_path.is_file():
        payload = source_path.read_bytes()
        if staff_geometry.get('source_sha256') != hashlib.sha256(payload).hexdigest():
            return {}
        source = json.loads(payload)
        if len(source.get('systems', [])) != len({s['system'] for s in staff_geometry['staves']}):
            return {}
    for si in sorted({s['system'] for s in staff_geometry['staves']}):
        staves = sorted([s for s in staff_geometry['staves'] if s['system'] == si], key=lambda s: s['index'])
        if not 1 <= len(staves) <= 2:
            return {}
        upper = staves[0]
        notes = []
        rows = [r for r in graph['systems'] if r['system'] == si and r['staff'] == 1]
        if len(rows) != 1 or len(rows[0]['groups']) > 2048:
            return {}
        group_heads=[]
        for gi,g in enumerate(rows[0]['groups']):
            if g.get('grace') or g.get('rest'):
                group_heads.append([])
                continue
            measured = []
            for h in g.get('heads', []):
                x, y = h['xy']
                line = upper['lines'][-1]['polyline']
                bottom = float(np.interp(x / FACTOR, *np.array(line).T)) * FACTOR
                top = float(np.interp(x / FACTOR, *np.array(upper['lines'][0]['polyline']).T)) * FACTOR
                space = (bottom - top) / 4
                pos = 1 + (bottom - y) * 2 / space
                measured.append((h, pos, space))
            group_items=[]
            for h, pos, space in measured:
                x, y = h['xy']
                if abs(pos - round(pos)) > .45:
                    # A clear octave partner can resolve subpixel track noise.
                    partners = [(hh, pp) for hh, pp, _ in measured
                                if hh is not h and hh['pitch'][0] == h['pitch'][0]
                                and abs(hh['pitch'][1] - h['pitch'][1]) == 1
                                and abs(pp - round(pp)) <= .3
                                and abs(abs(pp - pos) - 7) <= .3]
                    if len(partners) != 1:
                        continue
                    hh, pp = partners[0]
                    pos = round(pp) + 7 * (h['pitch'][1] - hh['pitch'][1])
                item={'x': x, 'y': y, 'position': round(pos), 'staff_space': space,
                      'stem': [g['stem'], 0] if g.get('stem') is not None else None,
                      'step':h['pitch'][0],'octave':int(h['pitch'][1]),'group':gi}
                notes.append(item);group_items.append(item)
            group_heads.append(group_items)
        # Homr's original segmentation can retain a chord head missed by the
        # relation detector. Use it only for one nearby, same-letter octave
        # partner and assign each recovered head to at most one image group.
        if source is not None:
            # Photo map system indices are one-based; Homr geometry arrays are zero-based.
            if not 1 <= si <= len(source.get('systems', [])):
                return {}
            src_staff=source['systems'][si-1]['staffs'][0];grid=src_staff.get('grid',[])
            if grid and len(grid[0]['y']) in (5,10):
                width,height=source['preprocessed_size'];cw,ch=staff_geometry.get('crop_size_px',staff_geometry['image_size_px'])
                left,top_offset=staff_geometry['crop_xy'];sx=cw/width*FACTOR;sy=ch/height*FACTOR
                border=(grid[0]['y'][4]+grid[0]['y'][5])/2 if len(grid[0]['y'])==10 else float('inf')
                raw_heads=[]
                for h in src_staff['notes']:
                    if len(grid[0]['y'])==10 and h['y']>=border:continue
                    p=int(h.get('position',0));d=30+p-1
                    if p<1 or h.get('extended_ledger') or h.get('size',[0])[0]<5:continue
                    raw_x=(h['x']*cw/width+left)*FACTOR
                    raw_stem=h.get('stem')
                    stem_x=((raw_stem[0]*cw/width+left)*FACTOR) if raw_stem and len(raw_stem)>=2 else None
                    raw_heads.append({'x':raw_x,
                                      'y':(h['y']*ch/height+top_offset)*FACTOR,
                                      'position':p,'staff_space':h['staff_space']*sy,
                                      'step':'CDEFGAB'[d%7],'octave':d//7,'source_recovered':True,
                                      'source_stem_direction':('up' if stem_x>raw_x else 'down') if stem_x is not None else None})
                # Calibrate Homr's local E4-based positions to the relation
                # graph's staff coordinate using co-located, same-pitch heads.
                offsets=[]
                for recovered in raw_heads:
                    for gi,g in enumerate(rows[0]['groups']):
                        for anchor,model_head in zip(group_heads[gi],g.get('heads',[])):
                            if (model_head.get('confidence',0)<.85 or recovered['step']!=anchor['step']
                                    or recovered['octave']!=anchor['octave']):continue
                            dx=abs(recovered['x']-anchor['x'])/anchor['staff_space']
                            dy=abs(recovered['y']-anchor['y'])/anchor['staff_space']
                            if dx<=.55 and dy<=.3:
                                offsets.append(anchor['position']-recovered['position'])
                if len(offsets)>=3:
                    offset=median(offsets)
                    if abs(offset-round(offset))<=.25 and sum(abs(v-offset)<=.25 for v in offsets)>=3:
                        for recovered in raw_heads:
                            recovered['position']+=round(offset)
                # Attach the independent Homr stem-side witness to relation
                # heads. Relation stem links can cross between interleaved
                # voices, so this source segmentation is retained separately.
                for gi,g in enumerate(rows[0]['groups']):
                    for anchor,model_head in zip(group_heads[gi],g.get('heads',[])):
                        matches=[]
                        for raw in raw_heads:
                            if raw['step']!=anchor['step'] or raw['octave']!=anchor['octave']:
                                continue
                            dx=abs(raw['x']-anchor['x'])/anchor['staff_space']
                            dy=abs(raw['y']-anchor['y'])/anchor['staff_space']
                            if dx<=.55 and dy<=.3 and raw.get('source_stem_direction'):
                                matches.append(raw)
                        if len(matches)==1:
                            anchor['source_stem_direction']=matches[0]['source_stem_direction']
                pending=[]
                for recovered in raw_heads:
                    choices=[]
                    represented=False
                    for gi,g in enumerate(rows[0]['groups']):
                        for anchor,model_head in zip(group_heads[gi],g.get('heads',[])):
                            dx=abs(recovered['x']-anchor['x'])/anchor['staff_space']
                            dy=abs(recovered['y']-anchor['y'])/anchor['staff_space']
                            # Do not duplicate a head already present in the relation graph.
                            if (recovered['step']==anchor['step'] and recovered['position']==anchor['position']
                                    and dx<=.55 and dy<=.3):
                                represented=True
                                break
                            if (model_head.get('confidence',0)<.85 or recovered['step']!=anchor['step']
                                    or abs(recovered['position']-anchor['position'])!=7):continue
                            dy=abs(abs(recovered['y']-anchor['y'])-3.5*anchor['staff_space'])/anchor['staff_space']
                            if dx<=.55 and dy<=.3:choices.append((dx+dy,gi,anchor))
                        if represented:
                            break
                    if represented:
                        continue
                    choices.sort(key=lambda x:x[0])
                    if choices and (len(choices)==1 or choices[1][0]-choices[0][0]>=.15):
                        pending.append((choices[0][0],choices[0][1],recovered))
                assigned=set()
                for _,gi,recovered in sorted(pending,key=lambda x:x[0]):
                    xy=(round(recovered['x']),round(recovered['y']))
                    if xy in assigned:continue
                    recovered['group']=gi;notes.append(recovered);group_heads[gi].append(recovered);assigned.add(xy)
        result['systems'].append({'staffs': [{'notes': notes,
            'x_range': [upper['lines'][0]['start'][0] * FACTOR, upper['lines'][0]['end'][0] * FACTOR],
            'y_range': [min(p[1] for p in upper['lines'][0]['polyline']) * FACTOR,
                        max(p[1] for p in staves[-1]['lines'][-1]['polyline']) * FACTOR]}],
            'group_heads':group_heads})
    return result


def fuse_ottavas(root, graph, photo, staff_geometry, mixed_only=False,global_group_indices=True):
    from pathlib import Path
    digest = hashlib.sha256(Path(photo).read_bytes()).hexdigest()
    if (graph.get('image_sha256') != digest or staff_geometry.get('image_sha256') != digest
            or root.find('.//octave-shift') is not None):
        return []
    geometry = geometry_from_graph(graph, staff_geometry)
    systems = _xml_systems(root)
    if not geometry or len(systems) != len(geometry['systems']):
        return []
    image = cv2.imread(str(photo), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size > 16000000 or max(image.shape) > 8192:
        return []
    image = cv2.resize(image, None, fx=FACTOR, fy=FACTOR)
    detected = spans(geometry, image)
    changes = []
    from omr_bar_geometry import measure_systems
    from omr_event_records import records
    measures_by_system = measure_systems(root)
    rec = {m: row for m, row in zip(root.findall('part/measure'), records(root))}
    if len(measures_by_system) != len(systems):
        return []
    for si, (notes, sys) in enumerate(zip(systems, geometry['systems'])):
        selected = [s for s in detected if s['system'] == si]
        if not selected:
            continue
        row = next(r for r in graph['systems'] if r['system'] == si + 1 and r['staff'] == 1)
        bounds = row['bounds']
        if len(bounds) != len(measures_by_system[si]) + 1 or len(bounds) > 129:
            continue
        for mi, measure in enumerate(measures_by_system[si]):
            events = [r for r in rec[measure] if r['staff'] == 1 and not r['grace']]
            # A staff bracket may be layer scoped in the source notation.
            # For mixed voices, only edit a long-duration octave chord whose
            # written head positions map uniquely to one relation group. This
            # keeps a bracketed upper voice from shifting the interleaved
            # melody. Single-note events are deliberately not inferred here.
            if len(events) > 128:
                continue
            mixed_voice = len({r['voice'] for r in events}) != 1
            groups = []
            if not mixed_voice:
                for onset in sorted({r['onset'] for r in events if r['midi'] is not None}):
                    group = [r['node'] for r in events if r['onset'] == onset and r['midi'] is not None]
                    groups.append(sorted(group, key=lambda n: _position(n)[1]))
            observed = sorted([(gi,g) for gi,g in enumerate(row['groups']) if bounds[mi] < g['x'] < bounds[mi+1]
                               and not g.get('grace') and g.get('heads')], key=lambda pair: pair[1]['x'])
            columns = []
            for local_index,(global_index,g) in enumerate(observed):
                # Geometry group_heads belongs to the complete image row,
                # not the measure-filtered list. Keep the row-global index.
                gi=global_index if global_group_indices else local_index
                if columns and g['x'] - columns[-1]['x'] <= .65 * row['space_px']:
                    columns[-1]['heads'].extend(g['heads'])
                    columns[-1]['indices'].append(gi)
                else:
                    columns.append({'x': g['x'], 'heads': list(g['heads']), 'indices':[gi]})
            observed = columns
            if mixed_voice:
                # Use chord geometry as the scope witness: a chord's note
                # letters and staff positions must agree exactly with one
                # image column, and both heads must lie under the same 8va.
                # Long duration separates the sustained voice from the
                # interleaved short-note line in the known polyphonic layout.
                short_durations = [float(r['duration']) for r in events
                                   if r['midi'] is not None and r['duration'] > 0]
                if not short_durations:
                    continue
                duration_floor = 2 * median(short_durations)
                positions_by_column = []
                # Keep relation groups separate here. The normal playback
                # aligner merges close x positions, but that would combine a
                # sustained chord with a neighboring 16th note in this layout.
                for gi, image_group in enumerate(row['groups']):
                    if not (bounds[mi] < image_group['x'] < bounds[mi + 1]
                            and not image_group.get('grace') and image_group.get('heads')):
                        continue
                    image_heads = sys['group_heads'][gi]
                    positions_by_column.append(({'x': image_group['x'], 'indices': [gi]}, image_heads))
                by_voice_onset_duration = {}
                for event in events:
                    if event['midi'] is None:
                        continue
                    key = (event['voice'], event['onset'], event['duration'])
                    by_voice_onset_duration.setdefault(key, []).append(event['node'])
                edits = []
                claimed_columns = set()
                for (_, _, duration), chord in by_voice_onset_duration.items():
                    if len(chord) < 2 or float(duration) < duration_floor:
                        continue
                    pitch_steps = [n.findtext('pitch/step') for n in chord]
                    staff_positions = [_position(n)[1] for n in chord]
                    if (not all(pitch_steps) or len(set(pitch_steps)) != 1
                            or len(set(staff_positions)) != len(staff_positions)
                            or sorted(staff_positions) != list(range(min(staff_positions), max(staff_positions) + 1, 7))):
                        continue
                    wanted = sorted(zip(pitch_steps, staff_positions))
                    candidates = []
                    for ci, (column, image_heads) in enumerate(positions_by_column):
                        found = sorted((h['step'], h['position']) for h in image_heads)
                        if found == wanted:
                            upper = max(image_heads, key=lambda h: h['position'])
                            # Require the long chord's top head to carry an
                            # independently recovered up-stem witness. This
                            # rejects octave-shaped chords belonging to the
                            # other, down-stem layer; the relation graph's
                            # stem direction is not trusted for this check.
                            if upper.get('source_stem_direction') != 'up':
                                continue
                            candidates.append((ci, column))
                    if len(candidates) != 1:
                        continue
                    ci, column = candidates[0]
                    head_by_position = {(h['step'], h['position']): h
                                        for h in positions_by_column[ci][1]}
                    spans_for_heads = [next((s for s in selected
                                             if s['left'] <= head_by_position[pair]['x'] <= s['right']), None)
                                       for pair in wanted]
                    if (not all(spans_for_heads) or len({id(s) for s in spans_for_heads}) != 1
                            or ci in claimed_columns):
                        continue
                    claimed_columns.add(ci)
                    edits.extend((n, head_by_position[(n.findtext('pitch/step'), _position(n)[1])],
                                  spans_for_heads[0]) for n in chord)
                for n, head, span in edits:
                    if n.find('tie') is not None or n.find('notations/tied') is not None:
                        continue
                    octave = n.find('pitch/octave')
                    old = int(octave.text)
                    octave.text = str(old + 1)
                    changes.append({'system': si + 1, 'measure': measure.get('number'), 'x': head['x'],
                                    'old_octave': old, 'new_octave': old + 1, 'span': span,
                                    'scope': 'unique_long_octave_chord'})
                continue
            if mixed_only:
                continue
            if not groups or len(groups) != len(observed) or len(groups) > 64:
                continue
            plan = []
            for group, observed_group in zip(groups, observed):
                heads=[]
                for gi in observed_group['indices']:
                    heads.extend(sys['group_heads'][gi])
                heads=list({(h['x'],h['y'],h['position']):h for h in heads}.values())
                heads.sort(key=lambda h: h['position'])
                if len(group) != len(heads) or not 0 < len(group) <= 8:
                    continue
                group_plan = []
                valid = True
                for n, head in zip(group, heads):
                    head_span = next((s for s in selected if s['left'] <= head['x'] <= s['right']), None)
                    target = head['position'] + (7 if head_span else 0)
                    diff = target - _position(n)[1]
                    if diff % 7 or abs(diff) > 14 or (not head_span and diff):
                        valid = False; break
                    if diff and n.find('tie') is None and n.find('notations/tied') is None:
                        group_plan.append((n, head, diff // 7, head_span))
                if valid:
                    plan.extend(group_plan)
            for n, head, delta, span in plan:
                octave = n.find('pitch/octave'); old = int(octave.text)
                octave.text = str(old + delta)
                changes.append({'system': si + 1, 'measure': measure.get('number'), 'x': head['x'],
                                'old_octave': old, 'new_octave': old + delta, 'span': span})
    return changes
