"""Bounded local clocks with segmented but unexported beam time slots.

A complete independent written-value staff and exact shared image columns
prove the bar clock. Unexported photographed heads reserve empty time slots;
this module never creates a note or changes pitches, staves or voice labels.
No source score, work identity, or reference notation is an inference input.
"""
import copy
from collections import Counter
from fractions import Fraction as F
from math import lcm
import xml.etree.ElementTree as E
import numpy as np
from align_homr_geometry import _position, _head_staff
from omr_bar_geometry import measure_systems, bar_bounds
from omr_event_records import records
from repair_grace_timing import _rebuild
from repair_tuplets import continuous_beams, scale_divisions
from .relation_group_clock import bounded_map
from .relation_clock_lattice import written, solve_grid
from .relation_beam_components import stem_end
from .relation_hollow_evidence import enclosed_hollow


def partner_written_evidence(rows, mapped, image, ink):
    """Prove the partner's written units independently in the photograph.

    A mistaken 16th XML label cannot establish a bar clock using a physical
    single beam. Unsupported flags, dots and chords conservatively abstain.
    """
    if not rows or len(rows)>128 or any(r['chord'] or r['node'].findall('dot') for r in rows):return False
    ordered=sorted([(mapped.get(r['node']),r) for r in rows],key=lambda p:p[0]['x'] if p[0] is not None else -1)
    if any(h is None for h,r in ordered):return False
    for i,(h,r) in enumerate(ordered):
        typ=r['node'].findtext('type');s=h['staff_space']
        others=[ordered[j][0] for j in (i-1,i+1) if 0<=j<len(ordered)]
        pairs=[(other,h) if other['x']<h['x'] else (h,other) for other in others]
        counts={n for a,b in pairs for n in (1,2) if any(continuous_beams(a,b,ink,up,n,maximum_gap=8) for up in (True,False))}
        shafts=[stem_end(h,ink,up) for up in (True,False)]
        if typ in ('eighth','16th'):
            expected=1 if typ=='eighth' else 2
            # A 16th head can adjoin a single primary beam across a beat
            # boundary and a double beam within its subdivision cell.
            # Its double-beam neighbor still proves the written value.
            if not _core(h,ink) or expected not in counts or expected==1 and 2 in counts:return False
        elif typ=='quarter':
            if not _core(h,ink) or not any(shafts) or counts:return False
        elif typ in ('half','whole'):
            if not enclosed_hollow({'xy':[h['x'],h['y']]},image,s):return False
            if typ=='half' and (not any(shafts) or counts):return False
            if typ=='whole' and any(shafts):return False
        else:return False
    return True


def beam_cells(heads, ink, count=1):
    """Every edge in each cell must have the same exact beam count."""
    if not 6 <= len(heads) <= 48:
        return None
    cells = [[0]]
    for i, (a, b) in enumerate(zip(heads, heads[1:]), 1):
        if any(continuous_beams(a, b, ink, up, count, maximum_gap=8) for up in (True, False)):
            cells[-1].append(i)
        else:
            cells.append([i])
    return cells


def _core(head, ink):
    x, y, s = head['x'], head['y'], head['staff_space']
    a, b = round(x-.3*s), round(x+.3*s)+1
    c, d = round(y-.15*s), round(y+.15*s)+1
    return 5 <= s <= 80 and min(a,c) >= 0 and b <= ink.shape[1] and d <= ink.shape[0] and bool(ink[c:d,a:b].size) and ink[c:d,a:b].mean() >= .85


def fuse(root, homr, image, diagnostics=None):
    diag = diagnostics if diagnostics is not None else Counter()
    if image.ndim != 2 or image.size > 16000000 or max(image.shape) > 8192 or len(root.findall('part')) != 1:
        return []
    systems = measure_systems(root)
    if len(systems) != len(homr.get('systems', [])):
        return []
    mapped = bounded_map(root, homr, image)
    ink = (image < 140).astype('uint8')
    events = records(root); measures = root.findall('part/measure')
    contexts=[]; division=1;meter=None
    for m in measures:
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)));t=a.find('time')
            if t is not None:
                b,u=t.findtext('beats'),t.findtext('beat-type')
                meter=F(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u) else None
        contexts.append((division,meter))
    jobs=[];offset=0
    for ms, sy in zip(systems, homr['systems']):
        if len(sy.get('staffs', [])) != 1:
            offset+=len(ms);continue
        st=sy['staffs'][0];bounds=bar_bounds(st,image)
        if bounds is None or len(bounds)!=len(ms)+1:
            offset+=len(ms);continue
        for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
            number=offset+j;rs=events[number];div,meter=contexts[number]
            if not rs or len(rs)>128 or not 0<div<=9600 or any(r['grace'] or r['node'].get('print-object')=='no' for r in rs):continue
            m=measures[number];first=next((k for k,n in enumerate(m) if n.tag=='note'),None)
            if first is None or any(n.tag in ('forward','backup') for n in list(m)[:first]):continue
            physical={sn:sorted([h for h in st['notes'] if lo<h['x']<hi and _head_staff(st,h)==sn],key=lambda h:h['x']) for sn in (1,2)}
            for target_sn,partner_sn in ((1,2),(2,1)):
                hs=physical[target_sn];es=[r for r in rs if r['staff']==target_sn];ps=[r for r in rs if r['staff']==partner_sn]
                if not 6<=len(hs)<=48 or len(hs)%3 or not es or not ps:continue
                if any(r['midi'] is None or r['chord'] or written(r['node'])!=F(1,2) or r['duration'] not in (F(1,2),F(1,3)) for r in es):continue
                if any(r['node'].find('time-modification') is not None and (r['node'].findtext('time-modification/actual-notes'),r['node'].findtext('time-modification/normal-notes'))!=('3','2') for r in es):continue
                if not .75*len(hs)<=len(es)<=len(hs) or len({r['voice'] for r in ps})!=1:continue
                s=float(np.median([h['staff_space'] for h in hs]))
                if any(b['x']-a['x']<.7*s for a,b in zip(hs,hs[1:])) or any(not _core(h,ink) for h in hs):continue
                diag['physical_slot_candidate']+=1
                cells=beam_cells(hs,ink)
                if cells is None or any(len(g)<3 or len(g)%3 for g in cells):diag['beam_cells_unproven']+=1;continue
                # Unique segmentation binding preserves one empty slot for
                # each photographed head omitted by the exporter.
                bindings=[];used=set()
                for r in es:
                    h=mapped.get(r['node']);p=_position(r['node'])
                    if h is None or p is None or p[1]!=h['position']:break
                    choices=[k for k,q in enumerate(hs) if abs(h['x']-q['x'])<.15*s and abs(h['y']-q['y'])<.15*s]
                    if len(choices)!=1 or choices[0] in used:break
                    k=choices[0];used.add(k);bindings.append((r,k))
                if len(bindings)!=len(es):diag['target_binding']+=1;continue
                # The partner staff must be complete and have one written
                # rhythmic lane. All pitched heads must bind bijectively.
                if any(r['midi'] is None or r['node'].find('time-modification') is not None or written(r['node'])!=r['duration'] for r in ps):continue
                if len(ps)!=len(physical[partner_sn]):diag['partner_cardinality']+=1;continue
                ph=[];partner_used=set()
                for r in ps:
                    h=mapped.get(r['node']);p=_position(r['node'])
                    if h is None or p is None or p[1]!=h['position'] or id(h) in partner_used:break
                    partner_used.add(id(h));ph.append((h['x'],r))
                if len(ph)!=len(ps):diag['partner_binding']+=1;continue
                if not partner_written_evidence(ps,mapped,image,ink):diag['partner_written_image_unproven']+=1;continue
                ph.sort(key=lambda z:(z[0],z[1]['onset']));pg=[]
                for x,r in ph:
                    if pg and abs(x-pg[-1][0])<.35*s:
                        if written(r['node'])!=pg[-1][2]:break
                        pg[-1][1].append(r)
                    else:pg.append([x,[r],written(r['node'])])
                if sum(len(g[1]) for g in pg)!=len(ps):continue
                local_meter=sum((g[2] for g in pg),F())
                if not 1<=local_meter<=16 or meter is not None and local_meter!=meter:continue
                anchors=[];cursor=F()
                for x,group,w in pg:anchors.append((x,cursor,.4*s));cursor+=w
                plan=solve_grid([h['x'] for h in hs],F(1,2),local_meter,anchors)
                if plan is None or (plan['actual'],plan['normal'])!=(3,2) or len(plan['columns'])<2:diag['shared_columns_unproven']+=1;continue
                edits=[(r,k*plan['duration'],plan['duration']) for r,k in bindings]
                if not any(r['onset']!=t or r['duration']!=d for r,t,d in edits):continue
                jobs.append(dict(number=number,edits=edits,div=div,physical_slots=len(hs),exported_heads=len(es),columns=plan['columns'],cells=[len(g) for g in cells],staff=target_sn,local_meter=str(local_meter)))
        offset+=len(ms)
    counts=Counter(j['number'] for j in jobs);jobs=[j for j in jobs if counts[j['number']]==1]
    if not jobs or len(jobs)>128:return []
    factor=1
    for job in jobs:
        for r,t,d in job['edits']:factor=lcm(factor,(t*job['div']).denominator,(d*job['div']).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for job in jobs:
        m=measures[job['number']];saved=copy.deepcopy(list(m));div=job['div']*factor
        desired={r['node']:int(r['onset']*div) for r in events[job['number']]}
        expected_duration={r['node']:r['duration'] for r in events[job['number']]}
        for r,t,d in job['edits']:
            n=r['node'];desired[n]=int(t*div);n.find('duration').text=str(int(d*div))
            expected_duration[n]=d
            for old in n.findall('time-modification'):n.remove(old)
            tm=E.Element('time-modification')
            for tag,val in [('actual-notes','3'),('normal-notes','2'),('normal-type','eighth')]:E.SubElement(tm,tag).text=val
            at=next((i for i,e in enumerate(n) if e.tag in ('stem','notehead','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
        if not _rebuild(m,desired):m[:]=saved;continue
        actual=records(root)[job['number']]
        if len(actual)!=len(desired) or any(r['node'] not in desired or r['onset']!=F(desired[r['node']],div) or r['duration']!=expected_duration[r['node']] for r in actual):m[:]=saved;continue
        changes.append({k:v for k,v in job.items() if k not in ('number','edits','div')}|dict(measure=job['number']+1,method='physical_empty_slots_and_independent_written_clock',voice_pitch_staff_cardinality_preserved=True))
    return changes
