"""Experimental cross-staff beam clocks, bounded image-only inference."""
import copy
from fractions import Fraction as F
from math import lcm
from collections import Counter,defaultdict
import xml.etree.ElementTree as E
import numpy as np
from omr_event_records import records
from repair_grace_timing import _rebuild
from repair_tuplets import scale_divisions
from .relation_beam_components import components,temporal_map
from .relation_clock_lattice import solve_grid,written

def fuse(root,homr,image,diag=None):
    diag=diag if diag is not None else Counter()
    if image.ndim!=2 or image.size>16000000 or max(image.shape)>8192 or len(root.findall('part'))!=1:return []
    mapped=temporal_map(root,homr);ink=(image<140).astype('uint8');all_events=records(root);meter=None;division=1;jobs=[]
    for mi,(m,rs) in enumerate(zip(root.findall('part/measure'),all_events)):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)));t=a.find('time')
            if t is not None:
                beats,unit=t.findtext('beats'),t.findtext('beat-type');meter=F(int(beats)*4,int(unit)) if beats and beats.isdecimal() and unit and unit.isdecimal() and int(unit) else None
        if meter is None or not 1<=meter<=16 or len(rs)>128 or division>9600 or any(r['grace'] or r['node'].get('print-object')=='no' for r in rs):continue
        groups,edges=components(rs,mapped,ink)
        cells=[g for g in groups if len(g)==3 and all(written(rs[i]['node'])==F(1,2) and rs[i]['duration'] in (F(1,2),F(1,3)) and not rs[i]['chord'] for i in g)]
        if not cells or len(cells)*F(1)!=meter:continue
        cells.sort(key=lambda g:mapped[rs[g[0]]['node']]['x']);sequence=[i for g in cells for i in g];heads=[mapped[rs[i]['node']] for i in sequence];s=float(np.median([h['staff_space'] for h in heads]));xs=[h['x'] for h in heads]
        if any(b-a<.7*s for a,b in zip(xs,xs[1:])):continue
        if any((ink[max(0,round(h['y']-.15*s)):round(h['y']+.15*s)+1,max(0,round(h['x']-.3*s)):round(h['x']+.3*s)+1]>0).mean()<.85 for h in heads):continue
        diag['complete_physical_cells']+=1
        target=set(sequence);others=defaultdict(list)
        for i,r in enumerate(rs):
            if i not in target:others[r['staff'],r['voice']].append((i,r))
        anchors=[];edits=[];valid=True
        for lane in others.values():
            gs=[]
            for i,r in lane:
                if r['chord'] and gs:gs[-1].append((i,r))
                else:gs.append([(i,r)])
            ws=[written(g[0][1]['node']) for g in gs]
            if None in ws or sum(ws,F())!=meter or any(any(written(r['node'])!=w or r['duration']!=w or r['node'].find('time-modification') is not None for i,r in g) for g,w in zip(gs,ws)):
                valid=False;diag['nonbeam_lane_incomplete']+=1;break
            t=F()
            for g,w in zip(gs,ws):
                hs=[mapped[r['node']] for i,r in g if r['node'] in mapped]
                if hs:
                    x=float(np.median([h['x'] for h in hs]))
                    if max(h['x'] for h in hs)-min(h['x'] for h in hs)>.5*s:valid=False;break
                    anchors.append((x,t,.4*s))
                elif any(r['midi'] is not None for i,r in g):valid=False;break
                edits.extend((r,t,r['duration']) for i,r in g);t+=w
            if not valid:break
        if not valid:continue
        # Co-located chord anchors from independent lanes count once.
        anchors=sorted(set(anchors),key=lambda a:(a[1],a[0]));unique=[]
        for a in anchors:
            if any(a[1]==b[1] and abs(a[0]-b[0])<.4*s for b in unique):continue
            unique.append(a)
        plan=solve_grid(xs,F(1,2),meter,unique)
        if plan is None or len(plan['columns'])<2:diag['independent_clock_unproven']+=1;continue
        edits.extend((rs[i],F(j,3),F(1,3)) for j,i in enumerate(sequence))
        if not any(r['onset']!=t or r['duration']!=d for r,t,d in edits):continue
        if len(edits)!=len(rs):continue
        # A connected rhythmic stream is one MusicXML voice across staves.
        # Reserve other written-value lanes, then use the lowest free voice.
        occupied={r['voice'] for i,r in enumerate(rs) if i not in target}
        free=[str(v) for v in range(1,9) if str(v) not in occupied]
        if not free:continue
        jobs.append(dict(mi=mi,division=division,edits=edits,target={rs[i]['node'] for i in sequence},voice=free[0],cells=len(cells),cross_staff_cells=sum(len({rs[i]['staff'] for i in g})>1 for g in cells),columns=plan['columns']))
    if not jobs or len(jobs)>128:return []
    factor=1
    for job in jobs:
        for r,t,d in job['edits']:factor=lcm(factor,(t*job['division']).denominator,(d*job['division']).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[];ms=root.findall('part/measure')
    for job in jobs:
        m=ms[job['mi']];saved=copy.deepcopy(list(m));desired={};div=job['division']*factor
        for r,t,d in job['edits']:
            n=r['node'];desired[n]=int(t*div);n.find('duration').text=str(int(d*div))
            if n in job['target']:
                if n.find('voice') is None:
                    v=E.Element('voice');v.text=job['voice'];n.insert(next((i for i,c in enumerate(n) if c.tag in ('type','dot','accidental','time-modification','stem','staff','beam','notations')),len(n)),v)
                else:n.find('voice').text=job['voice']
                for x in n.findall('time-modification'):n.remove(x)
                tm=E.Element('time-modification')
                for tag,val in [('actual-notes','3'),('normal-notes','2'),('normal-type','eighth')]:E.SubElement(tm,tag).text=val
                at=next((i for i,c in enumerate(n) if c.tag in ('stem','staff','beam','notations')),len(n));n.insert(at,tm)
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append(dict(measure=job['mi']+1,cells=job['cells'],cross_staff_cells=job['cross_staff_cells'],columns=job['columns'],voice=job['voice'],method='physical_cross_staff_components_independent_written_lanes'))
    return changes


