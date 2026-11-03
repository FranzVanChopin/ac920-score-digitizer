"""Complete single-lane tuplet clocks, with independent numbers and beam ink.

Graph beam labels are proposals, not the clock proof. Every three/four-note
cell must have its own photographed numeral and continuous counted beams.
"""
from collections import Counter
from fractions import Fraction
from math import lcm
import copy,xml.etree.ElementTree as E
import cv2
from omr_event_records import records
from repair_tuplets import printed_number,scale_divisions
from repair_grace_timing import _rebuild
from .relation_coverage import inputs,ordered
from .relation_measure_joint import align_columns,written
from .relation_tuplets import geometry_match


def proposal(groups,bound,row,track,image,ink,meter):
    if not 6<=len(groups)<=24 or len(groups)!=len(bound):return None
    if any(not 1<=len(g)<=8 or any(r['grace'] or r['midi'] is None or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].findall('dot') for r in g) for g in groups):return None
    kinds={r['node'].findtext('type') for g in groups for r in g}
    if len(kinds)!=1 or next(iter(kinds)) not in ('eighth','16th'):return None
    kind=next(iter(kinds));unit=Fraction(1,2) if kind=='eighth' else Fraction(1,4);beam_count=1 if kind=='eighth' else 2
    choices=[(count,normal) for count,normal in ((3,2),(4,3)) if len(groups)%count==0 and len(groups)*unit*normal/count==meter]
    if len(choices)!=1 or not geometry_match(groups,bound,row,track,image):return None
    count,normal=choices[0];duration=unit*normal/count
    if any(r['duration'] not in (unit,duration) or r['node'].find('time-modification') is not None and (r['node'].findtext('time-modification/actual-notes'),r['node'].findtext('time-modification/normal-notes'))!=(str(count),str(normal)) for g in groups for r in g):return None
    for i in range(0,len(bound),count):
        cell=bound[i:i+count];witnessed=[]
        for edge in (min,max):
            heads=[edge(g['heads'],key=lambda h:h['xy'][1]) for g in cell];mapped=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for h in heads]
            if printed_number(mapped,ink,count,recover_crossings=True,flexible_beams=True,beam_count=beam_count):witnessed.append(mapped)
        if not witnessed:return None
    return {'count':count,'normal':normal,'duration':duration,'type':kind,'beams':beam_count}


def fuse(root,graph,photo,geometry,proposal_fn=None):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];rows=records(root);ms=root.findall('part/measure');div=1;meter=None;divs=[];meters=[];jobs=[];offset=0
    for m in ms:
        for a in m.findall('attributes'):
            div=int(a.findtext('divisions',str(div)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        divs.append(div);meters.append(meter)
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in selected:
            if row['staff']>len(staffs):continue
            for mi,(l,r) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;m=ms[number];es=[r for r in rows[number] if r['staff']==row['staff']]
                if not meters[number] or m.get('implicit')=='yes' or len(es)>128 or len({r['voice'] for r in es})!=1 or any(r['midi'] is None or r['grace'] for r in es):continue
                groups=ordered(es)
                if not groups or not 6<=len(groups)<=24:continue
                seq=sorted([g for g in row['groups'] if l<g['x']<r and g.get('heads') and not g.get('grace')],key=lambda g:g['x'])
                pairs=align_columns([Counter(written(v) for v in g) for g in groups],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq])
                if len(pairs)!=len(groups) or [i for i,j in pairs]!=list(range(len(groups))):continue
                bound=[seq[j] for i,j in pairs]
                if any(Counter(written(v) for v in g)!=Counter(tuple(h['pitch'][:2]) for h in q['heads']) for g,q in zip(groups,bound)):continue
                plan=(proposal_fn or proposal)(groups,bound,row,staffs[row['staff']-1],image,ink,meters[number])
                if plan is None:continue
                durations=plan.get('durations',[plan['duration']]*len(groups));onsets=[];clock=Fraction()
                for duration in durations:onsets.append(clock);clock+=duration
                plan['durations']=durations;plan['onsets']=onsets
                if all(v['duration']==durations[i] and v['onset']==onsets[i] for i,g in enumerate(groups) for v in g):continue
                indices=[i for i,n in enumerate(m) if n.tag=='note']
                if not indices or any(n.tag in ('backup','forward') for n in list(m)[:indices[0]]) or any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
                jobs.append((number,groups,plan))
        offset+=len(selected[0]['bounds'])-1
    if not jobs or len(jobs)>128:return []
    factor=1
    for number,groups,p in jobs:
        for duration in p['durations']:factor=lcm(factor,(duration*divs[number]).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for number,groups,p in jobs:
        m=ms[number];saved=copy.deepcopy(list(m));division=divs[number]*factor;duration=int(p['duration']*division);desired={r['node']:int(r['onset']*division) for r in rows[number]}
        for i,g in enumerate(groups):
            cell=next((c for c in p.get('cells',[]) if c['start']<=i<c['end']),{'start':i//p['count']*p['count'],'end':(i//p['count']+1)*p['count'],'count':p['count'],'normal':p['normal']})
            duration=int(p['durations'][i]*division)
            for r in g:
                n=r['node'];desired[n]=int(p['onsets'][i]*division);n.find('duration').text=str(duration)
                for tag in ('time-modification','beam'):
                    for a in n.findall(tag):n.remove(a)
                if cell['count']!=1:
                    tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text=str(cell['count']);E.SubElement(tm,'normal-notes').text=str(cell['normal']);E.SubElement(tm,'normal-type').text=p['type'];at=next((j for j,c in enumerate(n) if c.tag in ('stem','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
                if not r['chord']:
                    for b in range(1,p['beams']+1):
                        beam=E.Element('beam',number=str(b));beam.text='begin' if i==cell['start'] else 'end' if i==cell['end']-1 else 'continue';at=next((j for j,c in enumerate(n) if c.tag in ('notations','lyric')),len(n));n.insert(at,beam)
                nt=n.find('notations')
                if nt is not None:
                    for a in nt.findall('tuplet'):nt.remove(a)
                if not r['chord'] and cell['count']!=1 and i in (cell['start'],cell['end']-1):
                    if nt is None:nt=E.SubElement(n,'notations')
                    E.SubElement(nt,'tuplet',type='start' if i==cell['start'] else 'stop',number=str(i//p['count']+1),bracket='no')
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'staff':groups[0][0]['staff'],'method':'joint_complete_lane_all_printed_numbers_and_beam_ink','actual':p['count'],'normal':p['normal'],'cells':len(groups)//p['count'],'existing_pitch_staff_voice_cardinality_preserved':True})
    return changes
