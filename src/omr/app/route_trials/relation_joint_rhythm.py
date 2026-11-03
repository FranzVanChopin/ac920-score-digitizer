"""Infer implicit 16th triplets from a complete, ink-backed partner staff.

One full ordinary lane, exact ordered group bindings, repeated rest/pair cells,
two photographed beams and matching cross-staff columns are all required.
No score labels, identifiers, renderer coordinates or learned models are used.
"""
import copy,hashlib
from collections import Counter,defaultdict
from fractions import Fraction
from math import lcm
from pathlib import Path
import xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from repair_tuplets import continuous_beams,scale_divisions
from repair_grace_timing import _rebuild
from .symbol_cnn import FACTOR


TYPES={'16th':Fraction(1,4),'eighth':Fraction(1,2),'quarter':Fraction(1)}


def written(group):
    values=[]
    for r in group:
        n=r['node'];kind=TYPES.get(n.findtext('type'))
        dots=len(n.findall('dot'))
        if kind is None or dots>1:return None
        values.append(kind*(Fraction(3,2) if dots else 1))
    return values[0] if len(set(values))==1 else None


def group_events(events):
    if not events or len(events)>128 or len({r['voice'] for r in events})!=1:return None
    groups=[]
    for r in events:
        if (r['grace'] or r['node'].get('print-object')=='no'
            or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None):return None
        if r['chord']:
            if not groups or groups[-1][0]['onset']!=r['onset']:return None
            groups[-1].append(r)
        else:groups.append([r])
    if len(groups)>64 or any(len(g)>8 or len({r['duration'] for r in g})!=1 for g in groups):return None
    return groups


def matched(groups,seq,row,staff,image):
    if len(groups)!=len(seq) or not groups or any(g.get('grace') for g in seq):return False
    if len({g.get('voice',1) for g in seq})!=1 or any(b['x']-a['x']<.65*row['space_px'] for a,b in zip(seq,seq[1:])):return False
    tokens=[]
    for g in seq:
        if g.get('rest'):
            if g['rest'].get('confidence',0)<.55:return False
            tokens.append(Counter([None]));continue
        heads=g.get('heads',[])
        if not heads or len(heads)>8:return False
        for h in heads:
            if h.get('class')!=0 or h.get('confidence',0)<.7:return False
            x,y=map(round,h['xy']);rx=max(2,round(.3*row['space_px']));ry=max(2,round(.15*row['space_px']))
            if min(x-rx,y-ry)<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
            if (image[y-ry:y+ry+1,x-rx:x+rx+1]<140).mean()<.8:return False
        tokens.append(Counter((h['pitch'][1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[h['pitch'][0]]+h['pitch'][2] for h in heads))
    if tokens==[Counter(r['midi'] for r in g) for g in groups]:return True
    # Spelling discrepancies never authorize pitch edits. Staff tracks only
    # prove the same physical written heads and cardinalities for clock edits.
    from .relation_tuplets import geometry_match
    return geometry_match(groups,seq,row,staff,image)


def beam_direction(a,b,ink,space,count,maximum_gap=4):
    choices=[]
    for up in (True,False):
        edge=min if up else max
        pair=[edge(g['heads'],key=lambda h:h['xy'][1]) for g in (a,b)]
        mapped=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=space) for h in pair]
        if continuous_beams(*mapped,ink,up,count,maximum_gap=maximum_gap):choices.append(up)
    return choices[0] if len(choices)==1 else None


def proposal(target,anchor,target_seq,anchor_seq,space,ink,expected):
    """Unique bounded hypothesis with independently conserved partner clocks."""
    if len(target)<9 or len(anchor)<3 or len(target)!=len(target_seq) or len(anchor)!=len(anchor_seq):return None
    tw=[written(g) for g in target];aw=[written(g) for g in anchor]
    if any(w!=Fraction(1,4) for w in tw) or any(w is None for w in aw):return None
    if sum(tw)!=expected*Fraction(3,2) or sum(aw)!=expected:return None
    if any(w not in (Fraction(1,2),Fraction(3,2)) for w in aw):return None
    for i,(w,g) in enumerate(zip(aw,anchor_seq)):
        if (not g.get('heads') or g.get('duration')!=w*480
            or bool(g.get('dotted'))!=(w==Fraction(3,2))):return None
        if w==Fraction(3,2) and (i!=len(aw)-1 or g.get('beams') or g.get('flags') or g.get('stem') is None):return None
    for group,w in zip(anchor,aw):
        if any(r['duration']!=w or r['node'].find('time-modification') is not None for r in group):return None
    for group in target:
        for r in group:
            tm=r['node'].find('time-modification')
            if tm is None:
                if r['duration']!=Fraction(1,4):return None
            elif (tm.findtext('actual-notes')!='3' or tm.findtext('normal-notes')!='2' or r['duration']!=Fraction(1,6)):return None
    rests=[i for i,g in enumerate(target_seq) if g.get('rest')]
    if any(target_seq[i]['rest'].get('class')!=6 for i in rests):return None
    if len(rests)<3 or any(i%3 or i+2>=len(target_seq) or target_seq[i+1].get('rest') or target_seq[i+2].get('rest') for i in rests):return None
    supported=set();links=0
    for i,(a,b) in enumerate(zip(target_seq,target_seq[1:])):
        if not a.get('heads') or not b.get('heads'):continue
        # A new cell may change stem direction and beam chain. Check only
        # pairs within each triplet; beam-chain predictions alone are weak.
        if i%3==2:continue
        direction=beam_direction(a,b,ink,space,2)
        if direction is None:return None
        supported.update((i,i+1));links+=1
    if links<3 or any(i not in supported for i,g in enumerate(target_seq) if g.get('heads')):return None
    # The partner's beam proof samples seven columns over larger spacings.
    anchor_supported=set();anchor_links=0
    for i,(a,b) in enumerate(zip(anchor_seq,anchor_seq[1:])):
        if aw[i]!=Fraction(1,2) or aw[i+1]!=Fraction(1,2):continue
        if a.get('beam_chain')!=b.get('beam_chain') or a.get('beams')!=1 or b.get('beams')!=1:continue
        if beam_direction(a,b,ink,space,1,maximum_gap=8) is None:return None
        anchor_supported.update((i,i+1));anchor_links+=1
    if anchor_links<2 or any(w==Fraction(1,2) and i not in anchor_supported for i,w in enumerate(aw)):return None
    # Every independently written partner onset must align with exactly one
    # corrected triplet boundary. No x-distance interpolation is used.
    clock=Fraction(0);columns=[]
    for w,g in zip(aw,anchor_seq):
        hits=[i for i,h in enumerate(target_seq) if abs(h['x']-g['x'])<=.3*space]
        if len(hits)!=1 or hits[0]%3 or Fraction(hits[0],6)!=clock:return None
        columns.append(hits[0]);clock+=w
    if len(columns)<3 or len(set(columns))!=len(columns) or columns[0]!=0:return None
    # Check the not-yet-timed tail as well as the aligned start columns.
    if Fraction(len(target),6)!=clock or clock!=expected:return None
    return {'target_durations':[Fraction(1,6)]*len(target),'anchor_durations':aw,
            'cross_staff_columns':len(columns),'target_beam_links':links,'anchor_beam_links':anchor_links}


def fuse(root,graph,photo,geometry):
    if len(root.findall('part'))!=1 or root.find('.//octave-shift') is not None:return []
    digest=hashlib.sha256(Path(photo).read_bytes()).hexdigest()
    if graph.get('image_sha256')!=digest or geometry.get('image_sha256')!=digest:raise ValueError('Joint rhythm photo mismatch')
    systems=graph['systems'];ids=sorted({r['system'] for r in systems})
    if len(ids)>64 or any(len(r['groups'])>2048 or len(r['bounds'])>129 for r in systems):return []
    measures=root.find('part').findall('measure');rows=records(root)
    if sum(len(next(r for r in systems if r['system']==s)['bounds'])-1 for s in ids)!=len(rows):return []
    raw=cv2.imread(str(photo),0)
    if raw is None or raw.size>16000000 or max(raw.shape)>8192:return []
    image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA);del raw
    ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1]
    from .testset3_staff_resize_fast import resize_staves
    staffs=resize_staves(geometry['staves'],FACTOR,FACTOR)
    divisions=1;meter=None;divs=[];meters=[]
    for m in measures:
        for a in m.findall('attributes'):
            divisions=int(a.findtext('divisions',str(divisions)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type')
                meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        divs.append(divisions);meters.append(meter)
    jobs=[];offset=0
    for system in ids:
        selected=sorted([r for r in systems if r['system']==system],key=lambda r:r['staff'])
        # This first implementation considers exactly two staves. Extra
        # lanes, mismatched bar boundaries and inline attributes abstain.
        if len(selected)!=2 or any(r['bounds']!=selected[0]['bounds'] for r in selected):
            offset+=len(selected[0]['bounds'])-1;continue
        tracks=[s for s in staffs if s['system']==system]
        for mi in range(len(selected[0]['bounds'])-1):
            number=offset+mi;expected=meters[number];m=measures[number]
            if expected is None or m.get('implicit')=='yes' or len(rows[number])>128 or not rows[number]:continue
            ns=[i for i,n in enumerate(m) if n.tag=='note']
            if ns and any(n.tag in ('backup','forward') for n in list(m)[:ns[0]]):continue
            if not ns or any(n.tag not in ('note','backup','forward') for n in list(m)[ns[0]:ns[-1]+1]):continue
            if any(n.tag not in ('barline',) for n in list(m)[ns[-1]+1:]):continue
            bindings=[]
            for row in selected:
                es=[r for r in rows[number] if r['staff']==row['staff']];groups=group_events(es)
                l,u=row['bounds'][mi:mi+2];seq=sorted([g for g in row['groups'] if l<g['x']<u],key=lambda g:g['x'])
                track=tracks[row['staff']-1] if len(tracks)>=row['staff'] else None
                if groups is None or not matched(groups,seq,row,track,image):break
                bindings.append((groups,seq,row['space_px']))
            if len(bindings)!=2 or len({r['staff'] for r in rows[number]})!=2:continue
            choices=[]
            for a,b in ((0,1),(1,0)):
                target,tseq,space=bindings[a];anchor,aseq,_=bindings[b]
                p=proposal(target,anchor,tseq,aseq,space,ink,expected)
                if p:choices.append((a,b,p))
            if len(choices)==1:jobs.append((number,bindings,*choices[0]))
        offset+=len(selected[0]['bounds'])-1
    if not jobs or len(jobs)>128:return []
    factor=1
    for number,bindings,a,b,p in jobs:
        for d in p['target_durations']+p['anchor_durations']:factor=lcm(factor,(d*divs[number]).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for number,bindings,a,b,p in jobs:
        m=measures[number];saved=copy.deepcopy(list(m));div=divs[number]*factor
        desired={r['node']:int(r['onset']*div) for r in rows[number]};changed=0
        for index,key in ((a,'target_durations'),(b,'anchor_durations')):
            cursor=0
            for group,d in zip(bindings[index][0],p[key]):
                for r in group:
                    n=r['node'];changed+=r['onset']!=Fraction(cursor,div) or r['duration']!=d
                    desired[n]=cursor;n.find('duration').text=str(int(d*div))
                    if index==a and n.find('time-modification') is None:
                        tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text='3';E.SubElement(tm,'normal-notes').text='2';E.SubElement(tm,'normal-type').text='16th'
                        at=next((i for i,c in enumerate(n) if c.tag in ('stem','notehead','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
                cursor+=int(d*div)
        if not changed or not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'method':'implicit_triplets_with_complete_partner_and_ink_beams','changed_events':changed,
                        'target_staff':bindings[a][0][0][0]['staff'],'partner_staff':bindings[b][0][0][0]['staff'],
                        'cross_staff_columns':p['cross_staff_columns'],'target_beam_links':p['target_beam_links'],'partner_beam_links':p['anchor_beam_links'],
                        'pitch_voice_cardinality_preserved':True})
    return changes
