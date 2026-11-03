"""Printed 2 plus two hooked bracket ends and one complete lane: 2:3 clocks.

No model/reference input, dot invention, piece-name or known-bar gate. Raster
coordinates must equal the supplied geometry frame. A chord is one group.
"""
import base64
from collections import Counter, defaultdict
import copy
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from math import lcm
from pathlib import Path
from statistics import median
import xml.etree.ElementTree as E
import cv2
import numpy as np

MAX_PIXELS=16_000_000
MAX_NOTES=128
MAX_STAFF_ROWS=128
MAX_JOBS=128
MAX_ROI_PIXELS=1_000_000


@lru_cache(maxsize=1)
def templates():
    rows=json.loads(Path(__file__).with_name('omr_duplet_templates_trial.json').read_text(encoding='utf8'))['glyphs']
    for r in rows:r['mask']=np.unpackbits(np.frombuffer(base64.b64decode(r['bits']),dtype=np.uint8)).reshape(64,48)
    return rows


def classify(mask, family):
    if mask.size==0 or max(mask.shape)>512:return None
    ratio=mask.shape[1]/mask.shape[0];small=cv2.resize(mask.astype('uint8'),(48,64),interpolation=cv2.INTER_NEAREST)>0
    dil=cv2.dilate(small.astype('uint8'),np.ones((3,3),np.uint8))>0;scores={}
    for r in templates():
        if not r['label'].startswith(family):continue
        ref=r['mask']>0;rd=cv2.dilate(r['mask'],np.ones((3,3),np.uint8))>0
        score=((small&rd).sum()/max(1,small.sum())+(ref&dil).sum()/max(1,ref.sum()))/2-.35*abs(ratio-r['ratio'])
        scores[r['label']]=max(scores.get(r['label'],0),float(score))
    ranked=sorted(scores.items(),key=lambda p:(-p[1],p[0]))
    if not ranked or ranked[0][1]<.86 or len(ranked)>1 and ranked[0][1]-ranked[1][1]<.08:return None
    return dict(label=ranked[0][0],score=ranked[0][1],margin=ranked[0][1]-ranked[1][1] if len(ranked)>1 else 1.)


def _hook(raw,x,y,side,space):
    cx=int(round(x));cy=int(round(y));radius=max(1,round(.12*space));limit=round(1.2*space)
    if cx-radius<0 or cx+radius>=raw.shape[1] or cy-limit<0 or cy+limit>=raw.shape[0]:return 0
    length=0
    for distance in range(1,limit+1):
        if not np.any(raw[cy+side*distance,cx-radius:cx+radius+1]):break
        length+=1
    return length if .3*space<=length<=1.2*space else 0


def detect_brackets(image,staff):
    """Bounded staff bands, generic digit shapes, two thin arms and two hooks."""
    results=[];grid=staff.get('grid',[])
    if not grid or len(grid[0]['y']) not in (5,10):return []
    point=min(grid,key=lambda p:abs(p['x']-sum(staff['x_range'])/2));left,right=map(round,staff['x_range'])
    for sn in range(1,len(point['y'])//5+1):
        lines=point['y'][(sn-1)*5:sn*5];space=median(b-a for a,b in zip(lines,lines[1:]))
        if not 5<=space<=40:continue
        for side,anchor in ((-1,lines[0]),(1,lines[-1])):
            top,bottom=sorted((round(anchor-.75*space*side),round(anchor+4*space*side)))
            l,r=max(0,left),min(image.shape[1],right);t,b=max(0,top),min(image.shape[0],bottom)
            if r<=l or b<=t or (r-l)*(b-t)>MAX_ROI_PIXELS:continue
            raw=(image[t:b,l:r]<140).astype('uint8')
            long=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,max(3,round(12*space))),np.uint8))
            clean=cv2.subtract(raw,long)
            horizontal=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((1,max(3,round(.8*space))),np.uint8))
            _,_,arms,_=cv2.connectedComponentsWithStats(horizontal,8)
            arms=[(int(x),int(y),int(x+w),int(y+h)) for x,y,w,h,_ in arms[1:] if space<=w<=12*space and h<=.22*space]
            _,labels,stats,_=cv2.connectedComponentsWithStats(clean,8)
            for ci,(x,y,w,h,_) in enumerate(stats[1:],1):
                x,y,w,h=map(int,(x,y,w,h))
                if not .4*space<=w<=1.5*space or not .7*space<=h<=2*space:continue
                glyph=classify((labels[y:y+h,x:x+w]==ci).astype('uint8'),'digit')
                if not glyph or glyph['label']!='digit2':continue
                combinations=[]
                for a in arms:
                    if not x-.9*space<=a[2]<=x+.12*space:continue
                    for d in arms:
                        if not x+w-.12*space<=d[0]<=x+w+.9*space:continue
                        ay=(a[1]+a[3]-1)/2;dy=(d[1]+d[3]-1)/2
                        if abs(ay-dy)>.3*space or not 2*space<=d[2]-a[0]<=16*space:continue
                        # Hooks point towards the associated staff.
                        ha=_hook(clean,a[0],ay,-side,space);hd=_hook(clean,d[2]-1,dy,-side,space)
                        if not ha or not hd:continue
                        combinations.append((a,d,ha,hd))
                if len(combinations)!=1:continue
                a,d,ha,hd=combinations[0]
                results.append(dict(staff=sn,space=float(space),left=l+a[0],right=l+d[2]-1,
                    y=t+(a[1]+a[3]+d[1]+d[3]-2)/4,side=side,digit_bbox=[l+x,t+y,l+x+w,t+y+h],
                    digit=glyph,hook_lengths=[ha,hd],lines=lines,roi_bbox=[l,t,r,b]))
    return results


def _rest_positions(image,bracket,right):
    s=bracket['space'];lines=bracket['lines'];l=max(0,round(bracket['left']-.3*s));r=min(image.shape[1],round(right-.7*s))
    t=max(0,round(lines[0]-s));b=min(image.shape[0],round(lines[-1]+s))
    if r-l<1 or b-t<1 or (r-l)*(b-t)>MAX_ROI_PIXELS:return []
    raw=(image[t:b,l:r]<140).astype('uint8')
    long=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,max(3,round(3*s))),np.uint8));clean=cv2.subtract(raw,long)
    closed=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((3,1),np.uint8));clean=np.maximum(clean,closed&raw)
    _,labels,stats,_=cv2.connectedComponentsWithStats(clean,8);found=[]
    for ci,(x,y,w,h,_) in enumerate(stats[1:],1):
        x,y,w,h=map(int,(x,y,w,h))
        if not .35*s<=w<=1.8*s or not 1*s<=h<=3*s:continue
        glyph=classify((labels[y:y+h,x:x+w]==ci).astype('uint8'),'rest')
        if glyph and glyph['label']=='rest_eighth':found.append(dict(x=l+x+w/2,y=t+y+h/2,bbox=[l+x,t+y,l+x+w,t+y+h],glyph=glyph))
    return found


def _core(image,x,y,s):
    l,r=round(x-.32*s),round(x+.32*s)+1;t,b=round(y-.17*s),round(y+.17*s)+1
    return l>=0 and t>=0 and r<=image.shape[1] and b<=image.shape[0] and float((image[t:b,l:r]<140).mean())>=.8


def _anchor(group,mapped,image):
    from align_homr_geometry import _position
    matched=[(r['node'],mapped[r['node']]) for r in group if r['node'] in mapped]
    if len(matched)<(1 if len(group)==1 else 2):return None
    s=median(h['staff_space'] for _,h in matched);x=median(h['x'] for _,h in matched)
    if any(abs(h['x']-x)>.4*s or _position(n) is None or _position(n)[1]!=h['position'] or not _core(image,h['x'],h['y'],s) for n,h in matched):return None
    bottoms=[h['y']+(_position(n)[1]-1)*s/2 for n,h in matched]
    bottom=median(bottoms)
    if max(bottoms)-min(bottoms)>.4*s:return None
    for r in group:
        if r['node'] not in mapped:
            pos=_position(r['node'])
            if pos is None or not _core(image,x,bottom-(pos[1]-1)*s/2,s):return None
    return dict(x=x,space=s,matched_heads=len(matched),optical_heads=len(group))


def _groups(rows):
    lanes=defaultdict(list)
    for r in rows:
        key=(r['staff'],r['voice'])
        if r['chord']:
            if not lanes[key] or lanes[key][-1][0]['onset']!=r['onset']:return None
            lanes[key][-1].append(r)
        else:lanes[key].append([r])
    return lanes


def _serialize(measure,desired,old_units,old_durations):
    """Retain node order/content; attach mid-measure clef context to its note."""
    nodes=list(measure);indices=[i for i,n in enumerate(nodes) if n.tag=='note'];first,last=indices[0],indices[-1]
    if any(n.tag not in ('note','backup','forward','attributes') for n in nodes[first:last+1]) or any(n.tag!='barline' for n in nodes[last+1:]):return False
    if any(n.find('divisions') is not None or n.find('time') is not None for n in nodes[first:last+1] if n.tag=='attributes'):return False
    old_cursor=0;old_times={}
    for n in nodes[first:last+1]:
        old_times[n]=old_cursor
        if n.tag=='backup':old_cursor-=int(n.findtext('duration','0'))
        elif n.tag=='forward':old_cursor+=int(n.findtext('duration','0'))
        elif n.tag=='note' and n.find('chord') is None:old_cursor=old_units[n]+old_durations[n]
    rebuilt=nodes[:first];cursor=0;previous=None
    def goto(target):
        nonlocal cursor
        if target<0:return False
        if target!=cursor:
            shift=E.Element('forward' if target>cursor else 'backup');E.SubElement(shift,'duration').text=str(abs(target-cursor));rebuilt.append(shift);cursor=target
        return True
    for i,n in enumerate(nodes[first:last+1],first):
        if n.tag in ('backup','forward'):continue
        if n.tag=='attributes':
            nxt=next((a for a in nodes[i+1:last+1] if a.tag=='note'),None)
            target=desired[nxt] if nxt is not None and old_times[n]==old_units[nxt] else old_times[n]
            if not goto(target):return False
            rebuilt.append(n);continue
        onset=desired[n]
        if n.find('chord') is None:
            if not goto(onset):return False
            previous=onset;cursor+=int(n.findtext('duration','0'))
        elif onset!=previous:return False
        rebuilt.append(n)
    measure[:]=rebuilt+nodes[last+1:];return True


def repair(root,geometry,image,initial_meter=None):
    from repair_augmentation_dots import _head_map
    from repair_duplicate_noteheads import _target_by_measure
    from align_homr_geometry import _xml_systems
    from omr_event_records import records
    from repair_tuplets import scale_divisions
    if image.ndim!=2 or image.dtype!=np.uint8 or image.size>MAX_PIXELS:raise ValueError('Bounded grayscale image required')
    if list(image.shape[::-1])!=geometry.get('preprocessed_size'):raise ValueError('Image must equal geometry pixel frame')
    if len(root.findall('part'))!=1:return dict(changes=[],abstain='single_part_required')
    if len(geometry.get('systems',[]))*2>MAX_STAFF_ROWS:return dict(changes=[],abstain='staff_row_budget')
    working=copy.deepcopy(root);systems=_xml_systems(working);mapped=_head_map(working,geometry);rows=records(working)
    if len(systems)!=len(geometry.get('systems',[])):return dict(changes=[],abstain='system_alignment')
    targets=_target_by_measure(working,initial_meter);system_of={n:si for si,ns in enumerate(systems) for n in ns}
    brackets={si:detect_brackets(image,sy['staffs'][0]) for si,sy in enumerate(geometry['systems']) if len(sy.get('staffs',[]))==1}
    jobs=[];factor=1;diagnostic=Counter()
    for mi,(measure,events) in enumerate(zip(working.findall('part/measure'),rows),1):
        target=targets.get(mi)
        if not target or target>256 or measure.get('implicit')=='yes' or not events or len(events)>MAX_NOTES or any(r['grace'] for r in events):continue
        pitched=[r['node'] for r in events if r['midi'] is not None];sis={system_of[n] for n in pitched if n in system_of}
        if len(sis)!=1:continue
        si=next(iter(sis));lanes=_groups(events)
        if lanes is None:continue
        proposals=[]
        for br in brackets.get(si,[]):
            options=[]
            for lane,groups in lanes.items():
                if lane[0]!=br['staff']:continue
                cursor=Fraction(0);continuous=True
                for group in groups:
                    if group[0]['onset']!=cursor or any(r['duration']!=group[0]['duration'] for r in group):continuous=False;break
                    cursor+=group[0]['duration']
                # Targets are native XML units; records are exact quarters.
                div=next((int(a.findtext('divisions')) for m in reversed(working.findall('part/measure')[:mi]) for a in reversed(m.findall('attributes')) if a.find('divisions') is not None),1)
                if not continuous or cursor>=Fraction(target,div):continue
                anchored=[]
                for gi,group in enumerate(groups):
                    if group[0]['midi'] is not None:
                        a=_anchor(group,mapped,image)
                        if a and br['left']-.25*br['space']<=a['x']<=br['right']+.25*br['space']:anchored.append((gi,a))
                if len(anchored)!=1:continue
                gi,anchor=anchored[0]
                if gi<1 or len(groups[gi-1])!=1 or groups[gi-1][0]['midi'] is not None:continue
                pair=groups[gi-1:gi+1]
                if any(r['node'].findtext('type')!='eighth' or r['node'].findall('dot') or r['node'].find('time-modification') is not None for group in pair for r in group):continue
                if pair[0][0]['duration']!=pair[1][0]['duration']:continue
                rests=_rest_positions(image,br,anchor['x'])
                if len(rests)!=1:continue
                addition=sum((group[0]['duration']/2 for group in pair),Fraction(0))
                if cursor+addition!=Fraction(target,div):continue
                options.append(dict(lane=lane,groups=groups,indices=[gi-1,gi],bracket=br,rest=rests[0],anchor=anchor,divisions=div))
            if len(options)==1:proposals.append(options[0])
            elif options:diagnostic['ambiguous_bracket_owner']+=1
        if not proposals:continue
        if len({p['lane'] for p in proposals})!=len(proposals):diagnostic['multiple_scopes_same_lane']+=1;continue
        if len(jobs)>=MAX_JOBS:diagnostic['job_budget']+=1;break
        for p in proposals:
            for gi in p['indices']:factor=lcm(factor,(p['groups'][gi][0]['duration']*p['divisions']*Fraction(3,2)).denominator)
        jobs.append((mi,measure,events,proposals))
    if factor>12:return dict(changes=[],abstain='division_factor_budget')
    if not jobs:return dict(changes=[],diagnostics=dict(diagnostic),brackets_detected=sum(map(len,brackets.values())),reference_used=False,model_calls=0)
    before=[Counter((r['staff'],r['voice'],r['midi'],r['duration'],r['onset'],r['grace']) for r in row) for row in records(working)]
    scale_divisions(working,factor);changes=[]
    for mi,measure,events,proposals in jobs:
        div=proposals[0]['divisions']*factor;desired={r['node']:int(r['onset']*div) for r in events};old_units=dict(desired)
        old_durations={r['node']:int(r['duration']*div) for r in events}
        for p in proposals:
            for gi in p['indices']:
                for r in p['groups'][gi]:
                    n=r['node'];n.find('duration').text=str(int(r['duration']*div*Fraction(3,2)))
                    tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text='2';E.SubElement(tm,'normal-notes').text='3';E.SubElement(tm,'normal-type').text='eighth'
                    at=next((i for i,e in enumerate(n) if e.tag in ('stem','notehead','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
            cursor=0
            for group in p['groups']:
                for r in group:desired[r['node']]=cursor
                cursor+=int(group[0]['node'].findtext('duration'))
            for gi,kind in zip(p['indices'],('start','stop')):
                n=p['groups'][gi][0]['node'];notation=n.find('notations')
                if notation is None:notation=E.SubElement(n,'notations')
                E.SubElement(notation,'tuplet',type=kind,number=str(len(changes)+1),bracket='yes')
        if not _serialize(measure,desired,old_units,old_durations):return dict(changes=[],abstain='unsupported_attribute_or_clock_serialization',input_unchanged=True)
        changes.append(dict(measure_index=mi-1,measure_number=measure.get('number'),
            scopes=[dict(staff=p['lane'][0],voice=p['lane'][1],group_indices=p['indices'],bracket=p['bracket'],rest=p['rest'],anchor=p['anchor'],ratio=[2,3]) for p in proposals]))
    after=records(working);affected={mi-1:{p['lane'] for p in ps} for mi,m,es,ps in jobs}
    for i,row in enumerate(after):
        if i not in affected:assert before[i]==Counter((r['staff'],r['voice'],r['midi'],r['duration'],r['onset'],r['grace']) for r in row)
        else:
            old_other=Counter({k:v for k,v in before[i].items() if (k[0],k[1]) not in affected[i]})
            new_other=Counter((r['staff'],r['voice'],r['midi'],r['duration'],r['onset'],r['grace']) for r in row if (r['staff'],r['voice']) not in affected[i])
            assert old_other==new_other
        assert Counter((r['staff'],r['voice'],r['midi'],r['grace']) for r in rows[i])==Counter((r['staff'],r['voice'],r['midi'],r['grace']) for r in row)
    root.clear();root.attrib.update(working.attrib);root.text=working.text;root.extend(list(working))
    return dict(changes=changes,diagnostics=dict(diagnostic),brackets_detected=sum(map(len,brackets.values())),
        division_factor=factor,original_pitch_staff_voice_cardinality_preserved=True,other_lane_clocks_preserved=True,
        no_augmentation_dots_added=True,reference_used=False,model_calls=0)
