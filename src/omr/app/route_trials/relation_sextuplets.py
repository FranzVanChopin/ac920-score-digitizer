"""Six printed sixteenths: numeral, two ink beams, staff and meter agree."""
import base64,copy,json
from functools import lru_cache
from pathlib import Path
from fractions import Fraction
from collections import Counter
from math import lcm
import cv2,numpy as np
import xml.etree.ElementTree as E
from .relation_coverage import inputs,core,ordered,safe_measure
from .relation_tuplets import geometry_match
from .relation_ensemble import flow_health
from repair_tuplets import printed_number,scale_divisions
from repair_unison_voices import _stem_end
from repair_grace_timing import _rebuild
from omr_event_records import records


@lru_cache(maxsize=1)
def templates():
    rows=json.loads(Path(__file__).resolve().parents[1].joinpath('omr_sextuplet_templates.json').read_text(encoding='utf8'))['glyphs']
    for r in rows:r['mask']=np.unpackbits(np.frombuffer(base64.b64decode(r['bits']),dtype='uint8')).reshape(64,48)
    return rows


def classify(mask):
    small=cv2.resize(mask,(48,64),interpolation=cv2.INTER_NEAREST)>0;dil=cv2.dilate(small.astype('uint8'),np.ones((3,3),np.uint8))>0;ratio=mask.shape[1]/mask.shape[0];scores={}
    for r in templates():
        ref=r['mask']>0;rd=cv2.dilate(r['mask'],np.ones((3,3),np.uint8))>0;score=((small&rd).sum()/max(1,small.sum())+(ref&dil).sum()/max(1,ref.sum()))/2-.35*abs(ratio-r['ratio']);scores[r['number']]=max(scores.get(r['number'],0),float(score))
    ranked=sorted(scores.items(),key=lambda x:x[1],reverse=True)
    return ranked[0][0] if ranked[0][1]>=.86 and ranked[0][1]-ranked[1][1]>=.08 else None


def double_beams(a,b,ink,up):
    """Broader outer-side search excludes thin staff ink and numeral below."""
    space=a['staff_space'];ends=[_stem_end(h,ink,up) for h in (a,b)]
    if any(e is None for e in ends):return False
    (xa,ya),(xb,yb)=ends
    if not .7*space<xb-xa<4*space or abs(ya-yb)>space:return False
    narrow=0
    for f in (.25,.5,.75):
        x=round(xa+(xb-xa)*f);y=ya+(yb-ya)*f;top=max(0,round(y-2*space));bottom=min(ink.shape[0],round(y+2*space)+1);bits=ink[top:bottom,x]>0;edges=np.diff(np.r_[False,bits,False].astype('int8'));spans=[]
        for lo,hi in zip(np.where(edges==1)[0],np.where(edges==-1)[0]):
            centre=top+(lo+hi-1)/2
            if .2*space<=hi-lo<=.8*space and (centre>=y-.25*space if up else centre<=y+.25*space):spans.append((top+lo,top+hi))
        if len(spans)!=2:return False
        gap=spans[1][0]-spans[0][1]
        if not 1<=gap<=.6*space:return False
        if gap<.1*space:narrow+=1
        outer=(spans[0][0]+spans[0][1]-1)/2 if up else (spans[-1][0]+spans[-1][1]-1)/2
        if abs(outer-y)>.55*space:return False
    return narrow<=1


def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];events=records(root);health=flow_health(root);ms=root.findall('part/measure');off=0;div=1;divs=[];known=False;meters=[];jobs=[]
    for m in ms:
        for a in m.findall('attributes'):
            div=int(a.findtext('divisions',str(div)))
            if a.find('time') is not None:known=a.findtext('time/beats') is not None and a.findtext('time/beat-type') is not None
        divs.append(div);meters.append(known)
    for system in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in rows:
            sn=row['staff']
            if sn>len(staffs):continue
            track=staffs[sn-1]
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=off+mi;es=[r for r in events[number] if r['staff']==sn];seq=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x']);old=ordered(es);expected=Fraction(health[number]['meter'][0]*4,health[number]['meter'][1])
                if not meters[number] or not safe_measure(events[number]) or not old or not 6<=len(seq)<=64 or len(old)!=len(seq) or len({r['voice'] for r in es})!=1 or any(r['midi'] is None or r['chord'] for r in es):continue
                seen=False;inline=False
                for child in ms[number]:
                    if child.tag=='note':seen=True
                    elif child.tag=='attributes' and seen:inline=True
                if inline or not geometry_match(old,seq,row,track,image) or any(g.get('grace') or g.get('rest') or g.get('dotted') or g.get('flags') or len(g.get('heads',[]))!=1 for g in seq):continue
                choices=[]
                for start in range(len(seq)-5):
                    window=seq[start:start+6]
                    if any(g['heads'][0].get('confidence',0)<.9 or not core(g['heads'][0],image,row['space_px'],.95) or g.get('tuplet_ratio') for g in window):continue
                    heads=[dict(x=g['heads'][0]['xy'][0],y=g['heads'][0]['xy'][1],staff_space=row['space_px']) for g in window]
                    if printed_number(heads,ink,6,recover_crossings=True,classify_fn=classify,beam_proof_fn=double_beams):choices.append(start)
                if len(choices)!=1:continue
                start=choices[0];durations=[Fraction(1,6) if start<=i<start+6 else group[0]['duration'] for i,group in enumerate(old)]
                if sum(durations)!=expected or any(r['node'].find('time-modification') is not None for group in old[start:start+6] for r in group):continue
                cursor=Fraction(0);valid=True
                for i,(group,g,duration) in enumerate(zip(old,seq,durations)):
                    if not start<=i<start+6 and (g['duration']!=group[0]['duration']*480 or group[0]['onset']!=cursor):valid=False;break
                    cursor+=duration
                if valid and any(group[0]['duration']!=duration for group,duration in zip(old,durations)):jobs.append((number,sn,old,durations,start))
        off+=len(rows[0]['bounds'])-1
    if not jobs or len(jobs)>128:return []
    factor=1
    for number,sn,old,durations,start in jobs:
        for duration in durations:factor=lcm(factor,(duration*divs[number]).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for number,sn,old,durations,start in jobs:
        m=ms[number];saved=copy.deepcopy(list(m));div=divs[number]*factor;desired={r['node']:int(r['onset']*div) for r in events[number]};cursor=Fraction(0)
        # A new begin requires the preceding open beam to end. Otherwise
        # importers can join the ordinary prefix to the six-note tuplet.
        if start:
            for beam in old[start-1][0]['node'].findall('beam'):
                if beam.get('number','1') in ('1','2') and beam.text=='continue':beam.text='end'
        for i,(group,duration) in enumerate(zip(old,durations)):
            n=group[0]['node'];desired[n]=int(cursor*div);n.find('duration').text=str(int(duration*div));cursor+=duration
            if start<=i<start+6:
                n.find('type').text='16th';tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text='6';E.SubElement(tm,'normal-notes').text='4';E.SubElement(tm,'normal-type').text='16th';at=next((j for j,child in enumerate(n) if child.tag in ('stem','staff','beam','notations')),len(n));n.insert(at,tm)
                for beam in n.findall('beam'):n.remove(beam)
                for layer in (1,2):
                    beam=E.Element('beam',number=str(layer));beam.text='begin' if i==start else 'end' if i==start+5 else 'continue';at=next((j for j,child in enumerate(n) if child.tag in ('notations','lyric')),len(n));n.insert(at,beam)
                if i in (start,start+5):
                    notations=n.find('notations')
                    if notations is None:notations=E.SubElement(n,'notations')
                    E.SubElement(notations,'tuplet',type='start' if i==start else 'stop',number='1',bracket='no')
        if not _rebuild(m,desired):m[:]=saved;continue
        after=records(root)[number];before_partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in events[number] if r['staff']!=sn);after_partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in after if r['staff']!=sn)
        if before_partner!=after_partner:m[:]=saved;continue
        changes.append({'measure':number+1,'staff':sn,'start_group':start,'groups':6,'ratio':[6,4],'method':'printed_six_and_continuous_double_beams','other_staff_clocks_preserved':True})
    return changes
