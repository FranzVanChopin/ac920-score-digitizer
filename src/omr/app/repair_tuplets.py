"""Read printed 3/4 tuplet numbers on complete single-voice eighth beams.

The numeral, continuous single beams, unique head mapping and exact printed
meter must agree. Integer MusicXML divisions are scaled before writing thirds.
No reference score or additional model is required.
"""
import base64
from fractions import Fraction
from decimal import Decimal
from functools import lru_cache
import json
from math import lcm
from pathlib import Path
import xml.etree.ElementTree as ET

from repair_augmentation_dots import _head_map
from repair_duplicate_noteheads import _target_by_measure
from repair_grace_timing import _rebuild
from repair_ledger_chords import _groups
from repair_pitch_context import _ink
from repair_unison_voices import _single_beam,_stem_end
from align_homr_geometry import _position


@lru_cache(maxsize=1)
def templates():
    import numpy as np
    rows=json.loads(Path(__file__).with_name('omr_tuplet_templates.json').read_text(encoding='utf8'))['glyphs']
    for r in rows:r['mask']=np.unpackbits(np.frombuffer(base64.b64decode(r['bits']),dtype='uint8')).reshape(64,48)
    return rows


def classify(mask):
    import cv2
    import numpy as np
    ratio=mask.shape[1]/mask.shape[0];small=cv2.resize(mask,(48,64),interpolation=cv2.INTER_NEAREST)>0
    dil=cv2.dilate(small.astype('uint8'),np.ones((3,3),dtype='uint8'))>0;scores={}
    for r in templates():
        ref=r['mask']>0;rd=cv2.dilate(r['mask'],np.ones((3,3),dtype='uint8'))>0
        score=((small&rd).sum()/max(1,small.sum())+(ref&dil).sum()/max(1,ref.sum()))/2-.35*abs(ratio-r['ratio'])
        scores[r['number']]=max(scores.get(r['number'],0),float(score))
    ranked=sorted(scores.items(),key=lambda p:p[1],reverse=True)
    return ranked[0][0] if ranked[0][1]>=.86 and ranked[0][1]-ranked[1][1]>=.08 else None


def continuous_beams(a,b,ink,up,count,maximum_gap=4):
    """Exact thick-band count at three interior columns, including raster seams."""
    import numpy as np
    s=a['staff_space'];ends=[_stem_end(h,ink,up) for h in (a,b)]
    if count not in (1,2) or not 4<=maximum_gap<=8 or any(e is None for e in ends):return False
    (xa,ya),(xb,yb)=ends
    if not .7*s<=xb-xa<=maximum_gap*s or abs(ya-yb)>s:return False
    fractions=(.125,.25,.375,.5,.625,.75,.875) if maximum_gap>4 else (.25,.5,.75)
    for fraction in fractions:
        x=round(xa+(xb-xa)*fraction);y=ya+(yb-ya)*fraction
        radius=1.35
        top=max(0,round(y-radius*s));bottom=min(ink.shape[0],round(y+radius*s)+1)
        bits=ink[top:bottom,x]>0;edges=np.diff(np.r_[False,bits,False].astype('int8'))
        spans=[(top+lo,top+hi) for lo,hi in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))
               if .2*s<=hi-lo<=.8*s and (((top+(lo+hi-1)/2)>=y-.4*s) if up else ((top+(lo+hi-1)/2)<=y+.4*s))]
        if len(spans)!=count:return False
        centers=[(lo+hi-1)/2 for lo,hi in spans]
        outer=min(centers) if up else max(centers)
        if abs(outer-y)>.55*s:return False
        if count==2 and not .1*s<=spans[1][0]-spans[0][1]<=.6*s:return False
    return True


def printed_number(heads,ink,count,recover_crossings=False,flexible_beams=False,beam_count=1,classify_fn=None,beam_proof_fn=None,stem_end_fn=None):
    import cv2
    import numpy as np
    space=heads[0].get('staff_space')
    if not space:return False
    for up in [True,False]:
        proof=(lambda a,b:beam_proof_fn(a,b,ink,up)) if beam_proof_fn else (lambda a,b:continuous_beams(a,b,ink,up,beam_count)) if flexible_beams else (lambda a,b:_single_beam(a,b,ink,up))
        if not all(proof(a,b) for a,b in zip(heads,heads[1:])):continue
        ends=[(stem_end_fn or _stem_end)(h,ink,up) for h in heads]
        if any(e is None for e in ends):continue
        cx=(ends[0][0]+ends[-1][0])/2
        left,right=round(cx-1.1*space),round(cx+1.1*space)
        top,bottom=(round(min(y for _,y in ends)-3*space),round(max(y for _,y in ends)-.25*space)) if up else (
                    round(min(y for _,y in ends)+.25*space),round(max(y for _,y in ends)+3*space))
        if min(left,top)<0 or right>ink.shape[1] or bottom>ink.shape[0]:continue
        raw=(ink[top:bottom,left:right]>0).astype('uint8')
        clean=cv2.subtract(raw,cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,round(3*space)),dtype='uint8')))
        if recover_crossings:
            # Staff-line subtraction can cut a numeral into two components.
            # Restore only removed line pixels with ink on both sides, rather
            # than closing all background gaps or lowering glyph confidence.
            bridge=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((5,3),dtype='uint8'))
            clean=np.maximum(clean,bridge&raw)
        _,_,stats,_=cv2.connectedComponentsWithStats(clean)
        found=[]
        for x,y,w,h,_ in stats[1:]:
            if .4*space<=w<=1.4*space and .65*space<=h<=1.8*space and abs(left+x+w/2-cx)<.7*space:
                number=(classify_fn or classify)(clean[y:y+h,x:x+w])
                if number is not None:found.append(number)
        if found==[count]:return True
    return False


def scale_divisions(root,factor):
    if factor==1:return
    for part in root.findall('part'):
        for tag in ['duration','divisions','offset']:
            for element in part.iter(tag):
                if element.text is not None:
                    element.text=str(Decimal(element.text)*factor) if tag=='offset' else str(int(element.text)*factor)


def repair(root,geometry,image,initial_meter=None):
    if any(len(s.get('staffs',[]))!=1 or not s['staffs'][0].get('grid') or len(s['staffs'][0]['grid'][0]['y'])!=5
           for s in geometry.get('systems',[])):return []
    mapped=_head_map(root,geometry);targets=_target_by_measure(root,initial_meter);ink=_ink(image)
    jobs=[];factor=1
    for mi,m in enumerate(root.findall('part/measure'),1):
        notes=m.findall('note')
        if (not notes or len(notes)>64 or len({n.findtext('voice','1') for n in notes})!=1
            or any(n.find('grace') is not None or n.find('time-modification') is not None for n in notes)):continue
        groups=_groups(notes);total=sum(int(g[0].findtext('duration','0')) for g in groups)
        target=targets.get(mi)
        if target is None or total<=target:continue
        plans=[]
        for start in range(len(groups)):
            for count,normal in [(3,2),(4,3)]:
                run=groups[start:start+count]
                if len(run)!=count or any(n.find('pitch') is None or n.findtext('type')!='eighth'
                    or n.findall('dot') or n not in mapped or _position(n)[1]!=mapped[n]['position'] for g in run for n in g):continue
                if any(len({n.findtext('duration') for n in g})!=1 for g in run):continue
                heads=[max((mapped[n] for n in g),key=lambda h:h['y']) for g in run]
                if not printed_number(heads,ink,count):
                    heads=[min((mapped[n] for n in g),key=lambda h:h['y']) for g in run]
                    if not printed_number(heads,ink,count):continue
                plans.append((start,count,normal))
        used=[i for start,count,_ in plans for i in range(start,start+count)]
        if not plans or len(used)!=len(set(used)) or len(jobs)>=128:continue
        proposed=[Fraction(int(g[0].findtext('duration','0'))) for g in groups]
        for start,count,normal in plans:
            for i in range(start,start+count):proposed[i]*=Fraction(normal,count)
        if sum(proposed)!=target:continue
        for d in proposed:factor=lcm(factor,d.denominator)
        jobs.append((m,groups,plans,proposed))
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for m,groups,plans,durations in jobs:
        desired={};cursor=0
        for g,duration in zip(groups,durations):
            d=int(duration*factor)
            for n in g:n.find('duration').text=str(d);desired[n]=cursor
            cursor+=d
        for number,(start,count,normal) in enumerate(plans,1):
            for g in groups[start:start+count]:
                for n in g:
                    tm=ET.Element('time-modification')
                    ET.SubElement(tm,'actual-notes').text=str(count);ET.SubElement(tm,'normal-notes').text=str(normal)
                    ET.SubElement(tm,'normal-type').text='eighth'
                    at=next((i for i,e in enumerate(n) if e.tag in {'stem','notehead','staff','beam','notations','lyric','play','listen'}),len(n))
                    n.insert(at,tm)
            for index,kind in [(start,'start'),(start+count-1,'stop')]:
                n=groups[index][0];notations=n.find('notations')
                if notations is None:notations=ET.SubElement(n,'notations')
                ET.SubElement(notations,'tuplet',number=str(number),type=kind,bracket='no')
        if not _rebuild(m,desired):raise ValueError('single-voice tuplet cursor reconstruction failed')
        changes.append({'measure':m.get('number'),'method':'printed_eighth_tuplet_number',
                        'ratios':[[count,normal] for _,count,normal in plans]})
    return changes
