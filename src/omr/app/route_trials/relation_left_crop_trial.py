"""Image-certified recovery of a left-clipped piano system.

Only the affected system may gain bars or have pitches repaired. Retained
clocks, voice labels and note nodes survive; other measures remain byte equal.
No score identity, reference XML or expected bar count is accepted.
"""
import copy
from collections import Counter
from statistics import median
import xml.etree.ElementTree as E
import cv2
import numpy as np
from homr_staff_bidirectional import extend_grid
from omr_bar_geometry import measure_systems,bar_bounds
from omr_event_records import records
from align_homr_geometry import _position
from repair_augmentation_dots import _head_map
from repair_pitch_context import _ink,no_accidental
from .relation_tail_splice_experiment import contexts
from .relation_coverage import core,hollow_core
from .relation_hollow_evidence import enclosed_hollow


def candidate_systems(geometry,image):
    """Bounded raster continuation, with actual omitted head evidence."""
    if image is None or image.ndim!=2 or image.size>16000000 or max(image.shape)>8192 or list(geometry.get('preprocessed_size',[]))!=[image.shape[1],image.shape[0]]:return []
    systems=geometry.get('systems',[])
    if not 2<=len(systems)<=64:return []
    staffs=[s['staffs'][0] for s in systems if len(s.get('staffs',[]))==1]
    starts={float(s['x_range'][0]) for s in staffs};out=[]
    for si,sy in enumerate(systems):
        if len(sy.get('staffs',[]))!=1:continue
        st=sy['staffs'][0];grid=st.get('grid',[])
        if not grid or any(len(p['y'])!=10 for p in grid) or len(grid)>1024 or len(st.get('notes',[]))>2048:continue
        values=[v for p in grid for v in [p['x'],*p['y']]]+[v for h in st.get('notes',[]) for v in (h['x'],h['y'],h['staff_space'])]+list(st['x_range'])+list(st['y_range'])
        if not all(np.isfinite(v) for v in values) or any(not 0<=p['x']<image.shape[1] or any(not 0<=y<image.shape[0] for y in p['y']) for p in grid):continue
        space=median(b-a for b,a in zip(grid[0]['y'][1:5],grid[0]['y'][:4]))
        if not 5<=space<=40:continue
        extended,reports=extend_grid(grid,starts,(),image)
        reports=[r for r in reports if r['direction']=='left' and r['from_x']-r['to_x']>4*space]
        if not reports:continue
        # Missing prefix glyphs alone do not justify a second OMR pass.
        omitted=[h for h in st.get('notes',[]) if extended[0]['x']+space<h['x']<st['x_range'][0]-.5*space]
        if len(omitted)<2 or any(not core(dict(xy=[h['x'],h['y']]),image,h['staff_space']) for h in omitted if h.get('size',[0])[0]<space*.9):continue
        out.append(dict(system=si+1,direction='left',old_x=st['x_range'][0],new_x=extended[0]['x'],omitted_segmented_heads=len(omitted),continuation=reports))
    return out


def _sharp_pairs(crop,space):
    """Two equally tall upright shafts plus two thick crossing strokes."""
    vertical=cv2.morphologyEx(crop,cv2.MORPH_OPEN,np.ones((round(1.3*space),1),np.uint8))
    count,_,stats,centres=cv2.connectedComponentsWithStats(vertical)
    if count>128:return []
    bars=[(float(centres[i,0]),float(centres[i,1]),int(stats[i,3])) for i in range(1,count) if 1.4*space<=stats[i,3]<=2.9*space and stats[i,2]<=.3*space]
    pairs=[]
    for a in bars:
        for b in bars:
            if not .25*space<=b[0]-a[0]<=.65*space or abs(a[1]-b[1])>.35*space or abs(a[2]-b[2])>.35*space:continue
            cx,cy=(a[0]+b[0])/2,(a[1]+b[1])/2
            lo,hi=round(a[0]-.24*space),round(b[0]+.24*space)
            if lo<0 or hi>=crop.shape[1]:continue
            crossed=True
            for sign in (-1,1):
                top,bottom=round(cy+sign*.45*space-.3*space),round(cy+sign*.45*space+.3*space)
                if top<0 or bottom>=crop.shape[0] or (crop[top:bottom+1,lo:hi+1]>0).mean(1).max()<.75:crossed=False;break
            if crossed:pairs.append((cx,cy))
    return pairs


def printed_sharps(staff,image):
    """Certify the ordered signature independently on both piano staves."""
    grid=staff.get('grid',[])
    if not grid or any(len(p['y'])!=10 for p in grid) or not staff.get('notes'):return None
    first=min(h['x'] for h in staff['notes']);point=min(grid,key=lambda p:abs(p['x']-(staff['x_range'][0]+first)/2));ink=_ink(image);counts=[];xs=[]
    for sn in (1,2):
        lines=point['y'][(sn-1)*5:sn*5];space=median(b-a for a,b in zip(lines,lines[1:]));left=round(staff['x_range'][0]+3.3*space);right=round(first-.65*space);top=round(lines[0]-2*space);bottom=round(lines[-1]+space)
        if not 5<=space<=40 or min(left,top)<0 or right>=image.shape[1] or bottom>=image.shape[0] or not 1.1*space<right-left<16*space:return None
        pairs=[(left+x,top+y) for x,y in _sharp_pairs(ink[top:bottom,left:right],space)];expected=(9,6,10,7,4,8,5) if sn==1 else (7,4,8,5,2,6,3);previous=None;positions=[]
        for position in expected:
            y=lines[-1]-(position-1)*space/2;choices=[x for x,cy in pairs if abs(cy-y)<=.3*space and (previous is None or .6*space<x-previous<1.5*space)]
            if len(choices)!=1:break
            previous=choices[0];positions.append(previous)
        counts.append(len(positions));xs.append(positions)
    if counts[0]!=counts[1] or not counts[0] or any(abs(a-b)>.4*space for a,b in zip(*xs)):return None
    return counts[0]


def _clock(r):return (r['staff'],r['voice'],r['onset'],r['duration'],r['chord'],r['grace'])


def no_accidental_except_proven_stems(head,staff,image,ink=None):
    """Remove only a single ink-proven shaft at each nearby head's rim.

    Both sides of a natural/sharp remain unless one is an independently
    continuous, head-attached shaft. Any other upright stroke vetoes repair.
    """
    s=head['staff_space'];x,y=head['x'],head['y']
    if ink is None:ink=_ink(image)
    if no_accidental(head,staff['notes'],ink):return True
    left,right=round(x-2.3*s),round(x-max(.65*s,head.get('size',[s])[0]/2+.15*s));top,bottom=round(y-1.3*s),round(y+1.3*s)
    previous=[h for h in staff['notes'] if h is not head and h['x']<x-3 and abs(h['y']-y)<1.7*s]
    if previous:left=max(left,round(max(h['x']+max(h.get('size',[s])[0]/2,.6*s)+1 for h in previous)))
    if min(left,top)<0 or right>=image.shape[1] or bottom>=image.shape[0] or right-left<.6*s:return False
    local=ink[top:bottom+1,left:right+1];horizontal=cv2.morphologyEx(local,cv2.MORPH_OPEN,np.ones((1,round(1.2*s)),np.uint8));clean=cv2.subtract(local,horizontal)
    # A full-height grand-staff stroke is independently a barline.
    grid=staff['grid']
    for bx in range(left,right+1):
        ys=[float(np.interp(bx,[p['x'] for p in grid],[p['y'][k] for p in grid])) for k in (0,9)];a,b=map(round,ys)
        if 0<=a<b<image.shape[0] and (image[a:b+1,bx]<140).mean()>=.95:clean[:,max(0,bx-left-1):min(clean.shape[1],bx-left+2)]=0
    # The head must join the same vertical column continuously at its rim.
    for other in staff['notes']:
        if not left-1.2*s<other['x']<right+1.2*s:continue
        for direction in (-1,1):
            cx=other['x']+direction*.55*s;hy=round(other['y']);a,b=sorted((hy,round(other['y']-direction*3*s)))
            if min(a,b)<0 or max(a,b)>=image.shape[0]:continue
            a+=round(.2*s);b-=round(.2*s)
            choices=[]
            for sx in range(round(cx-.2*s),round(cx+.2*s)+1):
                if not left<=sx<=right:continue
                if (image[a:b+1,sx]<140).mean()>=.97:choices.append(sx)
            if choices and max(choices)-min(choices)<=.3*s:
                for sx in choices:clean[:,max(0,sx-left-1):min(clean.shape[1],sx-left+2)]=0
    upright=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((max(3,round(.45*s)),1),np.uint8))
    return cv2.countNonZero(upright)==0 and cv2.countNonZero(clean)<2*s


def _written_rhythm_proof(rs,mapped,image):
    """New bars require physical two-beam runs and whole-note holes."""
    from repair_tuplets import continuous_beams
    ink=_ink(image);by_lane={}
    for r in rs:by_lane.setdefault((r['staff'],r['voice']),[]).append(r)
    for lane in by_lane.values():
        for index,r in enumerate(lane):
            h=mapped[r['node']];kind=r['node'].findtext('type')
            if r['node'].findall('dot') or r['node'].find('chord') is not None:return False
            if kind=='whole' and r['duration']==4 and r['onset']==0:continue
            if kind!='16th' or r['duration']!=__import__('fractions').Fraction(1,4):return False
            neighbors=[v for j,v in enumerate(lane) if abs(j-index)==1 and v['node'].findtext('type')==kind]
            directions=(r['node'].findtext('stem'),) if r['node'].findtext('stem') in ('up','down') else ('up','down')
            proven=[d for d in directions if any(continuous_beams(*sorted((h,mapped[v['node']]),key=lambda h:h['x']),ink,d=='up',2) for v in neighbors)]
            if len(proven)!=1:return False
    return True


def fuse(base,candidate,base_geometry,candidate_geometry,image):
    if len(base.findall('part'))!=1 or len(candidate.findall('part'))!=1 or base_geometry.get('preprocessed_size')!=candidate_geometry.get('preprocessed_size'):return []
    triggers={r['system'] for r in candidate_systems(base_geometry,image)}
    if not triggers:return []
    bs,cs=measure_systems(base),measure_systems(candidate)
    if not 2<=len(bs)==len(cs)==len(base_geometry.get('systems',[]))==len(candidate_geometry.get('systems',[]))<=64:return []
    be,ce=records(base),records(candidate);bc,cc=contexts(base),contexts(candidate);mapped=_head_map(candidate,candidate_geometry);old_by_id={id(m):(E.tostring(m),ctx) for m,ctx in zip(base.findall('part/measure'),bc)};jobs=[];a=b=0
    for si,(ms,ns,og,ng) in enumerate(zip(bs,cs,base_geometry['systems'],candidate_geometry['systems'])):
        oa,ob=a,b;a+=len(ms);b+=len(ns)
        if si+1 not in triggers or not 1<=len(ns)-len(ms)<=4 or len(ng.get('staffs',[]))!=1:continue
        old=og['staffs'][0];new=ng['staffs'][0];bounds=bar_bounds(new,image);extra=len(ns)-len(ms)
        if bounds is None or len(bounds)!=len(ns)+1 or not new.get('notes'):continue
        s=new['notes'][0]['staff_space']
        if not bounds[extra]<old['x_range'][0]-2*s or abs(old['x_range'][1]-new['x_range'][1])>.5*s:continue
        # Only a clipped system-start cancellation can be overturned, and the
        # omitted signature must physically agree with the inherited key.
        reset=ms[0].find('attributes/key/fifths');new_key=cc[ob][1].get(('key','1'));prior_key=bc[oa-1][1].get(('key','1')) if oa else None
        if reset is None or reset.text!='0' or prior_key is None or new_key!=prior_key:continue
        key=int(E.fromstring(prior_key).findtext('fifths','0'))
        if not 1<=key<=7 or printed_sharps(new,image)!=key:continue
        if bc[oa][0]!=cc[ob][0] or {k:v for k,v in bc[oa][1].items() if k[0]!='key'}!={k:v for k,v in cc[ob][1].items() if k[0]!='key'}:continue
        if any(m.findall('attributes') for m in ns) or any(m.findall('attributes') for m in ms[1:]):continue
        # Existing temporal/voice cardinality must exactly match each newly
        # visible physical bar. This trial never rewrites an old clock.
        if any([_clock(r) for r in be[oa+j]]!=[_clock(r) for r in ce[ob+extra+j]] for j in range(len(ms))):continue
        # Following-system context must be explicitly compatible; restoring a
        # prefix cannot silently reinterpret another system's notation.
        if a<len(bc) and bc[a]!=cc[b]:continue
        valid=True;ink=_ink(image)
        for j,m in enumerate(ns):
            rs=ce[ob+j];lo,hi=bounds[j:j+2]
            if not 2<=len(rs)<=128 or any(r['midi'] is None or r['grace'] or r['node'].find('tie') is not None or r['node'].find('time-modification') is not None or r['node'].get('print-object')=='no' or r['onset']<0 or r['duration']<=0 or r['onset']+r['duration']>16 for r in rs):valid=False;break
            hs=[mapped.get(r['node']) for r in rs]
            if any(h is None or not lo+.5*s<h['x']<hi-.5*s or _position(r['node'])[1]!=h['position'] for r,h in zip(rs,hs)):valid=False;break
            observed=[h for h in new['notes'] if lo+.5*s<h['x']<hi-.5*s]
            if Counter(map(id,hs))!=Counter(map(id,observed)):valid=False;break
            if any(not ((hollow_core(dict(xy=[h['x'],h['y']]),image,h['staff_space']) or enclosed_hollow(dict(xy=[h['x'],h['y']]),image,h['staff_space'])) if r['node'].findtext('type') in ('half','whole') else core(dict(xy=[h['x'],h['y']]),image,h['staff_space'])) for r,h in zip(rs,hs)):valid=False;break
            if j<extra and not _written_rhythm_proof(rs,mapped,image):valid=False;break
            for r,h in zip(rs,hs):
                letter=r['node'].findtext('pitch/step');alter=int(r['node'].findtext('pitch/alter','0'));expected=1 if letter in 'FCGDAEB'[:key] else 0
                # Printed accidentals in a newly introduced bar are outside
                # this trial's scope. Every proposed key-derived pitch needs
                # a clear accidental strip after proven shafts are masked.
                changed=[old for old in be[oa+j-extra] if j>=extra and _clock(old)==_clock(r) and E.tostring(old['node'].find('pitch'))!=E.tostring(r['node'].find('pitch'))]
                if j<extra or changed:
                    if any(old['node'].find('accidental') is not None for old in changed) or r['node'].find('accidental') is not None or alter!=expected or not no_accidental_except_proven_stems(h,new,image,ink):valid=False;break
            if not valid:break
        if valid:jobs.append((si,oa,ob,ms,ns,extra,key,bounds))
    changes=[];part=base.find('part');affected=set()
    for si,oa,ob,ms,ns,extra,key,bounds in jobs:
        pitch_changes=0;preserved=0
        for j,m in enumerate(ms):
            affected.add(id(m))
            for old,new in zip(be[oa+j],ce[ob+extra+j]):
                if old['midi']==new['midi'] and E.tostring(old['node'].find('pitch'))==E.tostring(new['node'].find('pitch')):preserved+=1;continue
                n=old['node'];at=list(n).index(n.find('pitch'));n.remove(n.find('pitch'));n.insert(at,copy.deepcopy(new['node'].find('pitch')))
                for acc in n.findall('accidental'):n.remove(acc)
                pitch_changes+=1
        ms[0].find('attributes/key/fifths').text=str(key)
        at=list(part).index(ms[0]);layout=ms[0].find('print')
        if layout is not None:ms[0].remove(layout)
        for j,m in enumerate(ns[:extra]):
            added=copy.deepcopy(m)
            for p in added.findall('print'):added.remove(p)
            if j==0:
                if layout is not None:added.insert(0,layout)
                # Move the repaired key to the physical start of the system.
                attrs=ms[0].findall('attributes')
                for attr in attrs:
                    if all(c.tag=='key' for c in attr):ms[0].remove(attr);added.insert(1 if layout is not None else 0,attr)
            part.insert(at,added);at+=1
        changes.append(dict(system=si+1,method='left_crop_visible_key_and_head_bijection',added_measures=extra,added_notes=sum(len(ce[ob+j]) for j in range(extra)),corrected_pitches=pitch_changes,preserved_existing_pitches=preserved,old_clocks_and_voices_preserved=True,key=key,physical_bounds=bounds))
    if changes:
        for m,ctx in zip(part.findall('measure'),contexts(base)):
            if id(m) in old_by_id and id(m) not in affected:
                assert E.tostring(m)==old_by_id[id(m)][0]
                assert ctx==old_by_id[id(m)][1]
    return changes
