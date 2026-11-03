"""Restore collapsed full-rest bars using visible boundaries and rest glyphs.

Only a complete ten-line system is supported. Pitched measures need unique
notehead anchors; every other interval must show one whole rest per staff.
No reference score, inferred empty time, or additional model is used.
"""
import copy
from fractions import Fraction
from statistics import median
import xml.etree.ElementTree as ET

from omr_bar_geometry import bar_bounds, measure_systems
from repair_augmentation_dots import _head_map
from align_homr_geometry import _position


def whole_rest(staff, image, a, b, number):
    import cv2
    import numpy as np
    grid = sorted(staff['grid'], key=lambda p: p['x'])
    space = median(q-p for row in grid for p,q in
                   zip(row['y'][(number-1)*5:number*5-1],row['y'][(number-1)*5+1:number*5]))
    left,right=max(0,round(a+.6*space)),min(image.shape[1],round(b-.6*space))
    if right-left<2*space or not 5<=space<=40:return False
    xs=[p['y'][(number-1)*5+1] for p in grid if left<=p['x']<=right]
    if not xs:return False
    top,bottom=max(0,round(min(xs)-.8*space)),min(image.shape[0],round(max(xs)+1.1*space))
    raw=(image[top:bottom,left:right]<140).astype('uint8')
    # A two-dimensional opening removes thin, slightly slanted staff lines
    # without subtracting the four-pixel body of the hanging rectangle.
    blocks=cv2.morphologyEx(raw,cv2.MORPH_OPEN,
                          np.ones((max(3,round(.22*space)),round(.8*space)),dtype='uint8'))
    removed=cv2.subtract(raw,cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,round(3*space)),dtype='uint8')))
    stats=[]
    for strip in (blocks,removed):stats.extend(cv2.connectedComponentsWithStats(strip)[2][1:])
    candidates=[]
    for x,y,w,h,area in stats:
        if not (.8*space<=w<=1.7*space and .18*space<=h<=.65*space and area/(w*h)>=.75):continue
        if abs(a-staff['x_range'][0])<space and left+x+w/2<a+8*space:continue
        local=np.interp(left+x+w/2,[p['x'] for p in grid],[p['y'][(number-1)*5+1] for p in grid])
        if not -.45*space<=top+y-local<=.6*space:continue
        # A whole rest hangs below a long printed line; a half rest stands
        # above one. Confirm the two side segments above the actual rectangle.
        good=False
        for row in range(max(0,y-3),y):
            segments=[raw[max(0,row-1):row+2,max(0,x-round(1.5*space)):x],
                      raw[max(0,row-1):row+2,x+w:min(raw.shape[1],x+w+round(1.5*space))]]
            if all(s.shape[1]>=.7*space and s.max(axis=0).mean()>=.75 for s in segments):good=True
        if good and not any(abs(cx-(x+w/2))<.5*space for cx in candidates):candidates.append(x+w/2)
    return len(candidates)==1


def _double_bars(staff,image,bounds):
    """Locate two thin full-height strokes for existing light-light markers."""
    space=median(b-a for p in staff['grid'] for a,b in zip(p['y'][:4],p['y'][1:5]))
    top,bottom=map(round,staff['y_range']);found=[]
    for i,x in enumerate(bounds[1:]):
        columns=[xx for xx in range(max(0,round(x-space)),min(image.shape[1],round(x+space)+1))
                 if (image[top:bottom,xx]<140).mean()>=.92]
        runs=[]
        for xx in columns:
            if not runs or xx-runs[-1][-1]>1:runs.append([])
            runs[-1].append(xx)
        if (len(runs)==2 and all(len(r)<.35*space for r in runs)
            and .2*space<=runs[1][0]-runs[0][-1]<=.8*space):found.append(i)
    return found


def _plain_rest_measure(measure):
    notes=measure.findall('note')
    return (len(notes)==2 and {n.findtext('staff','1') for n in notes}=={'1','2'}
            and all(n.find('rest') is not None and n.find('chord') is None
                    and (n.find('notations') is None or len(n.find('notations'))==0) and n.find('tie') is None
                    and (n.find('rest').get('measure')=='yes' or n.findtext('type')=='whole')
                    for n in notes))


def repair(root,geometry,image,initial_meter=None):
    import cv2
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    systems=measure_systems(root)
    if len(systems)!=len(geometry.get('systems',[])):return []
    parts=root.findall('part')
    if len(parts)!=1:return []
    mapped=_head_map(root,geometry)
    divisions=1;meter=initial_meter;changes=[]
    for si,(measures,system) in enumerate(zip(systems,geometry['systems']),1):
        for m in measures:
            for attr in m.findall('attributes'):
                divisions=int(attr.findtext('divisions',str(divisions)))
                t=attr.find('time')
                if t is not None:meter=(int(t.findtext('beats')),int(t.findtext('beat-type')))
        if len(system.get('staffs',[]))!=1 or meter is None:continue
        staff=system['staffs'][0];bounds=bar_bounds(staff,image)
        if bounds is None or len(bounds)-1<=len(measures):continue
        # Attributes/directions in the middle of an anonymous rest run cannot
        # be uniquely placed. Require unchanged state throughout this system.
        if any(m.findall('attributes') for m in measures[1:]):continue
        markers=[e for m in measures for e in m.findall('barline')]
        if any(e.get('location','right')!='right' or e.findtext('bar-style')!='light-light'
               or len(e)!=1 for e in markers):continue
        marker_bins=_double_bars(staff,image,bounds) if markers else []
        if len(marker_bins)!=len(markers):continue
        empty={};placed={};failed=False
        for j,(a,b) in enumerate(zip(bounds,bounds[1:])):
            empty[j]=(not any(a<h['x']<b for h in staff['notes'])
                      and all(whole_rest(staff,image,a,b,n) for n in (1,2)))
        previous=-1
        for m in measures:
            notes=[n for n in m.findall('note') if n.find('pitch') is not None and n.find('grace') is None]
            if not notes:
                if not _plain_rest_measure(m) or any(e.tag not in {'note','backup','forward','print','attributes','barline'} for e in m):
                    failed=True;break
                continue
            anchors=[mapped[n] for n in notes if n in mapped and _position(n)[1]==mapped[n]['position']]
            bins={j for h in anchors for j,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<h['x']<b}
            if len(anchors)<min(2,len(notes)) or len(bins)!=1:
                failed=True;break
            j=next(iter(bins))
            if j<=previous or j in placed or empty[j]:failed=True;break
            previous=j;placed[j]=m
        if failed or any(j not in placed and not empty[j] for j in range(len(bounds)-1)):continue
        duration=Fraction(divisions*meter[0]*4,meter[1])
        if duration.denominator!=1:continue
        exemplar=next((m for m in measures if _plain_rest_measure(m)),None)
        if exemplar is None:continue
        rebuilt=[]
        for j in range(len(bounds)-1):
            if j in placed:
                m=copy.deepcopy(placed[j])
            else:
                m=ET.Element('measure')
                for n in exemplar.findall('note'):
                    child=copy.deepcopy(n);child.find('duration').text=str(duration.numerator)
                    child.find('rest').set('measure','yes')
                    typ=child.find('type')
                    if typ is not None:child.remove(typ)
                    if list(m):
                        shift=ET.SubElement(m,'backup');ET.SubElement(shift,'duration').text=str(duration.numerator)
                    m.append(child)
            for e in list(m):
                if e.tag in {'print','barline'} or placed.get(j) is measures[0] and e.tag=='attributes':m.remove(e)
            rebuilt.append(m)
        for j,marker in zip(marker_bins,markers):rebuilt[j].append(copy.deepcopy(marker))
        # Header state and system marker belong at the physical first bar.
        for e in reversed([e for e in measures[0] if e.tag in {'attributes','print'}]):
            rebuilt[0].insert(0,copy.deepcopy(e))
        part=parts[0];at=list(part).index(measures[0])
        for m in measures:part.remove(m)
        for offset,m in enumerate(rebuilt):part.insert(at+offset,m)
        changes.append({'system':si,'method':'visible_whole_rest_bars','old_measures':len(measures),
                        'new_measures':len(rebuilt),'anchored_pitched_measures':len(placed)})
    if changes:
        for number,m in enumerate(parts[0].findall('measure'),1):m.set('number',str(number))
    return changes
