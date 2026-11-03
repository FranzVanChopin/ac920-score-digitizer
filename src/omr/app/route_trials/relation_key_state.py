"""Per-pitch printed-key state through physically observed graph columns.

XML spelling gaps do not erase unrelated photographed pitch states. Unknown
glyph ink still invalidates its own position. Clocks/cardinality stay fixed.
"""
from collections import Counter
from fractions import Fraction
import copy
import xml.etree.ElementTree as E
import cv2,numpy as np
from .relation_coverage import inputs,context_rows,ordered,spelling,detected,safe_measure,core,signature,empty_strip
from .relation_accidentals import associations,NAMES
from .relation_sequence import forced_pairs
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from align_homr_geometry import _position
from omr_event_records import records


def calibrated_signature(row,track,clef,key):
    if signature(row,track,clef,key):return True
    if not clef or not 3<=abs(key)<=7:return False
    first=min((g['x'] for g in row['groups'] if g.get('stem') is not None and any(h.get('confidence',0)>=.8 for h in g.get('heads',[]))),default=None)
    if first is None:return False
    cls=9 if key<0 else 8;space=row['space_px'];nodes=sorted([n for n in row['nodes'] if n['class']==cls and row['bounds'][0]<n['xy'][0]<first-.65*space],key=lambda n:n['xy'][0])
    if len(nodes)!=abs(key) or any(n.get('confidence',0)<.8 for n in nodes):return False
    residuals=[]
    for i,(n,letter) in enumerate(zip(nodes,('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)])):
        if i and not .6*space<n['xy'][0]-nodes[i-1]['xy'][0]<1.5*space:return False
        p=staff_step(track,n['xy'][0],n['xy'][1]+(.75*space if cls==9 else 0))
        if abs(p['residual'])>.4 or pitch_for_step(p['nearest_step'],clef)[0]!=letter:return False
        residuals.append(p['residual'])
    return max(residuals)-min(residuals)<=.12


def natural_component(crop,xy,space):
    """Isolated natural: two staggered upright bars, not a sharp/flat/stem."""
    horizontal=cv2.morphologyEx(crop,cv2.MORPH_OPEN,np.ones((1,max(3,round(3*space))),np.uint8))
    clean=cv2.morphologyEx(cv2.subtract(crop,horizontal),cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
    count,labels,stats,centres=cv2.connectedComponentsWithStats(clean);valid=[]
    if count>128:return None
    for i in range(1,count):
        x,y,w,hh,_=stats[i];cx,cy=centres[i]
        if not .35*space<=w<=.95*space or not 2*space<=hh<=3.3*space or abs(cx-xy[0])>.3*space or abs(cy-xy[1])>.3*space:continue
        if x<1 or y<1 or x+w>=crop.shape[1]-1 or y+hh>=crop.shape[0]-1:continue
        mask=(labels[y:y+hh,x:x+w]==i).astype('uint8');vertical=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((max(3,round(1.3*space)),1),np.uint8));columns=(vertical.sum(0)>=1.3*space);edges=np.diff(np.r_[False,columns,False].astype('int8'));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        if len(starts)!=2 or any(b-a>.25*space for a,b in zip(starts,ends)) or not .25*space<(starts[1]+ends[1]-starts[0]-ends[0])/2<.75*space:continue
        bars=[]
        for a,b in zip(starts,ends):
            ys=np.where(vertical[:,a:b].max(1)>0)[0];bars.append((ys.min(),ys.max()))
        if not .35*space<bars[1][0]-bars[0][0]<1.3*space or not .35*space<bars[1][1]-bars[0][1]<1.3*space:continue
        valid.append(labels==i)
    return valid[0] if len(valid)==1 else None


def padded_empty_strip(h,row,track,g,stems,image,ink,heads,attached=None):
    if empty_strip(h,row,track,g,stems,image,ink,heads):return True
    space=row['space_px'];x,y=h['xy'];left=round(x-2.3*space);right=round(x-.65*space);top=round(y-1.3*space);bottom=round(y+1.3*space)
    previous=[a for a in heads if a['x']<x-3 and abs(a['y']-y)<1.7*space]
    if previous:left=max(left,round(max(a['x']+.6*space for a in previous)+1))
    margin=round(1.3*space)
    if min(left-margin,top)<0 or right+margin>image.shape[1] or bottom>image.shape[0] or right-left<.75*space:return False
    if any(left-.5*space<a['x']<right and abs(a['y']-y)<1.7*space for a in heads):return False
    # Never erase a detector-supported accidental aimed at this position.
    if any(n['class'] in NAMES and left-.2*space<n['xy'][0]<right+.2*space and abs((n['xy'][1]-y)/space-(-.75 if n['class']==9 else 0))<.5 for n in row['nodes']):return False
    y0=max(0,round(y-3*space));y1=min(image.shape[0],round(y+3*space));x0=left-margin;crop=ink[y0:y1,x0:right+margin].copy()
    # A certified natural on another chord pitch can project into this strip.
    # Remove only its isolated connected component after validating both bars.
    # An accidental aimed at the target position already vetoed this path.
    if attached:
        for other in g.get('heads',[]):
            if other['id']==h['id'] or other.get('confidence',0)<.85 or not core(other,image,space):continue
            acc=attached.get(other['id'])
            if not acc or acc['class']!=10 or acc.get('confidence',0)<.9 or abs(other['xy'][1]-y)<1.4*space or abs(staff_step(track,*other['xy'])['residual'])>.3:continue
            ax,ay=acc['xy']
            if not x0+.6*space<ax<right+margin-.6*space or not y0+1.4*space<ay<y1-1.4*space:continue
            component=natural_component(crop,(ax-x0,ay-y0),space)
            if component is not None:crop[cv2.dilate(component.astype('uint8'),np.ones((3,3),np.uint8))>0]=0
    horizontal=cv2.morphologyEx(crop,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(round(1.2*space),1)))
    remaining=cv2.subtract(crop,horizontal)[top-y0:bottom-y0,margin:margin+right-left]
    vertical=cv2.morphologyEx(remaining,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(3,round(.45*space)))))
    return cv2.countNonZero(vertical)==0 and cv2.countNonZero(remaining)<2*space


def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];events=records(root);ctx=context_rows(root);ms=root.findall('part/measure');stems={s['id']:s for s in graph['stems']};changes=[];offset=0
    for system in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in rows:
            sn=row['staff']
            if sn>len(staffs):continue
            track=staffs[sn-1];first_key,first_clefs=ctx[offset];clef=first_clefs.get(sn)
            if first_key is None or not calibrated_signature(row,track,clef,first_key):continue
            heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for g in row['groups'] for h in g.get('heads',[])];attached=associations(row)
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;key,clefs=ctx[number];es=[r for r in events[number] if r['staff']==sn];m=ms[number]
                if key!=first_key or clefs.get(sn)!=clef or not safe_measure(es) or len({r['voice'] for r in es})!=1:continue
                seen=False;inline=False
                for c in m:
                    if c.tag=='note':seen=True
                    elif c.tag=='attributes' and seen:inline=True
                if inline:continue
                old=ordered(es);seq=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x'])
                if not old or not seq or len(seq)>64 or any(g.get('grace') for g in seq):continue
                pairs={hi:ni for ni,hi in forced_pairs([spelling(g) for g in old],[detected(g) for g in seq])}
                state={(letter,octave):(-1 if key<0 else 1) for letter in ('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)] for octave in range(10)}
                # All predicted columns participate in state. Only uniquely
                # aligned, clock-identical XML notes can actually be altered.
                for hi,g in enumerate(seq):
                    group=old[pairs[hi]] if hi in pairs else [];matched=group and all(r['onset']==Fraction(g.get('onset',-1),480) and r['duration']==Fraction(g['duration'],480) for r in group)
                    nodes={(r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))):r['node'] for r in group if r['midi'] is not None};updates={}
                    for h in g.get('heads',[]):
                        identity=tuple(h['pitch'][:2]);acc=attached.get(h['id']);absent=padded_empty_strip(h,row,track,g,stems,image,ink,heads,attached);pstep=staff_step(track,*h['xy']);n=nodes.get(identity) if matched else None
                        if acc:
                            if abs(pstep['residual'])>.3:
                                for step in (pstep['nearest_step']-1,pstep['nearest_step']+1):updates[pitch_for_step(step,clef)]=None
                            # Opening prefix symbols cannot become local seeds.
                            if mi==0 and acc['xy'][0]<seq[0]['x']-.9*row['space_px']:updates[identity]=None;continue
                            value=NAMES[acc['class']][0];updates[identity]=value if n is not None and int(n.findtext('pitch/alter','0'))==value and h.get('confidence',0)>=.85 else None
                            continue
                        if not absent:
                            updates[identity]=None
                            if abs(pstep['residual'])>.3:
                                for step in (pstep['nearest_step']-1,pstep['nearest_step']+1):updates[pitch_for_step(step,clef)]=None
                            continue
                        wanted=state.get(identity)
                        if n is not None and n.find('accidental') is not None:
                            updates[identity]=None;continue
                        if n is None or wanted is None or n.find('accidental') is not None or h['pitch'][2]!=wanted or h['pitch'][3] is not None:continue
                        old_alter=int(n.findtext('pitch/alter','0'))
                        if old_alter==wanted or h.get('class')!=0 or h.get('confidence',0)<.85 or not core(h,image,row['space_px']):continue
                        pos=_position(n)
                        if pos is None or abs(pstep['half_space_steps']+1-pos[1])>.3:continue
                        p=n.find('pitch');alter=p.find('alter')
                        if alter is None:alter=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,alter)
                        alter.text=str(wanted);changes.append({'measure':number+1,'staff':sn,'pitch':identity,'old_alter':old_alter,'new_alter':wanted,'xy':h['xy'],'method':'printed_key_per_pitch_observed_column_state'})
                    state.update(updates)
        offset+=len(rows[0]['bounds'])-1
    return changes
