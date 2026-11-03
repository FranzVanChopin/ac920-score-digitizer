"""Recover image-certified clef8 and upper/lower dashed octave brackets.

No reference, expected pitches or recognizer is accepted.  A two-counter 8,
thin repeated dashes or independent tall G-clef shape proposes a staff-local
scope.  A bounded unique sequence alignment then requires every changed pitch
to agree with an actual head's letter and physical row.  Timing, voice, note
types and cardinality stay unchanged.  Open parenthesized continuations need
an earlier accepted raster bracket or a previous predicted octave direction.
"""
from __future__ import annotations

import copy
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import median
import time
import xml.etree.ElementTree as E

import cv2
import numpy as np

from align_homr_geometry import _position, _xml_systems
from omr_event_records import records
from omr_bar_geometry import measure_systems
from .testset3_measure_recovery import _image_valid, _head_ink

MAX_ROI_PIXELS = 1_000_000
MAX_GROUPS = 512
MAX_HEADS = 2048
MAX_SCOPES = 128


def _tracks(geometry,image):
    result=[]
    for si,system in enumerate(geometry.get('systems',[])):
        if len(system.get('staffs',[]))!=1:
            continue
        staff=system['staffs'][0];grid=staff.get('grid',[])
        if not 2<=len(grid)<=1024 or len(staff.get('notes',[]))>MAX_HEADS:
            continue
        count=len(grid[0].get('y',[]))
        if count not in (5,10) or any(len(p.get('y',[]))!=count for p in grid):
            continue
        values=[v for p in grid for v in (p['x'],*p['y'])]
        if not all(np.isfinite(v) for v in values):
            continue
        if any(b['x']<=a['x'] for a,b in zip(grid,grid[1:])):
            continue
        border=median((p['y'][4]+p['y'][5])/2 for p in grid) if count==10 else float('inf')
        for sn in range(1,count//5+1):
            lines=[dict(x=p['x'],y=p['y'][(sn-1)*5:sn*5]) for p in grid]
            spacing=median(b-a for p in lines for a,b in zip(p['y'],p['y'][1:]))
            if not 5<=spacing<=40:
                continue
            notes=[h for h in staff.get('notes',[]) if (h['y']<border)==(sn==1) or count==5]
            result.append(dict(system=si,staff=sn,grid=lines,space=spacing,
                               left=float(staff['x_range'][0]),right=float(staff['x_range'][1]),notes=notes))
    return result


def _rows(track,x):
    return [float(np.interp(x,[p['x'] for p in track['grid']],[p['y'][i] for p in track['grid']])) for i in range(5)]


def _eights(image,box,space):
    left,top,right,bottom=box
    if left<0 or top<0 or right>image.shape[1] or bottom>image.shape[0] or right<=left or bottom<=top or (right-left)*(bottom-top)>MAX_ROI_PIXELS:
        return []
    ink=(image[top:bottom,left:right]<140).astype(np.uint8)
    contours,tree=cv2.findContours(ink,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None or len(contours)>4096:
        return []
    result=[]
    for i,c in enumerate(contours):
        if tree[0,i,3]!=-1:
            continue
        x,y,w,h=cv2.boundingRect(c)
        if not max(3,.3*space)<=w<=1.7*space or not max(7,.5*space)<=h<=2.1*space:
            continue
        child=tree[0,i,2];holes=[]
        while child>=0:
            if cv2.contourArea(contours[child])>=1:
                bx,by,bw,bh=cv2.boundingRect(contours[child]);mom=cv2.moments(contours[child])
                holes.append((cv2.contourArea(contours[child]),bx+bw/2,by+bh/2,bh))
            child=tree[0,child,0]
        if len(holes)!=2:
            continue
        a,b=sorted(holes,key=lambda q:q[2])
        if (min(a[0],b[0])/max(a[0],b[0])<.3 or min(a[3],b[3])/max(a[3],b[3])<.45
                or abs(a[1]-b[1])>.3*w or b[2]-a[2]<.25*h):
            continue
        result.append(dict(x=left+x,y=top+y,width=w,height=h,holes=2))
    return result


def _parentheses(image,eight,space):
    x,y,w,h=[eight[k] for k in ('x','y','width','height')]
    counts=[]
    for a,b in ((round(x-.6*space),x-1),(x+w+1,round(x+w+.6*space)+1)):
        if a<0 or b>image.shape[1] or b<=a:
            return False
        crop=(image[max(0,y-2):min(image.shape[0],y+h+3),a:b]<140).astype(np.uint8)
        n,_,stats,_=cv2.connectedComponentsWithStats(crop)
        counts.append(any(stats[i,3]>=.8*h and stats[i,2]<=.45*space for i in range(1,min(n,64))))
    return all(counts)


def _attached_eights(image,box,space,staff_top):
    """Two small upper counters can belong to an 8 touching the G ascender.

    Whole connected-component bounding boxes would merge the numeral and
    clef.  Isolate only a pair of small, aligned counters, then independently
    require the resulting local raster to contain the ordinary two-hole 8.
    """
    left,top,right,bottom=box
    crop=(image[top:bottom,left:right]<140).astype(np.uint8)
    contours,tree=cv2.findContours(crop,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if tree is None or len(contours)>4096:
        return []
    holes=[]
    for index,contour in enumerate(contours):
        x,y,w,h=cv2.boundingRect(contour);area=cv2.contourArea(contour)
        if (tree[0,index,3]>=0 and area>=1 and 2<=w<=.65*space and 2<=h<=.6*space
                and top+y+h<staff_top-.5*space):
            holes.append((left+x,top+y,w,h,area))
    if len(holes)>32:
        return []
    result=[];pad=max(1,round(.12*space))
    for a in holes:
        for b in holes:
            if b[1]<=a[1] or abs(a[0]+a[2]/2-b[0]-b[2]/2)>.25*space:
                continue
            if not .2*space<b[1]+b[3]/2-a[1]-a[3]/2<.85*space:
                continue
            if min(a[4],b[4])/max(a[4],b[4])<.3 or min(a[3],b[3])/max(a[3],b[3])<.45:
                continue
            x0=max(left,min(a[0],b[0])-pad);y0=max(top,a[1]-pad)
            x1=min(right,max(a[0]+a[2],b[0]+b[2])+pad);y1=min(bottom,b[1]+b[3]+pad)
            for eight in _eights(image,[x0,y0,x1,y1],space):
                if not any(abs(eight['x']-old['x'])<.3*space and abs(eight['y']-old['y'])<.3*space for old in result):
                    result.append(dict(eight,component_was_attached_to_clef=True))
    return result


def _dashes(image,track,eight,delta):
    """Thin separated strokes, with a directed hook or the actual row edge."""
    space=track['space'];height=image.shape[0]
    # An underside bracket is the vertically reflected upper bracket; the
    # numerator's two counters survive reflection unchanged.
    reflect=delta<0
    y=height-1-eight['y']-eight['height'] if reflect else eight['y']
    start=eight['x']+eight['width']+1
    stop=min(image.shape[1],round(track['right']+space))
    top,bottom=max(0,round(y-.6*space)),min(height,round(y+.6*space))
    raster=image[::-1] if reflect else image
    if stop<=start or (stop-start)*(bottom-top)>MAX_ROI_PIXELS:
        return None
    band=(raster[top:bottom,start:stop]<140).astype(np.uint8)
    columns=np.flatnonzero(band.max(axis=0))
    if not len(columns) or columns[0]>space:
        return None
    end=int(columns[0])
    for value in columns[1:]:
        if value-end>max(3,round(.6*space))+1:
            break
        end=int(value)
    if not 2*space<end<=160*space:
        return None
    local=band[:,:end+1]
    coverage=float(local.max(axis=0).mean())
    if not .3<=coverage<=.8:
        return None
    count,_,stats,centers=cv2.connectedComponentsWithStats(local)
    if count>1024:
        return None
    dashes=[tuple(centers[i]) for i in range(1,count)
            if .1*space<=stats[i,2]<=1.2*space and stats[i,3]<=.35*space]
    if len(dashes)<4:
        return None
    points=np.asarray(dashes);slope,intercept=np.polyfit(*points.T,1)
    for _ in range(3):
        good=np.abs(points[:,1]-(slope*points[:,0]+intercept))<=2
        if good.sum()<4:
            return None
        slope,intercept=np.polyfit(*points[good].T,1)
    good=np.abs(points[:,1]-(slope*points[:,0]+intercept))<=2
    if abs(slope)>.025 or good.sum()<max(4,.85*len(dashes)):
        return None
    ex=start+end;ey=round(top+slope*end+intercept)
    hook=(raster[max(0,ey):min(height,ey+round(space)),max(0,ex-2):min(image.shape[1],ex+3)]<140)
    closed=hook.shape[0]==round(space) and float(hook.max(axis=1).mean())>=.7
    edge=abs(ex-track['right'])<=space
    if not closed and not edge:
        return None
    return dict(left=eight['x']-.3*space,right=ex+.3*space,delta=delta,closed=closed,
                edge=edge,dash_count=len(dashes),dash_coverage=coverage,
                bracket_y=(height-1-ey if reflect else ey),eight=eight,
                method='two_counter_8_thin_dashes_directed_hook_or_row_edge',
                parenthesized=_parentheses(image,eight,space))


def _g_clef_shape(image,track,eight):
    space=track['space'];cx=eight['x']+eight['width']/2
    lines=_rows(track,cx)
    left,right=max(0,round(cx-1.5*space)),min(image.shape[1],round(cx+1.5*space)+1)
    top,bottom=max(0,round(lines[0]-2*space)),min(image.shape[0],round(lines[-1]+2*space)+1)
    crop=(image[top:bottom,left:right]<140).astype(np.uint8)
    for line in lines:
        cy=round(line)-top;radius=max(1,round(.2*space))
        rows=[value for value in range(max(0,cy-radius),min(crop.shape[0],cy+radius+1))
              if crop[value].mean()>=.8]
        # Refine to the actual long, thin raster row; a fitted grid may be
        # displaced by a pixel.  Whole-row erasure at that fitted value would
        # leave a true staff line joined to the clef's component.
        if rows and max(rows)-min(rows)+1<=max(3,round(.25*space)):
            crop[rows]=0
    crop=cv2.morphologyEx(crop,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
    n,_,stats,_=cv2.connectedComponentsWithStats(crop)
    if n>128:
        return False
    return any(4.6*space<=stats[i,3]<=7.8*space and .9*space<=stats[i,2]<=2.8*space
               and abs(left+stats[i,0]+stats[i,2]/2-cx)<=.65*space
               and top+stats[i,1]<lines[0]-.2*space
               and top+stats[i,1]+stats[i,3]>lines[-1]+.3*space for i in range(1,n))


def _previous_active(previous_score):
    if previous_score is None:
        return {}
    root=previous_score if isinstance(previous_score,E.Element) else E.parse(Path(previous_score)).getroot()
    if len(root.findall('part'))!=1:
        return {}
    active={}
    for measure in root.findall('part/measure'):
        for direction in measure.findall('direction'):
            staff=int(direction.findtext('staff','1'))
            for shift in direction.findall('direction-type/octave-shift'):
                if shift.get('size','8')!='8':
                    continue
                kind=shift.get('type')
                if kind=='stop':
                    active.pop(staff,None)
                elif kind in ('up','down'):
                    active[staff]=-1 if kind=='up' else 1
    return active


def detect(geometry,image,*,previous_score=None):
    report=dict(scopes=[],review=[],reference_supplied=False,model_calls=0,
                coordinate_frame='homr_preprocessed_pixels',maximum_roi_pixels=MAX_ROI_PIXELS)
    if not _image_valid(geometry,image) or not 1<=len(geometry.get('systems',[]))<=64:
        report['abstained']='frame_or_input_budget_mismatch';return report
    tracks=_tracks(geometry,image);raw=[];clefs=[]
    for track in tracks:
        space=track['space'];top=min(p['y'][0] for p in track['grid']);bottom=max(p['y'][-1] for p in track['grid'])
        for delta,a,b in ((1,top-8*space,top-.35*space),(-1,bottom+.35*space,bottom+8*space)):
            box=[max(0,round(track['left']-.5*space)),max(0,round(a)),min(image.shape[1],round(track['right']+space)),min(image.shape[0],round(b))]
            for eight in _eights(image,box,space):
                span=_dashes(image,track,eight,delta)
                if span is not None:
                    span.update(system=track['system'],staff=track['staff']);raw.append(span)
        prefix=[max(0,round(track['left'])),max(0,round(top-3*space)),min(image.shape[1],round(track['left']+4*space)),min(image.shape[0],round(bottom+3*space))]
        prefix_eights=_eights(image,prefix,space)+_attached_eights(image,prefix,space,top)
        for eight in prefix_eights:
            cx=eight['x']+eight['width']/2;lines=_rows(track,cx)
            above=.15*space<lines[0]-eight['y']-eight['height']<2*space
            below=.15*space<eight['y']-lines[-1]<2.6*space
            if (above or below) and eight['height']<=1.25*space and eight['width']<=1.1*space and _g_clef_shape(image,track,eight):
                if any(old['system']==track['system'] and old['staff']==track['staff']
                       and abs(old['eight']['x']-eight['x'])<.3*space for old in clefs):
                    continue
                clefs.append(dict(system=track['system'],staff=track['staff'],left=track['left'],right=track['right'],
                                  delta=1 if above else -1,eight=eight,method='two_counter_8_and_independent_tall_G_clef',kind='clef8'))
    if len(raw)+len(clefs)>MAX_SCOPES:
        report['abstained']='scope_budget_exceeded';return report
    active=_previous_active(previous_score)
    for candidate in sorted(raw,key=lambda span:(span['system'],span['staff'],span['left'])):
        staff=candidate['staff'];prior=active.get(staff)
        first_head=min((head['x'] for track in tracks if track['system']==candidate['system'] and track['staff']==staff for head in track['notes']),default=float('inf'))
        prefix=candidate['left']<=first_head+next(track['space'] for track in tracks if track['system']==candidate['system'] and track['staff']==staff)
        if candidate['parenthesized'] and (prior!=candidate['delta'] or not prefix):
            report['review'].append(dict(candidate,reason='parenthesized_continuation_without_previous_predicted_start'));continue
        candidate['kind']='bracket';candidate['review']=None if candidate['closed'] else 'open_at_row_or_page_edge'
        report['scopes'].append(candidate)
        if candidate['closed']:
            active.pop(staff,None)
        else:
            active[staff]=candidate['delta']
    report['scopes'].extend(clefs)
    report['active_predicted_brackets_out']={str(staff):delta for staff,delta in active.items()}
    return report


def _head_groups(track):
    groups=[];space=track['space']
    for raw in sorted(track['notes'],key=lambda head:head['x']):
        x,y=float(raw['x']),float(raw['y']);lines=_rows(track,x)
        pos=1+2*(lines[-1]-y)/space
        if abs(pos-round(pos))>.4:
            continue
        head=dict(raw,position=round(pos),staff_space=space)
        same_stem=(groups and raw.get('stem') and x-groups[-1][0]['x']<=1.5*space
                   and any(other.get('stem') and abs(other['stem'][0]-raw['stem'][0])<=1.5 for other in groups[-1]))
        if not groups or (x-groups[-1][-1]['x']>.55*space and not same_stem):
            groups.append([])
        groups[-1].append(head)
    return [sorted(group,key=lambda head:head['position']) for group in groups]


def _scope_for(head,scopes):
    found=[scope for scope in scopes if scope['left']<=head['x']<=scope['right']]
    return found[0] if len(found)==1 else None


def _clef_contexts(root):
    state={1:('G',2,0,0),2:('F',4,0,0)};notes={};opening={};generation=0
    for measure in root.findall('part/measure'):
        first=False
        for node in measure:
            if node.tag=='attributes':
                for clef in node.findall('clef'):
                    generation+=1;staff=int(clef.get('number','1'))
                    state[staff]=(clef.findtext('sign'),int(clef.findtext('line','0')),
                                  int(clef.findtext('clef-octave-change','0')),generation)
            elif node.tag=='note':
                if not first:
                    opening[measure]=dict(state);first=True
                notes[node]=state.get(int(node.findtext('staff','1')))
        if measure not in opening:
            opening[measure]=dict(state)
    return notes,opening


def _unique_alignment(groups,heads,scopes):
    """An edit is accepted only if its match appears on every optimal path."""
    n,m=len(groups),len(heads)
    if not n or not m or max(n,m)>MAX_GROUPS:
        return {},0
    choices={}
    for i,notes in enumerate(groups):
        ordered=sorted(notes,key=lambda row:_position(row['node'])[1])
        for j,column in enumerate(heads):
            if len(ordered)!=len(column):
                continue
            plan=[];cost=0
            for row,head in zip(ordered,column):
                position=_position(row['node'])
                if position is None:
                    break
                scope=_scope_for(head,scopes)
                if scope is not None and scope['kind']=='clef8' and row.get('_clef_context')!=scope.get('_clef_context'):
                    scope=None
                shift=scope['delta'] if scope else 0
                difference=head['position']+7*shift-position[1]
                if difference not in (0,7*shift) or (not scope and difference):
                    break
                cost+=abs(difference)//7
                plan.append((row,head,difference//7,scope))
            else:
                choices[(i,j)]=(cost,plan)
    inf=4*sum(map(len,groups))+4*sum(map(len,heads))+1
    forward=[[inf]*(m+1) for _ in range(n+1)];backward=[[inf]*(m+1) for _ in range(n+1)]
    forward[0][0]=0
    for i in range(n+1):
        for j in range(m+1):
            value=forward[i][j]
            if i<n:forward[i+1][j]=min(forward[i+1][j],value+4*len(groups[i]))
            if j<m:forward[i][j+1]=min(forward[i][j+1],value+4*len(heads[j]))
            if (i,j) in choices:forward[i+1][j+1]=min(forward[i+1][j+1],value+choices[(i,j)][0])
    backward[n][m]=0
    for i in range(n,-1,-1):
        for j in range(m,-1,-1):
            value=backward[i][j]
            if i:backward[i-1][j]=min(backward[i-1][j],value+4*len(groups[i-1]))
            if j:backward[i][j-1]=min(backward[i][j-1],value+4*len(heads[j-1]))
            if (i-1,j-1) in choices:backward[i-1][j-1]=min(backward[i-1][j-1],value+choices[(i-1,j-1)][0])
    accepted={};ambiguous=0;best=forward[n][m]
    for i in range(n):
        skip=any(forward[i][j]+4*len(groups[i])+backward[i+1][j]==best for j in range(m+1))
        matching=[j for j in range(m) if (i,j) in choices and forward[i][j]+choices[(i,j)][0]+backward[i+1][j+1]==best]
        if not skip and len(matching)==1:
            accepted[i]=choices[(i,matching[0])][1]
        elif matching:
            ambiguous+=len(groups[i])
    return accepted,ambiguous


def _direction(scope,kind,offset=0):
    node=E.Element('direction',placement='above' if scope['delta']>0 else 'below')
    notation=E.SubElement(node,'direction-type')
    shift_kind='down' if scope['delta']>0 else 'up'
    E.SubElement(notation,'octave-shift',type=shift_kind if kind=='start' else 'stop',size='8',number=str(scope['staff']))
    if offset:
        E.SubElement(node,'offset').text=str(offset)
    E.SubElement(node,'staff').text=str(scope['staff'])
    return node


def repair(root,geometry,image,*,previous_score=None):
    started=time.perf_counter();report=detect(geometry,image,previous_score=previous_score)
    report.update(enabled=True,changes=[],clocks_and_cardinality_preserved=True)
    systems=measure_systems(root)
    if len(root.findall('part'))!=1 or len(systems)!=len(geometry.get('systems',[])):
        report['abstained']='part_or_system_inventory_mismatch';return report
    if not report['scopes']:
        report['elapsed_s']=time.perf_counter()-started;return report
    _xml_systems(root)
    event_rows=records(root);by_measure={measure:rows for measure,rows in zip(root.findall('part/measure'),event_rows)}
    note_contexts,opening_contexts=_clef_contexts(root)
    for rows in event_rows:
        for row in rows:
            row['_clef_context']=note_contexts.get(row['node'])
    parents={row['node']:measure for measure,rows in by_measure.items() for row in rows}
    existing_shift_staffs={int(direction.findtext('staff','1')) for direction in root.findall('.//direction') if direction.find('direction-type/octave-shift') is not None}
    plans=[]
    for track in _tracks(geometry,image):
        si,sn=track['system'],track['staff']
        selected=[scope for scope in report['scopes'] if scope['system']==si and scope['staff']==sn]
        prefix_context=opening_contexts[systems[si][0]].get(sn)
        selected=[scope for scope in selected if scope['kind']!='clef8'
                  or prefix_context is not None and prefix_context[:3]==('G',2,0)]
        for scope in selected:
            if scope['kind']=='clef8':
                scope['_clef_context']=prefix_context
        if sn in existing_shift_staffs:
            selected=[scope for scope in selected if scope['kind']=='clef8']
        if not selected:
            continue
        groups=[]
        for measure in systems[si]:
            rows=[row for row in by_measure[measure] if row['staff']==sn and row['midi'] is not None and not row['grace']]
            for onset in sorted({row['onset'] for row in rows}):
                groups.append([row for row in rows if row['onset']==onset])
        accepted,ambiguous=_unique_alignment(groups,_head_groups(track),selected)
        report.setdefault('alignment',[]).append(dict(system=si+1,staff=sn,groups=len(groups),unique_groups=len(accepted),ambiguous_events=ambiguous))
        for plan in accepted.values():
            for row,head,delta,scope in plan:
                if not delta or scope is None:
                    continue
                if not _head_ink(head,image,row['node'].findtext('type') in ('half','whole')):
                    continue
                octave=int(row['node'].findtext('pitch/octave'))
                if not -1<=octave+delta<=9:
                    continue
                plans.append((row,head,delta,scope))
    if not plans:
        report['elapsed_s']=time.perf_counter()-started;return report
    old_clocks=[Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['chord'],r['grace']) for r in rows) for rows in event_rows]
    edited=set()
    for row,head,delta,scope in plans:
        if row['node'] in edited:
            continue
        edited.add(row['node']);octave=row['node'].find('pitch/octave');old=int(octave.text);octave.text=str(old+delta)
        report['changes'].append(dict(system=scope['system']+1,staff=scope['staff'],measure=parents[row['node']].get('number'),
                                      onset=str(row['onset']),old_octave=old,new_octave=old+delta,
                                      head_xy=[head['x'],head['y']],method=scope['method'],scope_kind=scope['kind']))
    # Add notation only for scopes with at least one physically anchored edit.
    # All metadata is derived from predictions and the newly witnessed symbol.
    notation_active={}
    for scope in sorted(report['scopes'],key=lambda value:(value['system'],value['staff'],value['left'])):
        accepted=[(row,head) for row,head,_,owner in plans if owner is scope]
        if not accepted:
            continue
        if scope['kind']=='clef8':
            measure=systems[scope['system']][0]
            attributes=E.Element('attributes');clef=E.SubElement(attributes,'clef',number=str(scope['staff']))
            E.SubElement(clef,'sign').text='G';E.SubElement(clef,'line').text='2';E.SubElement(clef,'clef-octave-change').text=str(scope['delta'])
            position=next((i for i,node in enumerate(measure) if node.tag not in ('print','attributes')),len(measure))
            measure.insert(position,attributes)
            # The positive marker proves this physical system only.  Restore
            # the original predicted context at the next unverified system;
            # absence of a detected 8 must never extend the new transposition.
            following=scope['system']+1
            next_verified=any(owner['kind']=='clef8' and owner['system']==following
                              and owner['staff']==scope['staff'] for _,_,_,owner in plans)
            if following<len(systems) and not next_verified:
                next_measure=systems[following][0]
                context=opening_contexts[next_measure].get(scope['staff'])
                explicit=any(int(clef.get('number','1'))==scope['staff']
                             for block in next_measure.findall('attributes') for clef in block.findall('clef'))
                if context is not None and not explicit:
                    restore=E.Element('attributes');restored=E.SubElement(restore,'clef',number=str(scope['staff']))
                    E.SubElement(restored,'sign').text=context[0];E.SubElement(restored,'line').text=str(context[1])
                    E.SubElement(restored,'clef-octave-change').text=str(context[2])
                    index=next((i for i,node in enumerate(next_measure) if node.tag not in ('print','attributes')),len(next_measure))
                    next_measure.insert(index,restore)
                    report.setdefault('context_restorations',[]).append(dict(system=following+1,staff=scope['staff'],
                        restored_original_predicted_clef=list(context[:3]),reason='next_system_has_no_positive_clef8_scope'))
        else:
            first=min(accepted,key=lambda pair:(root.findall('part/measure').index(parents[pair[0]['node']]),pair[0]['onset']))[0]
            measure=parents[first['node']];position=list(measure).index(first['node'])
            offset=-int(first['node'].findtext('duration','0')) if first['chord'] else 0
            key=scope['staff']
            if notation_active.get(key)!=scope['delta']:
                measure.insert(position,_direction(scope,'start',offset))
                notation_active[key]=scope['delta']
            if scope['closed']:
                last=max(accepted,key=lambda pair:(root.findall('part/measure').index(parents[pair[0]['node']]),pair[0]['onset']+pair[0]['duration']))[0]
                end=parents[last['node']];end.insert(list(end).index(last['node'])+1,_direction(scope,'stop'))
                notation_active.pop(key,None)
    after=records(root)
    assert old_clocks==[Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['chord'],r['grace']) for r in rows) for rows in after]
    report['elapsed_s']=time.perf_counter()-started
    return report
