"""Full-body G-axis and surrounding-text proof for a proposed clef8.

The two counters of a measure-number suffix are real 8 counters too.  They
cannot be bound to a partially clipped neighboring G.  Require an independently
contained complete G component, aligned to the numeral axis, and reject nearby
text components on the same text row.  Frozen V1/V2 algorithms are untouched.
"""
from __future__ import annotations

import copy
from collections import Counter
import time
import xml.etree.ElementTree as E
import cv2
import numpy as np

from align_homr_geometry import _position,_xml_systems
from omr_event_records import records
from omr_bar_geometry import measure_systems
from .testset3_measure_recovery import _head_ink
from .testset3_octave_recovery import (
    _tracks, _rows, _head_groups, _unique_alignment,
    _clef_contexts, _direction, MAX_ROI_PIXELS,
)
from .testset3_octave_recovery_v2 import detect as _v2_detect


def _neighbor_text(image,eight,space):
    x,y,width,height=[eight[name] for name in ('x','y','width','height')]
    left,top=max(0,round(x-2*space)),max(0,round(y-.3*space))
    right,bottom=min(image.shape[1],round(x+width+2*space)+1),min(image.shape[0],round(y+height+.3*space)+1)
    ink=(image[top:bottom,left:right]<140).astype(np.uint8)
    count,_,stats,_=cv2.connectedComponentsWithStats(ink)
    if count>128:
        return [dict(reason='surrounding_component_budget')]
    result=[]
    for index in range(1,count):
        bx,by,w,h,area=[int(value) for value in stats[index]]
        gx,gy=left+bx,top+by
        gap=x-(gx+w) if gx+w<x else gx-(x+width) if gx>x+width else None
        if (gap is not None and 0<=gap<=.65*space and .25*space<=w<=1.8*space
                and .6*height<=h<=1.4*height and abs(gy+h/2-y-height/2)<=.3*height
                and area>=.08*space*space):
            result.append(dict(bbox=[gx,gy,gx+w,gy+h],gap=gap,reason='separate_neighbor_on_same_text_row'))
    return result


def _full_g_body(image,track,eight):
    space=track['space'];left=max(0,round(track['left']-.25*space))
    right=min(image.shape[1],round(track['left']+7*space)+1)
    center=eight['x']+eight['width']/2;lines=_rows(track,center)
    # Padding must also contain a numeral touching the ascender before the
    # five-row closing; morphology near a crop border cannot certify a body.
    top=max(0,round(lines[0]-3.5*space));bottom=min(image.shape[0],round(lines[-1]+3*space)+1)
    if right<=left or bottom<=top or (right-left)*(bottom-top)>MAX_ROI_PIXELS:
        return []
    ink=(image[top:bottom,left:right]<140).astype(np.uint8)
    # A full prefix is wider than the old digit-centered crop.  Thin raster
    # lines can slope or be antialiased across that width; long horizontal
    # runs are staff ownership even when no single whole row reaches 80%.
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(5,round(4*space))),np.uint8))
    ink=cv2.subtract(ink,horizontal)
    for line in lines:
        row=round(line)-top;radius=max(1,round(.2*space))
        confirmed=[index for index in range(max(0,row-radius),min(ink.shape[0],row+radius+1)) if ink[index].mean()>=.8]
        if confirmed and max(confirmed)-min(confirmed)+1<=max(3,round(.25*space)):
            ink[confirmed]=0
    ink=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink)
    if count>256:
        return []
    result=[]
    for index in range(1,count):
        x,y,w,h,area=[int(value) for value in stats[index]]
        # A truncated component cannot supply its real center or body shape.
        contained=x>0 and y>0 and x+w<ink.shape[1]-1 and y+h<ink.shape[0]-1
        gx,gy=left+x,top+y
        # The G's lower spiral protrudes to the left of its ascender.  Its
        # whole bbox midpoint is therefore not the numeral's optical axis.
        # Measure the independently contained component above the top staff
        # row and below the numeral, rather than the digit's own pixels.
        axis_top=max(0,round(eight['y']+eight['height']+.1*space)-top)
        axis_bottom=min(ink.shape[0],round(lines[0]-.1*space)-top)
        ys,xs=np.where(labels[axis_top:axis_bottom]==index) if axis_bottom>axis_top else ([],[])
        if len(xs)<.15*space*space:
            continue
        axis=float(np.median(xs)+left)
        if (contained and .9*space<=w<=2.8*space and 4.6*space<=h<=8.5*space
                and gy<lines[0]-.2*space and gy+h>lines[-1]+.3*space
                and abs(axis-center)<=.35*space):
            result.append(dict(bbox=[gx,gy,gx+w,gy+h],body_axis=axis,axis_evidence='G_ascender_below_digit_above_staff',digit_axis=center,
                               axis_error_spaces=abs(axis-center)/space,complete_component_containment=True))
    return result


def detect(geometry,image,*,previous_score=None):
    report=_v2_detect(geometry,image,previous_score=previous_score)
    report['clef8_identity_guard']='complete_G_component_axis_and_separate_neighbor_text_v3'
    tracks={(track['system'],track['staff']):track for track in _tracks(geometry,image)} if not report.get('abstained') else {}
    kept=[]
    for scope in report['scopes']:
        if scope['kind']=='clef8':
            track=tracks.get((scope['system'],scope['staff']))
            neighbors=_neighbor_text(image,scope['eight'],track['space']) if track else []
            bodies=_full_g_body(image,track,scope['eight']) if track else []
            if neighbors or len(bodies)!=1:
                report['review'].append(dict(scope,reason='clef8_not_independently_centered_on_complete_G_or_part_of_text',
                                            neighboring_text=neighbors,complete_G_body_candidates=bodies))
                continue
            scope['independent_complete_G_body']=bodies[0]
        kept.append(scope)
    report['scopes']=kept
    return report


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
