"""Conservative optical inline-clef guard around frozen octave recovery.

An inline G can reset a G8 even when the recognizer omits an identical G/line
attribute.  When that physical boundary has no exact predicted serialization,
abstain the whole proposed clef8 scope.  No page identity, reference pitch or
score-based invocation gate enters this policy.
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
    detect as _original_detect, _tracks, _rows, _head_groups, _unique_alignment,
    _clef_contexts, _direction, MAX_ROI_PIXELS,
)


def _inline_g_candidates(image,track):
    space=track['space'];top=min(point['y'][0] for point in track['grid'])
    bottom=max(point['y'][-1] for point in track['grid'])
    left=max(0,round(track['left']+4*space));right=min(image.shape[1],round(track['right']))
    y0=max(0,round(top-2*space));y1=min(image.shape[0],round(bottom+2*space)+1)
    if right<=left or y1<=y0 or (right-left)*(y1-y0)>MAX_ROI_PIXELS:
        return []
    ink=(image[y0:y1,left:right]<140).astype(np.uint8)
    staff_rows=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(5,round(6*space))),np.uint8))
    clean=cv2.subtract(ink,staff_rows)
    clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
    count,labels,stats,centers=cv2.connectedComponentsWithStats(clean)
    if count>4096:
        return []
    result=[]
    for index in range(1,count):
        bx,by,width,height,area=stats[index]
        cx,cy=left+centers[index,0],y0+centers[index,1]
        lines=_rows(track,cx)
        if not (4.6*space<=height<=7.8*space and .9*space<=width<=2.8*space
                and y0+by<lines[0]-.2*space and y0+by+height>lines[-1]+.3*space):
            continue
        component=(labels[by:by+height,bx:bx+width]==index).astype(np.uint8)
        contours,tree=cv2.findContours(component,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if tree is None:
            continue
        upper=False;lower=False;hole_count=0
        for i,contour in enumerate(contours):
            if tree[0,i,3]<0 or cv2.contourArea(contour)<.08*space*space:
                continue
            hx,hy,hw,hh=cv2.boundingRect(contour);hole_y=y0+by+hy+hh/2
            if not .25*space<=hw<=1.7*space or not .25*space<=hh<=1.7*space:
                continue
            hole_count+=1
            upper|=lines[0]-1.3*space<hole_y<lines[0]+.6*space
            lower|=abs(hole_y-(lines[-1]-space))<.8*space
        # A long note stem, octave chord or ordinary text lacks the two
        # separated G loops at their independent staff-relative rows.
        if upper and lower and hole_count>=3:
            result.append(dict(bbox=[int(left+bx),int(y0+by),int(left+bx+width),int(y0+by+height)],
                               x=float(cx),upper_and_lower_G_loops=True,hole_count=hole_count,
                               method='inline_tall_G_with_two_staff_relative_loop_regions'))
    return result


def detect(geometry,image,*,previous_score=None):
    report=_original_detect(geometry,image,previous_score=previous_score)
    report['inline_clef_guard']='whole_clef8_scope_abstains_if_independent_inline_G_boundary_is_unserialized'
    retained=[]
    tracks={(track['system'],track['staff']):track for track in _tracks(geometry,image)} if not report.get('abstained') else {}
    for scope in report['scopes']:
        if scope['kind']=='clef8':
            track=tracks.get((scope['system'],scope['staff']))
            inline=_inline_g_candidates(image,track) if track is not None else []
            if inline:
                report['review'].append(dict(scope,reason='independent_inline_G_requires_exact_scope_boundary',inline_clefs=inline))
                continue
        retained.append(scope)
    report['scopes']=retained
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
