"""Preserve V5 and measure isolated short tie arcs on their actual support.

The fixed near-head sampling window can remove both ends of a short arc.
This supplement measures the connected thin component itself instead. It
requires two low ends, an interior convex extremum, a smooth measured ridge,
and uniquely owned, ink-backed heads on the same exact predicted clock lane.
No reference score, prior result IDs, or learned detector is an input.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import math
import time
from unittest.mock import patch

import cv2
import numpy as np

from align_homr_geometry import _position, _xml_systems
from omr_event_records import records
from . import testset3_tie_recovery_v5 as previous


def _terminal_meter_lanes(root):
    """No new clock claim in a predicted lane that ends short of its meter.

    Every interval, including any explicit opening rest, must cover the
    nominal meter without holes or overlaps. Equal-onset chord members must
    agree on duration. This does not fill gaps or alter any note.
    """
    parts=root.findall('part')
    if len(parts)!=1:return {}
    try:rows=records(root)
    except (ValueError,KeyError,TypeError,ZeroDivisionError):return {}
    result={};meter=None;divisions=1
    for mi,(measure,row) in enumerate(zip(parts[0].findall('measure'),rows)):
        try:
            cursor=Fraction(0);invalid=False;seen_note=False
            for node in measure:
                if node.tag=='attributes':
                    divisions=int(node.findtext('divisions',str(divisions)))
                    if divisions<=0:invalid=True;continue
                    for time_node in node.findall('time'):
                        candidate=Fraction(4*int(time_node.findtext('beats')),int(time_node.findtext('beat-type')))
                        if seen_note and candidate!=meter:invalid=True
                        meter=candidate
                elif node.tag in ('backup','forward'):
                    duration=Fraction(node.findtext('duration','0'))/divisions
                    if duration<0:invalid=True
                    cursor+=duration*(1 if node.tag=='forward' else -1)
                    if cursor<0:invalid=True
                elif node.tag=='note':
                    seen_note=True
                    if node.find('chord') is None and node.find('grace') is None:
                        cursor+=Fraction(node.findtext('duration','0'))/divisions
            if invalid or meter is None or not 0<meter<=16 or measure.get('implicit')=='yes' or len(row)>128:continue
            lanes=defaultdict(list)
            for event in row:
                if not event['grace']:lanes[str(event['staff']),event['voice']].append(event)
            for lane,events in lanes.items():
                if any(e['onset']<0 or e['duration']<=0 or e['onset']+e['duration']>meter for e in events):continue
                groups=defaultdict(set)
                for event in events:groups[event['onset']].add(event['duration'])
                terminal=Fraction(0);complete=True
                for onset,durations in sorted(groups.items()):
                    if len(durations)!=1 or onset!=terminal:complete=False;break
                    terminal=onset+next(iter(durations))
                if not complete or terminal!=meter:continue
                result[mi,*lane]=dict(nominal_meter_quarters=str(meter),
                    predicted_lane_terminal_quarters=str(meter),clock_modified=False,
                    interval_coverage='zero_to_meter_without_holes_overlaps_or_mixed_duration_chords')
        except (ValueError,TypeError,ZeroDivisionError):continue
    return result


def short_component_arc(image, first, last, staff, number):
    """Test bounded components of observed ink, without completing a curve."""
    x0,y0=float(first['x']),float(first['y'])
    x1,y1=float(last['x']),float(last['y'])
    space=float(np.mean([first['staff_space'],last['staff_space']]))
    span=x1-x0
    if (not all(math.isfinite(v) for v in (x0,y0,x1,y1,space))
            or not 4<=space<=64 or not 1.8*space<=span<=8*space
            or abs(y1-y0)>.35*space):
        return dict(accepted=False,reason='short_component_endpoint_window')
    left,right=math.floor(x0+.35*space),math.ceil(x1-.35*space)
    top,bottom=math.floor(min(y0,y1)-2*space),math.ceil(max(y0,y1)+2*space)
    if (left<0 or top<0 or right>image.shape[1] or bottom>image.shape[0]
            or right-left>1024 or bottom-top>160 or (right-left)*(bottom-top)>131072):
        return dict(accepted=False,reason='short_component_roi_budget_or_clipping')
    ink=image[top:bottom,left:right]<160
    xs=np.arange(left,right,dtype=float)
    for line in previous.original._staff_lines(staff,xs,number):
        rows=np.rint(line-top).astype(int)
        for delta in range(-max(1,round(.055*space)),max(1,round(.055*space))+1):
            yy=rows+delta;valid=(yy>=0)&(yy<ink.shape[0])
            ink[yy[valid],np.flatnonzero(valid)]=False
    count,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
    if count-1>128:
        return dict(accepted=False,reason='short_component_inventory_budget',component_count=count-1)
    accepted=[]
    for label in range(1,count):
        xx,yy,width,height,area=map(int,stats[label])
        if width<max(5,.35*span) or height<3 or area<6:continue
        start,end=left+xx,left+xx+width-1
        if not x0+.2*space<=start<=x0+1.1*space or not x1-1.1*space<=end<=x1-.2*space:continue
        mask=labels[yy:yy+height,xx:xx+width]==label
        columns=np.flatnonzero(mask.any(axis=0))
        if len(columns)/width<.95:continue
        ridge=np.asarray([np.flatnonzero(mask[:,column]).mean()+top+yy for column in columns])
        thickness=mask[:,columns].sum(axis=0)
        if np.median(thickness)>.25*space or np.quantile(thickness,.9)>.4*space:continue
        native_x=left+xx+columns;u=(native_x-start)/(end-start)
        baseline=y0+(y1-y0)*(native_x-x0)/span
        polynomial=np.polyfit(u,ridge-baseline,2)
        fitted=np.polyval(polynomial,u)
        errors=abs(ridge-baseline-fitted)
        if np.median(errors)>max(.7,.06*space) or np.quantile(errors,.9)>max(1.3,.12*space):continue
        if abs(polynomial[0])<1e-8:continue
        apex=-polynomial[1]/(2*polynomial[0])
        if not .25<=apex<=.75:continue
        side=-1 if polynomial[0]>0 else 1
        offsets=side*(ridge-baseline)
        edge_left=offsets[u<=.15];edge_right=offsets[u>=.85];middle=offsets[abs(u-apex)<=.12]
        if min(len(edge_left),len(edge_right),len(middle))<2:continue
        low,high=float(np.median(edge_left)),float(np.median(edge_right))
        peak=float(np.median(middle))
        if not .10*space<=min(low,high) or max(low,high)>1.1*space or peak>1.8*space:continue
        rise=peak-max(low,high)
        if rise<max(2.,.12*space):continue
        accepted.append(dict(side='above' if side<0 else 'below',component_bbox=[start,top+yy,end+1,top+yy+height],
            component_pixels=area,observed_width_px=width,observed_coverage=len(columns)/width,
            measured_rise_spaces=rise/space,endpoint_offsets_spaces=[low/space,high/space],
            median_thickness_spaces=float(np.median(thickness)/space),ridge_error_p90_px=float(np.quantile(errors,.9)),
            actual_ink_connected=True,extrapolated_curve_pixels=0,roi_bbox=[left,top,right,bottom]))
    if not accepted:return dict(accepted=False,reason='no_complete_short_convex_ink_component',roi_bbox=[left,top,right,bottom])
    if len(accepted)!=1:return dict(accepted=False,reason='multiple_short_arc_components',components=accepted)
    return dict(accepted=True,reason='unique_complete_short_convex_ink_component',**accepted[0])


def _shared_proposals(root,geometry,preprocessed):
    """One read-only mapping per call; changed/foreign inputs never reuse it."""
    original_map=previous.ridge.clock_head_map;cached=None
    def same_input_map(candidate_root,candidate_geometry):
        nonlocal cached
        if candidate_root is not root or candidate_geometry is not geometry:
            return original_map(candidate_root,candidate_geometry)
        if cached is None:cached=original_map(root,geometry)
        return cached
    with patch.object(previous.ridge,'clock_head_map',same_input_map):
        report=previous.recover(root,geometry,preprocessed,apply=False)
        mapped=None if 'reason' in report else same_input_map(root,geometry)
    return report,mapped


def recover(root,geometry,preprocessed,*,apply=True):
    """Same API as V5. Existing ties and every V5 decision are retained."""
    started=time.perf_counter()
    report,mapped=_shared_proposals(root,geometry,preprocessed)
    if 'reason' in report:return report
    image=cv2.cvtColor(preprocessed,cv2.COLOR_BGR2GRAY) if preprocessed.ndim==3 else preprocessed
    systems=_xml_systems(root);system_of={n:i for i,notes in enumerate(systems) for n in notes}
    owners=defaultdict(set)
    for note,head in mapped.items():
        if note in system_of:owners[previous._owner_key(head,system_of[note],note.findtext('staff','1'))].add(note)
    chosen={previous.previous._key(p):p for p in report['accepted_pairs']}
    supplements=[];counts=Counter();review=[];terminal_lanes=None
    for a,b,part in previous.original.adjacent_pairs(root):
        first,last=a['note'],b['note'];si,sj=system_of.get(first),system_of.get(last)
        info=dict(part=part,staff=first.findtext('staff','1'),voice=first.findtext('voice','1'),midi=str(a['midi']),
            start_measure=a['measure']+1,stop_measure=b['measure']+1,start_onset=str(a['local_onset']),stop_onset=str(b['local_onset']),
            first_xml_child_index=a['xml_child_index'],last_xml_child_index=b['xml_child_index'])
        key=previous.previous._key(info)
        if key in chosen or len(chosen)>=previous.original.MAXIMUM_CANDIDATES:continue
        reason=None
        if first not in mapped or last not in mapped:reason='short_endpoint_unmapped'
        elif si is None or si!=sj:reason='short_same_system_required'
        elif len(geometry['systems'][si].get('staffs',[]))!=1:reason='short_staff_inventory'
        elif any(_position(n) is None or _position(n)[1]!=int(mapped[n]['position']) for n in (first,last)):reason='short_staff_position_mismatch'
        elif any(n.find('notehead') is not None and n.find('notehead').get('parentheses')=='yes' for n in (first,last)):reason='short_parenthesized_endpoint'
        elif any(n.find('notations/articulations/'+name) is not None for n in (first,last) for name in previous.previous.template.DETACHED_ARTICULATIONS):reason='short_detached_articulation'
        elif any(len(owners[previous._owner_key(mapped[n],si,info['staff'])])!=1 for n in (first,last)):reason='short_shared_head_ownership'
        if reason:counts[reason]+=1;continue
        staff=geometry['systems'][si]['staffs'][0]
        heads=[previous._head_with_space(mapped[n],staff,int(info['staff'])) for n in (first,last)]
        if any(not previous._head_ink(h,image,n.findtext('type') in ('half','whole')) for n,h in zip((first,last),heads)):
            counts['short_endpoint_ink_unproved']+=1;continue
        evidence=short_component_arc(image,*heads,staff,int(info['staff']))
        if not evidence['accepted']:
            counts[evidence['reason']]+=1;review.append(dict(info,reason=evidence['reason']));continue
        if terminal_lanes is None:terminal_lanes=_terminal_meter_lanes(root)
        lane_keys=[(event['measure'],info['staff'],info['voice']) for event in (a,b)]
        if any(key not in terminal_lanes for key in lane_keys):
            counts['short_lane_terminal_meter_unproved']+=1;review.append(dict(info,reason='short_lane_terminal_meter_unproved'));continue
        info.update(system_index=si,head_centers=[[h['x'],h['y']] for h in heads],evidence=evidence,
            optical_detector='native_connected_short_arc_support',decision='exact_adjacent_clocks_unique_heads_complete_short_component',
            terminal_clock_proof=[terminal_lanes[key] for key in lane_keys])
        chosen[key]=info;supplements.append(info);counts['accepted_short_component_pairs']+=1
    if apply:
        for pair in chosen.values():
            measures=root.findall('part')[pair['part']].findall('measure')
            first=list(measures[pair['start_measure']-1])[pair['first_xml_child_index']]
            last=list(measures[pair['stop_measure']-1])[pair['last_xml_child_index']]
            previous.original._add_tie(first,'start');previous.original._add_tie(last,'stop')
            if pair not in supplements and pair not in report.get('system_edge_supplements',[]):
                starts={x.get('number','1'):x for x in first.findall('notations/slur') if x.get('type')=='start'}
                stops={x.get('number','1'):x for x in last.findall('notations/slur') if x.get('type')=='stop'}
                for number in starts.keys()&stops.keys():first.find('notations').remove(starts[number]);last.find('notations').remove(stops[number])
    report.update(algorithm='v5_union_native_short_component_arcs_v6',applied=bool(apply),accepted_pairs=list(chosen.values()),
        short_component_supplements=supplements,short_component_counts=dict(counts),short_component_review=review,
        v5_pairs_retained=True,supplemental_slurs_removed=0,elapsed_s=time.perf_counter()-started)
    report['counts']['accepted_pairs']=len(chosen)
    return report
