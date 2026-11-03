"""Image-backed, adjacent same-pitch ties, initially across measure boundaries.

Pitch equality is only a proposal. A shallow, thin, continuous near-head arc
must be observed in the exact Homr geometry raster. Same-measure repeated-note
slurs and split-system fragments remain unresolved in this first experiment.
No reference score, work name, learned model or existing slur is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import math
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from align_homr_geometry import _position, _xml_systems
from repair_augmentation_dots import _head_map

MAXIMUM_PIXELS=16_000_000
MAXIMUM_DIMENSION=8192
MAXIMUM_EVENTS_PER_MEASURE=128
MAXIMUM_CANDIDATES=512
MAXIMUM_ROI_PIXELS=131072
MAXIMUM_ROI_WIDTH=1024
MAXIMUM_ROI_HEIGHT=160
MAXIMUM_SYSTEMS=64
PITCH={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}


def _midi(note):
    pitch=note.find('pitch')
    if pitch is None or pitch.findtext('step') not in PITCH:return None
    try:return Fraction(12*(int(pitch.findtext('octave'))+1)+PITCH[pitch.findtext('step')])+Fraction(pitch.findtext('alter','0'))
    except (ValueError,TypeError,ZeroDivisionError):return None


def adjacent_pairs(root, diagnostics=None):
    """Exact XML clock adjacency, including chord members and barriers at rests."""
    diagnostics=diagnostics if diagnostics is not None else Counter()
    pairs=[]
    for part_index,part in enumerate(root.findall('part')):
        divisions=1;meter=Fraction(4);absolute=Fraction(0);lanes=defaultdict(list);segment=0
        for mi,measure in enumerate(part.findall('measure')):
            if len(measure.findall('note'))>MAXIMUM_EVENTS_PER_MEASURE:
                diagnostics['event_budget_measures']+=1;segment+=1;absolute+=meter;continue
            cursor=previous=Fraction(0);events=[];extent=Fraction(0);invalid=False
            for ni,element in enumerate(measure):
                try:
                    if element.tag=='attributes':
                        divisions=int(element.findtext('divisions',str(divisions)))
                        if divisions<=0:raise ValueError('divisions')
                        time_element=element.find('time')
                        if time_element is not None:
                            meter=Fraction(4*int(time_element.findtext('beats')),int(time_element.findtext('beat-type')))
                            if not 0<meter<=16:raise ValueError('meter')
                    elif element.tag in ('backup','forward'):
                        cursor+=Fraction(element.findtext('duration','0'))/divisions*(1 if element.tag=='forward' else -1)
                        if cursor<0:invalid=True
                    elif element.tag=='note':
                        grace=element.find('grace') is not None
                        chord=element.find('chord') is not None
                        onset=previous if chord else cursor
                        duration=Fraction(element.findtext('duration','0'))/divisions
                        if duration<0:invalid=True
                        if not chord:
                            previous=onset
                            if not grace:cursor+=duration
                        extent=max(extent,onset+duration)
                        if grace or element.get('print-object')=='no':continue
                        lane=(element.findtext('staff','1'),element.findtext('voice','1'),segment)
                        events.append(dict(note=element,lane=lane,midi=_midi(element),measure=mi,
                                           xml_child_index=ni,onset=absolute+onset,end=absolute+onset+duration,
                                           duration=duration,local_onset=onset))
                except (ValueError,TypeError,ZeroDivisionError):invalid=True
            if invalid or extent>meter and measure.get('implicit')!='yes':
                diagnostics['invalid_clock_measures']+=1;segment+=1;absolute+=meter;continue
            for event in sorted(events,key=lambda e:(e['onset'],e['xml_child_index'])):
                lanes[event['lane']].append(event)
            absolute+=extent if measure.get('implicit')=='yes' else meter
        for lane,events in lanes.items():
            groups=[]
            for event in events:
                if not groups or groups[-1][0]['onset']!=event['onset']:groups.append([])
                groups[-1].append(event)
            for first,last in zip(groups,groups[1:]):
                for a in first:
                    if a['midi'] is None or a['duration']<=0:continue
                    matches=[b for b in last if b['midi']==a['midi'] and a['end']==b['onset'] and b['duration']>0]
                    if len(matches)!=1 or sum(x['midi']==a['midi'] for x in first)!=1:continue
                    b=matches[0]
                    pairs.append((a,b,part_index))
                    if len(pairs)>=MAXIMUM_CANDIDATES:
                        diagnostics['candidate_budget_exhausted']+=1;return pairs
    return pairs


def _staff_lines(staff,xs,staff_number):
    grid=staff.get('grid',[])
    if not grid:return []
    points=sorted(grid,key=lambda p:p['x']);columns=np.asarray([p['x'] for p in points])
    count=len(points[0]['y'])
    indices=range((staff_number-1)*5,staff_number*5) if count==10 else range(5) if count==5 else []
    return [np.interp(xs,columns,[p['y'][j] for p in points]) for j in indices]


def near_head_arc(image, first, last, staff, staff_number):
    """Dense pixel support for a shallow convex arc close to both noteheads."""
    x0,y0=float(first['x']),float(first['y']);x1,y1=float(last['x']),float(last['y'])
    s=float(np.mean([first['staff_space'],last['staff_space']]))
    if (not math.isfinite(s) or not 4<=s<=64 or not 1.8*s<=x1-x0<=48*s
            or abs(y1-y0)>.25*s):return dict(accepted=False,reason='endpoint_geometry_outside_tie_window')
    left,right=math.floor(x0+.55*s),math.ceil(x1-.55*s)
    top,bottom=math.floor(min(y0,y1)-1.7*s),math.ceil(max(y0,y1)+1.7*s)
    if (left<0 or top<0 or right>image.shape[1] or bottom>image.shape[0]
            or right-left>MAXIMUM_ROI_WIDTH or bottom-top>MAXIMUM_ROI_HEIGHT
            or (right-left)*(bottom-top)>MAXIMUM_ROI_PIXELS):
        return dict(accepted=False,reason='arc_roi_budget_or_clipping')
    crop=image[top:bottom,left:right];ink=crop<160
    xs=np.arange(left,right,dtype=float)
    lines=_staff_lines(staff,xs,staff_number)
    staff_mask=np.zeros_like(ink)
    for line in lines:
        ys=np.rint(line-top).astype(int)
        for delta in range(-max(1,round(.06*s)),max(1,round(.06*s))+1):
            rows=ys+delta;valid=(rows>=0)&(rows<ink.shape[0]);staff_mask[rows[valid],np.flatnonzero(valid)]=True
    residual=ink&~staff_mask
    t=(xs-x0)/(x1-x0);baseline=y0+(y1-y0)*t
    radius=max(1,round(.10*s));deltas=np.arange(-radius,radius+1)
    observed=[]
    for sign in (-1,1):
        for offset in (.3,.45,.6,.75):
            for bend in (.3,.45,.6,.75,.9):
                predicted=baseline+sign*s*(offset+4*bend*t*(1-t))
                row=np.rint(predicted-top).astype(int)
                sample_rows=row[:,None]+deltas
                valid=(sample_rows>=0)&(sample_rows<ink.shape[0])
                samples=residual[np.clip(sample_rows,0,ink.shape[0]-1),np.arange(len(xs))[:,None]]&valid
                is_staff=staff_mask[np.clip(row,0,ink.shape[0]-1),np.arange(len(xs))]
                eligible=~is_staff
                if eligible.mean()<.75 or not eligible.any():continue
                hit=samples.any(axis=1)
                coverage=float(hit[eligible].mean())
                if coverage<.94:continue
                nearest=np.argmin(np.where(samples,np.abs(deltas)[None,:],1000),axis=1)
                actual=row+deltas[nearest]+top
                # Staff crossings are known missing observations, not ink.
                selected=hit&eligible
                middle=selected&(t>=.4)&(t<=.6)
                edge=selected&((t<=.2)|(t>=.8))
                if middle.sum()<3 or edge.sum()<3:continue
                offsets=sign*(actual-baseline)
                rise=float(np.median(offsets[middle])-np.median(offsets[edge]))
                if rise<.20*s:continue
                # A flat thick beam can fill an arched corridor. Require a
                # thin local ridge where the accepted path is observed.
                band=np.arange(-max(2,round(.3*s)),max(2,round(.3*s))+1)
                rows=np.rint(actual-top).astype(int)[:,None]+band
                local=residual[np.clip(rows,0,ink.shape[0]-1),np.arange(len(xs))[:,None]]
                thickness=np.median(local[selected].sum(axis=1))
                if thickness>.26*s:continue
                missing=~(hit|is_staff);edges=np.diff(np.r_[False,missing,False].astype(np.int8))
                longest=max((b-a for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))),default=0)
                if longest>max(3,round(.2*s)):continue
                observed.append(dict(side='above' if sign<0 else 'below',coverage=coverage,
                    staff_excluded_fraction=float(is_staff.mean()),rise_staff_spaces=rise/s,
                    median_ridge_thickness_staff_spaces=float(thickness/s),
                    maximum_observed_gap_px=int(longest),endpoint_offset_spaces=offset,
                    curve_bend_spaces=bend,roi_bbox=[left,top,right,bottom]))
    if not observed:return dict(accepted=False,reason='no_continuous_thin_shallow_near_head_arc',roi_bbox=[left,top,right,bottom])
    sides={x['side'] for x in observed}
    if len(sides)>1:return dict(accepted=False,reason='competing_above_and_below_arcs',roi_bbox=[left,top,right,bottom])
    best=max(observed,key=lambda e:(e['coverage'],-e['median_ridge_thickness_staff_spaces']))
    return dict(accepted=True,reason='unique_near_head_convex_arc',**best)


def _add_tie(note,kind):
    if not any(t.get('type')==kind for t in note.findall('tie')):
        tie=ET.Element('tie',{'type':kind});duration=note.find('duration')
        note.insert(list(note).index(duration)+1 if duration is not None else 1,tie)
    notations=note.find('notations')
    if notations is None:notations=ET.SubElement(note,'notations')
    if not any(t.get('type')==kind for t in notations.findall('tied')):
        ET.SubElement(notations,'tied',{'type':kind})


def recover(root,geometry,preprocessed,*,apply=True):
    """Add only independently evidenced paired tie/tied marks; preserve clocks."""
    tic=time.perf_counter();counts=Counter();accepted=[];review=[]
    if (not isinstance(preprocessed,np.ndarray) or preprocessed.dtype!=np.uint8
            or preprocessed.ndim not in (2,3)):
        raise ValueError('Expected uint8 actual Homr preprocessed raster')
    if preprocessed.ndim==3:preprocessed=cv2.cvtColor(preprocessed,cv2.COLOR_BGR2GRAY)
    h,w=preprocessed.shape
    if h*w>MAXIMUM_PIXELS or max(h,w)>MAXIMUM_DIMENSION:raise ValueError('Tie raster exceeds image budget')
    if geometry.get('preprocessed_size')!=[w,h]:
        return dict(applied=False,accepted_pairs=[],review=[],reason='preprocessed_coordinate_frame_mismatch',counts={},reference_used=False)
    if len(geometry.get('systems',[]))>MAXIMUM_SYSTEMS:raise ValueError('Tie geometry exceeds system budget')
    systems=_xml_systems(root)
    if len(systems)!=len(geometry.get('systems',[])):
        return dict(applied=False,accepted_pairs=[],review=[],reason='system_count_mismatch',counts={},reference_used=False)
    mapped=_head_map(root,geometry);system_of={note:i for i,notes in enumerate(systems) for note in notes}
    pairs=adjacent_pairs(root,counts);counts['adjacent_same_pitch_pairs']=len(pairs)
    for a,b,part in pairs:
        first,last=a['note'],b['note']
        info=dict(part=part,staff=first.findtext('staff','1'),voice=first.findtext('voice','1'),
                  midi=str(a['midi']),start_measure=a['measure']+1,stop_measure=b['measure']+1,
                  start_onset=str(a['local_onset']),stop_onset=str(b['local_onset']),
                  first_xml_child_index=a['xml_child_index'],last_xml_child_index=b['xml_child_index'])
        reason=None
        if a['measure']==b['measure']:reason='same_measure_tie_slur_semantic_ambiguity'
        elif b['measure']!=a['measure']+1:reason='nonadjacent_measure_boundaries'
        elif first not in mapped or last not in mapped:reason='head_localization_not_unique'
        elif system_of[first]!=system_of[last]:reason='split_system_arc_not_yet_verified'
        elif any(n.find('notehead') is not None and n.find('notehead').get('parentheses')=='yes' for n in (first,last)):
            reason='parenthesized_head'
        elif any(_position(n) is None or _position(n)[1]!=int(mapped[n]['position']) for n in (first,last)):
            reason='head_staff_position_mismatch'
        if reason:
            counts[reason]+=1;review.append(dict(info,reason=reason));continue
        heads=[dict(mapped[n]) for n in (first,last)];s=system_of[first];staff=geometry['systems'][s]['staffs'][0]
        for head in heads:
            if not head.get('staff_space'):
                line=_staff_lines(staff,[head['x']],int(info['staff']))
                head['staff_space']=float(np.median(np.diff([v[0] for v in line]))) if len(line)==5 else 0
        evidence=near_head_arc(preprocessed,*heads,staff,int(info['staff']))
        if not evidence['accepted']:
            counts[evidence['reason']]+=1;review.append(dict(info,evidence=evidence,reason=evidence['reason']));continue
        if apply:
            _add_tie(first,'start');_add_tie(last,'stop')
            # Remove only a paired slur joining these exact endpoints. Other
            # long/multi-note slurs and their numbers remain unchanged.
            starts={x.get('number','1'):x for x in first.findall('notations/slur') if x.get('type')=='start'}
            stops={x.get('number','1'):x for x in last.findall('notations/slur') if x.get('type')=='stop'}
            for number in starts.keys()&stops.keys():
                first.find('notations').remove(starts[number]);last.find('notations').remove(stops[number])
        counts['accepted_pairs']+=1;accepted.append(dict(info,system_index=s,
            head_centers=[[heads[0]['x'],heads[0]['y']],[heads[1]['x'],heads[1]['y']]],evidence=evidence))
    return dict(algorithm='adjacent_same_pitch_cross_measure_near_head_arc_v1',applied=bool(apply),
                accepted_pairs=accepted,review=review,counts=dict(counts),elapsed_s=time.perf_counter()-tic,
                reference_used=False,additional_model_used=False,additional_PL_used=False,
                existing_pitch_duration_voice_onset_preserved=True,
                limits=dict(maximum_candidates=MAXIMUM_CANDIDATES,maximum_events_per_measure=MAXIMUM_EVENTS_PER_MEASURE,
                            maximum_roi_pixels=MAXIMUM_ROI_PIXELS,maximum_roi_width=MAXIMUM_ROI_WIDTH,maximum_roi_height=MAXIMUM_ROI_HEIGHT),
                limitations=['Same-measure repeated-note slur/tie ambiguity abstains',
                             'Split-system and cross-page arcs remain unresolved',
                             'A same-pitch pair alone is never accepted without arc pixels'])
