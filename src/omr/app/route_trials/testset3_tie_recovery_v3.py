"""Observe real thin arc ridges and supplement missing clock-ordered heads.

Whole curve support is measured on actual vertical-run centres, so a flat
beam cannot imitate curvature by choosing different pixels inside its width.
Close endpoints, unique same-pitch clocks/heads and detached-attack rejection
remain mandatory. No reference score or model is an inference input.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
import time
from unittest.mock import patch

import numpy as np

from align_homr_geometry import _position,_xml_systems,_head_staff
from repair_ledger_chords import _align
from . import testset3_tie_recovery_v2 as previous

original=previous.original
_old_head_map=original._head_map
MAXIMUM_SAME_MEASURE_HEAD_SPAN_SPACES=32.


def _clock_locations(root):
    result={}
    for part in root.findall('part'):
        divisions=1
        for mi,measure in enumerate(part.findall('measure')):
            if len(measure.findall('note'))>original.MAXIMUM_EVENTS_PER_MEASURE:continue
            cursor=last=Fraction(0);local={};invalid=False
            try:
                for element in measure:
                    if element.tag=='attributes':
                        divisions=int(element.findtext('divisions',str(divisions)))
                        if divisions<=0:raise ValueError('divisions')
                    elif element.tag in ('backup','forward'):
                        cursor+=Fraction(element.findtext('duration','0'))/divisions*(1 if element.tag=='forward' else -1)
                        if cursor<0:invalid=True
                    elif element.tag=='note':
                        chord=element.find('chord') is not None
                        onset=last if chord else cursor
                        if not chord:
                            last=onset
                            if element.find('grace') is None:cursor+=Fraction(element.findtext('duration','0'))/divisions
                        if element.find('pitch') is not None and element.find('grace') is None:local[element]=(mi,onset)
            except (ValueError,TypeError,ZeroDivisionError):invalid=True
            if not invalid:result.update(local)
    return result


def clock_head_map(root,geometry):
    """Fill only missing unique positions; existing physical head claims win."""
    base=_old_head_map(root,geometry);systems=_xml_systems(root)
    if len(systems)!=len(geometry.get('systems',[])):return base
    clocks=_clock_locations(root)
    for si,(notes,system) in enumerate(zip(systems,geometry['systems'])):
        if len(system.get('staffs',[]))!=1:continue
        staff=system['staffs'][0]
        for number in (1,2):
            groups=defaultdict(list)
            for note in notes:
                p=_position(note)
                if p is not None and p[0]==number and note in clocks:groups[(*clocks[note],p[1])].append(note)
            keys=sorted(groups);representatives=[groups[k][0] for k in keys]
            raw=sorted((h for h in staff['notes'] if _head_staff(staff,h)==number),key=lambda h:(h['x'],h['position']))
            # Chord x jitter must not scramble physical diatonic order.
            headgroups=[]
            for head in raw:
                if not headgroups or head['x']-headgroups[-1][-1]['x']>3:headgroups.append([])
                headgroups[-1].append(head)
            heads=[h for group in headgroups for h in sorted(group,key=lambda h:h['position'])]
            supplemental=_align(representatives,heads)
            occupied={id(h):clocks.get(n) for n,h in base.items() if n in notes}
            for key,note in zip(keys,representatives):
                head=supplemental.get(note)
                if head is None or _position(note)[1]!=int(head['position']):continue
                if id(head) in occupied and occupied[id(head)]!=clocks[note]:continue
                for n in groups[key]:base.setdefault(n,head)
                occupied[id(head)]=clocks[note]
    return base


def near_head_arc(image,first,last,staff,staff_number):
    x0,y0=float(first['x']),float(first['y']);x1,y1=float(last['x']),float(last['y'])
    s=float(np.mean([first['staff_space'],last['staff_space']]))
    if (not math.isfinite(s) or not 4<=s<=64 or not 1.8*s<=x1-x0<=48*s
            or abs(y1-y0)>.35*s):return dict(accepted=False,reason='endpoint_geometry_outside_tie_window')
    left,right=math.floor(x0+.75*s),math.ceil(x1-.75*s)
    top,bottom=math.floor(min(y0,y1)-2.2*s),math.ceil(max(y0,y1)+2.2*s)
    if (right-left<6 or left<0 or top<0 or right>image.shape[1] or bottom>image.shape[0]
            or right-left>original.MAXIMUM_ROI_WIDTH or bottom-top>original.MAXIMUM_ROI_HEIGHT
            or (right-left)*(bottom-top)>original.MAXIMUM_ROI_PIXELS):
        return dict(accepted=False,reason='arc_roi_budget_or_clipping')
    ink=image[top:bottom,left:right]<160;xs=np.arange(left,right,dtype=float)
    staff_mask=np.zeros_like(ink)
    for line in original._staff_lines(staff,xs,staff_number):
        ys=np.rint(line-top).astype(int)
        for delta in range(-max(1,round(.055*s)),max(1,round(.055*s))+1):
            rows=ys+delta;valid=(rows>=0)&(rows<ink.shape[0]);staff_mask[rows[valid],np.flatnonzero(valid)]=True
    residual=ink&~staff_mask
    # Real vertical-run centres, bounded eight hypotheses per column. Long
    # head/stem blobs are excluded instead of supplying arbitrary path pixels.
    centers=np.full((len(xs),8),np.nan);widths=np.zeros((len(xs),8))
    for i in range(len(xs)):
        edges=np.diff(np.r_[False,residual[:,i],False].astype(np.int8))
        runs=[((a+b-1)/2+top,b-a) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a<=.5*s]
        baseline=y0+(y1-y0)*(xs[i]-x0)/(x1-x0)
        for j,(center,width) in enumerate(sorted(runs,key=lambda r:abs(r[0]-baseline))[:8]):centers[i,j]=center;widths[i,j]=width
    t=(xs-x0)/(x1-x0);baseline=y0+(y1-y0)*t;observed=[]
    for sign in (-1,1):
        for offset in (-.2,0.,.2,.4,.6):
            for bend in (.35,.55,.8,1.1,1.4,1.7):
                if offset+bend>1.85:continue
                predicted=baseline+sign*s*(offset+4*bend*t*(1-t))
                distances=np.where(np.isfinite(centers),abs(centers-predicted[:,None]),10000)
                index=distances.argmin(axis=1);actual=centers[np.arange(len(xs)),index]
                hit=distances[np.arange(len(xs)),index]<=max(1.5,.16*s)
                row=np.clip(np.rint(predicted-top).astype(int),0,residual.shape[0]-1)
                excluded=staff_mask[row,np.arange(len(xs))]
                eligible=~excluded
                if eligible.mean()<.70 or not eligible.any():continue
                coverage=float(hit[eligible].mean())
                if coverage<.90:continue
                valid=hit&eligible
                edge_window=(t<=.23)|(t>=.77)
                adapted_edge_sampling=(int(edge_window.sum())<3 and x1-x0<4.5*s)
                if adapted_edge_sampling:
                    # Select by nonstaff eligibility, never observed ink hits.
                    eligible_columns=np.flatnonzero(eligible)
                    edge_count=max(1,math.ceil(.15*len(eligible_columns)))
                    edge_window=np.zeros(len(xs),dtype=bool)
                    edge_window[eligible_columns[:edge_count]]=True
                    edge_window[eligible_columns[-edge_count:]]=True
                mid=valid&(t>=.4)&(t<=.6);edge=valid&edge_window
                if mid.sum()<3 or edge.sum()<3:continue
                offsets=sign*(actual-baseline)
                rise=float(np.median(offsets[mid])-np.median(offsets[edge]))
                if rise<.22*s:continue
                thickness=float(np.median(widths[np.arange(len(xs)),index][valid]))
                if thickness>.4*s:continue
                gaps=~(hit|excluded);e=np.diff(np.r_[False,gaps,False].astype(np.int8))
                longest=max((b-a for a,b in zip(np.flatnonzero(e==1),np.flatnonzero(e==-1))),default=0)
                if longest>max(3,round(.3*s)):continue
                observed.append(dict(side='above' if sign<0 else 'below',coverage=coverage,
                    staff_excluded_fraction=float(excluded.mean()),rise_staff_spaces=rise/s,
                    median_ridge_thickness_staff_spaces=thickness/s,maximum_observed_gap_px=int(longest),
                    endpoint_offset_spaces=offset,curve_bend_spaces=bend,roi_bbox=[left,top,right,bottom],
                    measured_path='actual_thin_vertical_run_centres_not_arbitrary_ink_pixels'))
                if adapted_edge_sampling:
                    observed[-1].update(edge_sampling='roi_nonstaff_eligible_outer_15_percent',
                        original_edge_capacity=int(((t<=.23)|(t>=.77)).sum()),
                        selected_edge_capacity=int(edge_window.sum()),
                        measured_edge_points=int(edge.sum()),measured_mid_points=int(mid.sum()))
    if not observed:return dict(accepted=False,reason='no_continuous_thin_near_head_ridge',roi_bbox=[left,top,right,bottom])
    if len({x['side'] for x in observed})>1:
        if any(x.get('edge_sampling') for x in observed):
            # Retain the old rejection instead of introducing new negative
            # evidence against independently supported template pairs.
            return dict(accepted=False,reason='no_continuous_thin_near_head_ridge',
                roi_bbox=[left,top,right,bottom],short_extension_competing_sides_abstained=True)
        return dict(accepted=False,reason='competing_above_and_below_arcs',roi_bbox=[left,top,right,bottom])
    return dict(accepted=True,reason='unique_near_head_convex_ridge',**max(observed,key=lambda e:(e['coverage'],-e['median_ridge_thickness_staff_spaces'])))


def recover(root,geometry,preprocessed,*,apply=True):
    tic=time.perf_counter()
    # The cross-bar and same-bar passes alter tie/slur marks only. Their
    # pitch/clock/head inputs stay identical, so share one mapping locally.
    # Do not retain mutable XML or image-derived heads across recover calls.
    mapped=None
    def head_map_once(candidate_root,candidate_geometry):
        nonlocal mapped
        if candidate_root is not root or candidate_geometry is not geometry:
            return clock_head_map(candidate_root,candidate_geometry)
        if mapped is None:mapped=clock_head_map(root,geometry)
        return mapped
    with patch.object(original,'near_head_arc',near_head_arc),patch.object(original,'_head_map',head_map_once), \
         patch.object(previous,'_head_map',head_map_once),patch.object(previous,'MAXIMUM_SAME_MEASURE_HEAD_SPAN_SPACES',MAXIMUM_SAME_MEASURE_HEAD_SPAN_SPACES):
        report=previous.recover(root,geometry,preprocessed,apply=apply)
    report.update(algorithm='adjacent_same_pitch_measured_near_head_ridge_v3',elapsed_s=time.perf_counter()-tic,
                  supplemental_clock_head_mapping_only_for_unmapped=True,
                  previous_unique_head_claims_preserved=True,
                  curve_centres_measured_before_curvature_test=True)
    return report
