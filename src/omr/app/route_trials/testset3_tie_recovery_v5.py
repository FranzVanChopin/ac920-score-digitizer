"""Retain V4 and verify conventional ties ending at a physical system edge.

Some engravers draw only the outgoing piece of a broken-system tie.  A
missing incoming piece supplies no positive evidence.  Acceptance instead
requires an observed thin convex ridge from the first head to the measured
right staff boundary, a unique first head of the next system, and exact
same-pitch/staff/voice clock adjacency.  No labels, counts or work IDs enter.
"""
from __future__ import annotations

from collections import Counter,defaultdict
import math
import time

import cv2
import numpy as np

from align_homr_geometry import _position,_xml_systems
from . import testset3_tie_recovery_v4 as previous
from .testset3_measure_recovery import _head_ink

ridge=previous.ridge
original=ridge.original


def _owner_key(head,system,staff):
    return system,staff,round(float(head['x'])),round(float(head['y'])),int(head['position'])


def _head_with_space(head,staff,number):
    head=dict(head)
    if not head.get('staff_space'):
        lines=original._staff_lines(staff,[head['x']],number)
        head['staff_space']=float(np.median(np.diff([row[0] for row in lines]))) if len(lines)==5 else 0.
    return head


def recover(root,geometry,preprocessed,*,apply=True):
    """V4-compatible bounded proposal/apply API; existing marks are retained."""
    started=time.perf_counter()
    report=previous.recover(root,geometry,preprocessed,apply=False)
    if 'reason' in report:return report
    image=cv2.cvtColor(preprocessed,cv2.COLOR_BGR2GRAY) if preprocessed.ndim==3 else preprocessed
    systems=_xml_systems(root);system_of={note:i for i,notes in enumerate(systems) for note in notes}
    mapped=ridge.clock_head_map(root,geometry);owners=defaultdict(set)
    for note,head in mapped.items():
        if note in system_of:owners[_owner_key(head,system_of[note],note.findtext('staff','1'))].add(note)
    chosen={previous._key(pair):pair for pair in report['accepted_pairs']}
    counts=Counter();review=[];supplements=[]
    for a,b,part in original.adjacent_pairs(root):
        first,last=a['note'],b['note'];si,sj=system_of.get(first),system_of.get(last)
        if si is None or sj is None or si==sj:continue
        info=dict(part=part,staff=first.findtext('staff','1'),voice=first.findtext('voice','1'),midi=str(a['midi']),
                  start_measure=a['measure']+1,stop_measure=b['measure']+1,start_onset=str(a['local_onset']),
                  stop_onset=str(b['local_onset']),first_xml_child_index=a['xml_child_index'],last_xml_child_index=b['xml_child_index'])
        def reject(reason,**details):
            counts[reason]+=1;review.append(dict(info,reason=reason,**details))
        if len(chosen)>=original.MAXIMUM_CANDIDATES:
            counts['supplement_candidate_budget_exhausted']+=1;break
        if sj!=si+1 or b['measure']!=a['measure']+1 or b['local_onset']!=0:
            reject('nonconsecutive_system_or_noninitial_target_clock');continue
        if any(n not in mapped for n in (first,last)):
            reject('split_endpoint_head_unmapped');continue
        if any(len(geometry['systems'][index].get('staffs',[]))!=1 for index in (si,sj)):
            reject('unsupported_split_staff_inventory');continue
        if any(_position(n) is None or _position(n)[1]!=int(mapped[n]['position']) for n in (first,last)):
            reject('split_physical_staff_position_mismatch');continue
        if any(n.find('notehead') is not None and n.find('notehead').get('parentheses')=='yes' for n in (first,last)):
            reject('split_parenthesized_endpoint');continue
        if any(n.find('notations/articulations/'+name) is not None for n in (first,last) for name in previous.template.DETACHED_ARTICULATIONS):
            reject('split_explicit_detached_articulation');continue
        if any(len(owners[_owner_key(mapped[n],system_of[n],info['staff'])])!=1 for n in (first,last)):
            reject('split_head_owned_by_multiple_events');continue
        st0=geometry['systems'][si]['staffs'][0];st1=geometry['systems'][sj]['staffs'][0];number=int(info['staff'])
        heads=[_head_with_space(mapped[n],staff,number) for n,staff in ((first,st0),(last,st1))]
        if any(not math.isfinite(float(h['staff_space'])) or not 4<=h['staff_space']<=64 for h in heads):
            reject('split_staff_space_unavailable');continue
        if any(not _head_ink(h,image,n.findtext('type') in ('half','whole')) for n,h in zip((first,last),heads)):
            reject('split_endpoint_notehead_ink_unproved');continue
        # A target later in the system could be a phrase/slur endpoint.  Its
        # incoming physical column must be the first on the target staff.
        s=heads[1]['staff_space']
        earlier=[h for n,h in mapped.items() if system_of.get(n)==sj and n.findtext('staff','1')==info['staff']
                 and float(h['x'])<heads[1]['x']-.7*s]
        if earlier:reject('split_target_not_first_physical_column');continue
        boundary=float(st0['x_range'][1]);space=heads[0]['staff_space']
        if not math.isfinite(boundary) or not heads[0]['x']+1.8*space<=boundary<image.shape[1]-.2*space:
            reject('split_terminal_span_or_frame_unproved');continue
        terminal=dict(heads[0],x=boundary+.1*space)
        evidence=ridge.near_head_arc(image,heads[0],terminal,st0,number)
        if not evidence['accepted']:
            reject('split_terminal_arc_unproved',evidence=evidence);continue
        info.update(system_index=si,last_system_index=sj,head_centers=[[h['x'],h['y']] for h in heads],
                    evidence=dict(evidence,physical_staff_right_boundary=boundary,virtual_endpoint_x=terminal['x'],
                                  target_unique_first_staff_column=True,both_endpoint_head_ink=True,
                                  incoming_arc_assumed=False),
                    optical_detector='unchanged_measured_ridge_to_physical_system_edge',
                    decision='exact_adjacent_system_clocks_and_observed_terminal_arc')
        key=previous._key(info)
        if key not in chosen:chosen[key]=info;supplements.append(info);counts['accepted_supplemental_pairs']+=1
    if apply:
        for pair in chosen.values():
            measures=root.findall('part')[pair['part']].findall('measure')
            first=list(measures[pair['start_measure']-1])[pair['first_xml_child_index']]
            last=list(measures[pair['stop_measure']-1])[pair['last_xml_child_index']]
            original._add_tie(first,'start');original._add_tie(last,'stop')
            # Preserve V4's exact original slur policy. New system-edge pairs
            # do not delete slurs: a later octave guard may revoke new ties.
            if pair not in supplements:
                starts={x.get('number','1'):x for x in first.findall('notations/slur') if x.get('type')=='start'}
                stops={x.get('number','1'):x for x in last.findall('notations/slur') if x.get('type')=='stop'}
                for number in starts.keys()&stops.keys():
                    first.find('notations').remove(starts[number]);last.find('notations').remove(stops[number])
    report.update(algorithm='v4_union_with_observed_system_terminal_ties_v5',applied=bool(apply),
                  accepted_pairs=list(chosen.values()),elapsed_s=time.perf_counter()-started,
                  system_edge_supplements=supplements,system_edge_review=review,system_edge_counts=dict(counts),
                  reference_used=False,additional_model_used=False,additional_PL_used=False,
                  v4_pairs_retained=True,existing_tie_and_tied_marks_removed=0,
                  previous_pipeline_slur_policy_retained=True,supplemental_slurs_removed=0)
    report['counts']['accepted_pairs']=len(chosen)
    report['limitations'].append('System-edge supplement requires observed outgoing ridge; short fragments, cross-page ties and unresolved head ownership abstain')
    return report
