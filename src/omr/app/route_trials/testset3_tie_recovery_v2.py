"""Add short same-measure tie evidence to the frozen cross-measure trial.

Same-pitch clock adjacency remains a proposal. Short arcs must pass the exact
same near-head continuity, convexity and thickness evidence as the frozen V1.
Explicit detached articulation, parenthesized heads and competing arcs abstain.
"""
from __future__ import annotations

from collections import Counter
import time
import xml.etree.ElementTree as ET

from align_homr_geometry import _position, _xml_systems
from repair_augmentation_dots import _head_map

from . import testset3_tie_recovery as original

MAXIMUM_SAME_MEASURE_HEAD_SPAN_SPACES=12.
DETACHED_ARTICULATIONS=('staccato','staccatissimo','detached-legato')


def recover(root,geometry,preprocessed,*,apply=True):
    tic=time.perf_counter()
    report=original.recover(root,geometry,preprocessed,apply=False)
    if 'reason' in report:return report
    def detached(notes):
        return any(n.find('notations/articulations/'+name) is not None for n in notes for name in DETACHED_ARTICULATIONS)
    def apply_pair(first,last):
        if not apply:return
        original._add_tie(first,'start');original._add_tie(last,'stop')
        starts={x.get('number','1'):x for x in first.findall('notations/slur') if x.get('type')=='start'}
        stops={x.get('number','1'):x for x in last.findall('notations/slur') if x.get('type')=='stop'}
        for number in starts.keys()&stops.keys():
            first.find('notations').remove(starts[number]);last.find('notations').remove(stops[number])
    retained=[]
    for pair in report['accepted_pairs']:
        measures=root.findall('part')[pair['part']].findall('measure')
        first=list(measures[pair['start_measure']-1])[pair['first_xml_child_index']]
        last=list(measures[pair['stop_measure']-1])[pair['last_xml_child_index']]
        if detached((first,last)):
            report['review'].append(dict(pair,reason='explicit_detached_articulation'))
            continue
        apply_pair(first,last);retained.append(pair)
    report['accepted_pairs']=retained
    image=preprocessed
    if image.ndim==3:
        import cv2
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    systems=_xml_systems(root);system_of={note:i for i,notes in enumerate(systems) for note in notes}
    mapped=_head_map(root,geometry);counts=Counter();new=[]
    for a,b,part in original.adjacent_pairs(root):
        if a['measure']!=b['measure']:continue
        first,last=a['note'],b['note']
        if first not in mapped or last not in mapped:counts['same_measure_unmapped']+=1;continue
        if any(_position(n) is None or _position(n)[1]!=int(mapped[n]['position']) for n in (first,last)):
            counts['same_measure_position_mismatch']+=1;continue
        if any(n.find('notehead') is not None and n.find('notehead').get('parentheses')=='yes' for n in (first,last)):
            counts['same_measure_parenthesized']+=1;continue
        if detached((first,last)):
            counts['explicit_detached_articulation']+=1;continue
        heads=[dict(mapped[n]) for n in (first,last)];s=system_of[first];staff=geometry['systems'][s]['staffs'][0]
        for head in heads:
            if not head.get('staff_space'):
                lines=original._staff_lines(staff,[head['x']],int(first.findtext('staff','1')))
                if len(lines)!=5:break
                import numpy as np
                head['staff_space']=float(np.median(np.diff([v[0] for v in lines])))
        if any(not h.get('staff_space') for h in heads):counts['same_measure_staff_space_unavailable']+=1;continue
        space=sum(h['staff_space'] for h in heads)/2
        if not 0<heads[1]['x']-heads[0]['x']<=MAXIMUM_SAME_MEASURE_HEAD_SPAN_SPACES*space:
            counts['same_measure_not_short_adjacent_arc']+=1;continue
        evidence=original.near_head_arc(image,*heads,staff,int(first.findtext('staff','1')))
        if not evidence['accepted']:counts[evidence['reason']]+=1;continue
        info=dict(part=part,staff=first.findtext('staff','1'),voice=first.findtext('voice','1'),midi=str(a['midi']),
                  start_measure=a['measure']+1,stop_measure=b['measure']+1,start_onset=str(a['local_onset']),
                  stop_onset=str(b['local_onset']),first_xml_child_index=a['xml_child_index'],
                  last_xml_child_index=b['xml_child_index'],system_index=s,
                  head_centers=[[h['x'],h['y']] for h in heads],evidence=evidence,
                  decision='short_same_measure_near_head_arc')
        apply_pair(first,last)
        counts['accepted_pairs']+=1;new.append(info)
    accepted_keys={(r['start_measure'],r['first_xml_child_index'],r['last_xml_child_index']) for r in new}
    report['review']=[r for r in report['review'] if (r['start_measure'],r['first_xml_child_index'],r['last_xml_child_index']) not in accepted_keys]
    report['accepted_pairs']+=new
    report['same_measure_counts']=dict(counts)
    report['counts']['accepted_pairs']=len(report['accepted_pairs'])
    report['applied']=bool(apply)
    report['elapsed_s']=time.perf_counter()-tic
    report['algorithm']='adjacent_same_pitch_cross_measure_and_short_near_head_arc_v2'
    report['limitations']=[l for l in report['limitations'] if not l.startswith('Same-measure')]
    report['limitations'].append('Close repeated-note slurs can be semantically ambiguous; acceptance is limited to short near-head arcs with exact clock adjacency and no detached articulation')
    return report
