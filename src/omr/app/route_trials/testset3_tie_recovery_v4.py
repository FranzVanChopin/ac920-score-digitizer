"""Union of two independent bounded optical tie profiles on the same clocks.

The measured-ridge detector supplies broad, near-head curve support. The
earlier thin-template detector complements shallow raster crossings. Both
require image evidence, unique same-pitch heads and exact adjacent clocks;
measured competing arcs are never overridden by the supplemental detector.
"""
from __future__ import annotations

import time

from . import testset3_tie_recovery_v3 as ridge
from . import testset3_tie_recovery_v2 as template


def _key(pair):
    return (pair['part'],pair['staff'],pair['voice'],pair['midi'],pair['start_measure'],pair['stop_measure'],
            pair['first_xml_child_index'],pair['last_xml_child_index'])


def recover(root,geometry,preprocessed,*,apply=True):
    tic=time.perf_counter()
    primary=ridge.recover(root,geometry,preprocessed,apply=False)
    if 'reason' in primary:return primary
    secondary=template.recover(root,geometry,preprocessed,apply=False)
    chosen={_key(p):dict(p,optical_detector='measured_vertical_ridge') for p in primary['accepted_pairs']}
    supplemental=0;competing=0
    for pair in secondary['accepted_pairs']:
        if _key(pair) in chosen:continue
        measures=root.findall('part')[pair['part']].findall('measure')
        first=list(measures[pair['start_measure']-1])[pair['first_xml_child_index']]
        last=list(measures[pair['stop_measure']-1])[pair['last_xml_child_index']]
        heads=ridge.clock_head_map(root,geometry)
        if first not in heads or last not in heads:continue
        system=pair['system_index'];staff=geometry['systems'][system]['staffs'][0]
        import cv2
        image=cv2.cvtColor(preprocessed,cv2.COLOR_BGR2GRAY) if preprocessed.ndim==3 else preprocessed
        evidence=ridge.near_head_arc(image,heads[first],heads[last],staff,int(pair['staff']))
        if evidence['reason']=='competing_above_and_below_arcs':competing+=1;continue
        chosen[_key(pair)]=dict(pair,optical_detector='supplemental_thin_template',
                                measured_ridge_counter_evidence=evidence)
        supplemental+=1
    if apply:
        for pair in chosen.values():
            measures=root.findall('part')[pair['part']].findall('measure')
            first=list(measures[pair['start_measure']-1])[pair['first_xml_child_index']]
            last=list(measures[pair['stop_measure']-1])[pair['last_xml_child_index']]
            template.original._add_tie(first,'start');template.original._add_tie(last,'stop')
            starts={x.get('number','1'):x for x in first.findall('notations/slur') if x.get('type')=='start'}
            stops={x.get('number','1'):x for x in last.findall('notations/slur') if x.get('type')=='stop'}
            for number in starts.keys()&stops.keys():
                first.find('notations').remove(starts[number]);last.find('notations').remove(stops[number])
    primary['accepted_pairs']=list(chosen.values())
    primary['review']=[p for p in primary['review'] if _key(p) not in chosen]
    primary['counts']['accepted_pairs']=len(chosen)
    primary.update(algorithm='adjacent_same_pitch_bounded_optical_profile_union_v4',applied=bool(apply),
                   elapsed_s=time.perf_counter()-tic,
                   optical_union=dict(measured_ridge_pairs=len(chosen)-supplemental,
                     independently_supported_thin_template_supplements=supplemental,
                     competing_ridge_pairs_not_overridden=competing,
                     no_reference_or_prior_prediction_ids_used=True,
                     thresholds_and_clock_staff_voice_head_uniqueness_gates_unchanged=True))
    return primary
