"""Recover physical staff structure while preserving decoded upper events.

A fully head-certified alternate view proves the common bar geometry and the
lower whole-rest glyph. Its additional pitched events remain deferred until
their timing has stronger evidence. Every existing upper note and time token
is copied byte-for-byte. Active and deferred head geometry remain separate so
ordinary sequence alignment cannot silently use an unexported symbol.
"""
import copy
import xml.etree.ElementTree as E

from omr_bar_geometry import measure_systems
from omr_event_records import records
from repair_augmentation_dots import _head_map
from align_homr_geometry import _position
from .relation_staff_pair_trial import fuse as prove_pair
from .relation_tail_splice_experiment import contexts


def _unique_subsequence(old,new):
    """Return one exact ordered event embedding, or abstain if ambiguous."""
    n,m=len(old),len(new)
    if not 1 <= n <= m <= 128:
        return None
    table=[[0]*(m+1) for _ in range(n+1)]
    for j in range(m+1):
        table[0][j]=1
    for i in range(1,n+1):
        for j in range(1,m+1):
            table[i][j]=min(2,table[i][j-1]+(table[i-1][j-1] if old[i-1]==new[j-1] else 0))
    if table[n][m]!=1:
        return None
    result=[];i=n;j=m
    while i:
        if not j:
            return None
        if table[i][j-1]:
            j-=1
        else:
            result.append(j-1);i-=1;j-=1
    return result[::-1]


def _signature(event):
    return event['midi'],_position(event['node'])


def fusion_geometry(base_geometry,candidate_geometry,changes):
    """Keep all detected heads, exporting a subset as active score geometry.

    ``notes`` remains the geometry eligible for the retained MusicXML events;
    ``deferred_noteheads`` stores complete detections for a later timing pass.
    No detected head is erased from the sidecar.
    """
    result=copy.deepcopy(base_geometry)
    for change in sorted(changes,key=lambda c:c['old_system_indices'][0],reverse=True):
        a,b=change['old_system_indices'];j=change['candidate_system_index']
        system=copy.deepcopy(candidate_geometry['systems'][j]);staff=system['staffs'][0]
        full=staff['notes'];retained=change['retained_pitch_head_indices'];deferred=change['deferred_pitch_head_indices']
        if sorted(retained+deferred)!=list(range(len(full))) or len(set(retained+deferred))!=len(full):
            raise ValueError('Head transaction does not partition every detection')
        staff['notes']=[full[i] for i in retained]
        staff['deferred_noteheads']=[full[i] for i in deferred]
        staff['notehead_geometry_scope']='retained_exported_events_with_separate_deferred_detections'
        staff['total_segmented_noteheads']=len(full)
        result['systems'][a:b+1]=[system]
    result['local_structure_recovery']=changes
    return result


def fuse(base,candidate,base_geometry,candidate_geometry,image):
    # V1's full candidate head/whole-rest proof runs only on a throwaway XML.
    # Its new pitched interpretation is never committed by this transaction.
    proof=prove_pair(copy.deepcopy(base),candidate,base_geometry,candidate_geometry,image)
    if not proof:
        return []
    old_systems=measure_systems(base);new_systems=measure_systems(candidate)
    old_events=records(base);new_events=records(candidate)
    old_start=[];new_start=[];at=0
    for group in old_systems:
        old_start.append(at);at+=len(group)
    at=0
    for group in new_systems:
        new_start.append(at);at+=len(group)
    old_context=contexts(base);new_context=contexts(candidate)
    old_heads=_head_map(base,base_geometry);new_heads=_head_map(candidate,candidate_geometry)
    jobs=[]
    for item in proof:
        oi,li=item['old_system_indices'];ni=item['candidate_system_index']
        upper=old_systems[oi];lower=old_systems[li];merged=new_systems[ni]
        old_after=old_start[li]+len(lower);new_after=new_start[ni]+len(merged)
        if old_after<len(old_context) and (new_after>=len(new_context) or old_context[old_after]!=new_context[new_after]):
            continue
        ng=candidate_geometry['systems'][ni]['staffs'][0]
        head_index={id(h):j for j,h in enumerate(ng['notes'])}
        retained=[];measures=[];valid=True
        for number,original in enumerate(upper):
            oidx=old_start[oi]+number;nidx=new_start[ni]+number
            if old_context[oidx]!=new_context[nidx]:
                valid=False;break
            previous=old_events[oidx];pitched=[r for r in new_events[nidx] if r['midi'] is not None]
            indices=_unique_subsequence([_signature(r) for r in previous],[_signature(r) for r in pitched])
            if indices is None:
                valid=False;break
            for r,index in zip(previous,indices):
                new_head=new_heads.get(pitched[index]['node'])
                old_head=old_heads.get(r['node'])
                if new_head is None or id(new_head) not in head_index:
                    valid=False;break
                space=new_head.get('staff_space',0)
                if not 5<=space<=40 or old_head is not None and (abs(new_head['x']-old_head['x'])>.25*space or abs(new_head['y']-old_head['y'])>.25*space):
                    valid=False;break
                retained.append(head_index[id(new_head)])
            if not valid:
                break
            rests=[r for r in new_events[nidx] if r['midi'] is None]
            rest=copy.deepcopy(rests[0]['node'])
            duration=rest.findtext('duration')
            if duration is None or int(duration)<=0 or rest.findtext('voice','1') in {r['voice'] for r in previous}:
                valid=False;break
            copied=copy.deepcopy(original)
            at=next((j for j,n in enumerate(copied) if n.tag=='note'),None)
            if at is None:
                valid=False;break
            backup=E.Element('backup');E.SubElement(backup,'duration').text=duration
            copied.insert(at,rest);copied.insert(at+1,backup)
            assert [E.tostring(n) for n in copied.findall('note') if n.find('pitch') is not None]==[E.tostring(n) for n in original.findall('note')]
            measures.append(copied)
        if not valid or len(set(retained))!=len(retained):
            continue
        deferred=[i for i in range(len(ng['notes'])) if i not in retained]
        change=dict(item,method='local_grandstaff_structure_with_original_upper_event_bytes',
                    retained_pitch_head_indices=retained,deferred_pitch_head_indices=deferred,
                    retained_exported_pitch_events=len(retained),deferred_pitch_events=len(deferred),
                    original_upper_note_bytes_preserved=True,original_upper_clock_and_voice_preserved=True)
        jobs.append((upper+lower,measures,change))
    if not jobs:
        return []
    trial=copy.deepcopy(base)
    part=trial.find('part');old_part=base.find('part')
    # Build the tentative XML on copies and validate the final retained-head
    # alignment before touching the caller's live score.
    before={id(m):E.tostring(m) for m in old_part.findall('measure')}
    positions={id(m):list(old_part).index(m) for m in old_part.findall('measure')}
    changes=[]
    for deleted,added,change in reversed(jobs):
        position=positions[id(deleted[0])]
        for _ in deleted:
            part.remove(list(part)[position])
        for offset,m in enumerate(added):
            part.insert(position+offset,m)
        changes.append(change)
    changes=changes[::-1]
    geo=fusion_geometry(base_geometry,candidate_geometry,changes)
    mapped=_head_map(trial,geo)
    new_groups=measure_systems(trial)
    for offset,change in enumerate(changes):
        index=change['old_system_indices'][0]-offset
        change['new_system_index']=index
        notes=[n for m in new_groups[index] for n in m.findall('note') if n.find('pitch') is not None]
        if len(notes)!=change['retained_exported_pitch_events'] or any(n not in mapped for n in notes):
            return []
        change['final_retained_pitch_head_matches']=len(notes)
    # Commit only the approved ranges; unaffected original objects and their
    # serialized content remain unchanged.
    live=base.find('part')
    for deleted,added,change in reversed(jobs):
        position=list(live).index(deleted[0])
        for m in deleted:
            live.remove(m)
        for offset,m in enumerate(added):
            live.insert(position+offset,copy.deepcopy(m))
    assert all(E.tostring(m)==before[id(m)] for m in live.findall('measure') if id(m) in before)
    return changes
