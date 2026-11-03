"""Preserve the primary upper voice through complete opposed-stem passages.

Only existing labels are permuted. Notes, rests, grace attachments and clocks
remain untouched. Partial lanes, ties, ambiguous stems and cross-staff voices
abstain. No neural model or reference score is needed.
"""
from collections import Counter,defaultdict
from fractions import Fraction
from repair_augmentation_dots import _head_map
from repair_unison_voices import _stem_end
from repair_pitch_context import _ink
from omr_event_records import records


def repair(root,geometry,image):
    if len(root.findall('part'))!=1:return []
    rows=records(root);mapped=_head_map(root,geometry);ink=_ink(image);changes=[]
    # Persistent voice labels belong to the part. Require a substantial
    # monophonic prefix to establish which existing voice is primary.
    prefix=defaultdict(Counter);established={};meter=Fraction(4)
    measures=root.find('part').findall('measure')
    for mi,(measure,row) in enumerate(zip(measures,rows)):
        time=measure.find('attributes/time')
        if time is not None:meter=Fraction(int(time.findtext('beats'))*4,int(time.findtext('beat-type')))
        for staff in sorted({r['staff'] for r in row}):
            ordinary=[r for r in row if r['staff']==staff and not r['grace']]
            labels={r['voice'] for r in ordinary}
            if len(labels)==1:
                voice=next(iter(labels));prefix[staff][voice]+=sum(r['midi'] is not None for r in ordinary)
                if prefix[staff][voice]>=24 and staff not in established:established[staff]=voice
                continue
            if len(labels)!=2 or staff not in established or established[staff] not in labels:continue
            # A voice shared between printed staves may be a cross-staff beam.
            if any(r['staff']!=staff and r['voice'] in labels and r['midi'] is not None for r in row):continue
            lanes={v:[r for r in ordinary if r['voice']==v] for v in labels}
            directions={};valid=True
            for v,events in lanes.items():
                leads=[r for r in events if not r['chord']];end=Fraction(0)
                for r in sorted(leads,key=lambda r:r['onset']):
                    if r['onset']!=end or r['duration']<=0:valid=False;break
                    end+=r['duration']
                if not valid or end!=meter:valid=False;break
                pitches=[r for r in events if r['midi'] is not None]
                if len(pitches)<2 or len(events)>96 or any(r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in events):valid=False;break
                votes=[]
                for r in pitches:
                    head=mapped.get(r['node'])
                    if head is None:continue
                    up=_stem_end(head,ink,True);down=_stem_end(head,ink,False)
                    if (up is None)==(down is None):continue
                    votes.append('up' if up else 'down')
                counts=Counter(votes)
                if len(votes)<2 or len(votes)/len(pitches)<.8 or len(counts)!=1:valid=False;break
                directions[v]=votes[0]
            if not valid or set(directions.values())!={'up','down'}:continue
            primary=established[staff];up=next(v for v,d in directions.items() if d=='up')
            if up==primary:continue
            remap={v:primary if d=='up' else up for v,d in directions.items()}
            changed=[]
            for r in row:
                if r['staff']==staff and r['voice'] in remap:
                    voice=r['node'].find('voice')
                    if voice is None:valid=False;break
                    changed.append((r,voice,remap[r['voice']]))
            if not valid:continue
            for r,node,v in changed:node.text=v;r['voice']=v
            changes.append({'measure':mi+1,'staff':staff,'notes_and_rests':len(changed),
                            'method':'complete_opposed_stems_after_primary_voice_prefix','mapping':remap})
    return changes
