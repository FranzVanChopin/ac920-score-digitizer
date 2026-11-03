"""Experimental physical streams with a bounded joint rational clock.

No score identity, source notation, or reference file is an inference input.
This module is independent of the accepted runtime while its benefit is
measured. Beam classification uses captured pixels and optional native ROIs.
"""
from collections import Counter,defaultdict
from fractions import Fraction as F
from math import lcm
import copy
import xml.etree.ElementTree as E
import numpy as np
from .relation_event_clock_trial import NativeBeamProof,_duration_proof
from .relation_polyrhythm_trial import _core
from .relation_beam_components import stem_end
from repair_tuplets import continuous_beams,scale_divisions
from repair_ledger_chords import _align
from repair_grace_timing import _rebuild
from omr_bar_geometry import measure_systems,bar_bounds
from align_homr_geometry import _head_staff,_position,_xml_systems
from omr_event_records import records


def physical_streams(heads,ink,native=None):
    """Separate single/double beams without using XML voice labels."""
    if not 6<=len(heads)<=48:return None
    ordered=sorted(heads,key=lambda h:(h['x'],h['y']));edges=[];kinds=defaultdict(set)
    for i,a in enumerate(ordered):
        candidates=[(j,b) for j,b in enumerate(ordered[i+1:],i+1) if .65*a['staff_space']<b['x']-a['x']<8*a['staff_space']]
        # At most the next four physical columns may be tested. No search
        # across a dense bar is allowed to connect arbitrary repeated heads.
        columns=[]
        for j,b in candidates:
            if not columns or b['x']-columns[-1]>.35*a['staff_space']:columns.append(b['x'])
            if len(columns)>4:break
            for up in (True,False):
                for number in (2,1):
                    if continuous_beams(a,b,ink,up,number,maximum_gap=8) or native is not None and native.test(a,b,up,number):
                        kinds[i].add(number);kinds[j].add(number);edges.append((i,j,up,number));break
    family={i:max(kinds[i]) if kinds[i] else 0 for i in range(len(ordered))}
    streams={n:[h for i,h in enumerate(ordered) if family[i]==n] for n in (0,1,2)}
    ports={i:{up:max(e[3] for e in edges if i in e[:2] and e[2]==up) for up in (True,False) if any(i in e[:2] and e[2]==up for e in edges)} for i in range(len(ordered))}
    # A double-beam stream may have a single primary beam across its beat
    # boundary. Opposite beamed stem ports represent separate voices and
    # cannot be collapsed by taking the largest layer count.
    dual_beamed_ports=any(len(p)>1 for p in ports.values())
    return dict(streams=streams,edges=edges,ordered=ordered,family=family,ports=ports,dual_beamed_ports=dual_beamed_ports)


def joint_ratios(streams,meter=None):
    """Enumerate at most nine two-stream ratio combinations."""
    if len(streams)!=2 or any(not 6<=len(s['heads'])<=48 for s in streams):return []
    plans=[]
    for a in ((1,1),(3,2),(4,3)):
        da=streams[0]['unit']*F(a[1],a[0]);span=len(streams[0]['heads'])*da
        if not 1<=span<=16 or meter is not None and span!=meter:continue
        for b in ((1,1),(3,2),(4,3)):
            db=streams[1]['unit']*F(b[1],b[0])
            if len(streams[1]['heads'])*db!=span:continue
            paired=[];valid=True
            for i,h in enumerate(streams[0]['heads']):
                matches=[j for j,g in enumerate(streams[1]['heads']) if abs(h['x']-g['x'])<=.4*min(h['staff_space'],g['staff_space'])]
                expected=i*da/db
                if not matches:
                    if expected.denominator==1 and 0<=expected<len(streams[1]['heads']):valid=False;break
                    continue
                if len(matches)!=1 or i*da!=matches[0]*db:valid=False;break
                paired.append(i*da)
            if valid and len(set(paired))>=2 and 0 in paired:
                plans.append(dict(durations=[da,db],ratios=[a,b],span=span,columns=len(set(paired))))
    return plans


def _cell_sizes(physical,kind):
    ids=[i for i in range(len(physical['ordered'])) if physical['family'][i]==kind];adj=defaultdict(set)
    for a,b,up,count in physical['edges']:
        if a in ids and b in ids and count==kind:adj[a].add(b);adj[b].add(a)
    todo=set(ids);sizes=[]
    while todo:
        found={next(iter(todo))};work=list(found)
        while work:
            for other in adj[work.pop()]-found:found.add(other);work.append(other)
        todo-=found;sizes.append(len(found))
    return sizes


def _short_bindings(rows,heads,duration,*,carriers=None):
    """Keep an already supported per-node clock, independent of voice blocks.

    A unique pitch edit path is unique only relative to its chosen order.
    Valid MusicXML may serialize complete voices in separate blocks, so an
    existing exact slot binding takes priority. For a partial clock its
    actual onset order is canonical, without assuming its absolute scale
    correct. Every existing supported slot must agree with the mapping.
    """
    eligible=heads if carriers is None else carriers
    anchors={};owners=defaultdict(list)
    for r in rows:
        slot=r['onset']/duration;p=_position(r['node'])
        if slot.denominator!=1 or not 0<=slot<len(heads) or p is None:continue
        h=heads[int(slot)]
        if p[1]!=h['position'] or not any(h is q for q in eligible):continue
        anchors[r['node']]=h;owners[int(slot)].append(r['node'])
    if any(len(nodes)>1 for nodes in owners.values()):return None
    if len(anchors)==len(rows)==len(eligible):return anchors
    ordered=sorted(rows,key=lambda r:r['onset'])
    # One physical lane has a strict head order. Overlapping old clocks do
    # not establish an order between simultaneous nodes, even with a stable
    # voice/pitch sorting key. Do not invent that relative placement.
    if any(a['onset']==b['onset'] for a,b in zip(ordered,ordered[1:])):return None
    mapping=_align([r['node'] for r in ordered],eligible)
    if any(mapping.get(n) is not h for n,h in anchors.items()):return None
    return mapping


def _prefix_clock_proof(measure,rows):
    elements=list(measure);first=next(k for k,n in enumerate(elements) if n.tag=='note')
    prefix=elements[:first];cursors=[k for k,n in enumerate(prefix) if n.tag in ('forward','backup')]
    if not cursors:return True
    # Equivalent voice-block serialization may begin at a nonzero onset
    # while another voice starts at zero. Timing metadata after the cursor
    # or a genuine whole-bar offset cannot be normalized this way.
    return min(r['onset'] for r in rows)==0 and all(n.tag in ('forward','backup') for n in prefix[cursors[0]:])


def _canonical_rebuild(measure,desired):
    elements=list(measure);first=next(k for k,n in enumerate(elements) if n.tag=='note')
    # _rebuild resets its cursor to zero. Keep prefix metadata, but remove
    # the old pure cursor prefix so the new leading shift is added once.
    measure[:]=[n for n in elements[:first] if n.tag not in ('forward','backup')]+elements[first:]
    return _rebuild(measure,desired)


def fuse(root,homr,image,diagnostics=None,*,photo=None):
    diag=diagnostics if diagnostics is not None else Counter()
    if image.ndim!=2 or image.size>16000000 or max(image.shape)>8192 or len(root.findall('part'))!=1:return []
    systems=measure_systems(root)
    if len(systems)!=len(homr.get('systems',[])) or len(systems)>128 or len(root.findall('part/measure'))>2048:return []
    if sum(len(st.get('notes',[])) for sy in homr.get('systems',[]) for st in sy.get('staffs',[]))>8192:return []
    try:
        if any(not np.isfinite([h['x'],h['y'],h['staff_space'],h['position']]).all() or not (0<=h['x']<image.shape[1] and 0<=h['y']<image.shape[0] and 5<=h['staff_space']<=80) for sy in homr['systems'] for st in sy.get('staffs',[]) for h in st.get('notes',[])):return []
    except (KeyError,TypeError,ValueError,OverflowError):return []
    _xml_systems(root);ink=(image<140).astype('uint8');native=NativeBeamProof(homr,image,photo,diag);ms=root.findall('part/measure');events=records(root);contexts=[];div=1;meter=None
    active_tuplets=set()
    for m in ms:
        for a in m.findall('attributes'):
            div=int(a.findtext('divisions',str(div)));t=a.find('time')
            if t is not None:
                b,u=t.findtext('beats'),t.findtext('beat-type');meter=F(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u) else None
        blocked=bool(active_tuplets)
        for n in m.findall('note'):
            for t in n.findall('notations/tuplet'):
                blocked=True;key=(n.findtext('staff','1'),n.findtext('voice','1'),t.get('number','1'))
                if t.get('type')=='start':active_tuplets.add(key)
                elif t.get('type')=='stop':active_tuplets.discard(key)
        contexts.append((div,meter,blocked))
    offset=0;jobs=[]
    for measures,sy in zip(systems,homr['systems']):
        if len(sy.get('staffs',[]))!=1:offset+=len(measures);continue
        staff=sy['staffs'][0];bounds=bar_bounds(staff,image)
        if bounds is None or len(bounds)!=len(measures)+1:offset+=len(measures);continue
        for j,(lo,hi) in enumerate(zip(bounds,bounds[1:])):
            number=offset+j;rs=events[number];division,meter,blocked=contexts[number];m=ms[number]
            if blocked:diag['existing_tuplet_scope_preserved']+=1;continue
            if not 1<=len(rs)<=128 or not 0<division<=9600 or any(r['grace'] or r['chord'] or r['midi'] is None or r['node'].get('print-object')=='no' or r['node'].findall('dot') for r in rs):continue
            if any(r['node'].findtext('type') not in ('eighth','16th','quarter') for r in rs):continue
            if not _prefix_clock_proof(m,rs) or m.get('implicit')=='yes':continue
            all_heads=[h for h in staff['notes'] if lo<h['x']<hi]
            if not 12<=len(all_heads)<=48 or any(not _core(h,ink) for h in all_heads):continue
            streams=[];valid=True
            for sn in (1,2):
                heads=[h for h in all_heads if _head_staff(staff,h)==sn];physical=physical_streams(heads,ink,native)
                if physical is None or physical['dual_beamed_ports']:valid=False;break
                families=[k for k in (1,2) if physical['streams'][k]]
                # This first trial supports one complete beam lane per staff;
                # a separate opposing, unbeamed quarter may share its head.
                if len(families)!=1 or physical['streams'][0]:valid=False;break
                kind=families[0];hs=physical['streams'][kind]
                if any(b['x']-a['x']<.65*min(a['staff_space'],b['staff_space']) for a,b in zip(hs,hs[1:])):valid=False;break
                streams.append(dict(staff=sn,kind=kind,unit=F(1,2**kind),heads=hs,physical=physical,cells=_cell_sizes(physical,kind)))
            if not valid:continue
            earliest=min(h['x'] for h in all_heads)
            if any(abs(s['heads'][0]['x']-earliest)>.4*s['heads'][0]['staff_space'] for s in streams):continue
            plans=joint_ratios(streams,meter)
            plans=[p for p in plans if all(all(c>=2 and (a==1 or c%a==0) for c in s['cells']) for s,(a,n) in zip(streams,p['ratios']))]
            diag['joint_physical_clock_candidates']+=1
            if len(plans)!=1:diag['joint_clock_ambiguous_or_absent']+=1;continue
            plan=plans[0];edits=[];quarter_edits=[];bound=set()
            for stream,duration,ratio in zip(streams,plan['durations'],plan['ratios']):
                sn=stream['staff'];heads=stream['heads'];short=[r for r in rs if r['staff']==sn and r['node'].findtext('type') in ('eighth','16th')];long=[r for r in rs if r['staff']==sn and r['node'].findtext('type')=='quarter']
                mapping=_short_bindings(short,heads,duration)
                if mapping is None:diag['existing_per_node_clock_order_unproven']+=1;valid=False;break
                if not short or len(mapping)<.75*len(short):valid=False;break
                carriers=[]
                for k,h in enumerate(heads):
                    pi=stream['physical']['ordered'].index(h);port=stream['physical']['ports'][pi]
                    if len(port)==1 and stem_end(h,ink,not next(iter(port))) is not None:carriers.append(h)
                qm=_short_bindings(long,heads,duration,carriers=carriers) if long else {}
                if qm is None:diag['existing_quarter_clock_order_unproven']+=1;valid=False;break
                if len(qm)!=len(long):valid=False;break
                if len({id(h) for h in [*mapping.values(),*qm.values()]})<.75*len(heads):valid=False;break
                for r in short:
                    h=mapping.get(r['node'])
                    if h is None:continue
                    k=next(k for k,q in enumerate(heads) if q is h);edits.append((r,k*duration,duration,ratio,'eighth' if stream['kind']==1 else '16th'));bound.add(r['node'])
                for r in long:
                    h=qm[r['node']];k=next(k for k,q in enumerate(heads) if q is h);t=k*duration
                    # A plain quarter cannot be shortened using a source's
                    # hidden playback convention. Preserve unresolved old
                    # long-note timing when the proposed dependency overflows.
                    if t+r['duration']<=plan['span'] and _duration_proof(r):quarter_edits.append((r,t,r['duration']))
                    else:diag['opposing_quarter_clock_unresolved']+=1
                    bound.add(r['node'])
            if not valid:continue
            if not any(r['onset']!=t or r['duration']!=d or r['node'].findtext('type')!=typ for r,t,d,ratio,typ in edits) and not any(r['onset']!=t for r,t,d in quarter_edits):continue
            jobs.append(dict(number=number,division=division,edits=edits,quarter_edits=quarter_edits,span=plan['span'],columns=plan['columns'],slot_counts=[len(s['heads']) for s in streams]))
        offset+=len(measures)
    if not jobs or len(jobs)>128:return []
    factor=1
    for job in jobs:
        for r,t,d,*_ in job['edits']+job['quarter_edits']:factor=lcm(factor,(t*job['division']).denominator,(d*job['division']).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for job in jobs:
        number=job['number'];m=ms[number];saved=copy.deepcopy(list(m));division=job['division']*factor;desired={r['node']:int(r['onset']*division) for r in events[number]};durations={r['node']:r['duration'] for r in events[number]};identity=[(r['staff'],r['voice'],r['midi'],r['chord'],r['grace']) for r in events[number]]
        for r,t,d,ratio,typ in job['edits']:
            n=r['node'];desired[n]=int(t*division);durations[n]=d;old_type=n.findtext('type');n.find('duration').text=str(int(d*division));n.find('type').text=typ
            for old in n.findall('time-modification'):n.remove(old)
            old_beams=n.findall('beam');layers=1 if typ=='eighth' else 2
            if old_beams and (old_type!=typ or max(int(b.get('number','1')) for b in old_beams)!=layers):
                # Preserve valid explicit grouping; contradictory layer tags
                # cannot override the image-proven type in music software.
                for old in old_beams:n.remove(old)
            if ratio!=(1,1):
                tm=E.Element('time-modification')
                for tag,val in [('actual-notes',str(ratio[0])),('normal-notes',str(ratio[1])),('normal-type',typ)]:E.SubElement(tm,tag).text=val
                at=next((k for k,n in enumerate(n) if n.tag in ('stem','staff','beam','notations')),len(n));n.insert(at,tm)
        for r,t,d in job['quarter_edits']:desired[r['node']]=int(t*division)
        if not _canonical_rebuild(m,desired):m[:]=saved;continue
        actual=records(root)[number]
        targets={r['node'] for r,t,d,ratio,typ in job['edits']}
        if len(actual)!=len(desired) or [(r['staff'],r['voice'],r['midi'],r['chord'],r['grace']) for r in actual]!=identity or any(r['onset']!=F(desired[r['node']],division) or r['duration']!=durations[r['node']] or r['node'] in targets and not _duration_proof(r) for r in actual):m[:]=saved;continue
        changes.append(dict(measure=number+1,method='independent_physical_beam_streams_joint_rational_clock',slots=job['slot_counts'],bar_beats=str(job['span']),columns=job['columns'],short_events=len(job['edits']),opposing_quarter_events=len(job['quarter_edits']),hidden_quarter_duration_invented=False,pitch_staff_voice_cardinality_preserved=True))
    return changes
