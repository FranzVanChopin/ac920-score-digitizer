"""Bounded voice continuity with beam-chain and rest-placement evidence."""
import copy
import numpy as np
from .relation_decode import voices,voice_offsets


def duration(g):return 0 if g.get('grace') else g['duration']


def beam_chains(groups,image,space):
    parent=list(range(len(groups)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i,a in enumerate(groups):
        if not a.get('beams'):continue
        for j in range(i+1,len(groups)):
            b=groups[j]
            if b['x']-a['x']>8*space:break
            if not b.get('beams') or bool(a.get('grace'))!=bool(b.get('grace')):continue
            for ea in a.get('beam_relations',[]):
                for eb in b.get('beam_relations',[]):
                    ya=float(np.mean(ea['y_range']));yb=float(np.mean(eb['y_range']))
                    if abs(ya-yb)>1.5*space:continue
                    xs=np.linspace(ea['stem_x'],eb['stem_x'],max(4,round(abs(eb['stem_x']-ea['stem_x']))))
                    ys=np.linspace(ya,yb,len(xs));hits=[]
                    for x,y in zip(xs,ys):
                        yy=round(y);xx=round(x)
                        crop=image[max(0,yy-2):min(image.shape[0],yy+3),max(0,xx-1):min(image.shape[1],xx+2)]
                        hits.append(bool(crop.size and (crop<155).any()))
                    if np.mean(hits)>=.88:parent[root(j)]=root(i)
    for i,g in enumerate(groups):g['beam_chain']=root(i)


def lane_cost(lanes,space,expected):
    # Shared columns constrain onset independently of a meter-fill heuristic.
    time={};ends={};cost=0.
    for lane,seq in lanes.items():
        t=0;time[lane]=[]
        for g in seq:time[lane].append((g['x'],t,g));t+=duration(g)
        ends[lane]=t;cost+=abs(t-expected)/480*.12+max(0,t-expected)/480*4
    if len(lanes)==2:
        for x,t,g in time[1]:
            if g.get('grace'):continue
            near=[(abs(x-xx),tt) for xx,tt,h in time[2] if abs(x-xx)<=.6*space and not h.get('grace')]
            if near:cost+=abs(t-min(near)[1])/120
    return cost


def attach_graces(lanes,graces,space):
    """Grace stems describe engraving, not the principal note's voice."""
    for grace in graces:
        following=[(g['x']-grace['x'],lane,g) for lane,seq in lanes.items()
                   for g in seq if not g.get('rest') and not g.get('grace') and g['x']>grace['x']]
        if not following:
            lane=next(iter(lanes),1);principal=None
        else:
            dx=min(v[0] for v in following)
            column=[v for v in following if v[0]<=dx+.6*space]
            _,lane,principal=min(column,key=lambda v:abs(v[2]['y']-grace['y']))
        grace['principal_relation']=None if principal is None else {'x':principal['x'],'y':principal['y'],'voice':lane,'method':'next_note_column_nearest_head'}
        if principal is not None and 'locked_onset' in principal:grace['locked_onset']=principal['locked_onset']
        lanes.setdefault(lane,[]).append(grace)
    for seq in lanes.values():seq.sort(key=lambda g:g['x'])
    return lanes


def continue_short_lane(lanes,space):
    """Keep a moving beamed melody's identity after an idle pedal voice.

    This bounded witness applies within an already proven two-voice measure:
    two earlier fast notes share a beam, the other lane's last physical column
    is older, and the new beam lies much closer to the moving melody. Stem
    reversal alone must not move that melody into the completed pedal lane.
    """
    if set(lanes)!={1,2}:return []
    starts,_=offsets(lanes,space);times={}
    for lane,seq in lanes.items():
        t=starts[lane]
        for g in seq:times[id(g)]=t;t+=duration(g)
    moves=[];ordinary=sorted((g for seq in lanes.values() for g in seq if not g.get('grace')),key=lambda g:g['x'])
    owner={id(g):lane for lane,seq in lanes.items() for g in seq};history={1:[],2:[]}
    for g in ordinary:
        lane=owner[id(g)];other=3-lane
        if not g.get('heads') or len(history[other])<2 or not history[lane]:history[lane].append(g);continue
        a,b=history[other][-2:];idle=history[lane][-1]
        continuing=bool(b.get('voice_continuity_witness'))
        if not a.get('heads') or not b.get('heads') or not idle.get('heads'):history[lane].append(g);continue
        if not continuing and (not g.get('beams') or a.get('beam_chain')!=b.get('beam_chain') or not a.get('beams') or not b.get('beams')):history[lane].append(g);continue
        # The two streams must already agree on the next musical onset. This
        # repairs identity without forcing a new rhythmic clock or filling rests.
        if times[id(g)]!=times[id(b)]+duration(b) or (not continuing and times[id(g)]!=times[id(idle)]+duration(idle)):history[lane].append(g);continue
        peers=[h for h in ordinary if h is not g and abs(h['x']-g['x'])<=.6*space]
        if peers or idle['x']>=b['x']-.6*space or not .6*space<g['x']-b['x']<4*space:history[lane].append(g);continue
        near=abs(g['y']-b['y']);far=abs(g['y']-idle['y'])
        if near>space or far<near+.6*space or duration(idle)<2*duration(b):history[lane].append(g);continue
        chain=[h for h in lanes[lane] if h.get('beam_chain')==g.get('beam_chain')] if g.get('beams') else [g]
        if not continuing and len(chain)<2:history[lane].append(g);continue
        for h in chain:
            lanes[lane].remove(h);lanes[other].append(h);owner[id(h)]=other
            h['voice_continuity_witness']={'method':'beamed_melody_after_completed_pedal','from_voice':lane,'to_voice':other,'previous_melody_x':b['x'],'idle_voice_x':idle['x']}
        moves.append({'from_voice':lane,'to_voice':other,'groups':len(chain)})
        history[other].append(g)
    for seq in lanes.values():seq.sort(key=lambda g:g['x'])
    if moves:
        # A voice-label correction must not silently recompute musical time.
        for g in ordinary:g['locked_onset']=times[id(g)]
    return moves


def decode_voices(groups,space,expected,state,attach=False,continuity=False):
    ordinary=[g for g in groups if not g.get('grace')];initial,poly=voices(ordinary,space)
    signs={g.get('direction') for g in ordinary if g.get('direction')}
    overflow=sum(duration(g) for g in ordinary)>1.35*expected
    # Staggered polyphony requires both printed stem directions plus positive
    # timing evidence. Direction changes in a simple melody remain one voice.
    poly=poly or (len(signs)==2 and overflow and state.get('poly',False))
    if not poly:
        # Pitch register is not a voice identity: a melody can move below the
        # previous lower voice. Without a continuing beam/tie witness abstain
        # from relabelling a whole solo measure as lane 2.
        return {1:groups},False
    # Search only ambiguous rests/un-stemmed heads. Fixed stem evidence and
    # actual beam chains constrain notes; keep at most 32 partial candidates.
    states=[{1:[],2:[]}];middle=float(np.median([g['y'] for g in ordinary])) if ordinary else 0
    chains={}
    for g in ordinary:
        if g.get('beams') and g.get('beam_chain') is not None:
            chains.setdefault(g['beam_chain'],[]).append(g)
    chain_lane={k:(1 if sum(g.get('direction')=='up' for g in seq)>=sum(g.get('direction')=='down' for g in seq) else 2) for k,seq in chains.items() if len(seq)>1}
    for g in ordinary if attach else groups:
        fixed=1 if g.get('direction')=='up' else 2 if g.get('direction')=='down' else None
        fixed=chain_lane.get(g.get('beam_chain'),fixed)
        choices=[fixed] if fixed else [1,2]
        candidates=[]
        for lanes in states:
            for lane in choices:
                new={k:list(v) for k,v in lanes.items()};new[lane].append(g)
                penalty=lane_cost(new,space,expected)
                if not fixed:penalty+=.1*int((g['y']<middle)!=(lane==1))
                candidates.append((penalty,new))
        states=[lanes for _,lanes in sorted(candidates,key=lambda p:p[0])[:32]]
    def final_cost(lanes):
        height=sum(.1*int((g['y']<middle)!=(lane==1)) for lane,seq in lanes.items() for g in seq if not g.get('direction'))
        return lane_cost(lanes,space,expected)+height
    selected=min(states,key=final_cost)
    selected={k:v for k,v in selected.items() if v}
    if continuity:continue_short_lane(selected,space)
    if attach:attach_graces(selected,[g for g in groups if g.get('grace')],space)
    starts,_=offsets(selected,space)
    if len(selected)==2 and all(starts[k]+sum(duration(g) for g in seq)==expected for k,seq in selected.items()):state['poly']=True
    state['means']={k:float(np.mean([g['y'] for g in v])) for k,v in selected.items()}
    return selected,True


def offsets(lanes,space):
    # The legacy aligner sums duration; use a temporary zero-time view of grace
    # groups while preserving the printed note type in the actual group.
    view={k:[dict(g,duration=duration(g)) for g in seq] for k,seq in lanes.items()}
    return voice_offsets(view,space)
