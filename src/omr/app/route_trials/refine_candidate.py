"""Reference-free notation/voice refinement; the separate runner owns scoring."""
import copy
from collections import Counter,defaultdict
from fractions import Fraction
import xml.etree.ElementTree as E
from repair_homr_tremolo import repair as tremolo_repair
from .xml_records_ticks_trial import records




def voice_refine(base,relation,graph):
    """Repair minority voice labels along an observed, matched beam chain."""
    br=records(base);changes=[];offset=0
    for system in sorted({s['system'] for s in graph['systems']}):
        rows=[s for s in graph['systems'] if s['system']==system]
        for row in rows:
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi
                if number>=len(br):continue
                lookup=defaultdict(list)
                for r in br[number]:
                    if not r['grace']:lookup[r['key']].append(r)
                chains=defaultdict(list)
                for g in row['groups']:
                    if left<g['x']<right and not g.get('grace') and g.get('beams'):
                        chains[g['beam_chain']].append(g)
                for chain,groups in chains.items():
                    if len(groups)<3:continue
                    members=[];valid=True
                    for g in groups:
                        matched=[]
                        for h in g['heads']:
                            letter,octave,alter,*_=h['pitch'];pitch=(octave+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[letter]+alter
                            key=(row['staff'],Fraction(g['onset'],480),pitch,Fraction(g['duration'],480))
                            hits=lookup[key]
                            if len(hits)!=1:valid=False;break
                            matched.append(hits[0])
                        if not valid or len({r['voice'] for r in matched})!=1:valid=False;break
                        members.append(matched)
                    if not valid:continue
                    vote=Counter(group[0]['voice'] for group in members);target,n=vote.most_common(1)[0]
                    if len(vote)<2 or n/len(members)<.75:continue
                    updates=[r for group in members for r in group if r['voice']!=target]
                    if any(r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in updates):continue
                    # Do not create overlaps with other destination-voice notes
                    # or rests. Chord tones in the same printed group are legal.
                    chosen={id(r) for group in members for r in group}
                    if any(r['staff']==row['staff'] and r['voice']==target and id(r) not in chosen and not r['grace'] and
                           any(max(r['onset'],u['onset'])<min(r['onset']+r['duration'],u['onset']+u['duration']) for u in updates) for r in br[number]):continue
                    for r in updates:r['node'].find('voice').text=target;r['voice']=target
                    changes.append({'measure':number+1,'staff':row['staff'],'beam_chain':chain,'notes_or_rests_relabelled':len(updates),'method':'observed_beam_majority_voice_continuity'})
        offset+=len(rows[0]['bounds'])-1
    return changes


def grace_refine(base,relation,graph):
    """Transfer a grace run only through its matching ordinary principal note.

    Principal pitch, staff and onset must agree in both routes; their printed
    duration may differ. Rhythmically incomplete candidate measures abstain.
    Every transferred head also needs the photographed small-head
    evidence. Existing graces must be an ordered subsequence of the candidate
    run; unrelated notes and all ordinary time streams are retained.
    """
    br=records(base);rr=records(relation);changes=[]
    if len(br)!=len(rr):return changes
    proof=Counter();offset=0
    unstable={(r.get('measure'),r.get('staff')) for r in graph.get('rhythm_issues',[])}
    for system in sorted({s['system'] for s in graph['systems']}):
        rows=[s for s in graph['systems'] if s['system']==system]
        for row in rows:
            for mi,(a,b) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                if (offset+mi+1,row['staff']) in unstable:continue
                for g in row['groups']:
                    if not a<g['x']<b or not g.get('grace'):continue
                    for h in g.get('heads',[]):
                        probability=h.get('refined_grace_probability',h.get('grace_probability',0))
                        width=h.get('core_width_spaces',h.get('head_width_spaces',0))
                        if h['class']!=0 or (probability<.99 and not g.get('grace_chord_witness')) or not .65<width<1.12:continue
                        letter,octave,alter,*_=h['pitch'];midi=(octave+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[letter]+alter
                        proof[(offset+mi,row['staff'],Fraction(g['onset'],480),midi)]+=1
        offset+=len(rows[0]['bounds'])-1
    measures=base.find('part').findall('measure')
    for mi,(a,b) in enumerate(zip(br,rr)):
        ordinary=defaultdict(list)
        for r in a:
            if not r['grace'] and not r['chord'] and r['midi'] is not None:ordinary[r['key'][:3]].append(r)
        lanes=defaultdict(list)
        for r in b:lanes[(r['staff'],r['voice'])].append(r)
        for lane,sequence in lanes.items():
            pending=[]
            for principal in sequence:
                if principal['grace']:pending.append(principal);continue
                if not pending:continue
                run=pending;pending=[]
                if principal['midi'] is None or principal['chord'] or len(ordinary[principal['key'][:3]])!=1:continue
                target=ordinary[principal['key'][:3]][0];staff=target['staff'];onset=target['onset']
                needed=Counter((mi,staff,onset,r['midi']) for r in run)
                if any(n>proof[k] for k,n in needed.items()) or any(r['onset']!=principal['onset'] for r in run):continue
                existing=[r for r in a if r['grace'] and r['staff']==staff and r['voice']==target['voice'] and r['onset']==onset]
                if any(r['grace'] and r['staff']==staff and r['onset']==onset and r['voice']!=target['voice'] for r in a):continue
                # Existing same-onset graces elsewhere in the XML stream are
                # ambiguous; only edit the contiguous run before the principal.
                children=list(measures[mi]);index=children.index(target['node']);start=index
                while start>0 and children[start-1].tag=='note' and children[start-1].find('grace') is not None:start-=1
                if set(r['node'] for r in existing)!=set(children[start:index]):continue
                pitches=[r['midi'] for r in run];old=[r['midi'] for r in existing];j=0
                for pitch in pitches:
                    if j<len(old) and pitch==old[j]:j+=1
                if j!=len(old) or len(pitches)<=len(old):continue
                output=[];j=0
                for r in run:
                    if j<len(existing) and r['midi']==existing[j]['midi']:
                        node=existing[j]['node'];j+=1
                    else:
                        node=copy.deepcopy(r['node'])
                        # Photographed layout coordinates do not belong to the
                        # reconstructed score's automatic engraving layout.
                        for attr in ('default-x','default-y','relative-x','relative-y'):node.attrib.pop(attr,None)
                        node.find('voice').text=target['voice']
                    output.append(node)
                measures[mi][start:index]=output
                changes.append({'measure':mi+1,'staff':staff,'principal_pitch':target['midi'],'onset':str(onset),'added_graces':len(pitches)-len(old),'method':'image_grace_run_matching_principal_event'})
    return changes


def refine(base,relation,graph):
    return {'tremolo':tremolo_repair(base),'voices':voice_refine(base,relation,graph),'graces':grace_refine(base,relation,graph),
            'reference_supplied_to_recognizer':False}
