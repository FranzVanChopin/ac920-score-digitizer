"""Carry explicit accidentals along the principal head of chromatic unisons.

An opposite-side head does not overwrite the principal's state. Both physical
lobes must occupy the same native written slot. Existing correct alternatives
remain; a missing member is added rather than substituted. Search is bounded.
"""
from collections import Counter,defaultdict
import copy,xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from .relation_coverage import inputs,core
from .relation_joint_dependencies import body_height
from .relation_joint_candidates import component,mandatory_assignment,empty
from .relation_joint_events import key_proof
from .relation_measure_joint import context_rows,align_columns,written,set_alter
from .relation_physical_streams import native_shaft,extended_shaft
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from .symbol_cnn import FACTOR

def slots(ident,g,row,track,clef,native):
    bottom=pitch_for_step(0,clef);step=(ident[1]-bottom[1])*7+'CDEFGAB'.index(ident[0])-'CDEFGAB'.index(bottom[0]);s=row['space_px'];hits=[]
    for h in g.get('heads',[]):
        if h.get('class')!=0 or h.get('confidence',0)<.45:continue
        p=staff_step(track,*h['xy']);y=p['bottom_line_y']-step*p['local_space_px']/2
        if tuple(h['pitch'][:2])!=ident or abs(p['residual'])>.3 or pitch_for_step(p['nearest_step'],clef)!=ident:continue
        if abs(y-h['xy'][1])>.7*s:continue
        real=dict(h,xy=[(h['xy'][0]+.5)/FACTOR-.5,(y+.5)/FACTOR-.5]);height=body_height(real,native,s/FACTOR)
        if height is not None and .85<=height<=1.65 and core(real,native,s/FACTOR,.95):hits.append(dict(h,xy=[h['xy'][0],y],observed_y=h['xy'][1]))
    clusters=[]
    for h in sorted(hits,key=lambda h:h['xy'][0]):
        if clusters and h['xy'][0]-clusters[-1][-1]['xy'][0]<.3*s:clusters[-1].append(h)
        else:clusters.append([h])
    return [max(c,key=lambda h:h['confidence']) for c in clusters] if len(clusters)<=2 else []

def fuse(root,graph,photo,geometry,slot_reader=None,empty_reader=None):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1];ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];rows=records(root);ctx=context_rows(root);stems={s['id']:s for s in graph['stems']};offset=0;changes=[]
    def glyph(n,legacy,s):
        proof=component(n,legacy,s)
        return proof if proof is not None else component(dict(n,xy=[(v+.5)/FACTOR-.5 for v in n['xy']]),native,s/FACTOR)
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=len(ts):return []
        for row,track in zip(rr,ts):
            sn=row['staff'];s=row['space_px'];all_heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=s) for g in row['groups'] for h in g.get('heads',[])]
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                i=offset+mi;rs=rows[i];key,clefs=ctx[i];clef=clefs.get(sn)
                if not clef or key is None or len(rs)>128:continue
                es=[r for r in rs if r['staff']==sn and r['midi'] is not None and not r['grace']]
                if not es or len({r['voice'] for r in es})!=1:continue
                old=[[r for r in es if r['onset']==t] for t in sorted({r['onset'] for r in es})];seq=sorted([g for g in row['groups'] if left<g['x']<right and g.get('heads') and not g.get('grace')],key=lambda g:g['x'])
                if max(len(old),len(seq))>64 or any(len(c)>8 for c in old) or any(len(g['heads'])>8 for g in seq):continue
                pairs=align_columns([Counter(written(r) for r in c) for c in old],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq]);mapping={b:a for a,b in pairs};state={};certified_unisons=set()
                # This stage activates only after a physical two-lobe unison;
                # ordinary accidental correction remains with the older stages.
                if key_proof(row,track,clef,key,row['groups'],glyph,image):state={(letter,o):(1 if key>0 else -1) if letter in ('FCGDAEB' if key>0 else 'BEADGCF')[:abs(key)] else 0 for letter in 'CDEFGAB' for o in range(10)}
                for gi,g in enumerate(seq):
                    identities=set(tuple(h['pitch'][:2]) for h in g['heads'])
                    if gi not in mapping:
                        for ident in identities:state[ident]=None
                        continue
                    cell=old[mapping[gi]];members=defaultdict(list)
                    for r in cell:members[written(r)].append(r)
                    physical={ident:(slot_reader or slots)(ident,g,row,track,clef,native) for ident in members};hs=[h for values in physical.values() for h in values]
                    if len(hs)>8:continue
                    options=[]
                    for n in row['nodes']:
                        if n['class'] not in (8,9,10) or n.get('confidence',0)<.65 or glyph(n,image,s) is None:continue
                        edges=[]
                        for j,h in enumerate(hs):
                            dx=(h['xy'][0]-n['xy'][0])/s;dy=(n['xy'][1]-h['xy'][1])/s;target=-.75 if n['class']==9 else 0
                            if .65<dx<4.5 and abs(dy-target)<.35 and not any(a['x']<h['xy'][0]-.5*s and a['x']>n['xy'][0]+.65*s and abs(a['y']-h['xy'][1])<.3*s for a in all_heads):edges.append((j,abs(dy-target)+.1*dx))
                        if edges:options.append((n,edges))
                    attached={hs[j]['id']:n for j,n in mandatory_assignment(options,len(hs),margin=.04).items()} if hs else {}
                    for ident,notes in members.items():
                        ss=physical[ident];prior=state.get(ident)
                        if not ss or any(r['onset']*480!=g.get('onset') or r['duration']*480!=g['duration'] for r in notes):state[ident]=None;continue
                        if len(ss)==2:
                            stem=stems.get(g.get('stem'));sx=np.array(stem['xy'])[:,0].mean() if stem else None
                            if sx is None or abs(ss[0]['observed_y']-ss[1]['observed_y'])>.2*s or not .4*s<sx-ss[0]['xy'][0]<.8*s or not .4*s<ss[1]['xy'][0]-sx<.8*s or not(extended_shaft(g,row,stems,image) or native_shaft(g,row,stems,native,native_ink)):state[ident]=None;continue
                            principal=ss[0] if g.get('direction')=='up' else ss[1];opposite=ss[1] if principal is ss[0] else ss[0]
                            sign=attached.get(opposite['id']);psign=attached.get(principal['id']);pvalue={8:1,9:-1,10:0}[psign['class']] if psign else prior
                            if sign is None or pvalue is None:state[ident]=None;continue
                            value={8:1,9:-1,10:0}[sign['class']];want=Counter((value,pvalue));have=Counter(int(r['node'].findtext('pitch/alter','0')) for r in notes)
                            if value==pvalue or have-want or len(notes)>2:state[ident]=None;continue
                            anchors=[r for r in cell if written(r)!=ident and any(tuple(h['pitch'][:2])==written(r) and h.get('confidence',0)>=.85 for h in g['heads'])]
                            if not anchors or any(r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].get('print-object')=='no' for r in notes):state[ident]=None;continue
                            missing=want-have
                            if sum(missing.values())==1 and len(cell)<8:
                                if len(root.findall('part/measure')[i].findall('note'))>=128:state[ident]=None;continue
                                target=next(iter(missing));n=copy.deepcopy(notes[0]['node'])
                                for tag in ('accidental','notations','lyric','tie'):
                                    for a in n.findall(tag):n.remove(a)
                                if n.find('chord') is None:n.insert(0,E.Element('chord'))
                                set_alter(n,target,None);m=root.findall('part/measure')[i];m.insert(max(list(m).index(r['node']) for r in cell)+1,n)
                                changes.append({'measure':i+1,'staff':sn,'onset':str(notes[0]['onset']),'pitch':ident,'alter':target,'method':'native_two_lobe_unison_explicit_opposite_and_retained_principal_state'})
                            elif missing:state[ident]=None;continue
                            state[ident]=pvalue;certified_unisons.add(ident)
                        elif len(notes)==1:
                            h=ss[0];sign=attached.get(h['id']);value=None
                            if sign:value={8:1,9:-1,10:0}[sign['class']]
                            elif (empty_reader(h,g,row,track,stems,image,ink,all_heads,attached,native) if empty_reader else empty(h,g,row,track,stems,image,ink,all_heads,attached)):value=prior
                            possible_sign=any(n['class'] in (8,9,10) and .65<(h['xy'][0]-n['xy'][0])/s<4.5 and abs((n['xy'][1]-h['xy'][1])/s-(-.75 if n['class']==9 else 0))<.4 and glyph(n,image,s) is not None for n in row['nodes'])
                            state[ident]=value if value is not None else None if possible_sign else prior
                            if ident not in certified_unisons or value is None or sign is not None:continue
                            r=notes[0];n=r['node'];old_value=int(n.findtext('pitch/alter','0'))
                            if old_value!=value and n.find('tie') is None and n.find('notations/tied') is None and n.get('print-object')!='no':
                                set_alter(n,value,None);changes.append({'measure':i+1,'staff':sn,'onset':str(r['onset']),'pitch':ident,'old_alter':old_value,'new_alter':value,'method':'principal_accidental_state_after_certified_chromatic_unison'})
                        else:state[ident]=None
                        if len(changes)>=128:break
                    if len(changes)>=128:break
                if len(changes)>=128:break
            if len(changes)>=128:break
        offset+=len(rr[0]['bounds'])-1
        if len(changes)>=128:break
    # Alter corrections retain all event clocks/labels; new unisons only add.
    clocks=lambda rows:Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['grace'],written(r) if r['midi'] is not None else None) for rs in rows for r in rs)
    if clocks(rows)-clocks(records(root)):raise ValueError('Unison state changed an event clock or written pitch')
    return changes
