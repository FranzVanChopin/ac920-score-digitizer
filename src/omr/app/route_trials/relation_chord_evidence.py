"""Bounded chord-level signed head multisets, without altering existing notes.

The image, predicted graph and recognized XML are the only inputs. Conflicting
simultaneous accidental states remain unknown rather than using musical priors.
"""
from collections import Counter,defaultdict
import copy,xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from .relation_coverage import inputs,core
from .relation_multivalue_state import slots
from .relation_measure_joint import context_rows,written,align_columns,set_alter
from .relation_joint_candidates import component
from .relation_physical_streams import extended_shaft,native_shaft
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from .symbol_cnn import FACTOR

VALUES={8:1,9:-1,10:0}

def real(n):return dict(n,xy=[(v+.5)/FACTOR-.5 for v in n['xy']])

def signed_nodes(ident,hs,g,row,track,clef,image,native,all_heads,cache):
    """Classify an accidental's staff slot before asking which chord owns it."""
    s=row['space_px'];result=[];members={h['id'] for h in g['heads']}
    for n in row['nodes']:
        if n['class'] not in VALUES or n.get('confidence',0)<.45:continue
        x,y=n['xy'];dx=(max(h['xy'][0] for h in hs)-x)/s
        if not .65<dx<4.5 or x>=min(h['xy'][0] for h in hs)-.65*s:continue
        p=staff_step(track,x,y+(.75*s if n['class']==9 else 0))
        if abs(p['residual'])>.35 or pitch_for_step(p['nearest_step'],clef)!=ident:continue
        if any(a['id'] not in members and x+.65*s<a['xy'][0]<min(h['xy'][0] for h in hs)-.5*s and abs(a['xy'][1]-hs[0]['xy'][1])<.3*s for a in all_heads):continue
        if n['id'] not in cache:
            low=component(n,image,s);high=component(real(n),native,s/FACTOR);cache[n['id']]=(low,high)
        low,high=cache[n['id']]
        if high is None or n['confidence']<.65 and low is None:continue
        result.append(n)
    return result if len(result)<=2 and len({n['class'] for n in result})==len(result) else []

def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096 or len(graph['systems'])>128:return []
    image,tracks=loaded;native=cv2.imread(str(photo),0);ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1];rs=records(root);ctx=context_rows(root);ms=root.findall('part/measure');stems={s['id']:s for s in graph['stems']};changes=[];offset=0;locations=[]
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy];tt=[t for t in tracks if t['system']==sy]
        if len(rows)!=len(tt):return []
        for row,track in zip(rows,tt):
            locations.append((row,track,offset,{}))
        offset+=len(rows[0]['bounds'])-1
    for row,track,offset,cache in locations:
        sn=row['staff'];s=row['space_px'];all_heads=[h for g in row['groups'] for h in g.get('heads',[])]
        for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
            i=offset+mi;key,clefs=ctx[i];clef=clefs.get(sn);es=[r for r in rs[i] if r['staff']==sn and r['midi'] is not None and not r['grace']]
            if not clef or key is None or not 2<=len(es)<=128 or len({r['voice'] for r in es})!=1:continue
            cells=[[r for r in es if r['onset']==t] for t in sorted({r['onset'] for r in es})];seq=sorted([g for g in row['groups'] if left<g['x']<right and g.get('heads') and not g.get('grace')],key=lambda g:g['x'])
            if max(len(cells),len(seq))>64 or any(len(g['heads'])>8 for g in seq):continue
            pairs=align_columns([Counter(written(r) for r in c) for c in cells],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq]);mapping={b:a for a,b in pairs}
            history={}
            for gi,g in enumerate(seq):
                observed={tuple(h['pitch'][:2]) for h in g['heads']}
                if gi not in mapping:
                    for ident in observed:history.pop(ident,None)
                    continue
                cell=cells[mapping[gi]];members=defaultdict(list)
                for r in cell:members[written(r)].append(r)
                if not 2<=len(cell)<=8 or len({(r['voice'],r['duration']) for r in cell})!=1 or sum((Counter(written(r) for r in cell)&Counter(tuple(h['pitch'][:2]) for h in g['heads'])).values())<2:
                    history.clear();continue
                if not (extended_shaft(g,row,stems,image) or native_shaft(g,row,stems,native,ink)):
                    history.clear();continue
                physical={ident:slots(ident,g,row,track,clef,native) for ident in observed};signs={ident:signed_nodes(ident,hs,g,row,track,clef,image,native,all_heads,cache) for ident,hs in physical.items() if hs}
                for ident,notes in members.items():
                    hs=physical.get(ident,[]);accs=signs.get(ident,[]);prior=history.pop(ident,None)
                    if not hs or any(r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in notes):continue
                    have=Counter(int(r['node'].findtext('pitch/alter','0')) for r in notes);values={VALUES[n['class']] for n in accs}
                    if len(hs)==2:
                        sx=np.mean(np.array(stems[g['stem']]['xy'])[:,0]);a,b=sorted(hs,key=lambda h:h['xy'][0])
                        if abs(a['observed_y']-b['observed_y'])>.2*s or not .4*s<sx-a['xy'][0]<.8*s or not .4*s<b['xy'][0]-sx<.8*s:continue
                        if len(values)==2:want=Counter(values)
                        elif len(values)==1 and len(notes)==1 and not values&set(have) and values|set(have)==prior:want=Counter(values|set(have))
                        else:continue
                        if have-want:continue
                        missing=want-have
                        if sum(missing.values())==1 and len(cell)<8 and len(ms[i].findall('note'))<128:
                            n=copy.deepcopy(notes[0]['node'])
                            for tag in ('accidental','notations','lyric','tie'):
                                for a in n.findall(tag):n.remove(a)
                            if n.find('chord') is None:n.insert(0,E.Element('chord'))
                            value=next(iter(missing));set_alter(n,value,None);ms[i].insert(max(list(ms[i]).index(r['node']) for r in cell)+1,n);changes.append({'measure':i+1,'staff':sn,'onset':str(notes[0]['onset']),'pitch':ident,'alter':value,'method':'two_physical_heads_signed_multiset_missing_member'})
                        elif missing:continue
                        history[ident]=set(want)
                    else:history.pop(ident,None)
                    if len(changes)>=128:return changes
    return changes
