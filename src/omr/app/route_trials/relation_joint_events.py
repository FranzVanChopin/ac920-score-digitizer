"""Joint physical chord ownership and native-resolution accidental witnesses.

Existing events keep their clock, pitch, voice and notation. New members need
unique parent binding, connected ink, retained anchors and proven pitch state.
No source score, reference count, page lookup, new model or PL is used.
"""
from collections import Counter
from fractions import Fraction
import copy
import xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from .symbol_cnn import FACTOR
from .relation_coverage import inputs
from .relation_key_state import calibrated_signature
from .relation_measure_joint import context_rows,align_columns,written
from .relation_measure_realign import hollow_on_line,traced_stem
from .relation_joint_candidates import component,chromatic_heads
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from repair_unison_voices import _stem_end


def hollow(h,row,track,clef,image,minimum=.85):
    # The classifier gate is checked here; the reused function below supplies
    # its independent enclosed-hole and staff-position proof.
    return h.get('confidence',0)>=minimum and hollow_on_line(dict(h,confidence=1.),row,track,clef,image)


def attach_hollows(row,track,clef,stems,image,ink):
    groups=copy.deepcopy(row['groups']);removed=set();edges=[];s=row['space_px']
    if sum(g.get('stem') is None and len(g.get('heads',[]))==1 and g['heads'][0].get('class')==1 for g in groups)>128:return groups,[]
    for i,orphan in enumerate(groups):
        hs=orphan.get('heads',[])
        if orphan.get('stem') is not None or orphan.get('rest') or orphan.get('grace') or len(hs)!=1 or hs[0].get('class')!=1:continue
        h=hs[0]
        if not hollow(h,row,track,clef,image):continue
        choices=[]
        for j,g in enumerate(groups):
            if i==j or g.get('stem') not in stems or g.get('grace') or not 2<=len(g.get('heads',[]))<8 or any(a.get('class')!=1 for a in g['heads']):continue
            if abs(g['x']-h['xy'][0])>1.4*s:continue
            if not any(l<h['xy'][0]<r and l<g['x']<r for l,r in zip(row['bounds'],row['bounds'][1:])):continue
            if any(tuple(a['pitch'][:2])==tuple(h['pitch'][:2]) for a in g['heads']):continue
            end=_stem_end(dict(x=h['xy'][0],y=h['xy'][1],staff_space=s),ink,g.get('direction')=='up')
            stem=stems[g['stem']];sx=sum(p[0] for p in stem['xy'])/2
            if end is None or abs(end[0]-sx)>.25*s:continue
            proposal=dict(g,heads=g['heads']+[h])
            if not traced_stem(proposal,row,stems,image) or any(not hollow(a,row,track,clef,image,.8) for a in g['heads']):continue
            choices.append(j)
        if len(choices)!=1:continue
        parent=choices[0];groups[parent]['heads'].append(h);removed.add(i);edges.append({'head':h['id'],'stem':groups[parent]['stem'],'method':'hollow_orphan_unique_continuous_ink_shaft'})
    return [g for i,g in enumerate(groups) if i not in removed],edges


def clear_sign_strip(h,row,image):
    s=row['space_px'];x,y=h['xy'];l,r=round(x-3.5*s),round(x-s);top,bottom=round(y-1.4*s),round(y+1.4*s)
    if min(l,top)<0 or r>=image.shape[1] or bottom>=image.shape[0]:return False
    if any(n['class'] in (8,9,10) and l-.2*s<n['xy'][0]<x-.65*s and abs(n['xy'][1]-y-(-.75*s if n['class']==9 else 0))<.4*s for n in row['nodes']):return False
    ink=(image[top:bottom,l:r]<140).astype('uint8');horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,round(1.2*s)),np.uint8));clean=cv2.subtract(ink,horizontal)
    return cv2.countNonZero(clean)<.2*s*s and cv2.countNonZero(cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((round(s),1),np.uint8)))==0


def key_proof(row,track,clef,key,groups,glyph_proof,image):
    if key==0 or calibrated_signature(row,track,clef,key):return True
    if not 1<=abs(key)<=7:return False
    # A clef can be proposed as a stemless head before the signature. Anchor
    # the prefix to a real stem rather than that false music node.
    first=min((g['x'] for g in groups if g.get('stem') is not None and not g.get('grace') and any(h.get('confidence',0)>=.85 for h in g.get('heads',[]))),default=None)
    if first is None:return False
    s=row['space_px'];cls=9 if key<0 else 8;nodes=sorted([n for n in row['nodes'] if n['class']==cls and row['bounds'][0]<n['xy'][0]<first-.65*s],key=lambda n:n['xy'][0])
    if len(nodes)!=abs(key) or any(n.get('confidence',0)<.8 for n in nodes):return False
    for i,(n,letter) in enumerate(zip(nodes,('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)])):
        if i and not .6*s<n['xy'][0]-nodes[i-1]['xy'][0]<1.5*s:return False
        p=staff_step(track,n['xy'][0],n['xy'][1]+(.75*s if cls==9 else 0))
        if abs(p['residual'])>.35 or pitch_for_step(p['nearest_step'],clef)[0]!=letter or glyph_proof(n,image,s) is None:return False
    return True


def fuse(root,graph,photo,geometry):
    if any(len(g.get('heads',[]))>8 for r in graph['systems'] for g in r['groups']):return [],{'abstained':'chord_size'}
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096 or len(graph['systems'])>128:return [],{}
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];native=None;native_hits=set()
    def glyph_proof(node,legacy,space):
        nonlocal native
        found=component(node,legacy,space)
        if found is not None:return found
        if native is None:native=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
        n=dict(node,xy=[(v+.5)/FACTOR-.5 for v in node['xy']]);proof=component(n,native,space/FACTOR)
        if proof is not None:native_hits.add((node['class'],tuple(node['xy'])))
        return proof
    events=records(root);ms=root.findall('part/measure');ctx=context_rows(root);stems={s['id']:s for s in graph['stems']};changes=[];edges=[];offset=0
    def verify_retained():
        token=lambda rs:Counter((r['key'],r['voice'],r['grace']) for r in rs)
        if any(token(a)-token(b) for a,b in zip(events,records(root))):raise ValueError('Joint event grouping changed a retained event')
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        for row in rows:
            sn=row['staff']
            if not 1<=sn<=len(ts) or len(row['groups'])>2048:continue
            track=ts[sn-1];clef=ctx[offset][1].get(sn)
            if not clef:continue
            groups,joined=attach_hollows(row,track,clef,stems,image,ink);edges.extend(joined)
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;m=ms[number];key,clefs=ctx[number];clef=clefs.get(sn)
                if key is None or not clef or len(events[number])>128:continue
                es=[r for r in events[number] if r['staff']==sn and r['midi'] is not None and not r['grace']]
                if not es or len({r['voice'] for r in es})!=1:continue
                old=[[r for r in es if r['onset']==t] for t in sorted({r['onset'] for r in es})]
                seq=sorted([g for g in groups if left<g['x']<right and g.get('heads') and not g.get('grace')],key=lambda g:g['x'])
                if max(len(old),len(seq))>64:continue
                pairs=align_columns([Counter(written(r) for r in cell) for cell in old],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq])
                for oi,gi in pairs:
                    cell=old[oi];g=seq[gi]
                    if len(changes)>=128 or len(m.findall('note'))>=128:
                        verify_retained();return changes,{'ownership_edges':edges,'native_glyph_witnesses':len(native_hits),'existing_events_preserved':True}
                    if not 2<=len(cell)<8 or len({(r['voice'],r['duration']) for r in cell})!=1 or any(r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in cell):continue
                    if sum(not r['chord'] for r in cell)!=1:continue
                    positions=[list(m).index(r['node']) for r in cell];members={r['node'] for r in cell}
                    if any(n.tag=='note' and n not in members for n in list(m)[min(positions):max(positions)+1]):continue
                    # The original recognized rhythm supplies the insertion
                    # clock only when the candidate's group agrees with it.
                    if any(r['onset']*480!=g.get('onset') or r['duration']*480!=g['duration'] for r in cell):continue
                    original=Counter(written(r) for r in cell);observed=Counter(tuple(h['pitch'][:2]) for h in g['heads'])
                    if original-observed or sum((observed-original).values())!=1:continue
                    before=len(changes)
                    chromatic_heads(cell,g,row,track,clef,stems,image,changes,number,m,glyph_proof=glyph_proof)
                    if len(changes)>before:continue
                    extra=next(iter(observed-original))
                    if observed[extra]!=1 or any(h.get('class')!=1 for h in g['heads']) or not traced_stem(g,row,stems,image):continue
                    h=next(h for h in g['heads'] if tuple(h['pitch'][:2])==extra)
                    if not hollow(h,row,track,clef,image,.85) or any(not hollow(a,row,track,clef,image,.8) for a in g['heads'] if tuple(a['pitch'][:2]) in original):continue
                    if not key_proof(row,track,clef,key,groups,glyph_proof,image) or not clear_sign_strip(h,row,image):continue
                    alter=(1 if key>0 else -1) if extra[0] in ('FCGDAEB' if key>0 else 'BEADGCF')[:abs(key)] else 0
                    prior=[r for r in es if r['onset']<cell[0]['onset'] and written(r)==extra]
                    if prior:alter=int(max(prior,key=lambda r:r['onset'])['node'].findtext('pitch/alter','0'))
                    if h['pitch'][2]!=alter or h['pitch'][3] is not None:continue
                    n=copy.deepcopy(cell[0]['node'])
                    for tag in ('pitch','accidental','notations','lyric','tie','chord'):
                        for a in n.findall(tag):n.remove(a)
                    n.insert(0,E.Element('chord'));pitch=E.Element('pitch');E.SubElement(pitch,'step').text=extra[0]
                    if alter:E.SubElement(pitch,'alter').text=str(alter)
                    E.SubElement(pitch,'octave').text=str(extra[1]);n.insert(1,pitch);m.insert(max(list(m).index(r['node']) for r in cell)+1,n)
                    changes.append({'measure':number+1,'staff':sn,'onset':str(cell[0]['onset']),'pitch':extra,'alter':alter,'method':'joint_hollow_chord_member_retained_clock_and_unique_ink_stem'})
        offset+=len(rows[0]['bounds'])-1
    # Adding an actual chord member cannot advance any retained event. Check
    # this explicitly instead of trusting XML insertion positions.
    verify_retained()
    return changes,{'ownership_edges':edges,'native_glyph_witnesses':len(native_hits),'existing_events_preserved':True}
