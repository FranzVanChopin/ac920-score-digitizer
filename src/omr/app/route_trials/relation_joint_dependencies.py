"""Ink-certified chord ownership, complete clocks and grace dependencies.

Inputs contain only recognized events and same-photo predictions. A small
same-pitch grace must precede a unique physical principal; meter completion
and a grace classification alone never justify moving it. Search is bounded.
"""
from collections import Counter
import copy
import xml.etree.ElementTree as E
import cv2
import numpy as np
from omr_event_records import records
from repair_grace_timing import _rebuild
from .relation_complete_clocks import plans, token, rest_ink
from .relation_coverage import inputs, core
from .relation_measure_joint import context_rows, head_witness
from .relation_measure_realign import traced_stem, hollow_on_line
from .symbol_cnn import FACTOR
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step


def body_height(head, image, space):
    x,y=head['xy']; yy=np.arange(round(y-.75*space),round(y+.75*space)+1)
    if yy.min()<0 or yy.max()>=image.shape[0]:return None
    spans=[]
    for offset in (-.12,-.06,0,.06,.12):
        xx=round(x+offset*space)
        if not 0<=xx<image.shape[1]:return None
        bits=image[yy,xx]<140; d=np.diff(np.r_[False,bits,False].astype('int8'))
        found=[(b-a,yy[a]+(b-a-1)/2) for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]
        nearest=min(found,key=lambda p:abs(p[1]-y),default=(0,1e9))
        spans.append(nearest[0]/space if abs(nearest[1]-y)<.25*space else 0)
    return float(np.median(spans))


def joined_columns(row, stems, image):
    groups=copy.deepcopy(row['groups']); removed=set(); edges=[]; s=row['space_px']
    if len(groups)>2048 or sum(g.get('stem') is None and not g.get('rest') and not g.get('grace') for g in groups)>128:return groups,edges
    for i,g in enumerate(groups):
        if g.get('stem') is not None or g.get('rest') or g.get('grace') or len(g.get('heads',[]))!=1:continue
        h=g['heads'][0]
        if h.get('class')!=0 or h.get('confidence',0)<.85:continue
        choices=[]
        for j,p in enumerate(groups):
            if i==j or j in removed or p.get('stem') is None or p.get('grace') or p.get('rest') or len(p.get('heads',[]))>=8 or p.get('voice')!=g.get('voice') or abs(p['x']-g['x'])>.25*s:continue
            proposal=dict(p,heads=p['heads']+[h])
            if traced_stem(proposal,row,stems,image):choices.append(j)
        if len(choices)==1:
            parent=choices[0]; groups[parent]['heads'].append(h); removed.add(i)
            edges.append({'head':h['id'],'stem':groups[parent]['stem'],'method':'unique_coincident_continuous_shaft'})
    return sorted([g for i,g in enumerate(groups) if i not in removed],key=lambda g:g['x']),edges


def written(r):
    return r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))


def attached_graces(rs):
    """Retain the recognized staff/voice and order when finding principals."""
    result=[]; pending={}
    for r in rs:
        k=(r['staff'],r['voice'])
        if r['grace']:pending.setdefault(k,[]).append(r)
        elif not r['chord']:
            for gr in pending.pop(k,[]):result.append((gr,r))
    return result


def fuse(root, graph, photo, geometry):
    before=records(root); jobs=plans(root)
    if not jobs and not any(r['grace'] for rs in before for r in rs):return [],{}
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096 or len(jobs)>128:return [],{}
    image,tracks=loaded; stems={s['id']:s for s in graph['stems']}; ctx=context_rows(root)
    locations={}; offset=0; native=None; edges=[]; changes=[]
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy]; ts=[t for t in tracks if t['system']==sy]
        if len(rows)!=len(ts):return [],{}
        for row in rows:
            row=dict(row); row['groups'],joins=joined_columns(row,stems,image); edges.extend(joins)
            for mi in range(len(row['bounds'])-1):locations[offset+mi,row['staff']]=(row,ts[row['staff']-1],mi)
        offset+=len(rows[0]['bounds'])-1

    def grace_proof(gr, principal, row, main, columns):
        nonlocal native
        if principal['midi'] is None or gr['midi'] is None or gr['voice']!=principal['voice']:return False
        pitch=written(gr); anchors=[h for h in main.get('heads',[]) if tuple(h['pitch'][:2])==pitch and h.get('class')==0]
        if len(anchors)!=1:return False
        s=row['space_px']; candidates=[]
        for g in columns:
            if not g.get('grace') or not .6*s<main['x']-g['x']<3*s:continue
            if any(not h.get('grace') and g['x']<h['x']<main['x']-.25*s for h in columns):continue
            candidates.extend(h for h in g.get('heads',[]) if tuple(h['pitch'][:2])==pitch and h.get('class')==0 and h.get('confidence',0)>=.45)
        if len(candidates)!=1:return False
        if native is None:native=cv2.imread(str(photo),0)
        def real(h):return dict(h,xy=[(v+.5)/FACTOR-.5 for v in h['xy']])
        small=body_height(real(candidates[0]),native,s/FACTOR); normal=body_height(real(anchors[0]),native,s/FACTOR)
        return bool(small is not None and normal is not None and .3<=small<=1 and .85<=normal<=1.6 and .35<=small/normal<=.83)

    def retained_head(h,g,row,track,clef):
        if head_witness(h,row,track,clef,image) or hollow_on_line(h,row,track,clef,image):return True
        # A weak member already retained in XML needs two strong chord anchors,
        # a normal-size dark body, staff position and a certified common shaft.
        if h.get('class')!=0 or h.get('confidence',0)<.7 or sum(a.get('confidence',0)>=.85 for a in g['heads'])<2:return False
        p=staff_step(track,*h['xy']); height=body_height(h,image,row['space_px'])
        return bool(height is not None and .85<=height<=1.6 and abs(p['residual'])<=.3 and pitch_for_step(p['nearest_step'],clef)==tuple(h['pitch'][:2]) and core(h,image,row['space_px'],.95))

    for job in jobs:
        i=job['index']; rs=before[i]; bound={}; valid=True; desired=dict(job['desired'])
        for sn,source in job['groups'].items():
            row,track,mi=locations[i,sn]; left,right=row['bounds'][mi:mi+2]
            cols=[g for g in row['groups'] if left<g['x']<right]; ordinary=[g for g in cols if not g.get('grace')]
            gt=[Counter([None]) if g.get('rest') else Counter(tuple(h['pitch'][:2]) for h in g.get('heads',[])) for g in ordinary]
            if len(ordinary)>64 or gt!=[token(c) for c in source] or any(b['x']-a['x']<.65*row['space_px'] for a,b in zip(ordinary,ordinary[1:])):valid=False;break
            for cell,g in zip(source,ordinary):
                if g.get('rest'):
                    if len(cell)!=1 or g['duration']!=cell[0]['duration']*480 or not rest_ink(g['rest'],image,row['space_px']):valid=False;break
                elif g.get('tuplet_ratio') or g['duration']!=cell[0]['duration']*480 or not traced_stem(g,row,stems,image) or any(not retained_head(h,g,row,track,ctx[i][1].get(sn,{})) for h in g['heads']):valid=False;break
                bound[cell[0]['node']]=(row,g,cols)
            if not valid:break
        if not valid:continue
        anchors=[]; a=job['groups'][1]; b=job['groups'][2]; s=min(locations[i,sn][0]['space_px'] for sn in (1,2))
        for cell in a:
            g=bound[cell[0]['node']][1]; matches=[c for c in b if abs(bound[c[0]['node']][1]['x']-g['x'])<=.35*s]
            if len(matches)>1 or matches and desired[cell[0]['node']]!=desired[matches[0][0]['node']]:valid=False;break
            if matches:anchors.append(desired[cell[0]['node']])
        if not valid or len(set(anchors))<2 or min(anchors)!=0:continue
        attachments=attached_graces(rs)
        if len(attachments)!=sum(r['grace'] for r in rs):continue
        for gr,principal in attachments:
            if principal['node'] not in bound or not grace_proof(gr,principal,*bound[principal['node']]):valid=False;break
            desired[gr['node']]=desired[principal['node']]
        if not valid or any((v*job['division']).denominator!=1 for v in desired.values()):continue
        m=root.findall('part/measure')[i]; saved=copy.deepcopy(list(m))
        if not _rebuild(m,{n:int(t*job['division']) for n,t in desired.items()}):m[:]=saved;continue
        conserved=lambda rows:Counter((r['staff'],r['voice'],r['midi'],r['duration'],r['grace']) for r in rows)
        if conserved(rs)!=conserved(records(root)[i]):raise ValueError('Joint dependencies changed event identity')
        changes.append({'measure':i+1,'method':'complete_staff_clocks_with_native_small_grace_principals','grace_dependencies':len(attachments),'coincident_columns':len(set(anchors))})

    # Reanchor a small grace to an already recognized, photograph-timed main
    # column. Ordinary events, rests, labels and all grace identities stay.
    divisions=1
    for i,(m,rs) in enumerate(zip(root.findall('part/measure'),records(root))):
        for a in m.findall('attributes'):divisions=int(a.findtext('divisions',str(divisions)))
        if not 0<divisions<=9600 or len(rs)>128 or len(changes)>=128 or m.get('implicit')=='yes':continue
        desired={r['node']:int(r['onset']*divisions) for r in rs}; moves=[]
        for gr,principal in attached_graces(rs):
            if gr['onset']==principal['onset'] or principal['midi'] is None or principal['node'].find('tie') is not None:continue
            row,track,mi=locations[i,principal['staff']]; left,right=row['bounds'][mi:mi+2]
            cols=[g for g in row['groups'] if left<g['x']<right]
            cell=[r for r in rs if not r['grace'] and r['voice']==principal['voice'] and r['staff']==principal['staff'] and r['onset']==principal['onset']]
            matches=[g for g in cols if not g.get('grace') and not g.get('rest') and Counter(tuple(h['pitch'][:2]) for h in g.get('heads',[]))==token(cell)]
            if len(matches)!=1 or len(cols)>64:continue
            g=matches[0]
            if g.get('onset')!=principal['onset']*480 or g['duration']!=principal['duration']*480 or not traced_stem(g,row,stems,image) or not grace_proof(gr,principal,row,g,cols):continue
            desired[gr['node']]=desired[principal['node']]; moves.append(gr)
        if not moves:continue
        saved=copy.deepcopy(list(m))
        if not _rebuild(m,desired):m[:]=saved;continue
        ordinary=lambda rows:Counter((r['key'],r['voice']) for r in rows if not r['grace'])
        grace_identity=lambda rows:Counter((r['staff'],r['voice'],r['midi']) for r in rows if r['grace'])
        after=records(root)[i]
        if ordinary(rs)!=ordinary(after) or grace_identity(rs)!=grace_identity(after):raise ValueError('Grace dependency changed ordinary events')
        changes.append({'measure':i+1,'method':'native_small_grace_to_unique_existing_physical_principal','graces_moved':len(moves)})
    return changes,{'ownership_edges':edges,'maximum_events':128,'maximum_columns':64,'reference_supplied':False}
