"""Join uniquely attached heads and bind complete physical columns for state carry.

Only prediction clocks change. Recognized XML clocks and event labels remain.
"""
import copy
from collections import Counter
import cv2,numpy as np
from omr_event_records import records
from .relation_coverage import inputs
from .relation_measure_joint import context_rows,written
from .relation_physical_streams import extended_shaft,native_shaft,filled_slot
from .relation_multivalue_state import fuse as carry
from .relation_label_rest import meters
from .symbol_cnn import FACTOR

def native_continuity(g,row,stems,native):
    stem=stems.get(g.get('stem'))
    if not stem or g.get('direction') not in ('up','down'):return False
    pts=np.array(stem['xy']);sy=pts[:,1];tip=sy.min() if g['direction']=='up' else sy.max();head=(max if g['direction']=='up' else min)(h['xy'][1] for h in g['heads']);lo,hi=sorted(((tip+.5)/FACTOR-.5,(head+.5)/FACTOR-.5));yy=np.arange(round(lo),round(hi)+1)
    radius=max(2,round(.15*row['space_px']/FACTOR));old_y=(yy+.5)*FACTOR-.5;order=np.argsort(sy);xx=np.rint((np.interp(old_y,sy[order],pts[order,0])+.5)/FACTOR-.5).astype(int)
    if not len(yy) or yy.min()<0 or yy.max()>=native.shape[0] or xx.min()<radius or xx.max()+radius>=native.shape[1]:return False
    return (native[yy[:,None],xx[:,None]+np.arange(-radius,radius+1)]<140).any(1).mean()>=.98

def canonical_graph(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return graph,[]
    image,tracks=loaded;native=cv2.imread(str(photo),0);ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1];candidate=copy.deepcopy(graph);stems={s['id']:s for s in graph['stems']};ctx=context_rows(root);rs_all=records(root);clock_context=meters(root);offset=0;repairs=[]
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in candidate['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=len(ts):return graph,[]
        for row,track in zip(rr,ts):
            sn=row['staff'];s=row['space_px']
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;rs=[r for r in rs_all[number] if r['staff']==sn];clef=ctx[number][1].get(sn)
                if not clef or not 3<=len(rs)<=128 or any(r['midi'] is None or r['grace'] or r['duration']*480!=240 or r['node'].get('print-object')=='no' for r in rs) or len({r['voice'] for r in rs})!=1:continue
                old=[[r for r in rs if r['onset']==t] for t in sorted({r['onset'] for r in rs})];gs=copy.deepcopy([g for g in row['groups'] if left<g['x']<right]);removed=set();joins=[]
                if max(len(old),len(gs))>64 or clock_context[number][1]!=len(old)/2:continue
                for k,g in enumerate(gs):
                    if g.get('stem') is not None or g.get('grace') or len(g.get('heads',[]))!=1:continue
                    h=g['heads'][0]
                    if h.get('confidence',0)<.65 or not filled_slot(h,row,track,clef,native):continue
                    options=[]
                    for j,p in enumerate(gs):
                        if k==j or p.get('stem') is None or p.get('grace') or p.get('flags') or p.get('beams')!=1 or p.get('duration')!=240 or not 2<=len(p.get('heads',[]))<8 or abs(p['x']-g['x'])>.25*s or p.get('voice')!=g.get('voice'):continue
                        if tuple(h['pitch'][:2]) in [tuple(a['pitch'][:2]) for a in p['heads']]:continue
                        proposal=dict(p,heads=p['heads']+[h])
                        if extended_shaft(proposal,row,stems,image) or native_shaft(proposal,row,stems,native,ink) and native_continuity(proposal,row,stems,native):options.append(j)
                    if len(options)==1:gs[options[0]]['heads'].append(h);removed.add(k);joins.append(h['id'])
                seq=sorted([g for j,g in enumerate(gs) if j not in removed],key=lambda g:g['x'])
                if not joins or len(seq)!=len(old) or any(g.get('rest') or g.get('grace') or g.get('flags') or g.get('dotted') or g.get('tuplet_ratio') or g.get('beams')!=1 or g.get('duration')!=240 for g in seq):continue
                valid=True
                for j,(cell,g) in enumerate(zip(old,seq)):
                    original=Counter(written(r) for r in cell);observed=Counter(tuple(h['pitch'][:2]) for h in g['heads']);extra=observed-original
                    if original-observed or sum(extra.values())>1 or extra and not any(observed[p]==2 and original[p]==1 for p in extra) or any(r['onset']*480!=j*240 for r in cell):valid=False;break
                if not valid:continue
                for j,g in enumerate(seq):g['onset']=j*240
                row['groups']=[g for g in row['groups'] if not left<g['x']<right]+seq;row['groups'].sort(key=lambda g:g['x']);repairs.append({'measure':number+1,'staff':sn,'orphan_heads_joined':joins,'method':'unique_native_shaft_complete_eighth_columns_rebind_only_predicted_clocks'})
        offset+=len(rr[0]['bounds'])-1
    return candidate,repairs

def fuse(root,graph,photo,geometry):
    canonical,repairs=canonical_graph(root,graph,photo,geometry)
    if not repairs:return []
    changes=carry(root,canonical,photo,geometry)
    for c in changes:c["canonical_group_repairs"]=[r for r in repairs if r["measure"]==c["measure"] and r["staff"]==c["staff"]]
    return changes
