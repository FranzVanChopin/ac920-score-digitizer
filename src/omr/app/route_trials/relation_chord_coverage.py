"""Ink-proven displaced seconds and conservative missing chord members."""
import copy
from collections import Counter
from fractions import Fraction
import xml.etree.ElementTree as E
import cv2,numpy as np
from .relation_coverage import inputs,core,context_rows,safe_measure,ordered,signature
from .relation_key_state import natural_component,padded_empty_strip
from .relation_accidentals import associations
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from omr_event_records import records


def attach_seconds(row,stems,image):
    if len(row['groups'])>2048 or len(row['bounds'])>129:return []
    space=row['space_px'];changes=[];removed=[]
    for orphan in row['groups']:
        if len(changes)>=128:break
        hs=orphan.get('heads',[])
        if orphan.get('stem') is not None or orphan.get('rest') or orphan.get('grace') or len(hs)!=1:continue
        h=hs[0]
        if h.get('class')!=0 or h.get('confidence',0)<.6 or not core(h,image,space,.95):continue
        hx,hy=h['xy'];candidates=[]
        for group in row['groups']:
            stem=stems.get(group.get('stem'));members=group.get('heads',[])
            if not stem or stem.get('confidence',0)<.97 or group.get('grace') or not 2<=len(members)<8 or group.get('direction') not in ('up','down') or group.get('flags'):continue
            if not any(a<hx<b and a<group['x']<b for a,b in zip(row['bounds'],row['bounds'][1:])):continue
            points=np.array(stem['xy']);sx=float(points[:,0].mean());lo,hi=sorted(points[:,1]);sign=1 if group['direction']=='up' else -1
            if not .35*space<sign*(sx-hx)<.85*space or not lo-.5*space<hy<hi+.5*space:continue
            adjacent=[a for a in members if a.get('class')==0 and a.get('confidence',0)>=.9 and .35*space<abs(a['xy'][1]-hy)<.68*space and -.85*space<sign*(sx-a['xy'][0])<-.35*space and core(a,image,space,.95)]
            if len(adjacent)!=1:continue
            a=adjacent[0]
            if any(abs(b['xy'][0]-hx)<.4*space and abs(b['xy'][1]-hy)<.3*space for b in members):continue
            valid=True
            for node in (h,a):
                x,y=map(round,node['xy']);left,right=sorted((x,round(sx)))
                if min(left,y-2)<0 or right>=image.shape[1] or y+2>=image.shape[0] or ((image[y-2:y+3,left:right+1]<140).max(0)).mean()<.95:valid=False
            top,bottom=round(min(lo,hy,a['xy'][1])),round(max(hi,hy,a['xy'][1]));xx=round(sx)
            if top<0 or bottom>=image.shape[0] or xx<2 or xx+2>=image.shape[1] or ((image[top:bottom+1,xx-2:xx+3]<140).max(1)).mean()<.95:valid=False
            if valid:candidates.append(group)
        if len(candidates)!=1:continue
        group=candidates[0];group['heads'].append(h);h['displaced_second_ink_proof']=True;removed.append(orphan);changes.append({'head':h['id'],'stem':group['stem'],'xy':h['xy'],'method':'displaced_second_filled_core_opposite_head_and_continuous_shared_stem'})
    for orphan in removed:row['groups'].remove(orphan)
    return changes


def natural_for(head,row,image):
    space=row['space_px'];x,y=head['xy'];hits=[]
    for n in row['nodes']:
        if n.get('class')!=10 or n.get('confidence',0)<.8:continue
        ax,ay=n['xy']
        if not .65<(x-ax)/space<3.3 or abs(ay-y)>.25*space:continue
        left,top=round(ax-1.5*space),round(ay-2*space);right,bottom=round(ax+1.5*space),round(ay+2*space)
        if min(left,top)<0 or right>=image.shape[1] or bottom>=image.shape[0]:continue
        ink=(image[top:bottom,left:right]<140).astype('uint8')
        if natural_component(ink,(ax-left,ay-top),space) is not None:hits.append(n)
    return hits[0] if len(hits)==1 else None


def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;events=records(root);ctx=context_rows(root);stems={s['id']:s for s in graph.get('stems',[])};ms=root.findall('part/measure');off=0;changes=[];div=1;divs=[];ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1]
    for m in ms:
        for a in m.findall('attributes'):div=int(a.findtext('divisions',str(div)))
        divs.append(div)
    for system in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in rows:
            if row['staff']>len(staffs):continue
            sn=row['staff'];track=staffs[sn-1];all_heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for g in row['groups'] for h in g.get('heads',[])];attached=associations(row)
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=off+mi;key,clefs=ctx[number];clef=clefs.get(sn);es=[r for r in events[number] if r['staff']==sn];seq=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x']);old=ordered(es)
                if key is None or not clef or not safe_measure(es) or len({r['voice'] for r in es})!=1 or not old or len(old)!=len(seq) or len(seq)>64:continue
                if any(g.get('grace') for g in seq):continue
                seen=False;inline=False
                for c in ms[number]:
                    if c.tag=='note':seen=True
                    elif c.tag=='attributes' and seen:inline=True
                if inline:continue
                for group,g in zip(old,seq):
                    if len(changes)>=128 or len(ms[number].findall('note'))>=128:continue
                    if len(group)<2 or not 3<=len(g.get('heads',[]))<=8 or len(g['heads'])!=len(group)+1 or any(r['midi'] is None or r['onset']!=Fraction(g.get('onset',-1),480) or r['duration']!=Fraction(g['duration'],480) for r in group):continue
                    original=Counter((r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))) for r in group);observed=Counter(tuple(h['pitch'][:2]) for h in g['heads'])
                    if original-observed or sum((observed-original).values())!=1 or any(v>1 for v in observed.values()):continue
                    target=next(iter(observed-original));h=next(h for h in g['heads'] if tuple(h['pitch'][:2])==target);p=staff_step(track,*h['xy'])
                    if pitch_for_step(p['nearest_step'],clef)!=target or abs(p['residual'])>.3 or h.get('class')!=0 or not core(h,image,row['space_px'],.95):continue
                    if h.get('confidence',0)<.92 and not h.get('displaced_second_ink_proof'):continue
                    stem=stems.get(g.get('stem'))
                    if not stem or stem.get('confidence',0)<.97 or g.get('direction') not in ('up','down'):continue
                    # Explicit naturals have an independent connected-component
                    # shape witness. Other accidental kinds abstain here.
                    glyph=natural_for(h,row,image)
                    if glyph is None or h['pitch'][2:]!=[0,'natural']:continue
                    template=group[0]['node']
                    if any(len(r['node'].findall('notations/*')) or r['node'].find('lyric') is not None for r in group):continue
                    n=copy.deepcopy(template)
                    for tag in ('pitch','accidental','notations','lyric','tie'):
                        for node in n.findall(tag):n.remove(node)
                    pnode=E.Element('pitch');E.SubElement(pnode,'step').text=target[0];E.SubElement(pnode,'octave').text=str(target[1]);n.insert(0,pnode)
                    if n.find('chord') is None:n.insert(0,E.Element('chord'))
                    acc=E.Element('accidental');acc.text='natural';at=list(n).index(n.find('type'))+1 if n.find('type') is not None else len(n);n.insert(at,acc)
                    m=ms[number];index=max(list(m).index(r['node']) for r in group)+1;m.insert(index,n)
                    changes.append({'measure':number+1,'staff':sn,'pitch':target,'onset':str(group[0]['onset']),'duration':str(group[0]['duration']),'xy':h['xy'],'method':'additional_ink_head_shared_stem_and_isolated_natural','all_existing_events_preserved':True})
        off+=len(rows[0]['bounds'])-1
    return changes
