"""Bounded exact accidental assignment and independently isolated glyph masks.

No score labels, renderer, learned harmony, or new model are used. Every
accepted extra edge is mandatory across all assignments within the margin.
"""
import cv2,numpy as np
from .relation_accidentals import NAMES
from .relation_key_state import natural_component
from .staff_preprocess import staff_step

MAX_HEADS=8
MAX_GLYPHS=8
MAX_MASKS=1<<MAX_HEADS


def component(n,image,space):
    x,y=n['xy'];l,r=round(x-1.6*space),round(x+1.6*space);t,b=round(y-2.2*space),round(y+2.2*space)
    if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0]:return None
    crop=(image[t:b,l:r]<140).astype('uint8')
    if n['class']==10:
        mask=natural_component(crop,(x-l,y-t),space)
        return (l,t,mask) if mask is not None else None
    horizontal=cv2.morphologyEx(crop,cv2.MORPH_OPEN,np.ones((1,max(3,round(3*space))),np.uint8))
    clean=cv2.morphologyEx(cv2.subtract(crop,horizontal),cv2.MORPH_CLOSE,np.ones((3,1),np.uint8))
    count,labels,stats,centres=cv2.connectedComponentsWithStats(clean);valid=[]
    if count>64:return None
    for i,(xx,yy,w,h,area) in enumerate(stats[1:],1):
        cx,cy=centres[i]
        if min(xx,yy)<1 or xx+w>=clean.shape[1]-1 or yy+h>=clean.shape[0]-1 or abs(cx-(x-l))>.35*space or abs(cy-(y-t))>.45*space:continue
        if not .45*space<=w<=1.35*space or not 2*space<=h<=3.6*space or not .15<=area/(w*h)<=.7:continue
        mask=(labels[yy:yy+h,xx:xx+w]==i).astype('uint8')
        upright=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((max(3,round(1.6*space)),1),np.uint8));cols=upright.sum(0)>=1.6*space;edges=np.diff(np.r_[False,cols,False].astype('int8'));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0]
        if n['class']==8:
            if len(starts)!=2 or any(e-a>.25*space for a,e in zip(starts,ends)) or not .25*space<(starts[1]+ends[1]-starts[0]-ends[0])/2<.85*space:continue
            bars=[]
            for a,e in zip(starts,ends):
                ys=np.flatnonzero(upright[:,a:e].max(1));bars.append((ys.min(),ys.max()))
            if max(abs(bars[0][j]-bars[1][j]) for j in (0,1))>.35*space:continue
            # Both short crossbars must connect the two vertical bars.
            bridge=mask[:,ends[0]:starts[1]];occupied=(bridge.mean(1)>=.7);runs=np.diff(np.r_[False,occupied,False].astype('int8'));aa=np.where(runs==1)[0];bb=np.where(runs==-1)[0]
            if len(aa)!=2 or not .5*space<(aa[1]+bb[1]-aa[0]-bb[0])/2<1.4*space:continue
        elif n['class']==9:
            if len(starts)!=1 or ends[0]-starts[0]>.25*space or starts[0]>.3*space:continue
            ys=np.flatnonzero(upright[:,starts[0]:ends[0]].max(1))
            if len(ys)<2*space:continue
            right=mask[:,min(w,ends[0]+round(.1*space)):]
            if right.shape[1]<.3*space or cv2.countNonZero(right[:round(.45*h)])>.3*space or cv2.countNonZero(right[round(.45*h):])<.5*space:continue
        else:continue
        valid.append(labels==i)
    return (l,t,valid[0]) if len(valid)==1 else None


def witness(n,row,image):
    from .relation_measure_joint import glyph_witness
    if glyph_witness(n,row,image):return True
    return (n.get('confidence',0)>=.65 or n.get('independent_shape_proposal',False)) and component(n,image,row['space_px']) is not None


def mandatory_assignment(options,head_count,margin=.15):
    """Exact bit-mask DP; no beam pruning can manufacture uniqueness."""
    if not 1<=head_count<=MAX_HEADS or not 1<=len(options)<=MAX_GLYPHS:return {}
    n=len(options);size=1<<head_count;pre=np.full((n+1,size),np.inf);post=np.full((n+1,size),np.inf);pre[0,0]=0;post[n,:]=0
    for k,(_,edges) in enumerate(options):
        for mask in range(size):
            value=pre[k,mask]
            if not np.isfinite(value):continue
            pre[k+1,mask]=min(pre[k+1,mask],value+.8)
            for i,cost in edges:
                if not mask&(1<<i):pre[k+1,mask|(1<<i)]=min(pre[k+1,mask|(1<<i)],value+cost)
    for k in range(n-1,-1,-1):
        for mask in range(size):
            post[k,mask]=.8+post[k+1,mask]
            for i,cost in options[k][1]:
                if not mask&(1<<i):post[k,mask]=min(post[k,mask],cost+post[k+1,mask|(1<<i)])
    optimum=post[0,0];result={}
    for i in range(head_count):
        if min(pre[n,mask] for mask in range(size) if not mask&(1<<i))<=optimum+margin:continue
        possible=[]
        for k,(node,edges) in enumerate(options):
            edge=next((cost for target,cost in edges if target==i),None)
            if edge is not None and min(pre[k,mask]+edge+post[k+1,mask|(1<<i)] for mask in range(size) if not mask&(1<<i))<=optimum+margin:possible.append(node)
        if len(possible)==1:result[i]=possible[0]
    return result


def associations(seq,row,image):
    from collections import Counter
    s=row['space_px'];heads=[h for g in seq for h in g.get('heads',[])];accs=[n for n in row['nodes'] if n['class'] in NAMES and witness(n,row,image)];result={}
    for g in seq:
        hs=g.get('heads',[])
        if not 1<=len(hs)<=MAX_HEADS:continue
        options=[]
        for n in accs:
            edges=[]
            for i,h in enumerate(hs):
                dx=(h['xy'][0]-n['xy'][0])/s;dy=(n['xy'][1]-h['xy'][1])/s;target=-.75 if n['class']==9 else 0
                if not .65<dx<4.5 or abs(dy-target)>(.45 if n['class']==9 else .3):continue
                if any(a['xy'][0]<h['xy'][0]-.5*s and a['xy'][0]>n['xy'][0]+.65*s and abs(a['xy'][1]-h['xy'][1])<.3*s for a in heads):continue
                edges.append((i,abs(dy-target)+.1*dx))
            if edges:options.append((n,edges))
        for i,n in mandatory_assignment(options,len(hs)).items():result[hs[i]['id']]=n
    counts=Counter(n['id'] for n in result.values())
    return {hid:n for hid,n in result.items() if counts[n['id']]==1}


def empty(h,g,row,track,stems,image,ink,heads,attached):
    from .relation_measure_joint import empty_joint
    if empty_joint(h,g,row,track,stems,image,ink,heads,attached):return True
    s=row['space_px'];x,y=h['xy'];l,r=round(x-3.5*s),round(x-.65*s);t,b=round(y-1.3*s),round(y+1.3*s)
    previous=[a for a in heads if a['x']<x-3 and abs(a['y']-y)<1.7*s]
    members={tuple(a['xy']) for a in g['heads']}
    previous=[a for a in previous if (a['x'],a['y']) not in members]
    if previous:l=max(l,round(max(a['x']+.6*s for a in previous)+1))
    if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0] or r-l<.75*s:return False
    if any(l-.5*s<a['x']<r and abs(a['y']-y)<1.7*s and (a['x'],a['y']) not in members for a in heads):return False
    # A sign at the target's physical position can never be erased.
    if any(n['class'] in NAMES and l-.2*s<n['xy'][0]<r+.2*s and abs((n['xy'][1]-y)/s-(-.75 if n['class']==9 else 0))<.4 for n in row['nodes']):return False
    remaining=ink[t:b,l:r].copy();removed=False
    for other in g['heads']:
        n=attached.get(other['id'])
        if not n or other['id']==h['id'] or abs(other['xy'][1]-y)<.4*s or abs(staff_step(track,*other['xy'])['residual'])>.3:continue
        proof=component(n,image,s)
        if proof is None:continue
        x0,y0,mask=proof;mask=cv2.dilate(mask.astype('uint8'),np.ones((3,3),np.uint8));a,c=max(l,x0),max(t,y0);z,d=min(r,x0+mask.shape[1]),min(b,y0+mask.shape[0])
        if a<z and c<d:remaining[c-t:d-t,a-l:z-l][mask[c-y0:d-y0,a-x0:z-x0]>0]=0;removed=True
    # A displaced second places its own chord head and stem in this strip.
    # Mask only the photograph-certified head core/ellipse and a continuous
    # thin stem belonging to this exact group. Target-position signs vetoed
    # this branch before any mask was made.
    from .relation_measure_joint import stem_witness
    from .relation_coverage import core
    if stem_witness(g,row,stems,image):
        for other in g['heads']:
            ox,oy=other['xy']
            if other['id']==h['id'] or not l-.7*s<ox<r+.7*s or abs(oy-y)<.4*s or other.get('confidence',0)<.7 or abs(staff_step(track,ox,oy)['residual'])>.3 or not core(other,image,s,.95):continue
            mask=np.zeros(remaining.shape,np.uint8);cv2.ellipse(mask,(round(ox-l),round(oy-t)),(round(.72*s),round(.40*s)),-25,0,360,1,-1);remaining[mask>0]=0;removed=True
        stem=stems[g['stem']];sx=round(sum(p[0] for p in stem['xy'])/2)
        if l<=sx<r:remaining[:,max(0,sx-l-2):min(r-l,sx-l+3)]=0;removed=True
    if not removed:return False
    horizontal=cv2.morphologyEx(remaining,cv2.MORPH_OPEN,np.ones((1,max(3,round(1.2*s))),np.uint8));remaining=cv2.subtract(remaining,horizontal)
    return cv2.countNonZero(remaining)<2*s and cv2.countNonZero(cv2.morphologyEx(remaining,cv2.MORPH_OPEN,np.ones((max(3,round(.45*s)),1),np.uint8)))==0


def chromatic_heads(group,g,row,track,clef,stems,image,changes,number,measure,glyph_proof=None):
    """Recover one missing chromatic unison from two detected physical lobes."""
    from collections import Counter
    import copy,xml.etree.ElementTree as E
    from .relation_measure_joint import written,set_alter,stem_witness
    from .relation_coverage import core
    from .relation_context import pitch_for_step
    if not 2<=len(group)<8 or len({(r['voice'],r['onset'],r['duration']) for r in group})!=1 or any(r['onset']*480!=g.get('onset') or r['duration']*480!=g['duration'] or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or len(r['node'].findall('notations/*')) or r['node'].find('lyric') is not None for r in group):return
    if not stem_witness(g,row,stems,image):return
    s=row['space_px'];sx=sum(p[0] for p in stems[g['stem']]['xy'])/2;observed=Counter(tuple(h['pitch'][:2]) for h in g['heads'])
    for ident,count in observed.items():
        hs=[h for h in g['heads'] if tuple(h['pitch'][:2])==ident];old=[r for r in group if written(r)==ident]
        if count!=2 or len(old)!=1 or any(h.get('confidence',0)<.7 or h.get('class')!=0 or not core(h,image,s,.95) for h in hs):continue
        positions=[staff_step(track,*h['xy']) for h in hs]
        if any(abs(p['residual'])>.3 or pitch_for_step(p['nearest_step'],clef)!=ident for p in positions) or abs(hs[0]['xy'][1]-hs[1]['xy'][1])>.2*s:continue
        xs=sorted(h['xy'][0] for h in hs)
        if not .4*s<sx-xs[0]<.8*s or not .4*s<xs[1]-sx<.8*s:continue
        signs=[n for n in row['nodes'] if n['class'] in NAMES and n.get('confidence',0)>=.65 and .65<(xs[1]-n['xy'][0])/s<4.5 and abs((n['xy'][1]-hs[0]['xy'][1])/s-(-.75 if n['class']==9 else 0))<.25 and (glyph_proof or component)(n,image,s) is not None]
        if len(signs)!=2 or {n['class'] for n in signs} not in ({8,10},{9,10}) or abs(signs[0]['xy'][0]-signs[1]['xy'][0])<.6*s:continue
        values={NAMES[n['class']][0]:n for n in signs};old_value=int(old[0]['node'].findtext('pitch/alter','0'))
        if old_value not in values:continue
        # A retained different written pitch binds the physical chord/clock.
        anchors=[h for h in g['heads'] if tuple(h['pitch'][:2])!=ident and h.get('confidence',0)>=.85 and core(h,image,s,.95) and any(written(r)==tuple(h['pitch'][:2]) for r in group)]
        if not anchors:continue
        value=next(v for v in values if v!=old_value);label=NAMES[values[value]['class']][1];n=copy.deepcopy(old[0]['node'])
        for tag in ('accidental','notations','lyric','tie'):
            for a in n.findall(tag):n.remove(a)
        if n.find('chord') is None:n.insert(0,E.Element('chord'))
        set_alter(n,value,label);measure.insert(max(list(measure).index(r['node']) for r in group)+1,n)
        changes.append({'measure':number+1,'staff':old[0]['staff'],'onset':str(old[0]['onset']),'pitch':ident,'alter':value,'method':'joint_two_detected_unison_heads_two_isolated_signs'});return
