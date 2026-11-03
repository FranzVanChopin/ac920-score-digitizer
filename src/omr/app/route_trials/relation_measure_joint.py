"""Bounded photograph-only partial-column and accidental/chord joint decoding.

Weighted sequence alignment permits missing chord members. Only correspondences
forced across competing paths can change XML. Glyph assignment is exclusive;
uncertain states abstain. Existing timing/voices are preserved in this phase.
"""
from collections import Counter,defaultdict
import copy
import xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from .relation_coverage import inputs,context_rows as basic_context_rows,core,hollow_core
from .relation_key_state import calibrated_signature,padded_empty_strip,natural_component
from .relation_accidentals import NAMES
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step

MAX_COLUMNS=64
MAX_STATES=64

def context_rows(root):
    """Independent context view extends the old G/F-only repair interface.

    Alto/tenor C clefs use the same physical diatonic staff calculation. No
    legacy decoder or its stored context is changed by this extension.
    """
    result=basic_context_rows(root);clefs={}
    for i,m in enumerate(root.findall('part/measure')):
        for a in m.findall('attributes'):
            for c in a.findall('clef'):
                sn=int(c.get('number','1'));sign=c.findtext('sign');line=int(c.findtext('line','0'));octave=int(c.findtext('clef-octave-change','0'))
                clefs[sn]={'sign':sign,'line':line,'octave':0} if octave==0 and ((sign,line) in (('G',2),('F',4)) or sign=='C' and 1<=line<=5) else None
        result[i]=(result[i][0],copy.deepcopy(clefs))
    return result

def anchored_pitch(group,g,row,track,clef,stems,image,values,changes,number):
    """One differing written pitch, one retained physical chord anchor."""
    if not 2<=len(group)==len(g['heads'])<=8 or len({(r['voice'],r['onset'],r['duration']) for r in group})!=1:return
    if any(r['onset']*480!=g.get('onset') or r['duration']*480!=g['duration'] for r in group):return
    old=Counter(written(r) for r in group);new=Counter(tuple(h['pitch'][:2]) for h in g['heads']);missing=old-new;extra=new-old
    if sum(missing.values())!=1 or sum(extra.values())!=1 or not old&new or any(v>1 for v in new.values()):return
    origin=next(iter(missing));target=next(iter(extra));r=next(r for r in group if written(r)==origin);h=next(h for h in g['heads'] if tuple(h['pitch'][:2])==target);anchors=[a for a in g['heads'] if tuple(a['pitch'][:2]) in old&new and head_witness(a,row,track,clef,image)]
    n=r['node']
    anchors=[a for a in anchors if any(written(r)==tuple(a['pitch'][:2]) and int(r['node'].findtext('pitch/alter','0'))==a['pitch'][2] for r in group)]
    if not anchors or not head_witness(h,row,track,clef,image) or not stem_witness(g,row,stems,image) or n.find('accidental') is not None or n.find('tie') is not None or n.find('notations/tied') is not None or n.get('print-object')=='no':return
    delta=(target[1]-origin[1])*7+'CDEFGAB'.index(target[0])-'CDEFGAB'.index(origin[0])
    # Written/played octave shifts are resolved by the established ottava
    # path. Large or same-letter octave substitutions cannot use this proof.
    if not 2<=abs(delta)<7:return
    # Positive evidence alone may be a different chord's head. Preserve an
    # XML pitch whenever either stem-side location contains its photographed
    # filled/hollow head. A contradicting empty original slot is required.
    bottom=pitch_for_step(0,clef);old_step=(origin[1]-bottom[1])*7+'CDEFGAB'.index(origin[0])-'CDEFGAB'.index(bottom[0]);position=staff_step(track,*h['xy']);old_y=position['bottom_line_y']-old_step*position['local_space_px']/2;s=row['space_px'];sx=sum(p[0] for p in stems[g['stem']]['xy'])/2
    if old_y<2*s or old_y+2*s>=image.shape[0]:return
    for x in (h['xy'][0],sx-.55*s,sx+.55*s):
        old_head={'xy':[x,old_y]}
        if core(old_head,image,s,.65) or hollow_core(old_head,image,s):return
    value=values.get(h['id'])
    if value is None:return
    n.find('pitch/step').text=target[0];n.find('pitch/octave').text=str(target[1]);set_alter(n,*value)
    changes.append({'measure':number+1,'staff':r['staff'],'onset':str(r['onset']),'old_written_pitch':origin,'new_written_pitch':target,'new_alter':value[0],'xy':h['xy'],'method':'joint_one_pitch_substitution_with_retained_ink_chord_anchor'})

def written(r):return (r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave')))

def align_columns(left,right,margin=.2):
    """Mandatory partial-pitch column pairs in all near-optimal DP paths."""
    n,m=len(left),len(right)
    if max(n,m)>MAX_COLUMNS:return []
    def score(a,b):
        common=sum((a&b).values())
        return 2*common-.7*sum((a-b).values())-.4*sum((b-a).values()) if common else -1000
    costs=[[score(a,b) for b in right] for a in left];pre=[[0.]*(m+1) for _ in range(n+1)];post=[[0.]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):pre[i+1][j+1]=max(pre[i][j+1],pre[i+1][j],pre[i][j]+costs[i][j])
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):post[i][j]=max(post[i+1][j],post[i][j+1],post[i+1][j+1]+costs[i][j])
    optimum=pre[n][m];result=[]
    for i in range(n):
        if any(pre[i][j]+post[i+1][j]>=optimum-margin for j in range(m+1)):continue
        possible=[j for j in range(m) if costs[i][j]>0 and pre[i][j]+costs[i][j]+post[i+1][j+1]>=optimum-margin]
        if len(possible)==1:result.append((i,possible[0]))
    return result

def glyph_witness(n,row,image):
    if n.get('confidence',0)<(.9 if n['class']!=10 else .85):return False
    s=row['space_px'];x,y=n['xy'];l,r=round(x-1.3*s),round(x+1.3*s);t,b=round(y-2*s),round(y+2*s)
    if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0]:return False
    crop=(image[t:b,l:r]<140).astype('uint8')
    if n['class']==10:return natural_component(crop,(x-l,y-t),s) is not None
    # Independent upright ink supports the detected flat/sharp; the complete
    # classifier and assignment, rather than harmony, determine the class.
    horizontal=cv2.morphologyEx(crop,cv2.MORPH_OPEN,np.ones((1,max(3,round(2*s))),np.uint8));clean=cv2.subtract(crop,horizontal)
    vertical=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((max(3,round(.7*s)),1),np.uint8))
    return cv2.countNonZero(vertical)>s and cv2.countNonZero(clean)>2*s

def joint_accidentals(seq,row,image):
    """Exclusive glyph-to-head assignment with explicit abstention options.

    Each local cluster has <=8 heads and <=8 glyphs. The best 64 assignments
    are retained; only assignments common to all near-best states are used.
    """
    heads=[h for g in seq for h in g.get('heads',[])];s=row['space_px'];result={}
    accs=[n for n in row['nodes'] if n['class'] in NAMES and glyph_witness(n,row,image)]
    for g in seq:
        hs=g.get('heads',[])
        if not hs or len(hs)>8:continue
        options=[]
        for n in accs:
            edges=[]
            for i,h in enumerate(hs):
                dx=(h['xy'][0]-n['xy'][0])/s;dy=(n['xy'][1]-h['xy'][1])/s;target=-.75 if n['class']==9 else 0
                if not .65<dx<3.5 or abs(dy-target)>(.45 if n['class']==9 else .3):continue
                # A preceding head at this printed position owns its sign.
                if any(a['xy'][0]<h['xy'][0]-.5*s and a['xy'][0]>n['xy'][0]+.65*s and abs(a['xy'][1]-h['xy'][1])<.3*s for a in heads):continue
                edges.append((i,abs(dy-target)+.1*dx))
            if edges:options.append((n,edges))
        if not options or len(options)>8:continue
        states=[(0.,0,())]
        for n,edges in options:
            nxt=[]
            for cost,mask,links in states:
                nxt.append((cost+.8,mask,links))
                for i,v in edges:
                    if not mask&(1<<i):nxt.append((cost+v,mask|(1<<i),links+((i,n['id']),)))
            # Pruning can manufacture apparent uniqueness. A cluster that
            # exceeds the search cap abstains instead of trusting beam rank.
            if len(nxt)>MAX_STATES:states=[];break
            states=sorted(nxt,key=lambda v:v[0])
        if not states:continue
        near=[v[2] for v in states if v[0]<=states[0][0]+.15];nodes={n['id']:n for n,_ in options}
        for i,h in enumerate(hs):
            values=[dict(v).get(i) for v in near]
            if values and values[0] is not None and len(set(values))==1:result[h['id']]=nodes[values[0]]
    # A glyph must not be consumed by neighboring groups as well.
    counts=Counter(n['id'] for n in result.values())
    return {hid:n for hid,n in result.items() if counts[n['id']]==1}

def set_alter(node,value,label=None):
    p=node.find('pitch');a=p.find('alter')
    if a is None:a=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,a)
    a.text=str(value)
    if label:
        a=node.find('accidental')
        if a is None:
            a=E.Element('accidental');at=list(node).index(node.find('type'))+1 if node.find('type') is not None else len(node);node.insert(at,a)
        a.text=label

def head_witness(h,row,track,clef,image):
    p=staff_step(track,*h['xy']);identity=pitch_for_step(p['nearest_step'],clef)
    physical=core(h,image,row['space_px'],.95) if h.get('class')==0 else hollow_core(h,image,row['space_px']) if h.get('class')==1 else False
    return h.get('confidence',0)>=.85 and abs(p['residual'])<=.3 and identity==tuple(h['pitch'][:2]) and physical

def empty_joint(h,g,row,track,stems,image,ink,all_heads,attached):
    if padded_empty_strip(h,row,track,g,stems,image,ink,all_heads,attached):return True
    s=row['space_px'];x,y=h['xy'];l,r=round(x-2.3*s),round(x-.65*s);t,b=round(y-1.3*s),round(y+1.3*s);margin=round(2*s)
    previous=[a for a in all_heads if a['x']<x-3 and abs(a['y']-y)<1.7*s]
    if previous:l=max(l,round(max(a['x']+.6*s for a in previous)+1))
    if min(l-margin,t-round(2*s))<0 or r+margin>=image.shape[1] or b+round(2*s)>=image.shape[0] or r-l<.75*s:return False
    if any(l-.5*s<a['x']<r and abs(a['y']-y)<1.7*s for a in all_heads):return False
    if any(n['class'] in NAMES and l-.2*s<n['xy'][0]<r+.2*s and abs((n['xy'][1]-y)/s-(-.75 if n['class']==9 else 0))<.4 for n in row['nodes']):return False
    x0=l-margin;y0=t-round(2*s);crop=ink[y0:b+round(2*s),x0:r+margin].copy();removed=False
    for other in g['heads']:
        acc=attached.get(other['id'])
        if not acc or acc['class']!=10 or abs(other['xy'][1]-y)<.4*s or other.get('confidence',0)<.88 or abs(staff_step(track,*other['xy'])['residual'])>.25:continue
        ax,ay=acc['xy'];component=natural_component(crop,(ax-x0,ay-y0),s)
        if component is not None:crop[cv2.dilate(component.astype('uint8'),np.ones((3,3),np.uint8))>0]=0;removed=True
    if not removed:return False
    horizontal=cv2.morphologyEx(crop,cv2.MORPH_OPEN,np.ones((1,round(1.2*s)),np.uint8));remaining=cv2.subtract(crop,horizontal)[t-y0:b-y0,l-x0:r-x0]
    return cv2.countNonZero(remaining)<2*s and cv2.countNonZero(cv2.morphologyEx(remaining,cv2.MORPH_OPEN,np.ones((max(3,round(.45*s)),1),np.uint8)))==0

def stem_witness(g,row,stems,image):
    stem=stems.get(g.get('stem'));s=row['space_px']
    if not stem or stem.get('confidence',0)<.97 or g.get('direction') not in ('up','down'):return False
    points=np.array(stem['xy']);sx=float(points[:,0].mean());lo,hi=sorted(points[:,1]);xx=round(sx)
    if hi-lo<2*s or lo<0 or hi>=image.shape[0] or xx<2 or xx+2>=image.shape[1]:return False
    if (image[round(lo):round(hi)+1,xx-2:xx+3]<140).max(1).mean()<.95:return False
    for h in g['heads']:
        x,y=map(round,h['xy']);left,right=sorted((x,xx))
        if abs(x-sx)>.95*s or not lo-.5*s<y<hi+.5*s or min(left,y-2)<0 or right>=image.shape[1] or y+2>=image.shape[0]:return False
        if (image[y-2:y+3,left:right+1]<140).max(0).mean()<.95:return False
    return True

def chromatic_sibling(group,g,row,track,clef,stems,image,changes,number,m):
    """Two separate printed signs and two opposite stem-side ink heads.

    This restores chromatic unison multiplicity, which a single heat-map
    maximum loses. Courtesy cancellation alone has only one physical head
    and cannot satisfy the opposite-side core plus neighboring-second proof.
    """
    if not 2<=len(group)<=8 or len({(r['voice'],r['duration']) for r in group})!=1 or any(r['node'].find('tie') is not None or len(r['node'].findall('notations/*')) or r['node'].find('lyric') is not None for r in group):return
    stem=stems.get(g.get('stem'));s=row['space_px']
    if not stem or stem.get('confidence',0)<.97 or g.get('direction') not in ('up','down'):return
    xy=np.array(stem['xy']);sx=float(xy[:,0].mean());lo,hi=sorted(xy[:,1]);xx=round(sx)
    if lo<0 or hi>=image.shape[0] or xx<2 or xx+2>=image.shape[1] or (image[round(lo):round(hi)+1,xx-2:xx+3]<140).max(1).mean()<.95:return
    for h in g['heads']:
        ident=tuple(h['pitch'][:2]);matches=[r for r in group if written(r)==ident]
        if len(matches)!=1 or h.get('confidence',0)<.9 or not head_witness(h,row,track,clef,image):continue
        hx,hy=h['xy'];mirror=2*sx-hx
        if not .4*s<abs(hx-sx)<.8*s or not lo-.5*s<hy<hi+.5*s:continue
        other=dict(h,xy=[mirror,hy])
        if not core(other,image,s,.95):continue
        # The second near this stem proves a displaced chord context, and is
        # distinct from both same-position lobes.
        if not any(a['id']!=h['id'] and abs(a['xy'][0]-mirror)<.3*s and .35*s<abs(a['xy'][1]-hy)<.7*s and (a.get('confidence',0)>=.8 or a.get('displaced_second_ink_proof')) and core(a,image,s,.95) for a in g['heads']):continue
        signs=[]
        for n in row['nodes']:
            if n['class'] not in NAMES or not glyph_witness(n,row,image):continue
            dx=(max(hx,mirror)-n['xy'][0])/s;dy=(n['xy'][1]-hy)/s;target=-.75 if n['class']==9 else 0
            if .65<dx<4.5 and abs(dy-target)<.25:signs.append(n)
        if len(signs)!=2 or {n['class'] for n in signs} not in ({8,10},{9,10}) or abs(signs[0]['xy'][0]-signs[1]['xy'][0])<.7*s:continue
        old=int(matches[0]['node'].findtext('pitch/alter','0'));values={NAMES[n['class']][0]:n for n in signs}
        if old not in values:continue
        value=next(v for v in values if v!=old);label=NAMES[values[value]['class']][1]
        n=copy.deepcopy(matches[0]['node'])
        for tag in ('accidental','notations','lyric','tie'):
            for a in n.findall(tag):n.remove(a)
        if n.find('chord') is None:n.insert(0,E.Element('chord'))
        set_alter(n,value,label);index=max(list(m).index(r['node']) for r in group)+1;m.insert(index,n)
        changes.append({'measure':number+1,'staff':matches[0]['staff'],'onset':str(matches[0]['onset']),'pitch':ident,'alter':value,'xy':[mirror,hy],'method':'joint_chromatic_unison_two_signs_opposite_ink_lobes'});return

def boundary_arc(head,image,space,above):
    """Small incoming tie hook beside a system-opening head, not its stem."""
    x,y=head['xy'];l,r=round(x-1.7*space),round(x-.75*space);t,b=(round(y-1.4*space),round(y-.2*space)) if above else (round(y+.2*space),round(y+1.4*space))
    if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0]:return False
    raw=(image[t:b,l:r]<140).astype('uint8');horizontal=cv2.morphologyEx(raw,cv2.MORPH_OPEN,np.ones((1,max(3,round(1.4*space))),np.uint8));clean=cv2.subtract(raw,horizontal);count,lab,stats,_=cv2.connectedComponentsWithStats(clean)
    if count>32:return False
    matches=[]
    for i,(xx,yy,w,h,area) in enumerate(stats[1:],1):
        if not .4*space<=w<=1.1*space or not .12*space<=h<=.75*space or w<h or not .15<=area/(w*h)<=.8:continue
        patch=(lab[yy:yy+h,xx:xx+w]==i);profile=[]
        for c in range(w):
            ys=np.flatnonzero(patch[:,c])
            if len(ys):profile.append((c,float(np.median(ys))))
        if len(profile)<.75*w:continue
        edges=[v for c,v in profile if c<w/3 or c>=2*w/3];middle=[v for c,v in profile if w/3<=c<2*w/3]
        if not edges or not middle:continue
        bend=np.mean(edges)-np.mean(middle)
        if (bend if above else -bend)>=.7:matches.append(i)
    return len(matches)==1

def boundary_chord(group,g,row,track,clef,stems,image,previous,changes,number):
    if len(group)!=2 or len(g['heads'])!=2 or len({(r['voice'],r['onset'],r['duration']) for r in group})!=1 or any(r['onset']!=0 or r['duration']*480!=g['duration'] or r['node'].get('print-object')=='no' or r['node'].find('accidental') is not None for r in group):return
    prior=[r for r in previous if r['staff']==row['staff'] and not r['grace'] and r['midi'] is not None]
    if not prior:return
    last=max(r['onset'] for r in prior);prior=[r for r in prior if r['onset']==last]
    if len(prior)!=2 or len({r['voice'] for r in prior})!=1 or Counter(written(r) for r in prior)!=Counter(tuple(h['pitch'][:2]) for h in g['heads']):return
    if not all(head_witness(h,row,track,clef,image) and h['confidence']>=.9 for h in g['heads']) or not stem_witness(g,row,stems,image):return
    hs=sorted(g['heads'],key=lambda h:h['xy'][1])
    if not boundary_arc(hs[0],image,row['space_px'],True) or not boundary_arc(hs[1],image,row['space_px'],False):return
    old={written(r):r for r in group};new={written(r):r for r in prior};common=set(old)&set(new)
    if len(old)!=2 or len(new)!=2 or not common:return
    mapping=[(old[k],new[k]) for k in common];remaining_old=[v for k,v in old.items() if k not in common];remaining_new=[v for k,v in new.items() if k not in common]
    if len(remaining_old)>1:return
    mapping+=list(zip(remaining_old,remaining_new))
    for r,target in mapping:
        a,b=written(r),written(target);delta=(b[1]-a[1])*7+'CDEFGAB'.index(b[0])-'CDEFGAB'.index(a[0])
        if abs(delta)>=7:return
    for r,target in mapping:
        a,b=written(r),written(target);old_alter=int(r['node'].findtext('pitch/alter','0'));wanted=int(target['node'].findtext('pitch/alter','0'))
        if a==b and old_alter==wanted:continue
        r['node'].find('pitch/step').text=b[0];r['node'].find('pitch/octave').text=str(b[1]);set_alter(r['node'],wanted)
        changes.append({'measure':number+1,'staff':r['staff'],'onset':'0','old_written_pitch':a,'new_written_pitch':b,'old_alter':old_alter,'new_alter':wanted,'method':'joint_system_boundary_two_incoming_arcs_and_previous_chord','tie_exported':False})

def fuse(root,graph,photo,geometry,chords=True,diagnostics=None,expanded_candidates=False):
    attach=joint_accidentals;negative=empty_joint
    if expanded_candidates:
        from .relation_joint_candidates import associations,empty
        attach=associations;negative=empty
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    if any(len(g.get('heads',[]))>8 for r in graph['systems'] for g in r['groups']):return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];rows=records(root);ctx=context_rows(root);ms=root.findall('part/measure');stems={s['id']:s for s in graph.get('stems',[])};changes=[];offset=0
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in selected:
            sn=row['staff']
            if sn>len(staffs):continue
            track=staffs[sn-1];all_heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for g in row['groups'] for h in g.get('heads',[])]
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;m=ms[number];key,clefs=ctx[number];clef=clefs.get(sn);es=[r for r in rows[number] if r['staff']==sn and not r['grace'] and r['midi'] is not None]
                if not clef or key is None or len(es)>128:continue
                note_seen=False;inline=False
                for n in m:
                    if n.tag=='note':note_seen=True
                    if n.tag=='attributes' and note_seen:inline=True
                if inline:continue
                old=[sorted([r for r in es if r['onset']==t],key=lambda r:written(r)) for t in sorted({r['onset'] for r in es})]
                seq=sorted([g for g in row['groups'] if left<g['x']<right and g.get('heads') and not g.get('grace')],key=lambda g:g['x'])
                if not old or not seq or max(len(old),len(seq))>64:continue
                pairs=align_columns([Counter(written(r) for r in g) for g in old],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq]);by_graph={hi:ni for ni,hi in pairs};attached=attach(seq,row,image)
                verified_key=key==0 or calibrated_signature(row,track,clef,key)
                state={(letter,octave):(int(key>0)-int(key<0) if letter in ('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)] else 0) for letter in 'CDEFGAB' for octave in range(10)} if verified_key else {}
                for hi,g in enumerate(seq):
                    group=old[by_graph[hi]] if hi in by_graph else [];nodes=defaultdict(list)
                    if mi==0 and hi==0 and number>0 and group and len(changes)<126:boundary_chord(group,g,row,track,clef,stems,image,rows[number-1],changes,number)
                    for r in group:nodes[written(r)].append(r)
                    before_chromatic=len(changes)
                    if chords and group and len(changes)<128 and len(m.findall('note'))<128:chromatic_sibling(group,g,row,track,clef,stems,image,changes,number,m)
                    if expanded_candidates and chords and group and len(changes)==before_chromatic and len(changes)<128 and len(m.findall('note'))<128:
                        from .relation_joint_candidates import chromatic_heads
                        chromatic_heads(group,g,row,track,clef,stems,image,changes,number,m)
                    values={};updates=defaultdict(set)
                    for h in g['heads']:
                        if len(changes)>=128:return changes
                        ident=tuple(h['pitch'][:2]);acc=attached.get(h['id']);absent=negative(h,g,row,track,stems,image,ink,all_heads,attached);value=None;label=None
                        position=staff_step(track,*h['xy'])
                        if acc:
                            ap=staff_step(track,acc['xy'][0],acc['xy'][1]+(.75*row['space_px'] if acc['class']==9 else 0))
                            if ap['nearest_step']!=position['nearest_step'] or abs(ap['residual'])>.35:acc=None
                        prefix=acc and mi==0 and acc['xy'][0]<seq[0]['x']-.9*row['space_px']
                        # A certified opening signature excludes local glyphs
                        # of its own class. An independently witnessed natural
                        # (or opposite sign) beside the first physical chord is
                        # a local accidental, rather than part of that prefix.
                        local_opening=expanded_candidates and verified_key and (key==0 or acc and acc['class']!=(9 if key<0 else 8))
                        if acc and (not prefix or local_opening):value,label=NAMES[acc['class']]
                        elif absent:value=state.get(ident)
                        if abs(position['residual'])>.3:
                            value=None
                            for step in (position['nearest_step']-1,position['nearest_step']+1):updates[pitch_for_step(step,clef)].add(None)
                        updates[ident].add(value)
                        if value is not None:values[h['id']]=(value,label)
                        matches=nodes.get(ident,[])
                        if len(matches)!=1 or sum(tuple(a['pitch'][:2])==ident for a in g['heads'])!=1 or value is None or not head_witness(h,row,track,clef,image):continue
                        r=matches[0];n=r['node'];old_alter=int(n.findtext('pitch/alter','0'))
                        if n.get('print-object')=='no' or n.find('tie') is not None or n.find('notations/tied') is not None or n.findtext('accidental') in ('double-sharp','flat-flat'):continue
                        if label is None and n.find('accidental') is not None and old_alter!=value:updates[ident].add(None);continue
                        if old_alter!=value and r['duration']*480==g['duration'] and r['onset']*480==g.get('onset'):
                            set_alter(n,value,label);changes.append({'measure':number+1,'staff':sn,'onset':str(r['onset']),'pitch':ident,'old_alter':old_alter,'new_alter':value,'xy':h['xy'],'method':'joint_partial_column_exclusive_accidental' if label else 'joint_observed_pitch_state'})
                    for ident,v in updates.items():state[ident]=next(iter(v)) if len(v)==1 else None
                    if group and len(changes)<128:anchored_pitch(group,g,row,track,clef,stems,image,values,changes,number)
                    if not chords or not group or len(changes)>=128 or len(m.findall('note'))>=128 or not 2<=len(g['heads'])<=8:continue
                    original=Counter(written(r) for r in group);observed=Counter(tuple(h['pitch'][:2]) for h in g['heads']);extra=observed-original
                    if original-observed or not 1<=sum(extra.values())<=2 or any(v>1 for v in observed.values()) or len({(r['voice'],r['duration']) for r in group})!=1:continue
                    if any(r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or len(r['node'].findall('notations/*')) or r['node'].find('lyric') is not None for r in group):continue
                    if not stem_witness(g,row,stems,image):continue
                    extras=[h for h in g['heads'] if tuple(h['pitch'][:2]) in extra]
                    anchors=[h for h in g['heads'] if tuple(h['pitch'][:2]) in original and head_witness(h,row,track,clef,image)]
                    if not anchors or any(h['id'] not in values or not head_witness(h,row,track,clef,image) for h in extras):continue
                    # Actual XML chord siblings, not simultaneous different lanes.
                    index=max(list(m).index(r['node']) for r in group)+1
                    for h in extras:
                        ident=tuple(h['pitch'][:2]);value,label=values[h['id']];n=copy.deepcopy(group[0]['node'])
                        for tag in ('pitch','accidental','notations','lyric','tie'):
                            for a in n.findall(tag):n.remove(a)
                        p=E.Element('pitch');E.SubElement(p,'step').text=ident[0];E.SubElement(p,'octave').text=str(ident[1]);n.insert(0,p)
                        if n.find('chord') is None:n.insert(0,E.Element('chord'))
                        set_alter(n,value,label);m.insert(index,n);index+=1
                        changes.append({'measure':number+1,'staff':sn,'onset':str(group[0]['onset']),'pitch':ident,'alter':value,'xy':h['xy'],'method':'joint_partial_chord_shared_stem_ink_head'})
                if diagnostics is not None:diagnostics.append({'measure':number+1,'staff':sn,'xml_columns':len(old),'graph_columns':len(seq),'forced_pairs':pairs,'exclusive_glyphs':len(attached)})
        offset+=len(selected[0]['bounds'])-1
    return changes
