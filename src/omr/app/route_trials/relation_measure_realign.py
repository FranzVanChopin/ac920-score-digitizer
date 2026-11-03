"""Repair local bar ownership when complete primary runs are visually shifted.

Matching is against photograph-derived bars, without reference scores. Two
distinct consecutive measures establish a shift; anchored duplicates are also
eligible. Correct matching notes keep their original XML and notation.
"""
from collections import Counter,defaultdict
from math import lcm
import copy,xml.etree.ElementTree as E
from omr_event_records import records
from repair_tuplets import scale_divisions
from repair_grace_timing import _rebuild
from .relation_coverage import inputs,rest_proof,core
from .relation_measure_joint import context_rows,head_witness,stem_witness
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
import numpy as np
import cv2


def hollow_on_line(h,row,track,clef,image):
    if h.get('class')!=1 or h.get('confidence',0)<.9:return False
    s=row['space_px'];x,y=map(round,h['xy']);p=staff_step(track,*h['xy'])
    if abs(p['residual'])>.35 or pitch_for_step(p['nearest_step'],clef)!=tuple(h['pitch'][:2]):return False
    rx=max(2,round(.8*s));ry=max(2,round(.7*s))
    if min(x-rx,y-ry)<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
    crop=image[y-ry:y+ry+1,x-rx:x+rx+1]
    # A staff line splits the enclosed white hole into two components.
    # Both remain enclosed by actual ink; no removal/inpainting is needed.
    _,labels,stats,centres=cv2.connectedComponentsWithStats((crop>=140).astype('uint8'),8)
    area=0
    for (cx,cy),(lx,ly,w,h,n) in zip(centres[1:],stats[1:]):
        if lx==0 or ly==0 or lx+w>=crop.shape[1] or ly+h>=crop.shape[0]:continue
        if n>=6 and abs(cx-rx)<.45*s and abs(cy-ry)<.45*s:area+=n
    return .15*s*s<=area<=.85*s*s


def traced_stem(g,row,stems,image):
    if stem_witness(g,row,stems,image):return True
    stem=stems.get(g.get('stem'));s=row['space_px']
    if not stem or stem.get('confidence',0)<.9 or g.get('direction') not in ('up','down'):return False
    pts=np.asarray(stem['xy']);pts=pts[np.argsort(pts[:,1])];lo,hi=pts[:,1]
    if hi-lo<2*s or lo<0 or hi>=image.shape[0]:return False
    # Detector boxes include a few blank pixels past the stem tip. Require
    # a continuous interior shaft of at least two staff spaces instead.
    margin=round(.3*s);yy=np.arange(round(lo)+margin,round(hi)-margin+1)
    if len(yy)<2*s:return False
    xx=np.rint(np.interp(yy,pts[:,1],pts[:,0])).astype(int)
    radius=min(round(.25*s),max(2,round(stem.get('bbox',[0,0,1])[2]/2)+1))
    if xx.min()<radius or xx.max()+radius>=image.shape[1]:return False
    if np.any(image[yy[:,None],xx[:,None]+np.arange(-radius,radius+1)]<140,axis=1).mean()<.98:return False
    for h in g['heads']:
        x,y=map(round,h['xy']);sx=round(np.interp(y,pts[:,1],pts[:,0]));l,r=sorted((x,sx));dy=round(.45*s)
        if abs(x-sx)>.95*s or not lo-.5*s<y<hi+.5*s or min(l,y-dy)<0 or r>=image.shape[1] or y+dy>=image.shape[0]:return False
        # The head joins at its rim, which may lie off the head centre line.
        if (image[y-dy:y+dy+1,l:r+1]<140).max(0).mean()<.98:return False
    return True


def local_staff_head(h,row,track,clef,image):
    """Recheck geometry using four/five independently visible local lines."""
    if h.get('class')!=0 or h.get('confidence',0)<.85 or not core(h,image,row['space_px'],.95):return False
    x,y=h['xy'];s=row['space_px'];ds=np.concatenate((np.arange(-4*s,-s),np.arange(s,4*s)));xs=np.rint(x+ds).astype(int)
    if xs.min()<0 or xs.max()>=image.shape[1]:return False
    points=[]
    for i,line in enumerate(track['lines']):
        ly=np.interp(xs,*np.array(line['polyline']).T);offsets=np.arange(-round(.35*s),round(.35*s)+1);yy=np.rint(ly[None,:]+offsets[:,None]).astype(int)
        if yy.min()<0 or yy.max()>=image.shape[0]:return False
        values=(image[yy,xs[None,:]]<140).mean(1);j=values.argmax()
        if values[j]>=.65:points.append((i,float(np.interp(x,*np.array(line['polyline']).T)+offsets[j])))
    if len(points)<4:return False
    indices,ys=np.array(points).T;space,top=np.polyfit(indices,ys,1)
    if abs(space-s)>.15*s or np.max(np.abs(ys-(top+indices*space)))>.2*s:return False
    step=8-2*(y-top)/space;nearest=round(step)
    return abs(step-nearest)<=.3 and pitch_for_step(nearest,clef)==tuple(h['pitch'][:2])


def signature(rows):return Counter(r['key'] for r in rows if not r['grace'] and r['midi'] is not None and r['node'].get('print-object')!='no')


def shifted_indices(base,visual):
    n=len(base);result=set()
    for shift in (-2,-1,1,2):
        run=[]
        for i in range(n+1):
            good=i<n and 0<=i+shift<n and bool(base[i]) and base[i]==visual[i+shift] and base[i]!=visual[i]
            if good:run.append(i);continue
            if len(run)>=2 and len({tuple(sorted(base[j].items())) for j in run})>=2:result.update(run)
            run=[]
    confirmed_run=set(result)
    for i in range(n):
        for j in (i-1,i+1):
            if 0<=j<n and not visual[i] and base[i] and base[i]==base[j]==visual[j]:result.add(i)
            k=j+(j-i)
            if not (0<=j<n and 0<=k<n and base[i] and base[i]==base[j]==visual[j] and base[k]==visual[k]):continue
            old_pitches={(key[0],key[2]) for key in base[i]};new_pitches={(key[0],key[2]) for key in visual[i]}
            if new_pitches and old_pitches.isdisjoint(new_pitches) and any(abs(i-r)<=2 for r in confirmed_run):result.add(i)
    return sorted(result)


def fuse(root,other,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(other.findall('part'))!=1 or len(graph['systems'])>128 or len(graph['stems'])>4096:return [],{}
    image,tracks=loaded;before=records(root);rr=records(other);ms=root.findall('part/measure')
    if len(before)!=len(rr):return [],{'abstained':'measure_count'}
    physical=[];row_maps=[];offset=0;stems={s['id']:s for s in graph['stems']};ctx=context_rows(root)
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy];staffs=[s for s in tracks if s['system']==sy]
        for mi in range(len(rows[0]['bounds'])-1):
            notes=Counter();mapped=[]
            for row in rows:
                sn=row['staff'];left,right=row['bounds'][mi:mi+2]
                if sn>len(staffs):continue
                gs=[g for g in row['groups'] if left<g['x']<right and not g.get('grace')]
                mapped.append((row,staffs[sn-1],gs))
                for g in gs:
                    for h in g.get('heads',[]):
                        p=h['pitch'];midi=(p[1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p[0]]+p[2]
                        from fractions import Fraction
                        notes[(sn,Fraction(g.get('onset',-1),480),midi,Fraction(g['duration'],480))]+=1
            physical.append(notes);row_maps.append(mapped)
        offset+=len(rows[0]['bounds'])-1
    selected=shifted_indices([signature(r) for r in before],physical);jobs=[];div=1;divs=[];rejections=[]
    for m in ms:
        for a in m.findall('attributes'):div=int(a.findtext('divisions',str(div)))
        divs.append(div)
    votes=defaultdict(Counter);labels=defaultdict(set)
    for a,b in zip(before,rr):
        for r in a:
            if r['midi'] is not None and not r['grace']:labels[r['staff']].add(r['voice'])
        for r in b:
            matches=[c for c in a if c['key']==r['key'] and not c['grace']]
            if len(matches)==1:votes[(r['staff'],r['voice'])][matches[0]['voice']]+=1
    mapping={}
    for key,c in votes.items():
        ranked=c.most_common()
        if ranked and (len(ranked)==1 or ranked[0][1]>ranked[1][1]):mapping[key]=ranked[0][0]
    for number in selected:
        if signature(rr[number])!=physical[number] or len(rr[number])>128 or any(r['grace'] or r['node'].get('print-object')=='no' for r in before[number]):rejections.append({'measure':number+1,'reason':'candidate_or_grace','candidate_graph_equal':signature(rr[number])==physical[number]});continue
        valid=True
        for row,track,gs in row_maps[number]:
            sn=row['staff']
            if Counter(r['key'] for r in before[number] if r['staff']==sn and r['midi'] is not None)==Counter(r['key'] for r in rr[number] if r['staff']==sn and r['midi'] is not None):continue
            clef=ctx[number][1].get(row['staff'])
            if not clef:valid=False;break
            for g in gs:
                if g.get('heads'):
                    stem_ok=bool(traced_stem(g,row,stems,image));heads_ok=[bool(head_witness(h,row,track,clef,image) or hollow_on_line(h,row,track,clef,image) or local_staff_head(h,row,track,clef,image)) for h in g['heads']]
                    if not stem_ok or not all(heads_ok):rejections.append({'measure':number+1,'reason':'head_or_stem','stem':stem_ok,'heads':heads_ok});valid=False;break
                elif g.get('rest') and not rest_proof(g['rest'],track,image,row['space_px']):valid=False;break
            if not valid:break
        if not valid:continue
        new=[r for r in rr[number] if r['midi'] is not None and not r['grace']]
        # When an anchored duplicated bar is physically empty, its certified
        # full-bar rest is missing from the primary XML. Existing rests stay.
        if not new:
            new += [r for r in rr[number] if r['midi'] is None and not r['grace'] and r['onset']==0 and any(c['staff']!=r['staff'] and c['midi'] is None and c['onset']==0 and c['duration']==r['duration'] for c in before[number]) and any(c['staff']==r['staff'] and c['midi'] is not None for c in before[number]) and not any(c['staff']==r['staff'] and c['midi'] is None for c in before[number]) and any(row['staff']==r['staff'] and len(gs)==1 and gs[0].get('rest',{}).get('class')==2 for row,track,gs in row_maps[number])]
        if any((r['staff'],r['voice']) not in mapping and len(labels[r['staff']])!=1 for r in new):continue
        jobs.append((number,new))
    factor=1
    for number,rs in jobs:
        for r in rs:factor=lcm(factor,(r['duration']*divs[number]).denominator,(r['onset']*divs[number]).denominator)
    if not jobs or len(jobs)>128 or factor>12:return [],{'shift_candidates':selected,'qualified':len(jobs),'rejections':rejections}
    scale_divisions(root,factor);changes=[]
    for number,rs in jobs:
        m=ms[number];saved=copy.deepcopy(list(m));division=divs[number]*factor;old=before[number];desired={};used=set();pitched=[]
        for r in rs:
            hits=[c for c in old if c['key']==r['key'] and not c['grace'] and id(c['node']) not in used]
            if hits:n=hits[0]['node'];used.add(id(n));voice=n.findtext('voice','1')
            else:
                n=copy.deepcopy(r['node']);voice=mapping.get((r['staff'],r['voice']),next(iter(labels[r['staff']]),'1'))
                for tag in ('tie','notations','lyric'):
                    for a in n.findall(tag):n.remove(a)
                for key in ('default-x','default-y','relative-x','relative-y'):n.attrib.pop(key,None)
                n.find('voice').text=voice
            for c in n.findall('chord'):n.remove(c)
            n.find('duration').text=str(int(r['duration']*division));desired[n]=int(r['onset']*division);pitched.append(n)
        kept=[r['node'] for r in old if r['midi'] is None or r['grace']]
        for r in old:
            if r['node'] in kept:desired[r['node']]=int(r['onset']*division)
        indices=[i for i,n in enumerate(m) if n.tag=='note']
        if not indices or any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
        m[indices[0]:indices[-1]+1]=kept+pitched
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'method':'photograph_bar_ownership_distinct_shifted_run_or_anchored_duplicate','old_pitched_notes':sum(signature(old).values()),'new_pitched_notes':sum(r['midi'] is not None for r in rs),'added_certified_full_bar_rests':sum(r['midi'] is None for r in rs),'old_rest_grace_clocks_preserved':True})
    return changes,{'shift_candidates':selected,'qualified':len(jobs),'rejections':rejections}
