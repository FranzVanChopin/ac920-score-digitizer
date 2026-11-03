"""Time-preserving two-voice partition from matched opposing ink stems.

Only complete, uniquely matched measures with two full lanes are eligible.
A preceding solo passage anchors the primary ID; this is not general voice
tracking, and incomplete groups, grace notes, ties and inline events abstain.
"""
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import copy,hashlib
import cv2,numpy as np
import xml.etree.ElementTree as E
from omr_event_records import records
from .symbol_cnn import FACTOR


def stem_proof(group,stems,image,space,ink_backed_heads=False):
    stem=stems.get(group.get('stem'));heads=group.get('heads',[])
    if not stem or stem.get('confidence',0)<.9 or not heads or len(heads)>8:return False
    if any(h.get('confidence',0)<(.7 if ink_backed_heads else .8) for h in heads):return False
    points=np.asarray(stem['xy']);sx=float(points[:,0].mean())
    up=group.get('direction')=='up';ys=[h['xy'][1] for h in heads]
    if group.get('direction') not in ('up','down'):return False
    if float(points[:,1].min())>min(ys)+.5*space or float(points[:,1].max())<max(ys)-.5*space:return False
    if up:
        first=max(ys);last=float(points[:,1].min())
        if first-last<1.4*space:return False
    else:
        first=min(ys);last=float(points[:,1].max())
        if last-first<1.4*space:return False
    for h in heads:
        x,y=h['xy'];dx=(sx-x)*(1 if up else -1)
        if not .15*space<abs(dx)<1.2*space:return False
        rx=max(2,round(.3*space));ry=max(2,round(.15*space));xx,yy=round(x),round(y)
        if min(xx-rx,yy-ry)<0 or xx+rx>=image.shape[1] or yy+ry>=image.shape[0]:return False
        density=(image[yy-ry:yy+ry+1,xx-rx:xx+rx+1]<140).mean()
        if h.get('class')==0 and density<.78:return False
        if h.get('confidence',0)<.8 and (h.get('class')!=0 or density<.95 or stem.get('confidence',0)<.97):return False
        if dx<0:
            # Seconds can put a filled head on the other side of its chord's
            # stem. Require an actual horizontal ink connection for that head.
            if h.get('class')!=0:return False
            left,right=sorted((xx,round(sx)))
            if ((image[yy-2:yy+3,left:right+1]<140).max(axis=0)).mean()<.85:return False
    top,bottom=round(min(first,last)),round(max(first,last));x=round(sx)
    if top<0 or bottom>=image.shape[0] or x<2 or x+2>=image.shape[1]:return False
    return bool(((image[top:bottom+1,x-2:x+3]<140).max(axis=1)).mean()>=.9)


def fuse(root,graph,photo,ink_backed_heads=False,stable_secondary_voice=False):
    if graph.get('image_sha256')!=hashlib.sha256(Path(photo).read_bytes()).hexdigest():raise ValueError('Voice partition photo mismatch')
    systems=graph['systems'];measure_rows=records(root);measures=root.find('part').findall('measure')
    from .relation_ensemble import flow_health
    health=flow_health(root);known=False;meter_known=[]
    for m in measures:
        for a in m.findall('attributes'):
            if a.find('time') is not None:known=a.findtext('time/beats') is not None and a.findtext('time/beat-type') is not None
        meter_known.append(known)
    if len(systems)>128 or any(len(r['groups'])>2048 or len(r['bounds'])>129 for r in systems):return []
    system_ids=sorted({r['system'] for r in systems})
    if sum(len(next(r for r in systems if r['system']==s)['bounds'])-1 for s in system_ids)!=len(measures):return []
    image=cv2.imread(str(photo),0)
    if image is None or image.size>16000000 or max(image.shape)>8192:return []
    image=cv2.resize(image,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA)
    stems={s['id']:s for s in graph.get('stems',[])};jobs=[];offset=0
    # A stable preceding solo passage is required to name the new two lanes.
    # This avoids independent per-measure permutations that hide discontinuity.
    prelude=defaultdict(Counter);primary={};secondary={};used=defaultdict(set);ordinary_usage=defaultdict(Counter)
    for row in measure_rows:
        for r in row:
            used[r['staff']].add(r['voice'])
            if not r['grace'] and r['midi'] is not None:ordinary_usage[r['staff']][r['voice']]+=1
    for system in system_ids:
        selected=[r for r in systems if r['system']==system]
        for mi in range(len(selected[0]['bounds'])-1):
            number=offset+mi;events=measure_rows[number];m=measures[number]
            for row in selected:
                sn=row['staff'];original=[r for r in events if r['staff']==sn]
                if not original or len(original)>128 or not meter_known[number]:continue
                voices={r['voice'] for r in original if r['midi'] is not None and not r['grace']}
                left,right=row['bounds'][mi:mi+2];groups=[g for g in row['groups'] if left<g['x']<right]
                graph_lanes={g.get('voice',1) for g in groups if g.get('heads') and not g.get('grace')}
                if len(voices)==1 and len(graph_lanes)==1:
                    prelude[sn][next(iter(voices))]+=1
                if graph_lanes!={1,2}:continue
                if sn not in primary:
                    votes=prelude[sn]
                    if len(votes)!=1 or sum(votes.values())<8:continue
                    main=next(iter(votes))
                    candidates=sorted((v for v in used[sn] if v!=main),key=lambda v:(not v.isdecimal(),int(v) if v.isdecimal() else v))
                    if stable_secondary_voice and len(candidates)>1:
                        support=ordinary_usage[sn];ranked=sorted(candidates,key=lambda v:-support[v])
                        # Numeric order carries no musical continuity. A sparse
                        # cross-staff label must not replace a well-established
                        # secondary lane. Ties in support remain unresolved.
                        if support[ranked[0]]<4 or support[ranked[0]]<=support[ranked[1]]:continue
                        candidates=ranked
                    primary[sn]=main
                    secondary[sn]=candidates[0] if candidates else next(str(v) for v in range(1,33) if str(v) not in {x for values in used.values() for x in values})
                    used[sn].add(secondary[sn])
                if any(r['grace'] or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in original):continue
                indices=[i for i,n in enumerate(m) if n.tag=='note']
                if any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
                by=defaultdict(list)
                for r in original:by[r['key']].append(r)
                mapped=[];seen=set();valid=True;ends={};starts={};directions=defaultdict(set)
                for g in groups:
                    if g.get('grace'):valid=False;break
                    lane=g.get('voice',1);onset=Fraction(g.get('onset',-1),480);duration=Fraction(g['duration'],480)
                    ends[lane]=max(ends.get(lane,0),onset+duration)
                    starts[lane]=min(starts.get(lane,onset),onset)
                    if g.get('heads'):
                        directions[lane].add(g.get('direction'))
                        if not stem_proof(g,stems,image,row['space_px'],ink_backed_heads=ink_backed_heads):valid=False;break
                        pitches=[(p['pitch'][1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p['pitch'][0]]+p['pitch'][2] for p in g['heads']]
                    elif g.get('rest'):pitches=[None]
                    else:valid=False;break
                    members=[]
                    for pitch in pitches:
                        hits=by[(sn,onset,pitch,duration)]
                        if len(hits)!=1 or id(hits[0]['node']) in seen:valid=False;break
                        members.append(hits[0]);seen.add(id(hits[0]['node']))
                    if not valid:break
                    mapped.append((lane,members))
                if not valid or len(seen)!=len(original) or directions!={1:{'up'},2:{'down'}} or set(ends)!={1,2} or ends[1]!=ends[2] or starts!={1:Fraction(0),2:Fraction(0)}:continue
                expected=Fraction(health[number]['meter'][0]*4,health[number]['meter'][1])
                if ends[1]!=expected:continue
                for lane in (1,2):
                    clock=Fraction(0)
                    for g in sorted((g for g in groups if g.get('voice',1)==lane),key=lambda g:g['x']):
                        if Fraction(g.get('onset',-1),480)!=clock:valid=False;break
                        clock+=Fraction(g['duration'],480)
                if not valid:continue
                if all(r['voice']==(primary[sn] if lane==1 else secondary[sn]) for lane,group in mapped for r in group):continue
                jobs.append((number,sn,mapped,primary[sn],secondary[sn]))
        offset+=len(selected[0]['bounds'])-1
    if len(jobs)>256:return []
    changes=[];divisions=1;divs=[]
    for m in measures:
        for a in m.findall('attributes'):divisions=int(a.findtext('divisions',str(divisions)))
        divs.append(divisions)
    grouped=defaultdict(list)
    for job in jobs:grouped[job[0]].append(job)
    from repair_grace_timing import _rebuild
    for number,plans in grouped.items():
        m=measures[number];saved=copy.deepcopy(list(m));nodes=m.findall('note');order={n:i for i,n in enumerate(nodes)}
        desired={r['node']:int(r['onset']*divs[number]) for r in measure_rows[number]}
        packs=[];chosen=set();changed=0
        for _,sn,mapped,main,other in plans:
            for lane,members in mapped:
                pack=sorted([r['node'] for r in members],key=lambda n:order[n]);packs.append(pack);chosen.update(pack)
                for i,n in enumerate(pack):
                    target=main if lane==1 else other
                    if n.findtext('voice','1')!=target:changed+=1
                    v=n.find('voice')
                    if v is None:v=E.Element('voice');n.insert(next((i for i,c in enumerate(n) if c.tag=='type'),len(n)),v)
                    v.text=target
                    for c in list(n.findall('chord')):n.remove(c)
                    if i:n.insert(0,E.Element('chord'))
        for n in nodes:
            if n not in chosen:
                if n.find('chord') is not None and packs and n.findtext('staff','1')==packs[-1][0].findtext('staff','1') and n.findtext('voice','1')==packs[-1][0].findtext('voice','1') and desired[n]==desired[packs[-1][0]]:packs[-1].append(n)
                else:packs.append([n])
        packs.sort(key=lambda pack:min(order[n] for n in pack))
        first=min(i for i,n in enumerate(m) if n.tag=='note');last=max(i for i,n in enumerate(m) if n.tag=='note')
        m[first:last+1]=[n for pack in packs for n in pack]
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'staves':[p[1] for p in plans],'relabelled_events':changed,'method':'matched_opposing_ink_stems_with_solo_prelude_identity','clocks_preserved':True})
    return changes
