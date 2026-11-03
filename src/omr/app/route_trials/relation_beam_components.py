"""Bounded physical beam components; no evaluator or source-score access."""
from collections import defaultdict
import numpy as np
from omr_event_records import records
from align_homr_geometry import _position,_head_staff,_xml_systems
from repair_ledger_chords import _align
from omr_bar_geometry import measure_systems

def temporal_map(root,homr):
    _xml_systems(root);systems=measure_systems(root);all_events=records(root);offset=0;result={}
    if len(systems)!=len(homr['systems']):return result
    for ms,sy in zip(systems,homr['systems']):
        if len(sy['staffs'])!=1:offset+=len(ms);continue
        staff=sy['staffs'][0]
        for sn in (1,2):
            ordered=[]
            for rs in all_events[offset:offset+len(ms)]:
                rs=[r for r in rs if r['staff']==sn and _position(r['node']) is not None]
                ordered.extend(r['node'] for r in sorted(rs,key=lambda r:(r['onset'],_position(r['node'])[1])))
            heads=sorted([h for h in staff['notes'] if _head_staff(staff,h)==sn],key=lambda h:h['x']);groups=[]
            for h in heads:
                if not groups or h['x']-groups[-1][-1]['x']>3:groups.append([])
                groups[-1].append(h)
            result.update(_align(ordered,[h for g in groups for h in sorted(g,key=lambda h:h['position'])]))
        offset+=len(ms)
    return result


def stem_end(head,ink,up):
    s,x,y=head['staff_space'],head['x'],head['y'];ends=[]
    nominal=round(x+(.6*s if up else -.6*s));sign=-1 if up else 1
    columns=set(range(nominal-2,nominal+3));stem=head.get('stem')
    if stem and (stem[0]-x)*(1 if up else -1)>0 and abs(stem[0]-x)<.85*s:
        columns.update(range(round(stem[0])-1,round(stem[0])+2))
    for col in sorted(columns):
        start=round(y+sign*.4*s);last=start;blanks=0
        for distance in range(round(16*s)):
            row=start+sign*distance
            if not 0<=row<ink.shape[0] or not 0<=col<ink.shape[1]:break
            if ink[row,col]:last=row;blanks=0
            else:
                blanks+=1
                if blanks>=2:break
        if 1.5*s<=abs(last-start)<15*s:ends.append((abs(last-start),col,last))
    return max(ends)[1:] if ends else None


def _component_stem_end(head,ink,up,cache):
    key=id(head),up
    if key not in cache:cache[key]=stem_end(head,ink,up)
    return cache[key]


def beam(a,b,ink,up,other_up=None,*,_stem_cache=None):
    if other_up is None:other_up=up
    s=a['staff_space']
    if _stem_cache is None:
        ea=stem_end(a,ink,up);eb=stem_end(b,ink,other_up)
    else:
        ea=_component_stem_end(a,ink,up,_stem_cache);eb=_component_stem_end(b,ink,other_up,_stem_cache)
    if ea is None or eb is None:return False
    xa,ya=ea;xb,yb=eb
    if not .7*s<xb-xa<8*s or abs(ya-yb)>2*s:return False
    for f in (.125,.25,.375,.5,.625,.75,.875):
        x=round(xa+(xb-xa)*f);y=ya+(yb-ya)*f;top=max(0,round(y-1.3*s));bottom=min(ink.shape[0],round(y+1.3*s)+1)
        bits=ink[top:bottom,x]>0;edges=np.diff(np.r_[False,bits,False].astype('int8'))
        spans=[(top+l,top+r) for l,r in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if .2*s<=r-l<=.8*s]
        spans=[(l,r) for l,r in spans if abs((l+r-1)/2-y)<=.55*s or (up==other_up and ((up and y<=(l+r-1)/2<=y+1.1*s) or (not up and y-1.1*s<=(l+r-1)/2<=y)))]
        if len(spans)!=1 or abs(sum(spans[0])/2-y)>.55*s:return False
    return True


def components(rs,mapped,ink):
    candidates=[]
    for i,r in enumerate(rs):
        n=r['node'];h=mapped.get(n)
        if h is None or _position(n) is None or h['position']!=_position(n)[1] or r['grace'] or r['midi'] is None:continue
        candidates.append((i,h))
    edges=[];adj=defaultdict(set);stem_cache={}
    candidates.sort(key=lambda p:p[1]['x'])
    for j,(i,h) in enumerate(candidates):
        for k,g in candidates[j+1:]:
            if g['x']-h['x']>=8*h['staff_space']:break
            if abs(g['x']-h['x'])<.7*h['staff_space']:continue
            for up,other in ((True,True),(False,False),(True,False),(False,True)):
                if beam(h,g,ink,up,other,_stem_cache=stem_cache):adj[i].add(k);adj[k].add(i);edges.append((i,k,up,other))
    remaining=set(adj);groups=[]
    while remaining:
        group={next(iter(remaining))};todo=list(group)
        while todo:
            for other in adj[todo.pop()]-group:group.add(other);todo.append(other)
        remaining-=group
        groups.append(sorted(group,key=lambda i:mapped[rs[i]['node']]['x']))
    return groups,edges


