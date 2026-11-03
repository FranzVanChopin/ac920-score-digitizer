"""Experimental shared bar clocks from physical beam cells and partner notation.

All hypotheses use photographed heads, shafts, beams and recognized written
values. A missing decoded note retains an empty time slot. No notes or pitches
are invented. Search is bounded; uncertainty returns the original events.
"""
import copy
from collections import Counter,defaultdict
from fractions import Fraction as F
from math import lcm
import xml.etree.ElementTree as E
import cv2,numpy as np
from align_homr_geometry import _position
from omr_event_records import records
from repair_augmentation_dots import _head_map
from repair_grace_timing import _rebuild
from repair_tuplets import continuous_beams,printed_number,scale_divisions
from .relation_coverage import inputs,core,rest_proof
from .relation_physical_streams import extended_shaft,native_shaft
from .relation_complete_clocks import rest_ink
from .relation_meter_decode import graph_beam
from .symbol_cnn import FACTOR
from repair_unison_voices import _stem_end

VALUES={'whole':F(4),'half':F(2),'quarter':F(1),'eighth':F(1,2),'16th':F(1,4),'32nd':F(1,8)}


def written(n):
    w=VALUES.get(n.findtext('type'));dots=len(n.findall('dot'))
    if w is None or dots>2:return None
    return w*sum((F(1,2**i) for i in range(dots+1)),F())


def recover_orphan_shaft(g,row,native,ink):
    """Recover an omitted detector edge only from a unique continuous shaft.

    The caller must additionally prove the beam from this head to its partners.
    The original predicted graph is never modified.
    """
    if g.get('stem') is not None or len(g.get('heads',[]))!=1:return g
    h=g['heads'][0];s=row['space_px']/FACTOR
    if h.get('confidence',0)<.9:return g
    x,y=[(v+.5)/FACTOR-.5 for v in h['xy']];found=[]
    for up in (True,False):
        end=_stem_end(dict(x=x,y=y,staff_space=s),ink,up)
        if end is None or not 2.5*s<abs(end[1]-y)<7*s or abs(end[0]-x)>.8*s:continue
        a,b=sorted((round(y),round(end[1])));sx=round(end[0]);radius=max(1,round(.12*s))
        if min(a,sx-radius)<0 or b>=ink.shape[0] or sx+radius>=ink.shape[1]:continue
        if (native[a:b+1,sx-radius:sx+radius+1]<140).any(1).mean()<.98:continue
        found.append(up)
    return dict(g,direction='up' if found[0] else 'down',native_shaft_recovered=True) if len(found)==1 else g


def solve_grid(xs,unit,meter,anchors):
    """Unique rational grid from exact cross-staff columns, never x interpolation."""
    if not 6<=len(xs)<=64 or not unit or not meter:return None
    found=[]
    for actual,normal in ((3,2),(4,3)):
        duration=unit*normal/actual
        if len(xs)%actual or len(xs)*duration!=meter:continue
        hits=[]
        for ax,t,tolerance in anchors:
            slots=[i for i,x in enumerate(xs) if abs(x-ax)<=tolerance]
            expected=t/duration
            if not slots:
                if expected.denominator==1 and 0<=expected<len(xs):break
                continue
            if len(slots)!=1 or slots[0]*duration!=t:break
            hits.append(slots[0])
        else:
            if len(set(hits))==len(hits):
                found.append(dict(actual=actual,normal=normal,duration=duration,columns=hits))
    return found[0] if len(found)==1 else None


def fuse(root,graph,photo,geometry,homr,diagnostics=None):
    diag=diagnostics if diagnostics is not None else Counter()
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1]
    mapped=_head_map(root,homr);ms=root.findall('part/measure');events=records(root)
    cw,ch=geometry['crop_size_px'];pw,ph=homr['preprocessed_size'];ox,oy=geometry['crop_xy']
    def xy(n):
        h=mapped.get(n)
        if h is None or _position(n) is None or _position(n)[1]!=h['position']:return None
        return (((h['x']+.5)*cw/pw+ox)*FACTOR-.5,((h['y']+.5)*ch/ph+oy)*FACTOR-.5)
    def group_x(group):
        coords=[xy(r['node']) for r in group if xy(r['node']) is not None]
        if not coords or max(p[0] for p in coords)-min(p[0] for p in coords)>12:return None
        return float(np.median([p[0] for p in coords]))
    stems={s['id']:s for s in graph['stems']};jobs=[];offset=0;meter=None;division=1;contexts=[]
    for m in ms:
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            t=a.find('time')
            if t is not None:
                b,u=t.findtext('beats'),t.findtext('beat-type')
                meter=F(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u) else None
        contexts.append((division,meter))
    native=None;native_ink=None
    def ensure_native():
        nonlocal native,native_ink
        if native is None:
            native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1]
    def cells_proof(seq,row,actual,beams):
        s=row['space_px'];witness=False
        for lo in range(0,len(seq),actual):
            cell=seq[lo:lo+actual];directions={g.get('direction') for g in cell}
            if len(directions)!=1 or not directions<=set(('up','down')):return None
            up=next(iter(directions))=='up';edge=min if up else max
            hs=[dict(x=(h:=edge(g['heads'],key=lambda h:h['xy'][1]))['xy'][0],y=h['xy'][1],staff_space=s,group=g) for g in cell]
            for a,b in zip(hs,hs[1:]):
                if continuous_beams(a,b,ink,up,beams,maximum_gap=8) or graph_beam(a,b,ink,up,beams):continue
                ensure_native()
                pair=[dict(x=(h['x']+.5)/FACTOR-.5,y=(h['y']+.5)/FACTOR-.5,staff_space=s/FACTOR) for h in (a,b)]
                if not continuous_beams(*pair,native_ink,up,beams,maximum_gap=8):return None
            witness=witness or printed_number(hs,ink,actual,recover_crossings=True,flexible_beams=True,beam_count=beams)
        return witness
    history={}
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=sorted([r for r in graph['systems'] if r['system']==sy],key=lambda r:r['staff'])
        if len(rr)!=2 or rr[0]['bounds']!=rr[1]['bounds']:
            offset+=len(rr[0]['bounds'])-1;continue
        for mi,(left,right) in enumerate(zip(rr[0]['bounds'],rr[0]['bounds'][1:])):
            number=offset+mi;m=ms[number];rs=events[number];division,meter=contexts[number]
            if not meter or not 0<meter<=16 or not 0<division<=9600 or len(rs)>128 or m.get('implicit')=='yes':continue
            ns=[i for i,n in enumerate(m) if n.tag=='note']
            if not ns or any(n.tag in ('backup','forward') for n in list(m)[:ns[0]]) or any(n.tag not in ('note','backup','forward') for n in list(m)[ns[0]:ns[-1]+1]) or any(n.tag!='barline' for n in list(m)[ns[-1]+1:]):continue
            for row,partner in (rr,rr[::-1]):
                sn=row['staff'];s=row['space_px'];es=[r for r in rs if r['staff']==sn]
                seq=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x'])
                if not 6<=len(seq)<=64 or not es or len({r['voice'] for r in es})>2:continue
                if any(not 1<=len(g.get('heads',[]))<=8 or g.get('grace') or g.get('rest') or g.get('dotted') or g.get('flags') for g in seq):continue
                if any(b['x']-a['x']<.65*s for a,b in zip(seq,seq[1:])):continue
                if any(r['grace'] or r['midi'] is None or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in es):continue
                units={written(r['node']) for r in es}
                if len(units)!=1 or next(iter(units)) not in (F(1,2),F(1,4)):continue
                unit=next(iter(units));beams=1 if unit==F(1,2) else 2
                plan=solve_grid([g['x'] for g in seq],unit,meter,[])
                if plan is None:continue
                diag['ratio_candidates']+=1
                if any(h.get('class')!=0 or h.get('confidence',0)<.8 or not core(h,image,s,.85) for g in seq for h in g['heads']):diag['heads']+=1;continue
                ensure_native()
                seq=[recover_orphan_shaft(g,row,native,native_ink) for g in seq]
                if any(not g.get('native_shaft_recovered') and not extended_shaft(g,row,stems,image) and not native_shaft(g,row,stems,native,native_ink) for g in seq):diag['shaft']+=1;continue
                witness=cells_proof(seq,row,plan['actual'],beams)
                if witness is None:diag['beams']+=1;continue
                # Bind each decoded head to one physical head by segmentation
                # coordinates. Unrecognized physical heads reserve time slots.
                bindings=[];used=set()
                for r in es:
                    p=xy(r['node'])
                    choices=[(j,k) for j,g in enumerate(seq) for k,h in enumerate(g['heads'])
                             if p is not None and abs(h['xy'][0]-p[0])<.55*s and abs(h['xy'][1]-p[1])<.45*s]
                    if len(choices)!=1 or choices[0] in used:break
                    j,k=choices[0];bindings.append((r,j));used.add((j,k))
                if len(bindings)!=len(es) or len(used)<.75*sum(len(g['heads']) for g in seq):diag['binding']+=1;continue
                if any(r['duration'] not in (unit,plan['duration']) or r['node'].find('time-modification') is not None and (r['node'].findtext('time-modification/actual-notes'),r['node'].findtext('time-modification/normal-notes'))!=(str(plan['actual']),str(plan['normal'])) for r,j in bindings):diag['written_conflict']+=1;continue
                # Independent partner lane: use written values and physical
                # order. Existing inter-staff onsets may themselves be wrong.
                prs=[r for r in rs if r['staff']==partner['staff']]
                if not prs or len({r['voice'] for r in prs})!=1 or any(r['grace'] or r['node'].get('print-object')=='no' or r['node'].find('time-modification') is not None for r in prs):diag['partner_lane']+=1;continue
                groups=[]
                for r in prs:
                    if r['chord'] and groups:groups[-1].append(r)
                    else:groups.append([r])
                ws=[written(g[0]['node']) for g in groups]
                if None in ws or sum(ws,F())!=meter or any(any(written(r['node'])!=w or r['duration']!=w for r in g) for g,w in zip(groups,ws)):diag['partner_written']+=1;continue
                whole_rest=len(groups)==1 and len(prs)==1 and prs[0]['midi'] is None and ws==[meter]
                xs=[group_x(g) for g in groups]
                if whole_rest:
                    pg=[g for g in partner['groups'] if left<g['x']<right]
                    track=next(t for t in tracks if t['system']==sy and t['index']==2*(sy-1)+partner['staff']-1)
                    if len(pg)!=1 or not pg[0].get('rest') or not (rest_proof(pg[0]['rest'],track,image,s) if pg[0]['rest'].get('class')==2 else rest_ink(pg[0]['rest'],image,s)):diag['rest_ink']+=1;continue
                    xs=[seq[0]['x']]
                if any(x is None for x in xs) or any(b-a<.65*s for a,b in zip(xs,xs[1:])):diag['partner_mapping']+=1;continue
                anchors=[];cursor=F();partner_changes=[]
                for g,x,w in zip(groups,xs,ws):
                    anchors.append((x,cursor,.35*s));partner_changes.extend((r,cursor,r['duration']) for r in g);cursor+=w
                checked=solve_grid([g['x'] for g in seq],unit,meter,anchors)
                if checked is None or not anchors or anchors[0][1]!=0:diag['column_conflict']+=1;continue
                key=(sn,unit,meter,plan['actual'],plan['normal'])
                explicit=witness
                if explicit:history[key]=number
                # A single whole-note anchor does not prove interior timing.
                # It additionally requires an independently printed ratio in
                # this measure or a recent unchanged recognized ratio scope.
                recent=key in history and 0<=number-history[key]<=4
                if len(checked['columns'])<2 and not (witness or recent):diag['insufficient_columns']+=1;continue
                edits=[(r,j*plan['duration'],plan['duration']) for r,j in bindings]+partner_changes
                changed=sum(r['onset']!=onset or r['duration']!=duration for r,onset,duration in edits)
                if not changed:continue
                jobs.append(dict(number=number,staff=sn,edits=edits,target_nodes={r['node'] for r,j in bindings},plan=plan,
                    changed=changed,columns=len(checked['columns']),witness=witness,history=recent,physical_slots=len(seq),decoded_heads=len(es)))
        offset+=len(rr[0]['bounds'])-1
    # Two competing target interpretations in one measure are ambiguous.
    counts=Counter(j['number'] for j in jobs);jobs=[j for j in jobs if counts[j['number']]==1]
    if not jobs or len(jobs)>128:return []
    factor=1
    for job in jobs:
        for _,onset,duration in job['edits']:
            div=contexts[job['number']][0];factor=lcm(factor,(onset*div).denominator,(duration*div).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for job in jobs:
        number=job['number'];m=ms[number];saved=copy.deepcopy(list(m));div=contexts[number][0]*factor
        desired={r['node']:int(r['onset']*div) for r in events[number]};p=job['plan']
        for r,onset,duration in job['edits']:
            n=r['node'];desired[n]=int(onset*div);n.find('duration').text=str(int(duration*div))
            if n not in job['target_nodes']:continue
            for old in n.findall('time-modification'):n.remove(old)
            tm=E.Element('time-modification')
            for tag,value in [('actual-notes',str(p['actual'])),('normal-notes',str(p['normal'])),('normal-type',n.findtext('type'))]:E.SubElement(tm,tag).text=value
            at=next((i for i,c in enumerate(n) if c.tag in ('stem','notehead','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({k:v for k,v in job.items() if k not in ('edits','target_nodes','plan')}|dict(measure=number+1,ratio=[p['actual'],p['normal']],method='physical_tuplet_grid_and_partner_written_clock',pitch_staff_voice_cardinality_preserved=True))
    return changes
