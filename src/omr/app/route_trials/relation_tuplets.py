"""Bounded, image-supported triplet/quartuplet interpretation on a graph.

No score, SVG, camera transform or training labels are accepted. A printed
numeral plus continuous single beam and matching head columns are required.
Recognition and fusion run on PS CPU; no neural model or PL logic is added.
"""
import copy
from collections import defaultdict
from fractions import Fraction
import cv2
import numpy as np
import xml.etree.ElementTree as E
from math import lcm
from collections import Counter
from repair_tuplets import printed_number
from .relation_context import update


def repair(groups, image, space, bounds, ctx, changes,local_spans=False,recover_crossings=False,expanded_rhythm=False):
    if len(groups)>2048 or len(bounds)>129:return []
    ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1]
    current=copy.deepcopy(ctx);report=[]
    for mi,(left,right) in enumerate(zip(bounds,bounds[1:])):
        if len(report)>=128:break
        inside=[g for g in groups if left<g['x']<right]
        if len(inside)>64:continue
        for change in changes:
            if change['measure']==mi and change['x']<min((g['x'] for g in inside),default=right):update(current,change)
        chains=defaultdict(list)
        for g in inside:
            allowed=(1,2) if expanded_rhythm else (1,)
            if (g.get('heads') and not g.get('grace') and g.get('beams') in allowed and g.get('flags',0)==0
                and g.get('duration')==480//(2**g['beams']) and not g.get('dotted') and g.get('beam_chain') is not None):
                chains[(g['beam_chain'],g['beams'])].append(g)
        planned=[]
        for (chain,beam_count),seq in chains.items():
            seq.sort(key=lambda g:g['x'])
            # A local span is supported by its own photographed numeral;
            # ambiguous overlapping windows are rejected below.
            count=len(seq)
            if (not local_spans and count not in (3,4)) or len({g.get('direction') for g in seq})!=1 or any(len(g['heads'])>8 for g in seq):continue
            if any(g.get('heads') and g.get('beam_chain')==chain and g not in seq for g in inside):continue
            spans=[]
            windows=[(0,count)] if not local_spans else [(start,size) for size in (3,4) for start in range(count-size+1)]
            if len(windows)>128:continue
            for start,size in windows:
                window=seq[start:start+size];choices=[]
                for edge in (min,max):
                    heads=[edge(g['heads'],key=lambda h:h['xy'][1]) for g in window]
                    mapped=[{'x':h['xy'][0],'y':h['xy'][1],'staff_space':space} for h in heads]
                    if printed_number(mapped,ink,size,recover_crossings=recover_crossings,flexible_beams=expanded_rhythm,beam_count=beam_count):choices.append(mapped)
                if choices:spans.append((start,size,window))
            # One number must not authorize overlapping interpretations. Every
            # accepted span retains its own numeral and continuous-beam proof.
            occupied=[i for start,size,_ in spans for i in range(start,start+size)]
            if len(occupied)!=len(set(occupied)):continue
            for start,size,window in spans:planned.append((window,size,2 if size==3 else 3))
        if not planned or len(planned)>8:continue
        for seq,count,normal in planned:
            for i,g in enumerate(seq):
                actual=Fraction(g['duration']*normal,count)
                if actual.denominator!=1:raise ValueError('Tuplet exceeds integer tick representation')
                g['written_duration']=g['duration'];g['duration']=int(actual)
                g['tuplet']='start' if i==0 else 'stop' if i==count-1 else 'continue'
                g['tuplet_ratio']=(count,normal)
                g['tuplet_witness']={'method':'printed_number_and_counted_beams' if expanded_rhythm else 'printed_number_and_single_beam','actual':count,'normal':normal,'beam_layers':g['beams']}
            report.append({'measure':mi+1,'groups':count,'normal':normal,'x_range':[seq[0]['x'],seq[-1]['x']]})
    return report


def geometry_match(seq,other,row,staff,image):
    """Written staff positions and ink anchor duration transfer, not pitch edits."""
    from align_homr_geometry import _position
    if not staff or len(seq)!=len(other):return False
    xs,ys=np.asarray(staff['lines'][-1]['polyline']).T;space=row['space_px']
    for group,g in zip(seq,other):
        if g.get('rest'):
            if len(group)!=1 or group[0]['midi'] is not None:return False
            continue
        heads=sorted(g.get('heads',[]),key=lambda h:h['xy'][1],reverse=True)
        positions=sorted(_position(r['node'])[1] for r in group if _position(r['node']) is not None)
        if len(heads)!=len(group) or len(positions)!=len(heads) or len(heads)>8:return False
        residual=[]
        for pos,h in zip(positions,heads):
            if h.get('class')!=0 or h.get('confidence',0)<.8:return False
            x,y=h['xy'];observed=1+2*(float(np.interp(x,xs,ys))-y)/space;residual.append(observed-pos)
            if abs(observed-pos)>.65:return False
            rx=max(2,round(.3*space));ry=max(2,round(.15*space));xx,yy=round(x),round(y)
            if min(xx-rx,yy-ry)<0 or xx+rx>=image.shape[1] or yy+ry>=image.shape[0]:return False
            if (image[yy-ry:yy+ry+1,xx-rx:xx+rx+1]<140).mean()<.8:return False
        if residual and max(residual)-min(residual)>.3:return False
    return True


def opening_noise(seq,left,space,image):
    """A weak, non-filled, unlinked header proposal before a confident rest."""
    if image is None or len(seq)<3:return False
    g,following=seq[:2];hs=g.get('heads',[])
    if (len(hs)!=1 or g.get('stem') is not None or g.get('beams',0) or g.get('flags',0)
        or g.get('grace') or not 0<g['x']-left<2*space
        or not following.get('rest') or following['rest'].get('confidence',0)<.85
        or following['x']-g['x']<3*space):return False
    h=hs[0]
    if h.get('class')!=0 or h.get('confidence',0)>=.7:return False
    xx,yy=map(round,h['xy']);rx=max(2,round(.3*space));ry=max(2,round(.15*space))
    if min(xx-rx,yy-ry)<0 or xx+rx>=image.shape[1] or yy+ry>=image.shape[0]:return False
    return bool((image[yy-ry:yy+ry+1,xx-rx:xx+rx+1]<140).mean()<.7)


def fuse(root, graph,photo=None,staff_geometry=None,geometry_alignment=False,replace_conflicting_ratio=False,expanded_rhythm=False):
    """Transfer only witnessed duration changes through an identical voice.

    All ordered pitch/chord/rest groups must match uniquely, unaffected
    durations must agree, and the corrected stream must fill the printed
    meter. No grace/tie/hidden notes or legal partial streams are edited.
    """
    from omr_event_records import records
    from .relation_ensemble import flow_health
    from repair_grace_timing import _rebuild
    from repair_tuplets import scale_divisions
    measures=root.find('part').findall('measure');rows=records(root);health=flow_health(root)
    staffs=[];image=None;known_clefs=set()
    if geometry_alignment:
        import hashlib
        from pathlib import Path
        from align_homr_geometry import _xml_systems
        from .testset3_staff_resize_fast import resize_staves
        from .symbol_cnn import FACTOR
        if photo is None or staff_geometry is None:raise ValueError('Geometry timing transfer needs its photograph and staff tracks')
        if graph.get('image_sha256')!=hashlib.sha256(Path(photo).read_bytes()).hexdigest() or staff_geometry.get('image_sha256')!=graph['image_sha256']:raise ValueError('Timing transfer photo mismatch')
        if root.find('.//octave-shift') is not None:geometry_alignment=False
        else:
            raw=cv2.imread(str(photo),0)
            if raw is None or raw.size>16000000 or max(raw.shape)>8192:return []
            image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA);del raw
            staffs=resize_staves(staff_geometry['staves'],FACTOR,FACTOR);_xml_systems(root)
            first_attributes=measures[0].findall('attributes') if measures else []
            known_clefs={int(c.get('number','1')) for a in first_attributes for c in a.findall('clef')
                         if c.findtext('sign') in ('G','F','C') and c.findtext('line','0') in ('1','2','3','4','5')}
    count=sum(len(next(r for r in graph['systems'] if r['system']==s)['bounds'])-1 for s in sorted({r['system'] for r in graph['systems']}))
    if count!=len(measures):return []
    jobs=[];offset=0;divisions=1;divs=[];known_meter=False;meter_known=[]
    for m in measures:
        for a in m.findall('attributes'):
            divisions=int(a.findtext('divisions',str(divisions)))
            if a.find('time') is not None:
                known_meter=a.findtext('time/beats') is not None and a.findtext('time/beat-type') is not None
        divs.append(divisions)
        meter_known.append(known_meter)
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system]
        for mi in range(len(selected[0]['bounds'])-1):
            number=offset+mi;events=rows[number]
            if not meter_known[number] or not events or len(events)>128 or any(r['grace'] or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in events):continue
            expected=Fraction(health[number]['meter'][0]*4,health[number]['meter'][1])
            lanes=defaultdict(list)
            for r in events:
                lane=(r['staff'],r['voice'])
                if r['chord']:
                    if not lanes[lane] or r['onset']!=lanes[lane][-1][0]['onset']:lanes[lane]=[];break
                    lanes[lane][-1].append(r)
                else:lanes[lane].append([r])
            candidates=defaultdict(list)
            for row in selected:
                left,right=row['bounds'][mi:mi+2];by=defaultdict(list)
                for g in row['groups']:
                    if left<g['x']<right and not g.get('grace'):by[g.get('voice',1)].append(g)
                for seq in by.values():
                    seq.sort(key=lambda g:g['x'])
                    if (expanded_rhythm and opening_noise(seq,left,row['space_px'],image)
                        and sum(Fraction(g['duration'],480) for g in seq[1:])==expected):seq=seq[1:]
                    if len(seq)<=64 and any(g.get('tuplet_witness') for g in seq) and sum(Fraction(g['duration'],480) for g in seq)==expected:candidates[row['staff']].append((seq,row))
            for lane,seq in lanes.items():
                if not seq or len(seq)>64:continue
                t=Fraction(0);valid=True
                for group in seq:
                    if group[0]['onset']!=t or len({r['duration'] for r in group})!=1:valid=False;break
                    t+=group[0]['duration']
                if not valid or t<expected or (t==expected and not replace_conflicting_ratio):continue
                token=[Counter(r['midi'] for r in group) for group in seq]
                choices=[]
                for other,row in candidates[lane[0]]:
                    if len(seq)!=len(other):continue
                    def pitches(g):
                        if g.get('rest'):return Counter([None])
                        return Counter((h['pitch'][1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[h['pitch'][0]]+h['pitch'][2] for h in g['heads'])
                    if token!=[pitches(g) for g in other]:
                        # Stored staff rows have no ordinal field; derive it
                        # within the image system when necessary.
                        same=[s for s in staffs if s['system']==row['system']]
                        staff=same[lane[0]-1] if len(same)>=lane[0] else None
                        if not geometry_alignment or lane[0] not in known_clefs or not geometry_match(seq,other,row,staff,image):continue
                    contradicted=False;compatible=True
                    for group,g in zip(seq,other):
                        written=Fraction(g.get('written_duration',g['duration']),480)
                        if not g.get('tuplet_witness'):
                            if any(r['duration']!=written for r in group):compatible=False
                            continue
                        for r in group:
                            n=r['node'];tm=n.find('time-modification')
                            kind={Fraction(1,2):'eighth',Fraction(1,4):'16th'}.get(written)
                            if kind is None or (kind!='eighth' and not expanded_rhythm) or n.findtext('type')!=kind:compatible=False;break
                            if tm is None:
                                if r['duration']!=written:compatible=False;break
                            elif replace_conflicting_ratio:
                                actual=int(tm.findtext('actual-notes','0'));normal=int(tm.findtext('normal-notes','0'))
                                if actual not in (3,4) or normal!=(2 if actual==3 else 3) or r['duration']!=written*Fraction(normal,actual):compatible=False;break
                                contradicted=contradicted or (actual,normal)!=tuple(g['tuplet_ratio'])
                            else:compatible=False;break
                    if not compatible or (t==expected and not contradicted):continue
                    choices.append(other)
                if len(choices)==1:jobs.append((number,lane,seq,choices[0]))
        offset+=len(selected[0]['bounds'])-1
    usage=Counter((number,id(other)) for number,lane,seq,other in jobs)
    jobs=[job for job in jobs if usage[(job[0],id(job[3]))]==1]
    if not jobs or len(jobs)>256:return []
    factor=1
    for number,lane,seq,other in jobs:
        for g in other:factor=lcm(factor,Fraction(g['duration']*divs[number],480).denominator)
    if factor>12:return []
    scale_divisions(root,factor);grouped=defaultdict(list)
    for job in jobs:grouped[job[0]].append(job)
    changes=[]
    for number,plans in grouped.items():
        m=measures[number];saved=copy.deepcopy(list(m));div=divs[number]*factor
        desired={r['node']:int(r['onset']*div) for r in rows[number]}
        for _,lane,seq,other in plans:
            cursor=0
            for group,g in zip(seq,other):
                duration=int(Fraction(g['duration']*div,480))
                for r in group:
                    n=r['node'];desired[n]=cursor;n.find('duration').text=str(duration)
                    if g.get('tuplet_witness'):
                        for old in list(n.findall('time-modification')):n.remove(old)
                        for notation in n.findall('notations'):
                            for old in list(notation.findall('tuplet')):notation.remove(old)
                        tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text=str(g['tuplet_ratio'][0]);E.SubElement(tm,'normal-notes').text=str(g['tuplet_ratio'][1]);E.SubElement(tm,'normal-type').text=n.findtext('type')
                        pos=next((i for i,c in enumerate(n) if c.tag in ('stem','notehead','staff','beam','notations')),len(n));n.insert(pos,tm)
                        if g['tuplet'] in ('start','stop'):
                            notation=n.find('notations')
                            if notation is None:notation=E.SubElement(n,'notations')
                            E.SubElement(notation,'tuplet',type=g['tuplet'],number='1',bracket='no')
                cursor+=duration
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'lanes':[list(p[1]) for p in plans],'method':'image_tuplet_unique_ordered_voice','changed_groups':sum(sum(bool(g.get('tuplet_witness')) for g in p[3]) for p in plans)})
    return changes
