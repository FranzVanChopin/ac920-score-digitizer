"""Remove false triplet prefix membership using exclusive printed scope.

No references: one conserved printed-meter lane, identical ordered spelling,
actual staff/head ink and an unambiguous numeral over exactly three heads.
"""
import copy
from collections import Counter,defaultdict
from fractions import Fraction as F
from math import lcm
import xml.etree.ElementTree as E
import cv2
from omr_event_records import records
from repair_tuplets import printed_number,scale_divisions,continuous_beams
from repair_grace_timing import _rebuild
from .relation_coverage import inputs,ordered,spelling,detected,safe_measure,context_rows
from .relation_tuplets import geometry_match
from .relation_ensemble import flow_health


def scope_proof(seq,start,row,ink):
    window=seq[start:start+3];chain=window[0].get('beam_chain');space=row['space_px']
    if len(window)!=3 or chain is None or any(g.get('beams')!=2 or g.get('beam_chain')!=chain or g.get('tuplet_ratio') not in ([3,2],(3,2)) or g.get('duration')!=80 or g.get('written_duration')!=120 or g.get('tuplet')!=('start','continue','stop')[i] or not g.get('tuplet_witness') for i,g in enumerate(window)):return False
    def number(gs):
        for edge in (min,max):
            hs=[edge(g['heads'],key=lambda h:h['xy'][1]) for g in gs]
            if printed_number([dict(x=h['xy'][0],y=h['xy'][1],staff_space=space) for h in hs],ink,3,recover_crossings=True,flexible_beams=True,beam_count=2):return True
        return False
    if not number(window):return False
    # A neighboring shifted interpretation must not share the same numeral.
    for j in (start-2,start-1,start+1,start+2):
        if j<0 or j+3>len(seq):continue
        shifted=seq[j:j+3]
        if all(g.get('heads') and g.get('beam_chain')==chain and g.get('beams')==2 for g in shifted) and number(shifted):return False
    return True


def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];events=records(root);ms=root.findall('part/measure');health=flow_health(root);ctx=context_rows(root);jobs=[];offset=0;div=1;divs=[];meter=False;known=[]
    for m in ms:
        for a in m.findall('attributes'):
            div=int(a.findtext('divisions',str(div)))
            if a.find('time') is not None:meter=a.findtext('time/beats') is not None and a.findtext('time/beat-type') is not None
        divs.append(div);known.append(meter)
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for mi in range(len(selected[0]['bounds'])-1):
            number=offset+mi;es=events[number];m=ms[number]
            if not known[number] or not safe_measure(es) or m.get('implicit')=='yes':continue
            seen=False;inline=False
            for child in m:
                if child.tag=='note':seen=True
                elif seen and child.tag=='attributes':inline=True
            if inline:continue
            expected=F(health[number]['meter'][0]*4,health[number]['meter'][1])
            for row in selected:
                sn=row['staff'];rs=[r for r in es if r['staff']==sn]
                if not rs or len({r['voice'] for r in rs})!=1 or ctx[number][1].get(sn) is None or len(staffs)<sn:continue
                original=ordered(rs);left,right=row['bounds'][mi:mi+2];seq=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x'])
                if not original or len(original)!=len(seq) or len(seq)>64 or any(g.get('grace') for g in seq) or [spelling(g) for g in original]!=[detected(g) for g in seq]:continue
                if len({g.get('voice',1) for g in seq})!=1 or sum(g['duration'] for g in seq)!=expected*480:continue
                cursor=F(0);valid=True
                for group in original:
                    if any(r['onset']!=cursor or r['duration']!=group[0]['duration'] for r in group):valid=False;break
                    cursor+=group[0]['duration']
                if not valid or cursor>=expected:continue
                false=[];starts=[]
                for i,(group,g) in enumerate(zip(original,seq)):
                    wanted=F(g['duration'],480)
                    if any(r['duration']!=wanted for r in group):
                        if g.get('tuplet_witness') or g.get('beams')!=2 or g.get('duration')!=120 or g.get('flags') or g.get('dotted') or not g.get('heads'):valid=False;break
                        if any(r['node'].findtext('type')!='16th' or r['duration']!=F(1,6) or r['node'].findtext('time-modification/actual-notes')!='3' or r['node'].findtext('time-modification/normal-notes')!='2' for r in group):valid=False;break
                        false.append(i)
                    if g.get('tuplet')=='start':starts.append(i)
                if not valid or not 1<=len(false)<=2 or len(starts)!=1:continue
                start=starts[0]
                if false!=list(range(start-len(false),start)) or any(seq[i].get('beam_chain')!=seq[start].get('beam_chain') for i in false):continue
                if not geometry_match(original,seq,row,staffs[sn-1],image) or not scope_proof(seq,start,row,ink):continue
                def prefix_beam(i):
                    a,b=seq[i],seq[i+1];space=row['space_px']
                    if a.get('direction') not in ('up','down') or a.get('direction')!=b.get('direction'):return False
                    ha=max(a['heads'],key=lambda h:h['xy'][1]) if a['direction']=='up' else min(a['heads'],key=lambda h:h['xy'][1]);hb=max(b['heads'],key=lambda h:h['xy'][1]) if b['direction']=='up' else min(b['heads'],key=lambda h:h['xy'][1])
                    return continuous_beams(dict(x=ha['xy'][0],y=ha['xy'][1],staff_space=space),dict(x=hb['xy'][0],y=hb['xy'][1],staff_space=space),ink,a['direction']=='up',2)
                if not all(prefix_beam(i) for i in false):continue
                jobs.append((number,sn,original,seq,false,start))
        offset+=len(selected[0]['bounds'])-1
    if not jobs or len(jobs)>128:return []
    factor=1
    for number,sn,original,seq,false,start in jobs:
        for g in seq:factor=lcm(factor,F(divs[number]*g['duration'],480).denominator)
    if factor>12:return []
    scale_divisions(root,factor);jobs_by=defaultdict(list)
    for job in jobs:jobs_by[job[0]].append(job)
    changes=[]
    for number,plans in jobs_by.items():
        m=ms[number];saved=copy.deepcopy(list(m));div=divs[number]*factor;desired={r['node']:int(r['onset']*div) for r in events[number]};changed_staffs={p[1] for p in plans};before_partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in events[number] if r['staff'] not in changed_staffs)
        for _,sn,original,seq,false,start in plans:
            cursor=0
            for i,(group,g) in enumerate(zip(original,seq)):
                duration=int(F(g['duration']*div,480))
                for r in group:
                    n=r['node'];desired[n]=cursor;n.find('duration').text=str(duration)
                    if i in false:
                        for tm in list(n.findall('time-modification')):n.remove(tm)
                    if i in false or start<=i<start+3:
                        for notation in n.findall('notations'):
                            for tuplet in list(notation.findall('tuplet')):notation.remove(tuplet)
                    if i in (start,start+2):
                        notation=n.find('notations')
                        if notation is None:notation=E.SubElement(n,'notations')
                        E.SubElement(notation,'tuplet',type='start' if i==start else 'stop',number='1',bracket='no')
                cursor+=duration
        if not _rebuild(m,desired):m[:]=saved;continue
        after=records(root)[number];partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in after if r['staff'] not in changed_staffs)
        if partner!=before_partner:m[:]=saved;continue
        changes.append({'measure':number+1,'staffs':sorted(changed_staffs),'false_prefix_groups_removed':sum(len(p[4]) for p in plans),'method':'exclusive_printed_triplet_scope_and_conserved_meter'})
    return changes
