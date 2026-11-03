"""Decode shared opening heads and complete printed triplet voices together.

Two real opposing stems, every printed numeral/beam chain, all physical heads
and recognized meter must agree. Preserves the sustained opening voice while
restoring the shorter voice instead of shifting a monophonic cursor blindly.
"""
import copy
from fractions import Fraction
from math import lcm
import xml.etree.ElementTree as E
import cv2
from omr_event_records import records
from repair_tuplets import printed_number,scale_divisions,continuous_beams
from repair_unison_voices import _stem_end
from repair_grace_timing import _rebuild
from .relation_coverage import inputs,context_rows,core
from .relation_measure_joint import align_columns,written
from collections import Counter
from .relation_tuplets import geometry_match

def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];rows=records(root);ms=root.findall('part/measure');div=1;meter=None;divs=[];meters=[];jobs=[];offset=0
    for m in ms:
        for a in m.findall('attributes'):
            div=int(a.findtext('divisions',str(div)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        divs.append(div);meters.append(meter)
    used={r['voice'] for row in rows for r in row};secondary=next((str(i) for i in range(1,33) if str(i) not in used),None)
    if secondary is None:return []
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in selected:
            if row['staff']>len(staffs):continue
            track=staffs[row['staff']-1];s=row['space_px']
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;m=ms[number];es=[r for r in rows[number] if r['staff']==row['staff']];length=meters[number]
                if not length or m.get('implicit')=='yes' or len(es)>24 or len({r['voice'] for r in es})!=1 or not 6<=len(es)<=24 or len(es)%3:continue
                if any(r['midi'] is None or r['grace'] or r['chord'] or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].findall('dot') for r in es):continue
                es=sorted(es,key=lambda r:r['onset'])
                short=es[0]['duration']==Fraction(1,3) and es[0]['node'].findtext('type')=='eighth'
                if es[0]['onset']!=0 or (not short and (es[0]['node'].findtext('type')!='quarter' or es[0]['duration'] not in (Fraction(1),Fraction(2,3)))) or length!=Fraction(len(es),3):continue
                if any(r['duration']!=Fraction(1,3) or r['node'].findtext('type')!='eighth' or r['node'].findtext('time-modification/actual-notes')!='3' or r['node'].findtext('time-modification/normal-notes')!='2' for r in es[1:]):continue
                seq=sorted([g for g in row['groups'] if left<g['x']<right and g.get('heads') and not g.get('grace')],key=lambda g:g['x']);pairs=align_columns([Counter([written(r)]) for r in es],[Counter(tuple(h['pitch'][:2]) for h in g['heads']) for g in seq])
                if len(pairs)!=len(es) or [i for i,j in pairs]!=list(range(len(es))):continue
                bound=[seq[j] for i,j in pairs]
                if any(len(g['heads'])!=1 for g in bound) or not geometry_match([[r] for r in es],bound,row,track,image):continue
                heads=[dict(x=g['heads'][0]['xy'][0],y=g['heads'][0]['xy'][1],staff_space=s) for g in bound]
                first=heads[0];up=_stem_end(first,ink,True);down=_stem_end(first,ink,False)
                if up is None or down is None or up[1]>first['y']-2.5*s or down[1]<first['y']+2.5*s:continue
                # Upper quarter has an independent recognized stem, not a
                # coincidental line through a head or a second beam voice.
                upper=[stem for stem in graph['stems'] if stem.get('confidence',0)>=.95 and abs(sum(p[0] for p in stem['xy'])/2-up[0])<.2*s and min(p[1] for p in stem['xy'])<first['y']-2*s and abs(max(p[1] for p in stem['xy'])-first['y'])<.5*s]
                if len(upper)!=1 or continuous_beams(heads[0],heads[1],ink,True,1):continue
                if not all(printed_number(heads[i:i+3],ink,3,recover_crossings=True,flexible_beams=True) for i in range(0,len(heads),3)):continue
                indices=[i for i,n in enumerate(m) if n.tag=='note']
                if any(n.tag in ('backup','forward') for n in list(m)[:indices[0]]):continue
                if any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
                candidates={r['voice'] for rs in rows for r in rs if r['staff']==row['staff'] and r['voice']!=es[0]['voice'] and r['duration']==Fraction(1,3)}
                other_voices={r['voice'] for rs in rows for r in rs if r['staff']==row['staff'] and r['voice']!=es[0]['voice']}
                if len(other_voices)==1:lane=next(iter(other_voices))
                elif not other_voices:lane=secondary
                else:continue
                jobs.append((number,es,lane,short))
        offset+=len(selected[0]['bounds'])-1
    if not jobs or len(jobs)>128:return []
    factor=1
    for number,es,voice,short in jobs:factor=lcm(factor,(Fraction(1,3)*divs[number]).denominator)
    if factor>12:return []
    scale_divisions(root,factor);changes=[]
    for number,es,voice,short in jobs:
        m=ms[number];saved=copy.deepcopy(list(m));division=divs[number]*factor;desired={r['node']:int(r['onset']*division) for r in rows[number]};original=es[0]['node']
        first=copy.deepcopy(original) if short else original;clone=original if short else copy.deepcopy(original)
        for n in first.findall('time-modification')+first.findall('beam')+first.findall('chord'):first.remove(n)
        if short:
            for n in first.findall('notations'):first.remove(n)
            first.find('voice').text=voice;m.insert(list(m).index(original),first);desired[first]=0
        else:
            for tag in ('time-modification','beam','notations','chord'):
                for n in clone.findall(tag):clone.remove(n)
            m.insert(list(m).index(first)+1,clone);desired[clone]=0
        for notations in first.findall('notations'):
            for n in notations.findall('tuplet'):notations.remove(n)
        lower_voice=original.findtext('voice','1') if short else voice
        first.find('duration').text=str(division)
        first.find('type').text='quarter'
        stem=first.find('stem')
        if stem is not None:stem.text='up'
        clone.find('duration').text=str(division//3);clone.find('type').text='eighth';clone.find('voice').text=lower_voice
        stem=clone.find('stem')
        if stem is not None:stem.text='down'
        lower=[clone]+[r['node'] for r in es[1:]]
        for i,n in enumerate(lower):
            desired[n]=i*division//3;n.find('voice').text=lower_voice;n.find('duration').text=str(division//3)
            for tag in ('time-modification','beam'):
                for old in n.findall(tag):n.remove(old)
            tm=E.Element('time-modification');E.SubElement(tm,'actual-notes').text='3';E.SubElement(tm,'normal-notes').text='2';E.SubElement(tm,'normal-type').text='eighth';at=next((j for j,c in enumerate(n) if c.tag in ('stem','staff','beam','notations','lyric')),len(n));n.insert(at,tm)
            beam=E.Element('beam',number='1');beam.text=('begin','continue','end')[i%3];at=next((j for j,c in enumerate(n) if c.tag in ('notations','lyric')),len(n));n.insert(at,beam)
            notations=n.find('notations')
            if notations is not None:
                for old in notations.findall('tuplet'):notations.remove(old)
            if i%3 in (0,2):
                if notations is None:notations=E.SubElement(n,'notations')
                E.SubElement(notations,'tuplet',type='start' if i%3==0 else 'stop',number=str(i//3+1),bracket='no')
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':number+1,'staff':es[0]['staff'],'method':'joint_shared_quarter_and_complete_printed_triplet_voice','triplet_groups':len(es)//3,'restored_shared_head_events':1,'existing_quarter_retained':not short,'changed_lower_events':0 if short else len(es)-1,'new_voice':voice})
    return changes
