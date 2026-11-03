"""Recover completely overflowed two-staff music through full physical lanes.

All old pitched events must extend past the meter. Their pitch/duration
multiset is retained; already admissible rests and every grace stay unchanged.
Independent ink, full voices and coincident melody columns certify the new
grouping. This is a narrow bounded hypothesis, not arbitrary bar replacement.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import copy
import xml.etree.ElementTree as E
import cv2
import numpy as np
from omr_event_records import records
from repair_grace_timing import _rebuild
from repair_tuplets import continuous_beams
from .relation_coverage import inputs,core
from .relation_measure_joint import context_rows,head_witness
from .relation_measure_realign import traced_stem
from .relation_joint_events import hollow
from .relation_complete_clocks import rest_ink
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from .relation_meter_decode import graph_beam
from .symbol_cnn import FACTOR


def plans(root):
    result=[];division=1;meter=None
    for i,(m,rs) in enumerate(zip(root.findall('part/measure'),records(root))):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        pitched=[r for r in rs if r['midi'] is not None and not r['grace']]
        if not meter or not 0<meter<=16 or not 0<division<=9600 or m.get('implicit')=='yes' or not 2<=len(pitched)<=128 or len(rs)>128 or {r['staff'] for r in pitched}!={1,2}:continue
        if not all(r['onset']>=0 and r['onset']+r['duration']>meter for r in pitched):continue
        if any(r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in rs):continue
        indices=[j for j,n in enumerate(m) if n.tag=='note']
        if not indices or any(n.tag in ('backup','forward') for n in list(m)[:indices[0]]) or any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
        result.append((i,division,meter,indices,rs))
    return result


def midi(h):
    p=h['pitch'];return (p[1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p[0]]+p[2]


def fuse(root,graph,photo,geometry):
    jobs=plans(root)
    if not jobs or len(jobs)>128:return []
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];ms=root.findall('part/measure');ctx=context_rows(root);stems={s['id']:s for s in graph['stems']};locations={};offset=0;usage=defaultdict(Counter);page_events=records(root)
    for rs in page_events:
        for r in rs:
            if r['midi'] is not None and not r['grace']:usage[r['staff']][r['voice']]+=1
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        for j in range(len(rows[0]['bounds'])-1):locations[offset+j]=(rows,ts,j)
        offset+=len(rows[0]['bounds'])-1
    changes=[];native_ink=None
    def beam_proof(hs,up,row,track):
        nonlocal native_ink
        if continuous_beams(hs[0],hs[1],ink,up,1,maximum_gap=8) or graph_beam(hs[0],hs[1],ink,up,1):return True
        # The same seven-point physical beam proof on original pixels avoids
        # aliasing a thin photographed band away in the smaller work image.
        if native_ink is None:native_ink=cv2.threshold(cv2.imread(str(photo),0),140,255,cv2.THRESH_BINARY_INV)[1]
        native=[]
        for h in hs:
            g=copy.deepcopy(h['group'])
            for b in g.get('beam_relations',[]):
                b['stem_x']=(b['stem_x']+.5)/FACTOR-.5;b['y_range']=[(y+.5)/FACTOR-.5 for y in b['y_range']]
            native.append(dict(x=(h['x']+.5)/FACTOR-.5,y=(h['y']+.5)/FACTOR-.5,staff_space=h['staff_space']/FACTOR,group=g))
        if continuous_beams(native[0],native[1],native_ink,up,1,maximum_gap=8) or graph_beam(native[0],native[1],native_ink,up,1):return True
        # Some fonts use a beam thinner than .2 staff spaces. Search a bounded
        # native window at both certified shaft tips, excluding every physical
        # staff line. Nine samples and a unique continuous band are required.
        s=native[0]['staff_space'];ends=[]
        if not 8<=s<=64:return False
        for h in hs:
            stem=stems.get(h['group'].get('stem'))
            if not stem:return False
            xy=np.asarray(stem['xy']);sx=float(xy[:,0].mean());sy=float((min if up else max)(xy[:,1]));ends.append(((sx+.5)/FACTOR-.5,(sy+.5)/FACTOR-.5))
        (xa,ya),(xb,yb)=ends
        if not .7*s<xb-xa<8*s or abs(ya-yb)>2*s:return False
        witnesses=[]
        for shift in range(-round(.75*s),round(.75*s)+1):
            hits=[]
            for f in (.02,.125,.25,.375,.5,.625,.75,.875,.98):
                x=round(xa+(xb-xa)*f);y=ya+(yb-ya)*f+shift;top=max(0,round(y-1.2*s));bottom=min(native_ink.shape[0],round(y+1.2*s))
                if x<0 or x>=native_ink.shape[1]:return False
                bits=native_ink[top:bottom,x]>0;d=np.diff(np.r_[False,bits,False].astype('int8'));spans=[]
                for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)):
                    centre=top+(a+b-1)/2
                    if not .14*s<=b-a<=.8*s or abs(centre-y)>.2*s:continue
                    legacy_x=(x+.5)*FACTOR-.5
                    lines=[(float(np.interp(legacy_x,*np.asarray(line['polyline']).T))+.5)/FACTOR-.5 for line in track['lines']]
                    if any(abs(centre-ly)<.25*s for ly in lines):continue
                    spans.append(centre)
                hits.append(len(spans)==1)
            if all(hits):witnesses.append(shift)
        return bool(witnesses and max(witnesses)-min(witnesses)<.5*s)
    for number,division,meter,indices,old in jobs:
        rows,ts,mi=locations[number]
        if len(rows)!=2 or len(ts)!=2 or rows[0]['bounds']!=rows[1]['bounds']:continue
        proposal=[];maps={};moving={};valid=True
        for row,t in zip(rows,ts):
            sn=row['staff'];s=row['space_px'];left,right=row['bounds'][mi:mi+2];gs=[g for g in row['groups'] if left<g['x']<right and not g.get('grace')]
            lanes=defaultdict(list)
            if not 1<=len(gs)<=64:valid=False;break
            for g in gs:lanes[g.get('voice',1)].append(g)
            if not 1<=len(lanes)<=2:valid=False;break
            ranks=usage[sn].most_common()
            if not ranks or ranks[0][1]<24 or len(ranks)>1 and ranks[0][1]==ranks[1][1]:valid=False;break
            main=ranks[0][0];others=[(v,n) for v,n in ranks if v!=main]
            normal=[v for v,seq in lanes.items() if len(seq)>=3]
            if len(normal)!=1:valid=False;break
            maps[sn]={normal[0]:main};moving[sn]=lanes[normal[0]]
            if len(lanes)==2:
                if not others or len(others)>1 and (others[0][1]<4 or others[0][1]==others[1][1]):valid=False;break
                # A unique secondary label can be sparse. Require a separate
                # already recognized full-bar half-note chord in that label,
                # rather than accepting it from numeric order alone.
                if others[0][1]<4:
                    witnessed=False
                    for prior in page_events:
                        hs=[r for r in prior if r['staff']==sn and r['voice']==others[0][0] and r['midi'] is not None and not r['grace']]
                        if len(hs)>=2 and all(r['onset']==0 and r['duration']==meter and r['node'].findtext('type')=='half' for r in hs):witnessed=True;break
                    if not witnessed:valid=False;break
                held=next(v for v in lanes if v!=normal[0]);seq=lanes[held]
                pitches=[midi(h) for g in lanes[normal[0]] for h in g.get('heads',[])]
                if len(seq)!=1 or seq[0].get('onset')!=0 or seq[0]['duration']!=meter*480 or not 2<=len(seq[0].get('heads',[]))<=8 or not pitches or max(midi(h) for h in seq[0]['heads'])>=min(pitches):valid=False;break
                maps[sn][held]=others[0][0]
            for lane,seq in lanes.items():
                clock=0
                for g in sorted(seq,key=lambda x:x.get('onset',-1)):
                    if g.get('onset')!=clock or g.get('tuplet_ratio') or g.get('dotted') or g.get('flags',0)>1 or len(g.get('heads',[]))>8:valid=False;break
                    clock+=g['duration']
                    if g.get('rest'):
                        if g['rest']['class']!=5 or g['duration']!=240 or not rest_ink(g['rest'],image,s):valid=False;break
                    else:
                        if not g.get('heads') or not traced_stem(g,row,stems,image):valid=False;break
                        for h in g['heads']:
                            if h.get('class')==1:
                                good=g['duration']==960 and hollow(h,row,t,ctx[number][1].get(sn,{}),image,.9)
                            else:
                                p=staff_step(t,*h['xy']);good=g['duration']==240 and h.get('confidence',0)>=.7 and core(h,image,s,.95) and abs(p['residual'])<=.35 and pitch_for_step(p['nearest_step'],ctx[number][1].get(sn,{}))==tuple(h['pitch'][:2])
                            if not good:valid=False;break
                    if not valid:break
                    proposal.append((row,g,maps[sn][lane]))
                if not valid or clock!=meter*480:valid=False;break
                pitched=[g for g in seq if g.get('heads')]
                if len(pitched)>1:
                    if len(pitched)!=2 or any(g.get('beams')!=1 or g.get('direction') not in ('up','down') for g in pitched) or pitched[0]['direction']!=pitched[1]['direction']:valid=False;break
                    up=pitched[0]['direction']=='up';edge=min if up else max;hs=[dict(x=(h:=edge(g['heads'],key=lambda h:h['xy'][1]))['xy'][0],y=h['xy'][1],staff_space=s,group=g) for g in pitched]
                    if not beam_proof(hs,up,row,t):valid=False;break
            if not valid:break
        if not valid:continue
        # At least two distinct pitched columns independently synchronize the
        # two moving lanes. Rest shapes alone cannot establish synchronization.
        columns=[];s=min(r['space_px'] for r in rows)
        for a in moving[1]:
            if not a.get('heads'):continue
            hits=[b for b in moving[2] if b.get('heads') and abs(a['x']-b['x'])<=.35*s]
            if len(hits)!=1 or hits[0]['onset']!=a['onset']:valid=False;break
            columns.append(a['onset'])
        if not valid or len(set(columns))<2:continue
        original=Counter((r['staff'],r['midi'],r['duration']) for r in old if r['midi'] is not None and not r['grace']);target=Counter((row['staff'],midi(h),Fraction(g['duration'],480)) for row,g,v in proposal for h in g.get('heads',[]))
        if original-target or not 0<=sum((target-original).values())<=2:continue
        # A new member must have the same existing written MIDI spelling in
        # this bar, preventing a new unanchored accidental/key hypothesis.
        if any(not any(r['staff']==sn and r['midi']==pitch and r['duration']==duration for r in old if not r['grace']) for sn,pitch,duration in target-original):continue
        rests=Counter((row['staff'],Fraction(g['onset'],480),Fraction(g['duration'],480)) for row,g,v in proposal if g.get('rest'))
        kept_rests=[r for r in old if not r['grace'] and r['midi'] is None and 0<=r['onset'] and r['onset']+r['duration']<=meter]
        if Counter((r['staff'],r['onset'],r['duration']) for r in kept_rests)-rests:continue
        if any((Fraction(g['duration'],480)*division).denominator!=1 or (Fraction(g['onset'],480)*division).denominator!=1 for row,g,v in proposal):continue
        m=ms[number];saved=copy.deepcopy(list(m));pool=defaultdict(list);grace_before=defaultdict(list);pending=defaultdict(list)
        for r in old:
            if r['grace']:pending[r['staff']].append(r)
            elif not r['chord']:
                grace_before[id(r['node'])]=pending[r['staff']];pending[r['staff']]=[]
            if not r['grace']:pool[(r['staff'],r['midi'],r['duration'])].append(r)
        desired={};new=[];used=set();graces_used=set()
        for row,g,voice in sorted(proposal,key=lambda x:(x[0]['staff'],x[2],x[1]['onset'])):
            sn=row['staff'];onset=int(Fraction(g['onset'],480)*division);duration=int(Fraction(g['duration'],480)*division);heads=g.get('heads',[]);keys=[(sn,midi(h),Fraction(g['duration'],480)) for h in heads] if heads else [(sn,None,Fraction(g['duration'],480))];bindings=[]
            for key in keys:
                r=next((r for r in pool[key] if id(r['node']) not in used),None)
                if r:used.add(id(r['node']))
                bindings.append((key,r))
            bindings.sort(key=lambda x:(x[1] is None,x[1]['chord'] if x[1] else True))
            for j,(key,r) in enumerate(bindings):
                if r:n=r['node']
                else:
                    template=next((r for r in old if r['staff']==sn and not r['grace'] and r['duration']==Fraction(g['duration'],480) and (r['midi'] is None)==(key[1] is None)),None)
                    if template is None:valid=False;break
                    n=copy.deepcopy(template['node'])
                    for tag in ('notations','tie','lyric','accidental'):
                        for a in n.findall(tag):n.remove(a)
                    if key[1] is not None:
                        h=next(h for h in heads if midi(h)==key[1]);p=n.find('pitch');p.find('step').text=h['pitch'][0];p.find('octave').text=str(h['pitch'][1]);alter=p.find('alter')
                        if h.get('confidence',0)<.9 or not core(h,image,row['space_px'],.95):valid=False;break
                        if alter is not None:p.remove(alter)
                        if h['pitch'][2]:a=E.Element('alter');a.text=str(h['pitch'][2]);p.insert(1,a)
                for gr in grace_before.get(id(n),[]):
                    if id(gr['node']) not in graces_used:new.append(gr['node']);desired[gr['node']]=int(gr['onset']*division);graces_used.add(id(gr['node']))
                for a in n.findall('chord'):n.remove(a)
                if j:n.insert(0,E.Element('chord'))
                n.find('duration').text=str(duration);n.find('voice').text=voice;desired[n]=onset;new.append(n)
            if not valid:break
        if not valid:m[:]=saved;continue
        for gr in old:
            if gr['grace'] and id(gr['node']) not in graces_used:new.append(gr['node']);desired[gr['node']]=int(gr['onset']*division);graces_used.add(id(gr['node']))
        m[:]=list(m)[:indices[0]]+new+list(m)[indices[-1]+1:]
        if not _rebuild(m,desired):m[:]=saved;continue
        after=records(root)[number]
        grace_keys=lambda rs:Counter((r['key'],r['voice']) for r in rs if r['grace'])
        if grace_keys(old)!=grace_keys(after) or Counter((r['staff'],r['midi'],r['duration']) for r in after if not r['grace'] and r['midi'] is not None)!=target:raise ValueError('Joint overflow recovery broke conserved identities')
        if Counter((r['staff'],r['onset'],r['duration']) for r in after if not r['grace'] and r['midi'] is None)!=rests:raise ValueError('Joint overflow recovery broke proposed rest clocks')
        changes.append({'measure':number+1,'method':'joint_full_ink_lanes_all_primary_pitched_events_overflow','added_pitched_events':sum((target-original).values()),'pitched_events_retained':sum(original.values()),'coincident_pitched_columns':len(set(columns)),'voice_assignment':maps,'valid_rests_and_all_grace_clocks_preserved':True})
    return changes
