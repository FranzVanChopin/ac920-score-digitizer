"""Bounded reconstruction of complete ink-certified simultaneous streams.

Only photographs, their predictions and recognized XML are accepted. A wrong
global clock is rebuilt only when complete independent physical lanes agree.
Small chord heads require native ink bodies and their own continuous shaft.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import copy,xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from repair_grace_timing import _rebuild
from repair_unison_voices import _stem_end
from .relation_coverage import inputs,core
from .relation_complete_clocks import rest_ink
from .relation_joint_dependencies import body_height
from .relation_joint_events import hollow,key_proof,clear_sign_strip
from .relation_joint_candidates import component
from .relation_measure_joint import context_rows,written
from .relation_measure_realign import traced_stem
from .staff_preprocess import staff_step
from .relation_context import pitch_for_step
from .symbol_cnn import FACTOR
from .relation_joint_lanes import empty_staff_before_rest

def accent_ink(g,row,stems,native):
    s=row['space_px']/FACTOR;x=(g['x']+.5)/FACTOR-.5
    bottom=max(h['xy'][1] for h in g['heads'])
    if g.get('direction')=='down' and g.get('stem') in stems:bottom=max(bottom,max(p[1] for p in stems[g['stem']]['xy']))
    y=(bottom+.5)/FACTOR-.5;l,r=round(x-1.5*s),round(x+1.5*s);t,b=round(y+.4*s),round(y+3*s)
    if min(l,t)<0 or r>=native.shape[1] or b>=native.shape[0]:return False
    ink=(native[t:b,l:r]<140).astype('uint8');_,_,stats,_=cv2.connectedComponentsWithStats(ink,8);hits=0
    for a,c,w,h,n in stats[1:]:
        if not .6*s<w<1.8*s or not .2*s<h<s or not .05*s*s<n<.6*s*s or abs(l+a+w/2-x)>.5*s:continue
        mask=ink[c:c+h,a:a+w];top=mask[:max(1,h//3),:max(1,w//3)];low=mask[-max(1,h//3):,:max(1,w//3)];mid=mask[h//3:max(h//3+1,2*h//3),:max(1,w//3)];tip=mask[h//3:max(h//3+1,2*h//3),-max(1,w//3):]
        if top.any() and low.any() and mid.mean()<.25 and tip.any():hits+=1
    return hits==1

def extended_shaft(g,row,stems,image):
    if traced_stem(g,row,stems,image):return True
    stem=stems.get(g.get('stem'));s=row['space_px']
    if not stem or stem.get('confidence',0)<.85 or not g.get('heads'):return False
    pts=sorted(stem['xy'],key=lambda p:p[1]);lo,hi=copy.deepcopy(pts)
    if g.get('direction')=='up':hi[1]=max(hi[1],max(h['xy'][1] for h in g['heads']))
    elif g.get('direction')=='down':lo[1]=min(lo[1],min(h['xy'][1] for h in g['heads']))
    else:return False
    if max(abs(lo[1]-pts[0][1]),abs(hi[1]-pts[1][1]))>2*s:return False
    proposal=dict(stem,xy=[lo,hi],confidence=max(.9,stem['confidence']))
    return traced_stem(g,row,{g['stem']:proposal},image)

def filled_slot(h,row,track,clef,native,small=False):
    if h.get('class')!=0:return False
    pos=staff_step(track,*h['xy']);s=row['space_px'];real=dict(h,xy=[(v+.5)/FACTOR-.5 for v in h['xy']])
    height=body_height(real,native,s/FACTOR)
    return bool(height is not None and (.5<=height<=.84 if small else .85<=height<=1.65) and abs(pos['residual'])<=.3 and pitch_for_step(pos['nearest_step'],clef)==tuple(h['pitch'][:2]) and core(real,native,s/FACTOR,.95))

def native_shaft(g,row,stems,native,ink):
    stem=stems.get(g.get('stem'));s=row['space_px']/FACTOR;up=g.get('direction')=='up'
    if not stem or stem.get('confidence',0)<.85 or not g.get('heads') or g.get('direction') not in ('up','down'):return False
    pts=np.array(stem['xy']);sx=(pts[:,0].mean()+.5)/FACTOR-.5;tip=((pts[:,1].min() if up else pts[:,1].max())+.5)/FACTOR-.5
    heads=[[(v+.5)/FACTOR-.5 for v in h['xy']] for h in g['heads']];ends=[]
    for x,y in heads:
        end=_stem_end(dict(x=x,y=y,staff_space=s*(.7 if g.get('grace') else 1)),ink,up)
        if end is not None and abs(end[0]-sx)<.35*s and abs(end[1]-tip)<.5*s:ends.append(end)
    if not ends or max(p[0] for p in ends)-min(p[0] for p in ends)>.25*s:return False
    xstem=round(np.median([p[0] for p in ends]))
    for x,y in heads:
        l,r=sorted((round(x),xstem));dy=round(.4*s)
        if r-l>.95*s or min(l,round(y)-dy)<0 or r>=native.shape[1] or round(y)+dy>=native.shape[0]:return False
        if (native[round(y)-dy:round(y)+dy+1,l:r+1]<140).max(0).mean()<.98:return False
    return True

def prepare(row,track,clef,stems,image,native):
    groups=copy.deepcopy(row['groups']);s=row['space_px'];remove=set()
    if sum(g.get('stem') is None and len(g.get('heads',[]))==1 for g in groups)>128:return groups
    # Coincident small orphan heads join only a unique continuous physical stem.
    for i,g in enumerate(groups):
        if g.get('stem') is not None or len(g.get('heads',[]))!=1 or not filled_slot(g['heads'][0],row,track,clef,native,True):continue
        candidates=[]
        for j,parent in enumerate(groups):
            if i==j or parent.get('stem') is None or parent.get('rest') or len(parent.get('heads',[]))>=8 or abs(g['x']-parent['x'])>.25*s:continue
            if all(filled_slot(h,row,track,clef,native,True) for h in parent['heads']) and extended_shaft(dict(parent,heads=parent['heads']+g['heads']),row,stems,image):candidates.append(j)
        if len(candidates)==1:groups[candidates[0]]['heads']+=g['heads'];remove.add(i)
    groups=[g for i,g in enumerate(groups) if i not in remove]
    for g in groups:
        hs=g.get('heads',[])
        if hs and all(filled_slot(h,row,track,clef,native,True) for h in hs):
            g['grace']=True
            if g.get('direction') is None:
                options=[]
                for ident,stem in stems.items():
                    pts=np.array(stem['xy']);sx=pts[:,0].mean()
                    if abs(sx-g['x'])>.7*s:continue
                    for direction in ('up','down'):
                        if (direction=='up' and pts[:,1].min()<min(h['xy'][1] for h in hs)-1.4*s or direction=='down' and pts[:,1].max()>max(h['xy'][1] for h in hs)+1.4*s) and extended_shaft(dict(g,stem=ident,direction=direction),row,stems,image):options.append((ident,direction))
                if len(options)==1:g['stem'],g['direction']=options[0]
    # A missing member of a small chord must be visible at its written slot,
    # join the same shaft and precede a unique containing principal chord.
    for g in groups:
        if not g.get('grace') or g.get('stem') is None or len(g.get('heads',[]))<2:continue
        following=sorted([p for p in groups if p.get('heads') and not p.get('grace') and p.get('direction')==g.get('direction') and .6*s<p['x']-g['x']<3*s],key=lambda p:p['x'])
        if not following:continue
        principal=following[0];old=Counter(tuple(h['pitch'][:2]) for h in g['heads']);new=Counter(tuple(h['pitch'][:2]) for h in principal['heads'])
        if old-new or sum((new-old).values())!=1 or len(g['heads'])>=8:continue
        target=next(iter(new-old));anchor=next(h for h in principal['heads'] if tuple(h['pitch'][:2])==target);p=staff_step(track,g['x'],anchor['xy'][1]);bottom=pitch_for_step(0,clef);step=(target[1]-bottom[1])*7+'CDEFGAB'.index(target[0])-'CDEFGAB'.index(bottom[0]);y=p['bottom_line_y']-step*p['local_space_px']/2
        hits=[]
        for dx in np.linspace(-1.1,1.1,23):
            h=dict(anchor,id='native-small-member',xy=[g['x']+dx*s,y],confidence=0.,proposal_source='bounded_native_slot')
            if filled_slot(h,row,track,clef,native,True) and extended_shaft(dict(g,heads=g['heads']+[h]),row,stems,image):hits.append(h)
        clusters=[]
        for h in hits:
            if clusters and h['xy'][0]-clusters[-1][-1]['xy'][0]<.15*s:clusters[-1].append(h)
            else:clusters.append([h])
        if len(clusters)==1:g['heads'].append(clusters[0][len(clusters[0])//2])
    return groups

def set_text(n,tag,value):
    child=n.find(tag)
    if child is None:
        child=E.Element(tag);idx=next((i for i,a in enumerate(n) if a.tag in ('beam','notations','lyric')),len(n));n.insert(idx,child)
    child.text=str(value)

def rebuild_with_context(measure,desired):
    elements=list(measure);last=max(i for i,n in enumerate(elements) if n.tag=='note');tail=elements[last+1:]
    if any(n.tag not in ('barline','attributes') for n in tail):return False
    measure[:]=elements[:last+1];success=_rebuild(measure,desired);measure.extend(tail)
    return success

def make_note(template,h,sn,voice,duration,grace,division,direction):
    n=copy.deepcopy(template)
    for tag in ('pitch','rest','chord','grace','duration','accidental','beam','notations','lyric','tie'):
        for a in n.findall(tag):n.remove(a)
    if h is None:n.insert(0,E.Element('rest'))
    else:
        p=E.Element('pitch');E.SubElement(p,'step').text=h['pitch'][0]
        if h['pitch'][2]:E.SubElement(p,'alter').text=str(h['pitch'][2])
        E.SubElement(p,'octave').text=str(h['pitch'][1]);n.insert(0,p)
    if grace:n.insert(0,E.Element('grace'))
    else:n.insert(1,E.Element('duration'));n.find('duration').text=str(int(duration*division))
    set_text(n,'voice',voice);set_text(n,'staff',sn);set_text(n,'type','eighth' if grace or duration==Fraction(1,2) else 'quarter' if duration==1 else 'half' if duration==2 else 'whole')
    if direction:set_text(n,'stem',direction)
    return n

def fuse(root,graph,photo,geometry):
    rs_all=records(root);division=1;meter=None;jobs=[]
    for i,(m,rs) in enumerate(zip(root.findall('part/measure'),rs_all)):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        if not meter or not 0<meter<=16 or not 0<division<=9600 or not 1<=len(rs)<=128 or m.get('implicit')=='yes':continue
        if any(r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].get('print-object')=='no' for r in rs):continue
        if any(r['node'].find('lyric') is not None or any(a.tag not in ('notations','articulations','accent') for a in r['node'].findall('notations//*')) for r in rs):continue
        if any(r['onset']+r['duration']>meter for r in rs if not r['grace']) and any(r['grace'] for r in rs):jobs.append((i,m,rs,division,meter))
    if not jobs or len(jobs)>128:return []
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1];stems={s['id']:s for s in graph['stems']};ctx=context_rows(root);locations={};offset=0;changes=[]
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=len(ts):return []
        for mi in range(len(rr[0]['bounds'])-1):locations[offset+mi]=(rr,ts,mi)
        offset+=len(rr[0]['bounds'])-1
    for i,m,rs,division,meter in jobs:
        rr,ts,mi=locations[i]
        if len(rr)!=2 or {r['staff'] for r in rs}!={1,2}:continue
        lanes=[];valid=True;main_columns={};small_count=0
        for row,track in zip(rr,ts):
            sn=row['staff'];key,clefs=ctx[i];clef=clefs.get(sn);s=row['space_px'];left,right=row['bounds'][mi:mi+2]
            if key is None or not clef:valid=False;break
            groups=[g for g in prepare(row,track,clef,stems,image,native) if left<g['x']<right]
            if len(groups)>64 or any(len(g.get('heads',[]))>8 or g.get('dotted') or g.get('tuplet_ratio') for g in groups):valid=False;break
            label_by_direction={}
            for direction in ('up','down'):
                current=[g for g in groups if g.get('heads') and g.get('direction')==direction]
                ordinary=[g for g in current if not g.get('grace')]
                if not ordinary:continue
                voices=Counter(r['voice'] for r in rs if not r['grace'] and r['midi'] is not None and any(written(r)==tuple(h['pitch'][:2]) and r['duration']*480==g['duration'] for g in ordinary for h in g['heads']))
                if not voices or len(voices)>1:valid=False;break
                voice=next(iter(voices));label_by_direction[direction]=voice
                if any(g.get('flags') or g['duration'] not in (240,480,960) or not(extended_shaft(g,row,stems,image) or native_shaft(g,row,stems,native,native_ink)) for g in ordinary):valid=False;break
                if any(not(filled_slot(h,row,track,clef,native) if h['class']==0 else hollow(h,row,track,clef,image,.85)) for g in ordinary for h in g['heads']):valid=False;break
                rests=[g for g in groups if g.get('rest')] if direction=='up' else []
                if any(not rest_ink(g['rest'],image,s) for g in rests):valid=False;break
                stream=sorted(ordinary+rests,key=lambda g:g['x']);clock=Fraction(0);slots=[]
                for g in stream:slots.append((g,clock));clock+=Fraction(g['duration'],480)
                if clock!=meter:valid=False;break
                graces=[]
                for g in current:
                    if not g.get('grace'):continue
                    following=[(p,t) for p,t in slots if p.get('heads') and p['x']>g['x']+.4*s]
                    if not following or following[0][0]['x']-g['x']>5*s or not all(filled_slot(h,row,track,clef,native,True) for h in g['heads']):valid=False;break
                    if g.get('stem') is not None:
                        if not extended_shaft(g,row,stems,image):
                            h=g['heads'][0];real=[(v+.5)/FACTOR-.5 for v in h['xy']]
                            if _stem_end(dict(x=real[0],y=real[1],staff_space=.7*s/FACTOR),native_ink,direction=='up') is None:valid=False;break
                    else:
                        # One detached small run member needs a unique predicted
                        # stem attached through its rim, rather than a size label.
                        options=[]
                        for ident,stem in stems.items():
                            sx=np.mean(np.array(stem['xy'])[:,0])
                            if abs(sx-g['x'])<.7*s and extended_shaft(dict(g,stem=ident,direction=direction),row,stems,image):options.append(ident)
                        if len(options)!=1:valid=False;break
                    graces.append((g,following[0][1]));small_count+=1
                if not valid:break
                main_columns.setdefault(sn,[]).extend((g['x'],t) for g,t in slots if direction=='up')
                lanes.append((sn,voice,slots,graces,row,track,clef,key))
            if not valid or len(set(label_by_direction.values()))!=len(label_by_direction):valid=False;break
        if not valid or len(lanes)<2 or small_count<2:continue
        anchors=[];s=min(row['space_px'] for row in rr)
        for x,t in main_columns.get(1,[]):
            match=[u for y,u in main_columns.get(2,[]) if abs(x-y)<.35*s]
            if len(match)>1 or match and match[0]!=t:valid=False;break
            if match:anchors.append(t)
        if not valid or len(set(anchors))<2 or min(anchors)!=0:continue
        # Every regular event must bind a physical lane. This phase does not
        # guess an alter from a neighboring pitch or remove hidden events.
        proposals=[]
        for sn,voice,slots,graces,row,track,clef,key in lanes:
            for g,t,grace in [(g,t,False) for g,t in slots]+[(g,t,True) for g,t in graces]:
                for h in g.get('heads',[]) or [None]:
                    ident=tuple(h['pitch'][:2]) if h else None
                    old=[r for r in rs if r['staff']==sn and (written(r) if r['midi'] is not None else None)==ident]
                    alters={int(r['node'].findtext('pitch/alter','0')) for r in old} if h else {0}
                    if grace and t==0 and h and old and all(r['grace'] for r in old):
                        matching=[r for r in old if int(r['node'].findtext('pitch/alter','0'))==h['pitch'][2]]
                        if matching:old=matching;alters={h['pitch'][2]}
                    if not old and h:
                        if not key_proof(row,track,clef,key,row['groups'],component,image) or not clear_sign_strip(h,row,image):valid=False;break
                        alters={(1 if key>0 else -1) if ident[0] in ('FCGDAEB' if key>0 else 'BEADGCF')[:abs(key)] else 0}
                    if len(alters)!=1 or h and h['pitch'][2]!=next(iter(alters)):valid=False;break
                    template=(old[0] if old else next(r for r in rs if r['midi'] is not None))['node']
                    duration=Fraction(g['duration'],480)
                    if (t*division).denominator!=1 or (duration*division).denominator!=1:valid=False;break
                    proposals.append((make_note(template,h,sn,voice,duration,grace,division,g.get('direction')),t,g,grace))
                if not valid:break
            if not valid:break
        if not valid or len(proposals)>128:continue
        # Preserve recognized grace-run members missed by the detector, only
        # when a certified same-pitch small opening run already binds its lane.
        for r in rs:
            if not r['grace']:continue
            exists=any(n.findtext('pitch/step')==r['node'].findtext('pitch/step') and n.findtext('pitch/octave')==r['node'].findtext('pitch/octave') and int(n.findtext('pitch/alter','0'))==int(r['node'].findtext('pitch/alter','0')) and int(n.findtext('staff'))==r['staff'] for n,t,g,grace in proposals)
            if exists:continue
            choices=[(sn,v,gr) for sn,v,sl,gr,*_ in lanes if sn==r['staff'] and gr and gr[0][1]==0 and r['onset']==0 and any(tuple(h['pitch'][:2])==written(r) for g,t in gr for h in g['heads'])]
            if len(choices)!=1:valid=False;break
            sn,v,gr=choices[0];n=copy.deepcopy(r['node']);set_text(n,'voice',v);proposals.append((n,Fraction(0),gr[0][0],True))
        if not valid:continue
        # Existing pitches must remain, either regular or photograph-certified
        # small; unsupported leftover notes veto the complete reconstruction.
        identity=lambda rows:Counter((r['staff'],written(r),int(r['node'].findtext('pitch/alter','0'))) for r in rows if r['midi'] is not None)
        available=Counter((int(n.findtext('staff')), (n.findtext('pitch/step'),int(n.findtext('pitch/octave'))),int(n.findtext('pitch/alter','0'))) for n,t,g,gr in proposals if n.find('pitch') is not None)
        deficit=identity(rs)-available
        if any(not available[k] or sum(not r['grace'] and r['onset']>=meter and (r['staff'],written(r),int(r['node'].findtext('pitch/alter','0')))==k for r in rs if r['midi'] is not None)<count for k,count in deficit.items()):continue
        proposals.sort(key=lambda q:(int(q[0].findtext('staff')),q[0].findtext('voice'),q[1],not q[3],q[2]['x']))
        notes=[];desired={};last=None
        accents_seen=set();recognized_accent=any(r['node'].find('notations/articulations/accent') is not None for r in rs)
        for n,t,g,grace in proposals:
            k=(n.findtext('staff'),n.findtext('voice'),t,id(g),grace)
            if k==last and n.find('pitch') is not None:n.insert(1 if grace else 0,E.Element('chord'))
            if recognized_accent and not grace and g.get('heads') and id(g) not in accents_seen:
                accents_seen.add(id(g));row=next(row for row in rr if row['staff']==int(n.findtext('staff')))
                if accent_ink(g,row,stems,native):E.SubElement(E.SubElement(E.SubElement(n,'notations'),'articulations'),'accent')
            notes.append(n);desired[n]=int(t*division);last=k
        indices=[j for j,n in enumerate(m) if n.tag=='note'];saved=copy.deepcopy(list(m))
        if any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]):continue
        m[:]=list(m)[:indices[0]]+notes+list(m)[indices[-1]+1:]
        if not _rebuild(m,desired):m[:]=saved;continue
        changes.append({'measure':i+1,'method':'native_small_heads_complete_simultaneous_streams','lanes':len(lanes),'small_groups':small_count,'independent_coincident_columns':len(set(anchors)),'outside_bar_duplicate_events_removed':sum(deficit.values())})
    return changes

def hollow_unisons(root,graph,photo,geometry):
    """Add a distinct half-note head; migrate only a uniquely active empty staff voice."""
    all_rs=records(root);jobs=[];division=1;meter=None
    for i,(m,rs) in enumerate(zip(root.findall('part/measure'),all_rs)):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        if i==0 or meter!=2 or not 0<division<=9600 or not 2<=len(rs)<=8 or m.get('implicit')=='yes':continue
        if any(r['midi'] is None or r['grace'] or r['onset']!=0 or r['duration']!=2 or r['node'].findtext('type')!='half' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].get('print-object')=='no' for r in rs):continue
        if len({(r['staff'],r['voice']) for r in rs})!=1 or sum(not r['chord'] for r in rs)!=1:continue
        jobs.append((i,m,rs,division))
    if not jobs or len(jobs)>128:return []
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;native=cv2.imread(str(photo),0);ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1];stems={s['id']:s for s in graph['stems']};ctx=context_rows(root);locations={};offset=0;changes=[]
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=len(ts):return []
        for mi in range(len(rr[0]['bounds'])-1):locations[offset+mi]=(rr,ts,mi)
        offset+=len(rr[0]['bounds'])-1
    for i,m,rs,division in jobs:
        rr,ts,mi=locations[i];sn=rs[0]['staff'];row=next((r for r in rr if r['staff']==sn),None)
        if row is None or sn<=1 or sn>len(ts):continue
        track=ts[sn-1];s=row['space_px'];left,right=row['bounds'][mi:mi+2];clef=ctx[i][1].get(sn)
        if not clef:continue
        gs=[g for g in row['groups'] if left<g['x']<right]
        if len(gs)!=2 or any(g.get('grace') or g.get('rest') or g.get('beams') or g.get('flags') for g in gs):continue
        main=[g for g in gs if g.get('stem') is not None and len(g.get('heads',[]))==len(rs)];extra=[g for g in gs if g.get('stem') is None and len(g.get('heads',[]))==1]
        if len(main)!=1 or len(extra)!=1:continue
        main,extra=main[0],extra[0];h=extra['heads'][0];ident=tuple(h['pitch'][:2]);matched=[r for r in rs if written(r)==ident]
        if len(matched)!=1 or not .8*s<main['x']-extra['x']<1.7*s or main['duration']!=960 or main.get('direction')!='up':continue
        if Counter(written(r) for r in rs)!=Counter(tuple(a['pitch'][:2]) for a in main['heads']) or not traced_stem(main,row,stems,image) or not all(hollow(a,row,track,clef,image,.85) for a in main['heads']+[h]):continue
        real=[(v+.5)/FACTOR-.5 for v in h['xy']];end=_stem_end(dict(x=real[0],y=real[1],staff_space=s/FACTOR),ink,True)
        stem=stems[main['stem']];sx=(np.array(stem['xy'])[:,0].mean()+.5)/FACTOR-.5;tip=(min(p[1] for p in stem['xy'])+.5)/FACTOR-.5
        if end is None or abs(end[0]-sx)<.7*s/FACTOR or abs(end[1]-tip)>.35*s/FACTOR or abs(end[1]-real[1])<2.5*s/FACTOR:continue
        upper=next((r for r in rr if r['staff']==sn-1),None)
        if upper is None or any(left<g['x']<right for g in upper['groups']) or not empty_staff_before_rest(upper,ts[sn-2],left,right,image):continue
        recent=next((rs0 for rs0 in reversed(all_rs[max(0,i-8):i]) if any(r['staff']==sn-1 for r in rs0)),[])
        prior=[r for r in recent if r['staff']==sn-1 and r['midi'] is not None and not r['grace'] and r['onset']+r['duration']==2]
        labels={r['voice'] for r in prior}
        if len(labels)!=1 or next(iter(labels))==rs[0]['voice']:continue
        original=matched[0];n=copy.deepcopy(original['node'])
        for tag in ('chord','notations','lyric','tie','accidental','beam'):
            for a in n.findall(tag):n.remove(a)
        set_text(n,'voice',next(iter(labels)));set_text(n,'stem','up');saved=copy.deepcopy(list(m));m.insert(max(j for j,a in enumerate(m) if a.tag=='note')+1,n)
        desired={r['node']:0 for r in rs};desired[n]=0
        if not rebuild_with_context(m,desired):m[:]=saved;continue
        if Counter((r['key'],r['voice'],r['grace']) for r in rs)-Counter((r['key'],r['voice'],r['grace']) for r in records(root)[i]):raise ValueError('Hollow unison changed a retained event')
        changes.append({'measure':i+1,'method':'distinct_hollow_head_independent_native_shaft_empty_adjacent_staff_voice','added_notes':1,'staff':sn,'voice':next(iter(labels))})
    return changes
