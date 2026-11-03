"""Late whole-bar rests and ink-certified sustained/cross-staff voice labels.

Pitch, ordinary clocks, grace dependencies and all notations are preserved.
References, piece names and expected voice IDs are never inputs.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import copy,xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from .relation_coverage import inputs
from .relation_measure_joint import context_rows,written
from .relation_joint_events import hollow
from .relation_physical_streams import extended_shaft,native_shaft,rebuild_with_context,set_text,filled_slot
from .relation_joint_dependencies import body_height
from .relation_joint_lanes import empty_staff_before_rest
from repair_unison_voices import _stem_end
from .symbol_cnn import FACTOR

def meters(root):
    division=1;meter=None;result=[]
    for m in root.findall('part/measure'):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type')
                try:meter=Fraction(sum(map(int,b.split('+')))*4,int(u))
                except (ValueError,TypeError,ZeroDivisionError,AttributeError):meter=None
        result.append((division,meter))
    return result

def ordinary(rs):return Counter((r['key'],r['voice'],r['grace']) for r in rs if r['midi'] is not None)

def rest_body(g,row,image):
    n=g.get('rest');s=row['space_px']
    if not n or n.get('class') not in (2,3) or n.get('confidence',0)<.85:return False
    height=body_height(n,image,s);x,y=map(round,n['xy']);rx=round(.3*s);ry=max(1,round(.1*s))
    if height is None or not .2<=height<=.7 or min(x-rx,y-ry)<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
    return bool((image[y-ry:y+ry+1,x-rx:x+rx+1]<140).mean()>=.85)

def rebuild_labels(measure,rs,labels):
    """Split/rejoin chord markers only within a common recognized clock."""
    saved=copy.deepcopy(list(measure));previous=None;indices=[i for i,n in enumerate(measure) if n.tag=='note']
    if any(n.tag not in ('note','backup','forward') for n in list(measure)[indices[0]:indices[-1]+1]):return False
    ordered=sorted(enumerate(rs),key=lambda p:(p[1]['staff'],labels.get(p[1]['node'],p[1]['voice']),p[1]['onset'],not p[1]['grace'],p[0]))
    for _,r in ordered:
        n=r['node'];voice=labels.get(n,r['voice']);set_text(n,'voice',voice)
        for c in n.findall('chord'):n.remove(c)
        k=(r['staff'],voice,r['onset'],r['duration'],r['grace'])
        if previous==k and r['midi'] is not None:n.insert(1 if r['grace'] else 0,E.Element('chord'))
        previous=k
    measure[:]=list(measure)[:indices[0]]+[r['node'] for _,r in ordered]+list(measure)[indices[-1]+1:]
    desired={r['node']:int(r['onset']*r['_division']) for r in rs}
    if not rebuild_with_context(measure,desired):measure[:]=saved;return False
    return True

def fuse(root,graph,photo,geometry):
    ms=root.findall('part/measure');before=records(root);context=meters(root);changes=[];rest_jobs=[];voice_jobs=[]
    for i,(m,rs,(division,meter)) in enumerate(zip(ms,before,context)):
        if not meter or not 0<meter<=16 or not 0<division<=9600 or (meter*division).denominator!=1 or not 1<=len(rs)<=128 or m.get('implicit')=='yes':continue
        rests=[r for r in rs if r['midi'] is None and not r['grace'] and (r['node'].findtext('type')=='whole' or r['node'].find('rest').get('measure')=='yes') and r['onset']==0 and r['duration']!=meter]
        for r in rests:
            same=[a for a in rs if a['staff']==r['staff']]
            if len(same)==1:
                saved=copy.deepcopy(list(m));r['node'].find('duration').text=str(int(meter*division));r['node'].find('rest').set('measure','yes')
                for t in r['node'].findall('type'):r['node'].remove(t)
                desired={a['node']:int(a['onset']*division) for a in rs}
                if rebuild_with_context(m,desired) and ordinary(rs)==ordinary(records(root)[i]):changes.append({'measure':i+1,'staff':r['staff'],'method':'late_full_measure_rest_active_meter_preserved_other_clocks','old_duration':str(r['duration']),'new_duration':str(meter)})
                else:m[:]=saved
            else:rest_jobs.append((i,r,division,meter))
        voice_jobs.append(i)
    if not rest_jobs and not any(len({r['voice'] for r in rs if r['staff']>1 and r['midi'] is not None})<=2 and any(r['staff']>1 and r['midi'] is not None for r in rs) for rs in before):return changes
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return changes
    image,tracks=loaded;native=None;native_ink=None;stems={s['id']:s for s in graph['stems']};ctx=context_rows(root);locations={};offset=0
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=len(ts):return changes
        for mi in range(len(rr[0]['bounds'])-1):locations[offset+mi]=(rr,ts,mi)
        offset+=len(rr[0]['bounds'])-1
    for i,r,division,meter in rest_jobs[:128]:
        rr,ts,mi=locations[i];rs=records(root)[i];r=rs[before[i].index(r)];target=[]
        for row,track in zip(rr,ts):
            sn=row['staff'];left,right=row['bounds'][mi:mi+2];gs=[g for g in row['groups'] if left<g['x']<right]
            if sn==r['staff'] or any(a['staff']==sn for a in rs if a['node'] is not r['node']) or len(gs)!=1 or not rest_body(gs[0],row,image):continue
            target.append(sn)
        if len(target)!=1:continue
        m=ms[i];saved=copy.deepcopy(list(m));n=r['node'];set_text(n,'staff',target[0]);n.find('duration').text=str(int(meter*division));n.find('rest').set('measure','yes')
        for t in n.findall('type'):n.remove(t)
        if not rebuild_with_context(m,{a['node']:int(a['onset']*division) for a in rs}) or ordinary(rs)!=ordinary(records(root)[i]):m[:]=saved;continue
        changes.append({'measure':i+1,'method':'unique_empty_staff_full_rest_body','old_staff':r['staff'],'new_staff':target[0],'duration':str(meter)})
    initial=records(root);used={r['voice'] for rs in initial for r in rs};auxiliary={}
    def auxiliary_label(primary):
        if primary not in auxiliary:
            if not primary.isdecimal():return None
            n=int(primary)+1
            while str(n) in used and n<=64:n+=1
            if n>64:return None
            auxiliary[primary]=str(n);used.add(str(n))
        return auxiliary[primary]
    def hollow_proof(h,row,track,clef):return hollow(h,row,track,clef,image,.6)
    def certify(g,row,track,clef):
        nonlocal native,native_ink
        if not g.get('heads') or g.get('grace') or g.get('tuplet_ratio') or g.get('dotted') or len(g['heads'])>8:return False
        if any(not hollow_proof(h,row,track,clef) for h in g['heads']):return False
        if extended_shaft(g,row,stems,image):return True
        if native is None:native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1]
        return native_shaft(g,row,stems,native,native_ink)
    def ending_proof(gs,row,track,clef):
        nonlocal native,native_ink
        if len(gs)!=2 or any(not all(hollow_proof(h,row,track,clef) for h in g['heads']) for g in gs):return False
        if native is None:native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1]
        ends=[];s=row['space_px']/FACTOR
        for g in gs:
            candidates=[]
            for h in g['heads']:
                real=[(v+.5)/FACTOR-.5 for v in h['xy']];end=_stem_end(dict(x=real[0],y=real[1],staff_space=s),native_ink,True)
                if end is not None and real[1]-end[1]>2.5*s:candidates.append(end)
            if not candidates:return False
            ends.append(candidates[0])
        return abs(ends[0][0]-ends[1][0])>.7*s and abs(ends[0][1]-ends[1][1])<.5*s
    for i in voice_jobs[:128]:
        rs=records(root)[i];division,meter=context[i];rr,ts,mi=locations[i]
        if len(rr)!=2 or len(rs)>128 or any(r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['grace'] or r['node'].get('print-object')=='no' for r in rs):continue
        upper,lower=rr;left,right=lower['bounds'][mi:mi+2];s=lower['space_px'];sn=lower['staff'];track=ts[sn-1];clef=ctx[i][1].get(sn)
        if not clef or sn!=upper['staff']+1:continue
        lr=[r for r in rs if r['staff']==sn and r['midi'] is not None];ur=[r for r in rs if r['staff']==sn-1 and r['midi'] is not None]
        gs=[g for g in lower['groups'] if left<g['x']<right];hs=[g for g in gs if g.get('heads')]
        if not lr or len(gs)!=len(hs) or len(hs)>64:continue
        labels={};method=None
        # An upper down-stem sustained lane above a complete bass up-stem
        # octave is an auxiliary lane, while the upper primary staff is active.
        if len({r['voice'] for r in lr})==1 and len(hs)==2 and all(r['onset']==0 and r['duration']==meter for r in lr) and ur:
            down=next((g for g in hs if g.get('direction')=='down'),None);up=next((g for g in hs if g.get('direction')=='up'),None)
            if down and up and abs(down['x']-up['x'])<.5*s and all(g['duration']==meter*480 for g in hs) and all(certify(g,lower,track,clef) for g in hs):
                low=[h['pitch'] for h in up['heads']];high=[h['pitch'] for h in down['heads']];pm=lambda p:(p[1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p[0]]+p[2]
                octave=any(a[0]==b[0] and abs(a[1]-b[1])==1 and a[2]==b[2] for a in low for b in low)
                primary={r['voice'] for r in ur}
                if octave and min(map(pm,high))>max(map(pm,low)) and len(primary)==1 and Counter(written(r) for r in lr)==Counter(tuple(h['pitch'][:2]) for g in hs for h in g['heads']):
                    new=auxiliary_label(next(iter(primary)));targets={tuple(h['pitch'][:2]) for h in down['heads']}
                    if new:labels={r['node']:new for r in lr if written(r) in targets};method='opposed_sustained_upper_auxiliary_lane_over_complete_bass_octave'
        # A migrating upper primary is anchored on both sides: the preceding
        # upper staff and the next independent hollow ending above the exact
        # same bass octave. No voice is guessed from pitch similarity alone.
        if not labels and i>0 and i+1<len(ms) and not any(r['staff']==sn-1 for r in rs) and len({r['voice'] for r in lr})==2 and not any(left<g['x']<right for g in upper['groups']) and empty_staff_before_rest(upper,ts[sn-2],left,right,image):
            slow=[r for r in lr if r['onset']==0 and r['duration']==meter];fast=[r for r in lr if r['duration']<meter]
            prev=[r for r in initial[i-1] if r['staff']==sn-1 and r['midi'] is not None and r['onset']+r['duration']==meter];future=[r for r in initial[i+1] if r['staff']==sn and r['midi'] is not None and r['onset']==0 and r['duration']==meter]
            if len(slow)==2 and abs(slow[0]['midi']-slow[1]['midi'])==12:
                pairs=Counter(written(r) for r in slow);future_bass=defaultdict(list)
                for r in future:future_bass[r['voice']].append(r)
                bass=[v for v,c in future_bass.items() if Counter(written(r) for r in c)==pairs];primary={r['voice'] for r in prev}
                if fast and min(r['midi'] for r in fast)>max(r['midi'] for r in slow) and len({r['voice'] for r in slow})==len({r['voice'] for r in fast})==1 and len(primary)==len(bass)==1 and next(iter(primary))!=bass[0]:
                    # The next independent principal and bass must themselves
                    # have distinct graph columns with continuous hollow shafts.
                    nr,nt,nmi=locations[i+1];nrow=next(r for r in nr if r['staff']==sn);nl,nrgt=nrow['bounds'][nmi:nmi+2];ng=[g for g in nrow['groups'] if nl<g['x']<nrgt and g.get('heads')]
                    fclock=sum(r['duration'] for r in fast if not r['chord'])
                    if native is None:native=cv2.imread(str(photo),0);native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1]
                    current_proof=all((all(hollow_proof(h,lower,track,clef) for h in g['heads']) if g['duration']==meter*480 else all(filled_slot(h,lower,track,clef,native) for h in g['heads'])) and (extended_shaft(g,lower,stems,image) or native_shaft(g,lower,stems,native,native_ink)) for g in hs)
                    if ending_proof(ng,nrow,nt[sn-1],ctx[i+1][1].get(sn)) and current_proof and fclock==meter and all(g.get('direction')=='up' for g in hs) and Counter(written(r) for r in lr)==Counter(tuple(h['pitch'][:2]) for g in hs for h in g['heads']):
                        labels={r['node']:(bass[0] if r in slow else next(iter(primary))) for r in lr};method='cross_staff_primary_bidirectional_bass_and_ending_anchors'
        if not labels or all(r['voice']==labels.get(r['node'],r['voice']) for r in rs):continue
        for r in rs:r['_division']=division
        old_keys=Counter(r['key'] for r in rs);m=ms[i]
        if not rebuild_labels(m,rs,labels):continue
        if old_keys!=Counter(r['key'] for r in records(root)[i]):raise ValueError('Voice labels changed ordinary event identity')
        changes.append({'measure':i+1,'method':method,'voices':sorted(set(labels.values())),'notes_relabelled':sum(r['voice']!=labels.get(r['node'],r['voice']) for r in rs)})
    return changes
