"""Recover opposed physical lanes with one missing paired column.

Both lanes need complete ink-proven rhythms, a shared zero-time anchor and
terminal rests. Existing pitches/durations are retained. Each new head needs
an explicit, isolated accidental at either image scale. No score identifiers
are consulted.
"""
from collections import Counter
from fractions import Fraction
import copy,xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from repair_grace_timing import _rebuild
from .relation_coverage import inputs
from .relation_complete_clocks import rest_ink,token
from .relation_measure_joint import context_rows,head_witness,align_columns
from .relation_measure_realign import traced_stem,local_staff_head
from .relation_meter_decode import graph_beam
from .relation_joint_candidates import component
from .symbol_cnn import FACTOR
from repair_unison_voices import _stem_end


def explicit_alter(h,row,native,image=None):
    s=row['space_px']; target={'sharp':8,'flat':9,'natural':10}.get(h['pitch'][3])
    if target is None:return None
    signs=[]
    for n in row['nodes']:
        dx=(h['xy'][0]-n['xy'][0])/s; dy=(n['xy'][1]-h['xy'][1])/s
        if n['class']!=target or n.get('confidence',0)<.8 or not .65<dx<3 or abs(dy-(-.75 if target==9 else 0))>.25:continue
        real=dict(n,xy=[(v+.5)/FACTOR-.5 for v in n['xy']])
        if (image is not None and component(n,image,s) is not None) or component(real,native,s/FACTOR) is not None:signs.append(n)
    return {8:1,9:-1,10:0}[target] if len(signs)==1 else None


def empty_staff_before_rest(row,track,left,last,image):
    s=row['space_px'];l,r=round(left+s),round(last-1.8*s)
    lines=[float(np.interp((l+r)/2,*np.asarray(t['polyline']).T)) for t in track['lines']]
    top,bottom=round(min(lines)-.75*s),round(max(lines)+.75*s)
    if r-l<4*s or min(l,top)<0 or r>=image.shape[1] or bottom>=image.shape[0]:return False
    ink=(image[top:bottom,l:r]<140).astype('uint8')
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,round(3*s)),np.uint8))
    return cv2.countNonZero(cv2.subtract(ink,horizontal))<.25*s*s


def fuse(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1]
    stems={s['id']:s for s in graph['stems']};ctx=context_rows(root);ms=root.findall('part/measure');rows=records(root)
    locations={};offset=0;changes=[];division=1;meter=None;native=None;native_ink=None;usage=Counter((r['staff'],r['voice']) for rs in rows for r in rs if r['midi'] is not None and not r['grace'])
    def ensure_native():
        nonlocal native,native_ink
        if native is None:native=cv2.imread(str(photo),0)
        if native_ink is None:native_ink=cv2.threshold(native,140,255,cv2.THRESH_BINARY_INV)[1]
    def shaft(g,row):
        if traced_stem(g,row,stems,image):return True
        stem=stems.get(g.get('stem'))
        if not stem or stem.get('confidence',0)<.85 or len(g['heads'])!=1:return False
        ensure_native();h=g['heads'][0];s=row['space_px']/FACTOR
        end=_stem_end(dict(x=(h['xy'][0]+.5)/FACTOR-.5,y=(h['xy'][1]+.5)/FACTOR-.5,staff_space=s),native_ink,g['direction']=='up')
        points=np.asarray(stem['xy']);sx=(points[:,0].mean()+.5)/FACTOR-.5;sy=((min if g['direction']=='up' else max)(points[:,1])+.5)/FACTOR-.5
        return bool(end is not None and abs(end[0]-sx)<.35*s and abs(end[1]-sy)<.5*s and np.ptp(points[:,1])>2.5*row['space_px'])
    for sy in sorted({r['system'] for r in graph['systems']}):
        rr=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        if len(rr)!=2 or len(ts)!=2:offset+=len(rr[0]['bounds'])-1;continue
        for mi in range(len(rr[0]['bounds'])-1):locations[offset+mi]=(rr,ts,mi)
        offset+=len(rr[0]['bounds'])-1
    for i,(m,rs) in enumerate(zip(ms,rows)):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type');meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        pitched=[r for r in rs if r['midi'] is not None];rest=[r for r in rs if r['midi'] is None]
        if any(r['node'].get('print-object')=='no' for r in rs):continue
        if not meter or not 0<meter<=16 or not 0<division<=9600 or i not in locations or m.get('implicit')=='yes' or len(rs)>128 or len(changes)>=128:continue
        if not 16<=len(pitched)<=126 or any(r['staff']!=1 or r['grace'] or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].get('print-object')=='no' for r in rs if r['midi'] is not None):continue
        if len({r['voice'] for r in pitched})!=1 or len(rest) not in (2,3):continue
        old=[sorted([r for r in pitched if r['onset']==t],key=lambda r:r['midi']) for t in sorted({r['onset'] for r in pitched})]
        if any(len(c)!=2 or len({r['duration'] for r in c})!=1 for c in old) or old[0][0]['onset']!=0:continue
        rr,ts,mi=locations[i];row,lower=rr;track,lower_track=ts
        if row['staff']!=1 or lower['staff']!=2 or row['bounds']!=lower['bounds']:continue
        left,right=row['bounds'][mi:mi+2];s=row['space_px'];gs=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x'])
        notes=[g for g in gs if g.get('heads')];rests=[g for g in gs if g.get('rest')]
        bg=[g for g in lower['groups'] if left<g['x']<right]
        if len(gs)>64 or len(notes)%2 or len(rests)!=1 or len(bg)!=1 or not bg[0].get('rest') or abs(bg[0]['x']-rests[0]['x'])>.35*s:continue
        pairs=[];valid=True;clock=Fraction(0)
        for a,b in zip(notes[::2],notes[1::2]):
            if abs(a['x']-b['x'])>.25*s or {a.get('direction'),b.get('direction')}!={'up','down'} or a['duration']!=b['duration'] or any(len(g['heads'])!=1 or g.get('grace') or g.get('tuplet_ratio') or g.get('dotted') for g in (a,b)):valid=False;break
            up=next(g for g in (a,b) if g['direction']=='up');down=next(g for g in (a,b) if g['direction']=='down')
            if any(g['duration']!=480//(2**(g.get('beams') or g.get('flags') or 0)) for g in (up,down)):valid=False;break
            if not all(shaft(g,row) and (head_witness(g['heads'][0],row,track,ctx[i][1].get(1,{}),image) or local_staff_head(g['heads'][0],row,track,ctx[i][1].get(1,{}),image)) for g in (up,down)):valid=False;break
            pairs.append((up,down,clock));clock+=Fraction(a['duration'],480)
        if not valid or len(pairs)!=len(old)+1 or len(pairs)<8 or rests[0]['duration']!=bg[0]['duration'] or clock+Fraction(rests[0]['duration'],480)!=meter:continue
        if not all(rest_ink(g['rest'],image,s) for g in (rests[0],bg[0])) or not empty_staff_before_rest(lower,lower_track,left,bg[0]['x'],image):continue
        beam_edges=0
        for lane in (0,1):
            sequence=[p[lane] for p in pairs]
            for a,b in zip(sequence,sequence[1:]):
                if a.get('beam_chain') is None or a.get('beam_chain')!=b.get('beam_chain'):continue
                if a.get('beams')!=b.get('beams') or a['beams'] not in (1,2):valid=False;break
                hs=[dict(x=g['heads'][0]['xy'][0],y=g['heads'][0]['xy'][1],staff_space=s,group=g) for g in (a,b)]
                if not graph_beam(hs[0],hs[1],ink,lane==0,a['beams']):valid=False;break
                beam_edges+=1
        if not valid or beam_edges<4:continue
        if (clock*division).denominator!=1 or any((p[2]*division).denominator!=1 or (Fraction(p[0]['duration'],480)*division).denominator!=1 for p in pairs):continue
        observed=[Counter(tuple(g['heads'][0]['pitch'][:2]) for g in p[:2]) for p in pairs]
        if token(old[0])!=observed[0]:continue
        alignment=[(0,0)]+[(a+1,b+1) for a,b in align_columns([token(c) for c in old[1:]],observed[1:])]
        if len(alignment)!=len(old) or {a for a,b in alignment}!=set(range(len(old))) or any(token(old[a])!=observed[b] or old[a][0]['duration']*480!=pairs[b][0]['duration'] for a,b in alignment):continue
        missing=set(range(len(pairs)))-{b for a,b in alignment}
        if len(missing)!=1:continue
        bass_rests=[r for r in rest if r['staff']==2 and r['duration']*480==bg[0]['duration']];top_rests=[r for r in rest if r['staff']==1 and r['duration']*480==rests[0]['duration']]
        if len(bass_rests)!=1 or len(top_rests)!=1:continue
        primary=pitched[0]['voice'];secondary=bass_rests[0]['voice']
        if primary==secondary or usage[2,secondary]<24 or top_rests[0]['voice']!=primary:continue
        if native is None:native=cv2.imread(str(photo),0)
        extra_index=next(iter(missing));new_pitches=[explicit_alter(g['heads'][0],row,native,image) for g in pairs[extra_index][:2]]
        if any(a is None for a in new_pitches):continue
        old_by={b:old[a] for a,b in alignment};new=[];desired={};used=set();saved=copy.deepcopy(list(m));indices=[j for j,n in enumerate(m) if n.tag=='note']
        if not indices or any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]):continue
        for j,(up,down,onset) in enumerate(pairs):
            for lane,(g,voice) in enumerate(((up,primary),(down,secondary))):
                h=g['heads'][0];ident=tuple(h['pitch'][:2]);existing=[r for r in old_by.get(j,[]) if (r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave')))==ident]
                if len(existing)>1:valid=False;break
                if existing:n=existing[0]['node'];used.add(id(n))
                else:
                    n=copy.deepcopy(pitched[0]['node'])
                    for tag in ('pitch','accidental','notations','lyric','tie'):
                        for a in n.findall(tag):n.remove(a)
                    pitch=E.Element('pitch');E.SubElement(pitch,'step').text=ident[0];E.SubElement(pitch,'alter').text=str(new_pitches[lane]);E.SubElement(pitch,'octave').text=str(ident[1]);n.insert(0,pitch)
                    E.SubElement(n,'accidental').text=h['pitch'][3]
                for a in n.findall('chord'):n.remove(a)
                n.find('voice').text=voice; n.find('duration').text=str(int(Fraction(g['duration'],480)*division))
                stem=n.find('stem')
                if stem is None:
                    stem=E.Element('stem');index=next((k for k,a in enumerate(n) if a.tag in ('beam','notations','lyric')),len(n));n.insert(index,stem)
                stem.text=g['direction']
                for a in n.findall('beam'):n.remove(a)
                for layer in range(1,(g.get('beams') or 0)+1):
                    previous=j>0 and pairs[j-1][lane].get('beam_chain')==g.get('beam_chain')
                    following=j+1<len(pairs) and pairs[j+1][lane].get('beam_chain')==g.get('beam_chain')
                    beam=E.Element('beam',number=str(layer));beam.text='continue' if previous and following else 'end' if previous else 'begin'
                    index=next((k for k,a in enumerate(n) if a.tag in ('notations','lyric')),len(n));n.insert(index,beam)
                desired[n]=int(onset*division);new.append(n)
            if not valid:break
        if not valid or len(used)!=len(pitched):m[:]=saved;continue
        for r in (top_rests[0],bass_rests[0]):desired[r['node']]=int(clock*division);new.append(r['node'])
        m[:]=list(m)[:indices[0]]+new+list(m)[indices[-1]+1:]
        if not _rebuild(m,desired):m[:]=saved;continue
        identity=lambda rs:Counter((r['staff'],r['midi'],r['duration']) for r in rs if r['midi'] is not None)
        if identity(pitched)-identity(records(root)[i]):raise ValueError('Opposed lanes lost an existing pitch/duration')
        changes.append({'measure':i+1,'method':'opposed_ink_lanes_missing_paired_column_explicit_multiscale_glyphs','added_notes':2,'beam_edges':beam_edges,'existing_pitches_durations_retained':True,'voices':[primary,secondary]})
    return changes
