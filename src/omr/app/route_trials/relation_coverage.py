"""Bounded physical note coverage and printed-key context refinements.

Only photograph, image-predicted graph/tracks and recognized XML are inputs.
Reference scores and engraving labels are accepted by separate diagnostics.
"""
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import hashlib,copy
import xml.etree.ElementTree as E
import cv2,numpy as np
from align_homr_geometry import _xml_systems,_position,_base
from omr_event_records import records
from repair_pitch_context import no_accidental
from .symbol_cnn import FACTOR
from .staff_preprocess import staff_step
from .testset3_staff_resize_fast import resize_staves
from .relation_context import pitch_for_step
from .relation_sequence import forced_pairs
from .relation_accidentals import associations
from .relation_page_inputs import photo_digest, scaled_photo


def inputs(root,graph,photo,geometry):
    if len(root.findall('part'))!=1 or root.find('.//octave-shift') is not None:return None
    digest=photo_digest(photo)
    if graph.get('image_sha256')!=digest or geometry.get('image_sha256')!=digest:raise ValueError('Coverage refinement photo mismatch')
    systems=graph['systems'];ids=sorted({r['system'] for r in systems});ms=root.findall('part/measure')
    if len(ids)>64 or any(len(r['groups'])>2048 or len(r.get('nodes',[]))>2048 or len(r['bounds'])>129 for r in systems):return None
    if sum(len(next(r for r in systems if r['system']==s)['bounds'])-1 for s in ids)!=len(ms):return None
    image=scaled_photo(photo,FACTOR,FACTOR)
    if image is None:return None
    _xml_systems(root)
    return image,resize_staves(geometry['staves'],FACTOR,FACTOR)


def core(head,image,space,minimum=.85):
    x,y=map(round,head['xy']);rx=max(2,round(.3*space));ry=max(2,round(.15*space))
    if min(x-rx,y-ry)<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
    return bool((image[y-ry:y+ry+1,x-rx:x+rx+1]<140).mean()>=minimum)


def empty_strip(h,row,track,group,stems,image,ink,all_heads):
    """Negative glyph evidence with only independently ink-proven line masks.

    A photographed bar line or the current note's own stem can enter the
    accidental search window. Remove at most a thin vertical column, after
    testing its uninterrupted ink over the full staff/stem height. Accidental
    nodes in this window always veto the mask; curves and other ink remain.
    """
    space=row['space_px'];head=dict(x=h['xy'][0],y=h['xy'][1],staff_space=space)
    if no_accidental(head,all_heads,ink):return True
    x,y=h['xy'];left=round(x-2.3*space);right=round(x-.65*space);top=round(y-1.3*space);bottom=round(y+1.3*space)
    if min(left,top)<0 or bottom>image.shape[0] or right>image.shape[1]:return False
    if any(n['class'] in (8,9,10) and left-.2*space<n['xy'][0]<right+.2*space and abs(n['xy'][1]-y)<2*space for n in row['nodes']):return False
    candidates=[]
    for bound in row['bounds']:
        if not left<bound<right:continue
        ys=[float(np.interp(bound,*np.array(line['polyline']).T)) for line in (track['lines'][0],track['lines'][-1])]
        candidates.append((bound,round(min(ys)),round(max(ys))))
    stem=stems.get(group.get('stem'))
    if stem and stem.get('confidence',0)>=.9:
        xy=np.array(stem['xy']);sx=float(xy[:,0].mean());sy=xy[:,1]
        if abs(sx-x)<.9*space and sy.min()<=y+.25*space and sy.max()>=y-.25*space and sy.max()-sy.min()>2.5*space and right-.3*space<sx<right+.3*space:
            candidates.append((sx,round(sy.min()+.2*space),round(sy.max()-.2*space)))
    if not candidates:return False
    # Work in a bounded local window; copying a full image per note is avoided.
    margin=round(3*space);x0=max(0,left-margin);y0=max(0,top-margin);x1=min(image.shape[1],right+margin);y1=min(image.shape[0],bottom+margin);local=ink[y0:y1,x0:x1].copy();masked=False
    for sx,a,b in candidates:
        if a<0 or b>=image.shape[0] or b-a<2.5*space:continue
        cols=np.arange(round(sx)-2,round(sx)+3)
        if cols.min()<0 or cols.max()>=image.shape[1]:continue
        hits=(image[a:b,cols]<140).mean(axis=0);best=int(cols[int(np.argmax(hits))])
        if max(hits)<.9:continue
        lo=best-1;hi=best+2
        if lo<x0 or hi>x1:continue
        local[:,lo-x0:hi-x0]=0;masked=True
    if not masked:return False
    local_heads=[dict(v,x=v['x']-x0,y=v['y']-y0) for v in all_heads]
    return no_accidental(dict(head,x=x-x0,y=y-y0),local_heads,local)


def hollow_core(h,image,space):
    x,y=map(round,h['xy'])
    if min(x,y)<space or x+space>=image.shape[1] or y+space>=image.shape[0]:return False
    # A light centre plus dark opposing wings distinguishes hollow heads from
    # filled heads/ledger strokes. The stem is independently checked by caller.
    if (image[y-1:y+2,x-1:x+2]>=140).mean()<.65:return False
    wings=[]
    for sign in (-1,1):
        xx=round(x+sign*.42*space);yy=round(y-sign*.10*space)
        wings.append((image[yy-2:yy+3,xx-2:xx+3]<140).mean())
    return min(wings)>=.65


def rest_proof(rest,track,image,space):
    x,y=rest['xy'];line=float(np.interp(x,*np.array(track['lines'][1]['polyline']).T))
    return rest.get('confidence',0)>=.8 and .03*space<y-line<.4*space and core(rest,image,space,.9)


def rhythm_proof(seq,row,image,ink):
    from repair_tuplets import continuous_beams
    space=row['space_px']
    for i,g in enumerate(seq):
        h=g['heads'][0];dur=g['duration']
        if g.get('flags') or g.get('tuplet_ratio'):return False
        if dur==240 and h['class']==0 and g.get('beams')==1 and not g.get('dotted'):
            neighbors=[v for j,v in enumerate(seq) if abs(j-i)==1 and v.get('duration')==240 and v.get('beams')==1 and v.get('beam_chain')==g.get('beam_chain') and g.get('beam_chain') is not None and v.get('direction')==g.get('direction')]
            def proven(v):
                a,b=sorted((h,v['heads'][0]),key=lambda head:head['xy'][0])
                return continuous_beams(dict(x=a['xy'][0],y=a['xy'][1],staff_space=space),dict(x=b['xy'][0],y=b['xy'][1],staff_space=space),ink,g.get('direction')=='up',1)
            if not neighbors or not any(proven(v) for v in neighbors):return False
        elif dur==480 and h['class']==0 and not g.get('beams') and not g.get('dotted'):pass
        elif dur in (960,1440) and h['class']==1 and not g.get('beams'):
            if bool(g.get('dotted'))!=(dur==1440):return False
            if dur==1440:
                dots=[n for n in row['nodes'] if n['class']==11 and n.get('confidence',0)>=.85 and .7*space<n['xy'][0]-h['xy'][0]<2*space and abs(n['xy'][1]-h['xy'][1])<.75*space]
                if len(dots)!=1:return False
                x,y=map(round,dots[0]['xy'])
                if min(x,y)<2 or x+2>=image.shape[1] or y+2>=image.shape[0] or (image[y-1:y+2,x-1:x+2]<140).mean()<.8:return False
        else:return False
    return True


def signature(row,track,clef,fifths):
    if not clef or not 1<=abs(fifths)<=7:return False
    first=min((g['x'] for g in row['groups'] if g.get('stem') is not None and any(h.get('confidence',0)>=.8 for h in g.get('heads',[]))),default=None)
    if first is None:return False
    cls=9 if fifths<0 else 8;space=row['space_px'];count=abs(fifths)
    glyphs=sorted([n for n in row['nodes'] if n['class']==cls and row['bounds'][0]<n['xy'][0]<first-.65*space],key=lambda n:n['xy'][0])
    if len(glyphs)!=count or any(n.get('confidence',0)<.8 for n in glyphs):return False
    expected=('BEADGCF' if fifths<0 else 'FCGDAEB')[:count]
    for i,(n,letter) in enumerate(zip(glyphs,expected)):
        if i and not .6*space<n['xy'][0]-glyphs[i-1]['xy'][0]<1.5*space:return False
        p=staff_step(track,n['xy'][0],n['xy'][1]+(.75*space if cls==9 else 0))
        if abs(p['residual'])>.3 or pitch_for_step(p['nearest_step'],clef)[0]!=letter:return False
    return True


def context_rows(root):
    key=None;clefs={};rows=[]
    for m in root.findall('part/measure'):
        for a in m.findall('attributes'):
            if a.find('key/fifths') is not None:key=int(a.findtext('key/fifths'))
            for c in a.findall('clef'):
                sn=int(c.get('number','1'));sign=c.findtext('sign');line=int(c.findtext('line','0'));octave=int(c.findtext('clef-octave-change','0'))
                clefs[sn]={'sign':sign,'line':line,'octave':octave} if (sign,line) in (('G',2),('F',4)) and octave==0 else None
        rows.append((key,copy.deepcopy(clefs)))
    return rows


def ordered(events):
    groups=[]
    for r in events:
        if r['chord']:
            if not groups or r['onset']!=groups[-1][0]['onset']:return []
            groups[-1].append(r)
        else:groups.append([r])
    return groups if len(groups)<=64 and all(len(g)<=8 for g in groups) else []


def spelling(group):
    return tuple(sorted((r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))) for r in group)) if all(r['midi'] is not None for r in group) else ((None,0),)


def detected(group):
    return tuple(sorted((h['pitch'][0],h['pitch'][1]) for h in group.get('heads',[]))) if group.get('heads') else ((None,0),)


def safe_measure(events):
    return bool(events and len(events)<=128 and not any(r['grace'] or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in events))


def fuse_key_context(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];events=records(root);ms=root.findall('part/measure');ctx=context_rows(root);changes=[];offset=0;stems={s['id']:s for s in graph.get('stems',[])}
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in selected:
            sn=row['staff'];track=staffs[sn-1] if len(staffs)>=sn else None
            if track is None:continue
            first_key,first_clefs=ctx[offset]
            if first_key is None or not signature(row,track,first_clefs.get(sn),first_key):continue
            all_heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for g in row['groups'] for h in g.get('heads',[])];attached=associations(row)
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;key,clefs=ctx[number];m=ms[number];es=[r for r in events[number] if r['staff']==sn]
                if key!=first_key or clefs.get(sn)!=first_clefs.get(sn) or not safe_measure(es) or len({r['voice'] for r in es})!=1:continue
                seen=False;inline=False
                for child in m:
                    if child.tag=='note':seen=True
                    elif seen and child.tag=='attributes':inline=True
                if inline:continue
                original=ordered(es);other=sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x'])
                if not original or len(other)>64 or any(g.get('grace') for g in other):continue
                pairs=forced_pairs([spelling(g) for g in original],[detected(g) for g in other]);state={(letter,octave):(-1 if key<0 else 1) for letter in ('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)] for octave in range(10)};previous=(-1,-1)
                for ni,hi in pairs:
                    if ni!=previous[0]+1 or hi!=previous[1]+1:state.clear()
                    previous=(ni,hi);group,g=original[ni],other[hi]
                    if g.get('rest'):continue
                    if any(r['onset']!=Fraction(g.get('onset',-1),480) or r['duration']!=Fraction(g['duration'],480) for r in group):state.clear();continue
                    nodes=sorted(group,key=lambda r:(r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))));heads=sorted(g['heads'],key=lambda h:(h['pitch'][0],h['pitch'][1]))
                    for r,h in zip(nodes,heads):
                        n=r['node'];p=n.find('pitch');identity=(p.findtext('step'),int(p.findtext('octave')));old=int(p.findtext('alter','0'));wanted=state.get(identity);acc=attached.get(h['id']);absent=empty_strip(h,row,track,g,stems,image,ink,all_heads)
                        if acc or n.find('accidental') is not None or not absent:
                            state[identity]=old if acc and old=={8:1,9:-1,10:0}[acc['class']] else None;continue
                        if wanted is None or old==wanted or h['pitch'][2]!=wanted or h['pitch'][3] is not None:continue
                        if h.get('class')!=0 or h.get('confidence',0)<.85 or not core(h,image,row['space_px']):continue
                        pos=_position(n);step=staff_step(track,*h['xy'])
                        if pos is None or abs(step['half_space_steps']+1-pos[1])>.3:continue
                        alter=p.find('alter')
                        if alter is None:alter=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,alter)
                        alter.text=str(wanted);changes.append({'measure':number+1,'staff':sn,'pitch':identity,'old_alter':old,'new_alter':wanted,'method':'printed_key_and_empty_strip_conserved_bar_state','xy':h['xy']})
        offset+=len(selected[0]['bounds'])-1
    return changes


def fuse_written_low_octaves(root,graph,photo,geometry):
    """Correct one written low position using an unchanged written octave mate.

    Existing alterations are copied from that mate. Printed-key reconciliation
    is a separate step, so a spelling mismatch cannot guess a semitone value.
    """
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;events=records(root);ctx=context_rows(root);changes=[];stems={s['id']:s for s in graph.get('stems',[])};offset=0
    from .relation_pitch import stem_evidence
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=[r for r in graph['systems'] if r['system']==system];staffs=[s for s in tracks if s['system']==system]
        for row in selected:
            sn=row['staff'];track=staffs[sn-1] if len(staffs)>=sn else None
            if track is None:continue
            for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
                number=offset+mi;key,clefs=ctx[number]
                if clefs.get(sn)!={'sign':'F','line':4,'octave':0}:continue
                es=[r for r in events[number] if r['staff']==sn]
                if not safe_measure(es) or len({r['voice'] for r in es})!=1:continue
                original=ordered(es);other=sorted([g for g in row['groups'] if left<g['x']<right and not g.get('grace')],key=lambda g:g['x'])
                if not original or len(original)!=len(other) or len(original)>64:continue
                for group,g in zip(original,other):
                    if len(group)!=len(g.get('heads',[])) or not 2<=len(group)<=8 or any(r['midi'] is None for r in group):continue
                    if any(r['onset']!=Fraction(g.get('onset',-1),480) or r['duration']!=Fraction(g['duration'],480) for r in group):continue
                    old=Counter((r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))) for r in group);new=Counter((h['pitch'][0],h['pitch'][1]) for h in g['heads']);missing=new-old;extra=old-new
                    if sum(missing.values())!=1 or sum(extra.values())!=1:continue
                    target=next(iter(missing));wrong=next(iter(extra));h=next(h for h in g['heads'] if tuple(h['pitch'][:2])==target);n=next(r['node'] for r in group if (r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave')))==wrong)
                    pos=target[1]*7+'CDEFGAB'.index(target[0])-18+1
                    if not -15<=pos<=-7 or not 1<=abs(pos-_position(n)[1])<=4 or h['pitch'][3] is not None or n.find('accidental') is not None:continue
                    partners=[a for a in g['heads'] if tuple(a['pitch'][:2])==(target[0],target[1]+1) and (target[0],target[1]+1) in old and a['pitch'][3] is None]
                    if len(partners)!=1 or h.get('class')!=0 or h.get('confidence',0)<.92 or partners[0].get('class')!=0 or partners[0].get('confidence',0)<.8:continue
                    a=partners[0];anchor=next(r['node'] for r in group if (r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave')))==(target[0],target[1]+1))
                    if anchor.find('accidental') is not None or anchor.findtext('type')!=n.findtext('type') or anchor.findtext('duration')!=n.findtext('duration'):continue
                    # The retained upper head calibrates the local ledger
                    # scale: five-line scale drift accumulates far below staff.
                    # Both centres must still form an octave within .4 steps.
                    if abs(staff_step(track,*a['xy'])['half_space_steps']+1-(pos+7))>.3 or abs(2*(h['xy'][1]-a['xy'][1])/row['space_px']-7)>.4:continue
                    if not stem_evidence(h,a,stems.get(g.get('stem')),image,row['space_px']):continue
                    p=n.find('pitch');p.find('step').text=target[0];p.find('octave').text=str(target[1]);alter=p.find('alter');value=anchor.findtext('pitch/alter','0')
                    if alter is None and value!='0':alter=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,alter)
                    if alter is not None:alter.text=value
                    changes.append({'measure':number+1,'staff':sn,'old_written':wrong,'new_written':target,'alter_from_unchanged_octave_partner':int(value),'method':'written_low_position_two_heads_and_unchanged_octave_partner','xy':h['xy']})
        offset+=len(selected[0]['bounds'])-1
    return changes


def fuse_isolated_melody(root,graph,photo,geometry,physical_geometry=False,rest_conflicts=False):
    """Recover an ink-proven simple melody beside a retained full-bar rest.

    Existing pitched spelling must have no physical counterpart in the new
    lane at the same clock. Mixed voices, tuplets and uncertain glyphs abstain.
    """
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return []
    image,tracks=loaded;ink=cv2.threshold(image,140,255,cv2.THRESH_BINARY_INV)[1];events=records(root);ctx=context_rows(root);ms=root.findall('part/measure');stems={s['id']:s for s in graph.get('stems',[])};offset=0;jobs=[];div=1;divs=[]
    for m in ms:
        for a in m.findall('attributes'):div=int(a.findtext('divisions',str(div)))
        divs.append(div)
    from .relation_voice_partition import stem_proof
    from repair_grace_timing import _rebuild
    for system in sorted({r['system'] for r in graph['systems']}):
        selected=sorted([r for r in graph['systems'] if r['system']==system],key=lambda r:r['staff']);staffs=[s for s in tracks if s['system']==system]
        if len(selected)!=2 or len(staffs)!=2 or selected[0]['bounds']!=selected[1]['bounds']:
            offset+=len(selected[0]['bounds'])-1;continue
        for mi,(left,right) in enumerate(zip(selected[0]['bounds'],selected[0]['bounds'][1:])):
            number=offset+mi;all_events=events[number];m=ms[number]
            if not safe_measure(all_events) or m.get('implicit')=='yes' or {r['staff'] for r in all_events}!={row['staff'] for row in selected}:continue
            by={row['staff']:sorted([g for g in row['groups'] if left<g['x']<right],key=lambda g:g['x']) for row in selected}
            rests=[(row,by[row['staff']][0]) for row in selected if len(by[row['staff']])==1 and by[row['staff']][0].get('rest') and by[row['staff']][0]['rest'].get('class')==2 and rest_proof(by[row['staff']][0]['rest'],staffs[row['staff']-1],image,row['space_px'])]
            if len(rests)!=1:continue
            partner,rg=rests[0];other=next(row for row in selected if row is not partner);sn=other['staff'];seq=by[sn];track=staffs[sn-1] if len(staffs)>=sn else None;key,clefs=ctx[number]
            if track is None or clefs.get(sn)!={'sign':'G','line':2,'octave':0} or key is None or not signature(other,track,clefs.get(sn),key):continue
            preserved=[r for r in all_events if r['staff']==partner['staff']];original=[r for r in all_events if r['staff']==sn]
            if len(preserved)!=1 or preserved[0]['midi'] is not None or preserved[0]['onset']!=0 or preserved[0]['duration']!=Fraction(rg['duration'],480):continue
            expected=preserved[0]['duration']
            rest_conflict=rest_conflicts and len(original)==1 and original[0]['midi'] is None and original[0]['node'].find('rest').get('measure')=='yes' and original[0]['onset']==0 and original[0]['duration']==expected
            if not original or len(original)>8 or len({r['voice'] for r in original})!=1 or any((r['midi'] is None and not rest_conflict) or r['chord'] or len(r['node'].findall('notations/*')) or r['node'].find('lyric') is not None for r in original):continue
            expected=preserved[0]['duration']
            if sum(r['duration'] for r in original)!=expected or min(r['onset'] for r in original)!=0 or max(r['onset']+r['duration'] for r in original)!=expected:continue
            if not 1<=len(seq)<=8 or sum(g['duration'] for g in seq)!=expected*480 or len({g.get('voice',1) for g in seq})!=1:continue
            # Only replace a wholly unbound lane. If even one existing event
            # matches a predicted pitch and clock, keep the original measure.
            if any(r['midi'] is not None and tuple(h['pitch'][:2])==(r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))) and r['onset']==Fraction(g['onset'],480) and r['duration']==Fraction(g['duration'],480) for g in seq for h in g.get('heads',[]) for r in original):continue
            all_heads=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=other['space_px']) for g in other['groups'] for h in g.get('heads',[])];valid=True
            cursor=0
            for i,g in enumerate(seq):
                hs=g.get('heads',[])
                if g.get('grace') or g.get('rest') or len(hs)!=1 or g.get('onset')!=cursor:valid=False;break
                cursor+=g['duration']
                h=hs[0];p=staff_step(track,*h['xy']);letter,octave=pitch_for_step(p['nearest_step'],clefs[sn]);expected_alter=(-1 if key<0 else 1) if letter in ('BEADGCF' if key<0 else 'FCGDAEB')[:abs(key)] else 0
                if h.get('class') not in (0,1) or h.get('confidence',0)<.9 or h['pitch'][3] is not None or tuple(h['pitch'][:3])!=(letter,octave,expected_alter) or abs(p['residual'])>.3:valid=False;break
                physical=core(h,image,other['space_px']) if h['class']==0 else hollow_core(h,image,other['space_px']);shaft=stem_proof(g,stems,image,other['space_px'])
                if physical_geometry:
                    from .relation_measure_realign import hollow_on_line,traced_stem
                    physical=physical or hollow_on_line(h,other,track,clefs[sn],image);shaft=shaft or traced_stem(g,other,stems,image)
                if not physical or not shaft:valid=False;break
                if not empty_strip(h,other,track,g,stems,image,ink,all_heads):valid=False;break
                if i and g['x']-seq[i-1]['x']<1.4*other['space_px']:valid=False;break
            if valid and not rhythm_proof(seq,other,image,ink):valid=False
            indices=[i for i,n in enumerate(m) if n.tag=='note']
            if any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):valid=False
            if any(divs[number]*g['duration']%480 or divs[number]*g['onset']%480 for g in seq):valid=False
            if valid:jobs.append((number,sn,original,seq))
        offset+=len(selected[0]['bounds'])-1
    if len(jobs)>128:return []
    changes=[]
    for number,sn,original,seq in jobs:
        m=ms[number];saved=copy.deepcopy(list(m));div=divs[number];voice=original[0]['voice'];new=[];desired={r['node']:int(r['onset']*div) for r in events[number]}
        for i,g in enumerate(seq):
            h=g['heads'][0];letter,octave,alter=h['pitch'][:3];n=E.Element('note');p=E.SubElement(n,'pitch');E.SubElement(p,'step').text=letter
            if alter:E.SubElement(p,'alter').text=str(alter)
            E.SubElement(p,'octave').text=str(octave);E.SubElement(n,'duration').text=str(div*g['duration']//480);E.SubElement(n,'voice').text=voice;E.SubElement(n,'type').text={240:'eighth',480:'quarter',960:'half',1440:'half'}[g['duration']]
            if g.get('dotted'):E.SubElement(n,'dot')
            E.SubElement(n,'stem').text=g['direction'];E.SubElement(n,'staff').text=str(sn)
            if g['duration']==240:
                previous=i>0 and seq[i-1]['duration']==240 and seq[i-1].get('beam_chain')==g.get('beam_chain')
                following=i+1<len(seq) and seq[i+1]['duration']==240 and seq[i+1].get('beam_chain')==g.get('beam_chain')
                E.SubElement(n,'beam',number='1').text='continue' if previous and following else 'end' if previous else 'begin'
            # Keep recognized onset-bound notation at the same musical point.
            for r in original:
                if r['onset']==i:
                    for c in r['node']:
                        if c.tag in ('notations','lyric'):n.append(copy.deepcopy(c))
            desired[n]=g['onset']*div//480;new.append(n)
        selected_nodes={r['node'] for r in original};rebuilt=[];inserted=False
        for c in m:
            if c in selected_nodes:
                if not inserted:rebuilt.extend(new);inserted=True
            else:rebuilt.append(c)
        m[:]=rebuilt
        before_partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in events[number] if r['staff']!=sn)
        if not _rebuild(m,desired):m[:]=saved;continue
        after_events=records(root)[number]
        after_partner=Counter((r['staff'],r['voice'],r['onset'],r['duration'],r['midi'],r['grace']) for r in after_events if r['staff']!=sn)
        target_events=[r for r in after_events if r['staff']==sn]
        expected_clocks=[(Fraction(g['onset'],480),Fraction(g['duration'],480)) for g in seq]
        if before_partner!=after_partner or [(r['onset'],r['duration']) for r in target_events]!=expected_clocks or any(r['voice']!=voice for r in target_events):m[:]=saved;continue
        changes.append({'measure':number+1,'staff':sn,'old_events':len(original),'new_events':len(new),'method':'isolated_ink_melody_beside_unchanged_full_bar_rest','partner_rest_preserved':True,'voice_preserved':voice})
    return changes
