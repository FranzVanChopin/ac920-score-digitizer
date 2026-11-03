"""Bounded low-octave pitch fusion using photographed chord/stem evidence.

No reference events, renderer, camera transform or training labels. Only an
ordered single voice is considered. Partial-measure extension also requires
the target group's onset and duration to agree. Every change needs an unchanged
octave partner and a continuous ink stem.
"""
import hashlib
from collections import Counter
from fractions import Fraction
from pathlib import Path
import cv2,numpy as np
from omr_event_records import records
from align_homr_geometry import _xml_systems,_position
from .testset3_staff_resize_fast import resize_staves
from .symbol_cnn import FACTOR


def midi(p):return (p[1]+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[p[0]]+p[2]


def stem_evidence(lower,upper,stem,image,space):
    if not stem or stem.get('confidence',0)<.9:return False
    lx,ly=lower['xy'];ux,uy=upper['xy']
    if abs(lx-ux)>.28*space or abs(2*(ly-uy)/space-7)>.4:return False
    sx=float(np.mean(np.array(stem['xy'])[:,0]));ys=np.array(stem['xy'])[:,1]
    if not .25*space<sx-lx<1.1*space or ys.min()>uy+.3*space or ys.max()<ly-.3*space:return False
    xx=np.arange(round(sx)-2,round(sx)+3);yy=np.arange(round(uy)+2,round(ly)-1)
    if not len(yy) or xx.min()<0 or yy.min()<0 or xx.max()>=image.shape[1] or yy.max()>=image.shape[0]:return False
    if ((image[yy[:,None],xx[None,:]]<140).max(1)).mean()<.86:return False
    # Thin ledger lines cannot pass a two-dimensional head-core test.
    for h in (lower,upper):
        x,y=map(round,h['xy']);rx=max(2,round(.35*space));ry=max(2,round(.18*space))
        if x-rx<0 or y-ry<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
        if (image[y-ry:y+ry+1,x-rx:x+rx+1]<140).mean()<.82:return False
    return True


def fuse_low_octaves(root,graph,photo,staff_geometry,partial_measures=False,independent_partner=False):
    parts=root.findall('part')
    if len(parts)!=1 or root.find('.//octave-shift') is not None:return []
    clefs=root.findall('part/measure/attributes/clef')
    if not clefs or any(c.findtext('sign')!='F' or c.findtext('line')!='4' or c.findtext('clef-octave-change','0')!='0' for c in clefs):return []
    if graph.get('image_sha256')!=hashlib.sha256(Path(photo).read_bytes()).hexdigest() or staff_geometry.get('image_sha256')!=graph['image_sha256']:raise ValueError('Pitch fusion photo mismatch')
    systems=graph['systems']
    if len(systems)>64 or any(len(r['groups'])>2048 or len(r['bounds'])>129 for r in systems):return []
    if any(sum(r['system']==s['system'] for r in systems)!=1 for s in systems):return []
    rows=records(root);measures=parts[0].findall('measure')
    if sum(len(r['bounds'])-1 for r in systems)!=len(rows):return []
    if independent_partner:
        legacy=fuse_low_octaves(root,graph,photo,staff_geometry,partial_measures=partial_measures)
    else:
        legacy=fuse_low_octaves(root,graph,photo,staff_geometry,partial_measures=False) if partial_measures else []
    if partial_measures or independent_partner:rows=records(root)
    _xml_systems(root);staffs=resize_staves(staff_geometry['staves'],FACTOR,FACTOR)
    stems={s['id']:s for s in graph.get('stems',[])};jobs=[];offset=0
    for row in sorted(systems,key=lambda r:r['system']):
        staff=next((s for s in staffs if s['system']==row['system']),None)
        if staff is None:return []
        for mi,(left,right) in enumerate(zip(row['bounds'],row['bounds'][1:])):
            events=rows[offset+mi]
            if not events or len(events)>128 or len({(e['staff'],e['voice']) for e in events})!=1 or any(e['grace'] or e['node'].get('print-object')=='no' or e['node'].find('tie') is not None or e['node'].find('notations/tied') is not None for e in events):continue
            original=[]
            for e in events:
                if e['chord']:
                    if not original or e['onset']!=original[-1][0]['onset']:original=[];break
                    original[-1].append(e)
                else:original.append([e])
            other=sorted([g for g in row['groups'] if left<g['x']<right and not g.get('grace')],key=lambda g:g['x'])
            if not original or len(original)!=len(other) or len(original)>64:continue
            plans=[];valid=True
            for group,g in zip(original,other):
                if partial_measures and any(e['onset']!=Fraction(g.get('onset',-1),480)
                        or e['duration']!=Fraction(g.get('duration',0),480) for e in group):continue
                if g.get('rest'):
                    if len(group)!=1 or group[0]['midi'] is not None:
                        if partial_measures:continue
                        valid=False;break
                    continue
                hs=g.get('heads',[])
                if len(hs)!=len(group) or not all(e['midi'] is not None for e in group):
                    if partial_measures:continue
                    valid=False;break
                old=Counter(e['midi'] for e in group);new=Counter(midi(h['pitch']) for h in hs)
                if old==new:continue
                missing=new-old;extra=old-new
                if sum(missing.values())!=1 or sum(extra.values())!=1 or not 2<=len(hs)<=8:
                    if partial_measures:continue
                    valid=False;break
                target=next(iter(missing));wrong=next(iter(extra))
                low=next(h for h in hs if midi(h['pitch'])==target)
                node=next(e['node'] for e in group if e['midi']==wrong)
                pos=low['pitch'][1]*7+'CDEFGAB'.index(low['pitch'][0])-18+1
                if not -15<=pos<=-7 or not 1<=abs(pos-_position(node)[1])<=4 or low['pitch'][3] is not None or node.find('accidental') is not None:continue
                partners=[h for h in hs if h is not low and tuple(h['pitch'][:3])==(low['pitch'][0],low['pitch'][1]+1,low['pitch'][2]) and midi(h['pitch']) in old and h['pitch'][3] is None]
                if len(partners)!=1 or any(h['class']!=0 for h in (low,partners[0])):continue
                # A retained production note supplies independent identity for
                # the anchor. Validate its five-line position and actual ink,
                # rather than requiring its detector score to equal the target's.
                # The changed lower head still needs the original high score.
                if low.get('confidence',0)<.92 or partners[0].get('confidence',0)<(.8 if independent_partner else .92):continue
                anchor=next(e['node'] for e in group if e['midi']==midi(partners[0]['pitch']))
                if anchor.find('accidental') is not None or anchor.findtext('duration')!=node.findtext('duration') or anchor.findtext('type')!=node.findtext('type'):continue
                # Require the graph's quantised target to agree with the actual
                # five-line track before trusting the spelling/context.
                xs,ys=np.array(staff['lines'][-1]['polyline']).T;bottom=float(np.interp(low['xy'][0],xs,ys))
                if abs(1+2*(bottom-low['xy'][1])/row['space_px']-pos)>.3:continue
                if independent_partner:
                    upper=partners[0];upper_bottom=float(np.interp(upper['xy'][0],xs,ys))
                    if abs(1+2*(upper_bottom-upper['xy'][1])/row['space_px']-(pos+7))>.3:continue
                    # New anchor evidence is valid only with an unchanged time
                    # and an ordered, same-cardinality physical chord.
                    if any(e['onset']!=Fraction(g.get('onset',-1),480) or e['duration']!=Fraction(g['duration'],480) for e in group):continue
                plans.append((node,anchor,low,partners[0],stems.get(g.get('stem')),row['space_px'],offset+mi))
            if valid:jobs.extend(plans)
        offset+=len(row['bounds'])-1
    if not jobs or len(jobs)>256:return legacy
    raw=cv2.imread(str(photo),cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.size>16000000 or max(raw.shape)>8192:return []
    image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA);del raw
    changes=list(legacy)
    for node,anchor,low,upper,stem,space,mi in jobs:
        if not stem_evidence(low,upper,stem,image,space):continue
        old=node.findtext('pitch/step')+node.findtext('pitch/octave');p=node.find('pitch');source=anchor.find('pitch')
        p.find('step').text=source.findtext('step');p.find('octave').text=str(int(source.findtext('octave'))-1)
        alter=p.find('alter');value=source.findtext('alter','0')
        if alter is None and value!='0':
            import xml.etree.ElementTree as E
            alter=E.Element('alter');p.insert(list(p).index(p.find('step'))+1,alter)
        if alter is not None:alter.text=value
        changes.append({'measure':mi+1,'method':'independent_chord_anchor_with_two_staff_positions_and_ink' if independent_partner else 'ordered_low_octave_with_unchanged_partner_and_ink_stem','old_pitch':old,'new_pitch':p.findtext('step')+p.findtext('octave'),'alter':int(value),'xy':low['xy'],
            'anchor_detector_confidence':upper['confidence'],'target_detector_confidence':low['confidence']})
    return changes
