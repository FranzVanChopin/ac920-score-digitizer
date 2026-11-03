"""Restore only image-supported small noteheads before relation decoding."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
from .runtime_inference import probabilities
from .relation_grace import patch,head_width
from .relation_decode import staff_distance
from .symbol_cnn import FACTOR


def small_stem(image,xy,space):
    x,y=xy;ink=image<155;found=[]
    for direction,sign in (('up',-1),('down',1)):
        for col in range(round(x+.25*space),round(x+.65*space)+1):
            # Up stems attach to the right; down stems attach to the left.
            if direction=='down':col=round(2*x-col)
            if not 0<=col<image.shape[1]:continue
            start=round(y+sign*.15*space);last=start;blanks=0
            for d in range(round(4.5*space)):
                yy=start+sign*d
                if not 0<=yy<image.shape[0]:break
                if ink[yy,col]:last=yy;blanks=0
                else:
                    blanks+=1
                    if blanks>=2:break
            length=abs(last-start)/space
            if length>=1.2:found.append((length,col,last,direction))
    if not found:return None
    length,x,tip,direction=max(found)
    if length>3.:return None
    return {'length_spaces':length,'x':x,'tip':tip,'direction':direction}


def core_ink(image,xy,space):
    x,y=xy;t=max(0,round(y-.12*space));b=min(image.shape[0],round(y+.12*space)+1)
    a=max(0,round(x-.22*space));z=min(image.shape[1],round(x+.22*space)+1)
    crop=image[t:b,a:z]
    return float((crop<155).mean()) if crop.size else 0.


def rescue(groups,nodes,image,staff,bounds,proposals,model,threshold):
    """The raw small-stem witness rejects text/rest-shaped head proposals."""
    space=staff['space_px'];existing=[h for g in groups for h in g.get('heads',[])];candidates=[]
    for c,x,y,p in proposals:
        if not staff['lines'][0]['start'][0]<x<staff['lines'][0]['end'][0] or staff_distance(staff,x,y)>12:continue
        if any(np.linalg.norm(np.array(h['xy'])-[x,y])<6 for h in existing):continue
        candidates.append({'id':len(nodes)+len(candidates),'class':0,'xy':[x,y],'confidence':p})
    # Recover oversized row spans on existing filled grace heads. Quantile
    # suppresses a few merged staff/ledger rows; it never overrides the CNN.
    all_heads=existing+candidates
    if not all_heads:return []
    prob=np.concatenate([probabilities(model,np.stack([patch(image,h['xy'],space) for h in all_heads[i:i+128]])).flatten() for i in range(0,len(all_heads),128)])
    for h,p in zip(all_heads,prob):
        h['refined_grace_probability']=float(p);h['core_width_spaces']=head_width(image,h['xy'],space,.5)
    for g in groups:
        hs=g.get('heads',[])
        if hs and any(h['class']!=0 for h in hs):g['grace']=False
        elif hs and all(h['refined_grace_probability']>=max(threshold,.99) and .65<h['core_width_spaces']<1.12 for h in hs):g['grace']=True
    changes=[]
    for h in candidates:
        if h['refined_grace_probability']<max(threshold,.99) or not .65<h['core_width_spaces']<1.12 or core_ink(image,h['xy'],space)<.75:continue
        witness=small_stem(image,h['xy'],space)
        if witness is None:continue
        x,y=h['xy'];interval=next(((a,b) for a,b in zip(bounds,bounds[1:]) if a<x<b),None)
        if interval is None:continue
        principals=[g for g in groups if g.get('heads') and not g.get('grace') and x+.45*space<g['x']<min(x+5*space,interval[1])]
        if not principals:continue
        g={'heads':[h],'stem':None,'direction':witness['direction'],'tip':witness['tip'],'beams':0,'flags':1,'duration':240,'x':x,'y':y,'grace':True,'rescue_evidence':{'core_ink':core_ink(image,h['xy'],space),'small_stem':witness,'source':'low_threshold_image_proposal'}}
        nodes.append(h);groups.append(g);changes.append({'xy':h['xy'],'probability':h['refined_grace_probability'],'evidence':g['rescue_evidence']})
    groups.sort(key=lambda g:g['x'])
    return changes


def read_proposals(path,photo):
    payload=json.loads(Path(path).read_text(encoding='utf8'))
    if payload['image_sha256']!=hashlib.sha256(Path(photo).read_bytes()).hexdigest() or payload['coordinate_scale']!=FACTOR:raise ValueError('Grace proposals belong to another photograph or coordinate scale')
    return payload['predictions']


def join_grace_chords(groups,stems,space):
    """A short stem at the top head may serve a taller small-head chord.

    Require the photographed stem tip, side and x coordinate to agree. A
    normal-sized or opposing-stem head in the column makes it ambiguous.
    This is applied before clocks are decoded, never as a meter-fill rule.
    """
    changes=[];removed=set()
    for anchor in list(groups):
        witness=anchor.get('rescue_evidence',{}).get('small_stem')
        if not anchor.get('grace') or not witness or id(anchor) in removed:continue
        neighbours=[g for g in groups if id(g) not in removed and g is not anchor and
                    abs(g['x']-anchor['x'])<.35*space and g.get('heads')]
        if not neighbours:continue
        column=[anchor]+neighbours;heads=[h for g in column for h in g['heads']]
        if (len(heads)>8 or max(h['xy'][1] for h in heads)-min(h['xy'][1] for h in heads)>4*space or
            any(h['class']!=0 or not .65<h.get('core_width_spaces',0)<1.02 for h in heads)):continue
        valid=True
        for g in neighbours:
            if g.get('direction')!=witness['direction']:valid=False;break
            if g.get('grace'):
                other=g.get('rescue_evidence',{}).get('small_stem')
                if not other or abs(other['x']-witness['x'])>.2*space or abs(other['tip']-witness['tip'])>.4*space:valid=False;break
            else:
                index=g.get('stem')
                if index is None or not 0<=index<len(stems):valid=False;break
                xy=stems[index]['xy'];sx=sum(p[0] for p in xy)/2
                if abs(sx-witness['x'])>.2*space or abs(g['tip']-witness['tip'])>.4*space:valid=False;break
        if not valid:continue
        anchor['heads']=sorted(heads,key=lambda h:-h['xy'][1])
        anchor['grace_chord_witness']={'method':'small_column_shared_short_stem_tip',
                                      'stem_x':witness['x'],'tip':witness['tip'],'head_count':len(heads)}
        for g in neighbours:removed.add(id(g))
        changes.append({'xy':[anchor['x'],anchor['y']],**anchor['grace_chord_witness']})
    groups[:]=[g for g in groups if id(g) not in removed]
    return changes
