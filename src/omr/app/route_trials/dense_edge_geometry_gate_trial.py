"""Traditional positive head/stem proof; only ambiguous links query edge MLP.

This isolated experiment never reads a score or reference. It preserves the
original candidate window and learned decision for every uncertain head.
Geometric positives carry no invented model probability. The caller supplies
the ORIGINAL raster in exactly the head/stem coordinate frame.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np

MAX_PIXELS = 16_000_000
MAX_ITEMS = 4096
MAX_ROI = 65_536


def _stem_data(stems):
    result=[]
    for stem in stems:
        xy=np.asarray(stem['xy'])
        if xy.shape!=(2,2) or not np.isfinite(xy).all():
            raise ValueError('Finite two-endpoint stems required')
        sx=xy[:,0].mean();lo,hi=sorted(xy[:,1])
        points=xy[np.argsort(xy[:,1],kind='stable')]
        result.append((stem,sx,lo,hi,points))
    return result


def _proof(gray,head,stem,space):
    _,sx,lo,hi,points=stem
    hx,hy=head['xy'];delta=float(sx-hx)
    evidence=dict(stem_length_spaces=float((hi-lo)/space),rim_dx_spaces=abs(delta)/space,
                  endpoint_distance_spaces=float(min(abs(hy-lo),abs(hy-hi))/space))
    def reject(reason):return dict(accepted=False,reason=reason,**evidence)
    if head['class']!=0:
        return reject('hollow_head_requires_model')
    if abs(points[1,0]-points[0,0])>max(2,.08*(hi-lo)):
        return reject('shaft_angle_outside_vertical_stroke_contract')
    if not 2.2*space<=hi-lo<=5.2*space:
        return reject('stem_length_not_single_note_shaft')
    if not .35*space<=abs(delta)<=.8*space or min(abs(hy-lo),abs(hy-hi))>.55*space:
        return reject('head_is_not_unique_shaft_endpoint')
    upward=abs(hy-hi)<abs(hy-lo)
    if upward!=(delta>0):
        return reject('nonstandard_or_opposed_stem_corner')
    x0,x1=math.floor(hx-1.5*space),math.ceil(hx+1.5*space)+1
    y0,y1=math.floor(min(lo,hy-.7*space)),math.ceil(max(hi,hy+.7*space))+1
    if x0<0 or y0<0 or x1>gray.shape[1] or y1>gray.shape[0] or max(x1-x0,y1-y0)>512 or (x1-x0)*(y1-y0)>MAX_ROI:
        return reject('physical_proof_roi_clipped_or_over_budget')
    evidence['roi_bbox']=[x0,y0,x1,y1]
    ink=gray[y0:y1,x0:x1]<155
    cx,cy=int(round(hx))-x0,int(round(hy))-y0
    rx,ry=max(1,round(.2*space)),max(1,round(.12*space))
    core=float(ink[cy-ry:cy+ry+1,cx-rx:cx+rx+1].mean())
    evidence['filled_core_fraction']=core
    if core<.85:return reject('head_core_not_filled')
    widths=[]
    for y in range(round(hy-.35*space),round(hy+.35*space)+1):
        row=ink[y-y0]
        if not row[cx]:continue
        left=right=cx
        while left>0 and row[left-1]:left-=1
        while right+1<len(row) and row[right+1]:right+=1
        width=right-left+1
        # A full-width staff crossing supplies no head-shape evidence.
        if .65*space<=width<=1.55*space and abs((left+right)/2-cx)<=.25*space:
            widths.append(width)
    evidence['compact_head_rows']=len(widths)
    if len(widths)<max(3,round(.28*space)):
        return reject('no_independent_compact_head_body')
    # Exclude the head and opposite beam/flag endpoint when proving the shaft.
    a,b=(lo+.35*space,hy-.65*space) if upward else (hy+.65*space,hi-.35*space)
    if b-a<1.2*space:return reject('shaft_interior_too_short')
    ys=np.arange(math.ceil(a),math.floor(b)+1)
    xs=np.interp(ys,points[:,1],points[:,0])
    radius=max(1,round(.08*space))
    columns=np.rint(xs).astype(int)[:,None]+np.arange(-radius,radius+1)
    shaft=(gray[ys[:,None],columns]<155).any(axis=1)
    support=float(shaft.mean());evidence['continuous_shaft_row_fraction']=support
    starts=np.flatnonzero(np.diff(np.r_[False,~shaft,False].astype('int8'))==1)
    stops=np.flatnonzero(np.diff(np.r_[False,~shaft,False].astype('int8'))==-1)
    gap=max((int(z-y) for y,z in zip(starts,stops)),default=0)
    evidence['maximum_shaft_gap_px']=gap
    if support<.94 or gap>max(1,round(.12*space)):
        return reject('shaft_not_continuous_in_original_ink')
    # Actual short ink bridge, not just proximity of fitted boxes.
    near_y=np.arange(round(hy-.15*space),round(hy+.15*space)+1)
    end_x=float(np.interp(hy,points[:,1],points[:,0]))
    first,last=sorted((hx+math.copysign(.15*space,delta),end_x))
    bridge=gray[near_y[:,None],np.arange(math.ceil(first),math.floor(last)+1)[None,:]]<155
    contact=float(bridge.mean(axis=1).max()) if bridge.shape[1] else 0.
    evidence['head_to_shaft_bridge_fraction']=contact
    if contact<.95:return reject('no_original_ink_endpoint_contact')
    return dict(accepted=True,reason='unique_filled_head_rim_continuous_ink_shaft_endpoint',**evidence)


def propose_link_gates(gray,nodes,stems,space=16.):
    """Input-only proposal; save before any baseline edge-model comparison."""
    if not isinstance(gray,np.ndarray) or gray.dtype!=np.uint8 or gray.ndim!=2 or gray.size>MAX_PIXELS or max(gray.shape)>8192:
        raise ValueError('Bounded grayscale raster required')
    if not math.isfinite(space) or not 4<=space<=64 or len(nodes)>MAX_ITEMS or len(stems)>MAX_ITEMS:
        raise ValueError('Node/stem/spacing budget exceeded')
    if len({s['id'] for s in stems})!=len(stems) or len({n['id'] for n in nodes})!=len(nodes):
        raise ValueError('Unique node and stem IDs required')
    stem_data=_stem_data(stems);rows=[]
    for head in nodes:
        if head['class']>=2:continue
        hx,hy=head['xy']
        if not np.isfinite([hx,hy]).all():raise ValueError('Finite head position required')
        eligible=[stem for stem in stem_data if abs(stem[1]-hx)<=1.4*space and stem[2]-.35*space<hy<stem[3]+.35*space]
        if not eligible:proof=dict(accepted=False,reason='no_original_window_stem_candidates')
        elif len(eligible)>1:proof=dict(accepted=False,reason='multiple_original_window_stems')
        else:proof=_proof(gray,head,eligible[0],space)
        rows.append(dict(head=head['id'],head_xy=list(head['xy']),head_class=int(head['class']),
                         candidates=[s[0]['id'] for s in eligible],proof=proof))
    identity=json.dumps(dict(nodes=nodes,stems=stems,space=space),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(schema='physical_endpoint_edge_gate_v1',coordinate_frame='supplied_original_raster_pixels',
                input_shape=list(gray.shape),input_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
                geometry_sha256=hashlib.sha256(identity).hexdigest(),space=space,heads=rows,
                reference_used=False,model_calls=0,geometric_scores_are_not_calibrated_probabilities=True)


def resolve_link_gates(proposal,nodes,stems,net,threshold,*,probability_fn=None):
    """Only unproved candidates call the original edge probability function."""
    from .relation_net import edge_features
    if probability_fn is None:
        from .runtime_inference import probabilities as probability_fn
    identity=json.dumps(dict(nodes=nodes,stems=stems,space=proposal['space']),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(identity).hexdigest()!=proposal['geometry_sha256']:
        raise ValueError('Proposal belongs to different geometry')
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Finite threshold required')
    by_head={n['id']:n for n in nodes};by_stem={s['id']:s for s in stems};links=[];decisions=[]
    stats=dict(heads=len(proposal['heads']),original_candidate_queries=0,model_calls=0,
               skipped_model_queries=0,geometry_links=0,model_links=0,model_s=0.)
    for row in proposal['heads']:
        ids=row['candidates'];stats['original_candidate_queries']+=len(ids)
        record=dict(row,model_queries=[])
        if row['proof']['accepted']:
            if len(ids)!=1:raise ValueError('Geometric proof must own one original-window stem')
            links.append(dict(head=row['head'],stem=ids[0],confidence=None,
                              confidence_calibrated=False,source='traditional_ink_endpoint_proof'))
            stats['skipped_model_queries']+=1;stats['geometry_links']+=1
        else:
            candidates=[]
            for stem_id in ids:
                features=edge_features(by_head[row['head']]['xy'],by_stem[stem_id]['xy'],proposal['space'])
                tick=time.perf_counter();value=float(probability_fn(net,np.asarray(features,dtype='float32')).reshape(-1)[0])
                stats['model_s']+=time.perf_counter()-tick;stats['model_calls']+=1
                record['model_queries'].append(dict(stem=stem_id,probability=value,threshold=threshold))
                if value>=threshold:candidates.append((value,stem_id))
            if candidates:
                confidence,stem=max(candidates)
                links.append(dict(head=row['head'],stem=stem,confidence=confidence))
                stats['model_links']+=1
        decisions.append(record)
    return dict(links=links,decisions=decisions,statistics=stats,reference_used=False)
