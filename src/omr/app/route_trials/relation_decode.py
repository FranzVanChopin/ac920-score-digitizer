"""Image-only stroke graph and multi-voice music reconstruction trial.

Inputs: photo, learned nodes/strokes/edge weights, frozen image-derived staff
geometry, predicted system headers. No reference sequence, camera transform,
SVG labels, reference bar count, or score/page-specific lookup is accepted.
"""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as E
import cv2
import numpy as np
from .runtime_inference import torch,probabilities
from experimental_omr import theory
from experimental_omr.model import Score,Part,Staff,Measure,Voice,Event,BBox
from experimental_omr.export import to_musicxml
from . import relation_net as rn, symbol_cnn as sn
from .staff_preprocess import staff_step
from .testset3_staff_resize_fast import resize_staves
from .relation_export import export_score


def strokes(prob,threshold,space=16.,minimum=1.6):
    mask=(prob>=round(threshold*255)).astype('uint8')
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,1),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((round(minimum*space),1),np.uint8))
    # Distinct vertical ridges can touch through a beam/head. Rejecting a wide
    # connected component loses the true bar as well as its nearby stem.
    cv2.setRNGSeed(7123)
    lines=cv2.HoughLinesP(mask,1,np.pi/720,threshold=round(minimum*space*.65),minLineLength=round(minimum*space),maxLineGap=round(.3*space))
    groups=[]
    for x1,y1,x2,y2 in sorted(lines.reshape(-1,4).tolist() if lines is not None else [],key=lambda p: -abs(p[3]-p[1])):
        if abs(y2-y1)<minimum*space or abs(x2-x1)>max(2,.08*abs(y2-y1)):continue
        if y1>y2:x1,y1,x2,y2=x2,y2,x1,y1
        x=(x1+x2)/2
        selected=None
        for group in groups:
            gx=np.mean([(p[0]+p[2])/2 for p in group]);lo=min(p[1] for p in group);hi=max(p[3] for p in group)
            if abs(x-gx)<.35*space and y1<hi+.4*space and y2>lo-.4*space:selected=group;break
        if selected is None:groups.append([[x1,y1,x2,y2]])
        else:selected.append([x1,y1,x2,y2])
    result=[]
    for group in groups:
        xy=np.array(group).reshape(-1,2);lo=int(xy[:,1].min());hi=int(xy[:,1].max())
        slope,intercept=np.polyfit(xy[:,1],xy[:,0],1)
        a,b=float(slope*lo+intercept),float(slope*hi+intercept)
        yy=np.arange(lo,hi+1);xx=np.clip(np.rint(slope*yy+intercept).astype(int),0,prob.shape[1]-1)
        result.append({'id':len(result),'xy':[[a,float(lo)],[b,float(hi)]],
                       'confidence':float(prob[yy,xx].mean()/255),'bbox':[int(min(a,b)),lo,max(1,int(abs(b-a))+1),hi-lo+1]})
    return result


def cluster_x(values,tolerance):
    groups=[]
    for x in sorted(values):
        if not groups or x-groups[-1][-1]>tolerance:groups.append([])
        groups[-1].append(x)
    return [float(np.mean(g)) for g in groups]


def bar_bounds(staffs,bars,prob=None,image=None,music_nodes=None):
    space=float(np.median([s['space_px'] for s in staffs]));left=min(s['lines'][0]['start'][0] for s in staffs);right=max(s['lines'][0]['end'][0] for s in staffs)
    if image is not None:
        # Segmentation sometimes extends a short system to the page-wide
        # right border. Confirm the five horizontal strokes in original ink.
        xs=np.arange(max(0,round(right-6*space)),min(image.shape[1],round(right+space)),dtype='float32')
        vertical=cv2.dilate((image<155).astype('uint8'),np.ones((5,1),np.uint8)).astype('float32')
        ends=[]
        for staff in staffs:
            yy=np.array([np.interp(xs,*np.array(line['polyline']).T) for line in staff['lines']],dtype='float32')
            xx=np.broadcast_to(xs,yy.shape).copy();q=cv2.remap(vertical,xx,yy,cv2.INTER_LINEAR)
            supported=xs[(q>.5).sum(0)>=4]
            if len(supported)>space:ends.append(float(supported[-1]))
        if ends:right=max(ends)
    hits=[]
    for bar in bars:
        a,b=np.array(bar['xy']);x=float((a[0]+b[0])/2);lo,hi=sorted((a[1],b[1]))
        # Opening system stroke/bracket is not a completed first measure.
        # Frozen staff extent can begin slightly to its left after autocrop.
        if not left+2.5*space<x<right+.8*space:continue
        covered=[]
        for staff in staffs:
            top=np.interp(x,*np.array(staff['lines'][0]['polyline']).T);bottom=np.interp(x,*np.array(staff['lines'][-1]['polyline']).T)
            covered.append(lo<top+.35*space and hi>bottom-.35*space)
        if any(covered):hits.append(x)
    if prob is not None:
        hits=[]
        radius=max(1,round(.12*space))
        likelihood=cv2.dilate(prob[2],np.ones((1,2*radius+1),np.uint8)).astype('float32')/255
        physical=cv2.dilate((image<155).astype('uint8'),np.ones((1,2*radius+1),np.uint8)).astype('float32')
        xs=np.arange(max(0,round(left+2.5*space)),min(image.shape[1],round(right+.8*space)),dtype='float32')
        fractions=np.linspace(.05,.95,48,dtype='float32')[:,None]
        for staff in staffs:
            top=np.interp(xs,*np.array(staff['lines'][0]['polyline']).T)
            bottom=np.interp(xs,*np.array(staff['lines'][-1]['polyline']).T)
            yy=(top[None]+fractions*(bottom-top)[None]).astype('float32');xx=np.broadcast_to(xs,yy.shape).copy()
            q=cv2.remap(likelihood,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            ink=cv2.remap(physical,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            # Partial model strokes remain useful when original ink continues
            # across the staff (e.g. a notehead occludes the predicted bar tip).
            accepted=(q.mean(0)>=.55)&((q>.6).mean(0)>=.45)&(ink.mean(0)>=.82)
            hits.extend(xs[accepted].tolist())
    # Engraved repeat/double bars may be ~0.8 spaces apart.
    boundaries=cluster_x(hits,1.1*space)
    if boundaries and music_nodes is not None:
        last=boundaries[-1]
        trailing=[n for n in music_nodes if n['class']<=7 and last+.3*space<n['xy'][0]<right]
        # Courtesy clefs/meters can extend the five lines past the final bar.
        # A short tail without musical events is not an extra empty measure.
        if .85*space<right-last<4*space and not trailing:right=last
    # Geometry provides the physical system border, not a requested bar count.
    if not boundaries or right-boundaries[-1]>.85*space:boundaries.append(float(right))
    else:boundaries[-1]=float(right)
    return [float(left)]+boundaries


def contexts(headers, nsystems):
    from omr_bar_geometry import measure_systems
    groups=measure_systems(E.parse(headers).getroot())
    if len(groups)!=nsystems:raise ValueError('Predicted header/system alignment unavailable')
    state={'clefs':{1:'treble',2:'bass'},'meter':(4,4),'key':0};result=[]
    for system in groups:
        for attr in system[0].findall('attributes'):
            for clef in attr.findall('clef'):
                state['clefs'][int(clef.get('number','1'))]={('G','2'):'treble',('F','4'):'bass',('C','3'):'alto',('C','4'):'tenor'}.get((clef.findtext('sign'),clef.findtext('line')),'treble')
            if attr.findtext('key/fifths') is not None:state['key']=int(attr.findtext('key/fifths'))
            if attr.find('time') is not None:state['meter']=(int(attr.findtext('time/beats')),int(attr.findtext('time/beat-type')))
        result.append({'clefs':dict(state['clefs']),'meter':state['meter'],'key':state['key']})
    return result


def staff_distance(staff,x,y):
    from .relation_geometry_index import indexed_array,query_index
    index=query_index(x,y)
    top=np.interp(x,*indexed_array(staff['lines'][0]['polyline'],index).T);bottom=np.interp(x,*indexed_array(staff['lines'][-1]['polyline'],index).T)
    return max(top-y,y-bottom,0)/staff['space_px']


def assign_nodes(pred,staffs,thresholds):
    result=[[] for _ in staffs]
    for c,x,y,confidence in pred:
        c=int(c)
        if confidence<thresholds[c]:continue
        eligible=[(staff_distance(s,x,y),i) for i,s in enumerate(staffs) if s['lines'][0]['start'][0]-.5*s['space_px']<x<s['lines'][0]['end'][0]+.5*s['space_px']]
        if not eligible:continue
        distance,index=min(eligible)
        if distance>(5 if 2<=c<=7 else 12):continue
        result[index].append({'id':len(result[index]),'class':c,'xy':[float(x),float(y)],'confidence':float(confidence)})
    return result


def assign_graph(pred,staffs,thresholds,stems,net,threshold,gray=None,gate_report=None):
    # Link before assigning staff: high ledger chord tones must follow their
    # shared stem, rather than being dropped or moved independently.
    all_nodes=[]
    for c,x,y,confidence in pred:
        c=int(c)
        if confidence>=thresholds[c]:all_nodes.append({'id':len(all_nodes),'class':c,'xy':[float(x),float(y)],'confidence':float(confidence)})
    if gray is None:
        global_links=link_heads(all_nodes,stems,net,threshold,16.)
    else:
        from .dense_edge_geometry_gate_trial import propose_link_gates,resolve_link_gates
        tick=time.perf_counter()
        proposal=propose_link_gates(gray,all_nodes,stems,16.)
        geometry_s=time.perf_counter()-tick
        tick=time.perf_counter()
        gated=resolve_link_gates(proposal,all_nodes,stems,net,threshold,probability_fn=probabilities)
        resolving_s=time.perf_counter()-tick
        global_links=gated['links']
        if gate_report is not None:
            from collections import Counter
            gate_report.update(enabled=True,proof=proposal,decisions=gated['decisions'],
                statistics=gated['statistics'],geometry_s=geometry_s,resolving_s=resolving_s,
                threshold=threshold,edge_model_calls_before_geometry_proof=0,
                uncertainty_reason_counts=dict(Counter(row['proof']['reason'] for row in proposal['heads'] if not row['proof']['accepted'])),
                geometric_confidence_is_not_model_probability=True,
                decision_head_id_space='pre_staff_assignment_global_node_ids')
    by_stem=defaultdict(list)
    for link in global_links:by_stem[link['stem']].append(all_nodes[link['head']])
    ownership={}
    for heads in by_stem.values():
        def cost(index):
            staff=staffs[index];distances=[staff_distance(staff,*h['xy']) for h in heads]
            x=float(np.mean([h['xy'][0] for h in heads]));left=staff['lines'][0]['start'][0];right=staff['lines'][0]['end'][0]
            if not left-staff['space_px']<x<right+staff['space_px']:return float('inf')
            return -sum(d<=.35 for d in distances)+.05*sum(min(d,12) for d in distances)
        index=min(range(len(staffs)),key=cost)
        for head in heads:ownership[head['id']]=index
    result=[[] for _ in staffs];mapping={}
    for node in all_nodes:
        x,y=node['xy'];eligible=[(staff_distance(s,x,y),i) for i,s in enumerate(staffs) if s['lines'][0]['start'][0]-.5*s['space_px']<x<s['lines'][0]['end'][0]+.5*s['space_px']]
        if not eligible:continue
        distance,index=min(eligible);index=ownership.get(node['id'],index)
        if distance>(5 if 2<=node['class']<=7 else 12):continue
        new=dict(node,id=len(result[index]));mapping[node['id']]=(index,new['id']);result[index].append(new)
    links=[[] for _ in staffs]
    for link in global_links:
        if link['head'] in mapping:
            index,head=mapping[link['head']];links[index].append(dict(link,head=head))
    if gray is not None and gate_report is not None:
        gate_report['assigned_head_ids']={str(old):dict(staff_index=index,head_id=head) for old,(index,head) in mapping.items()}
    return result,links


def link_heads(nodes,stems,net,threshold,space):
    links=[]
    for head in nodes:
        if head['class']>=2:continue
        hx,hy=head['xy'];candidates=[]
        for stem in stems:
            xy=np.array(stem['xy']);sx=xy[:,0].mean();lo,hi=sorted(xy[:,1])
            if abs(sx-hx)>1.4*space or not lo-.35*space<hy<hi+.35*space:continue
            features=rn.edge_features(head['xy'],stem['xy'],space)
            p=float(probabilities(net,np.asarray(features,dtype='float32')).reshape(-1)[0])
            if p>=threshold:candidates.append((p,stem['id']))
        if candidates:
            confidence,stem=max(candidates);links.append({'head':head['id'],'stem':stem,'confidence':confidence})
    return links


def beam_relations(stem,heads,prob,thresholds,space):
    xy=np.array(stem['xy']);sx=float(xy[:,0].mean());hy=float(np.mean([h['xy'][1] for h in heads]))
    lo,hi=sorted(xy[:,1]);direction='up' if hy-lo>hi-hy else 'down';tip=lo if direction=='up' else hi
    y0=max(0,round(tip-.45*space if direction=='up' else tip-2.1*space));y1=min(prob.shape[1],round(tip+2.1*space if direction=='up' else tip+.45*space)+1)
    rows=[]
    for y in range(y0,y1):
        # Probe either side of the stem; include short secondary beam hooks.
        left=max(0,round(sx-.6*space));right=min(prob.shape[2],round(sx+.6*space)+1)
        a=prob[1,y,left:right]>=round(thresholds[1]*255)
        rows.append(bool(a.mean()>=.32) if len(a) else False)
    runs=np.diff(np.r_[False,rows,False].astype('int8'))
    starts=np.where(runs==1)[0];ends=np.where(runs==-1)[0]
    beams=min(3,sum(end-start>=.14*space for start,end in zip(starts,ends)))
    flags=0
    if not beams:
        left=max(0,round(sx+.05*space));right=min(prob.shape[2],round(sx+1.1*space))
        for c in range(3,6):
            crop=prob[c,y0:y1,left:right]
            if crop.size and (crop>=round(thresholds[c]*255)).sum()>.08*space*space:flags=c-2
    evidence=[{'y_range':[int(y0+a),int(y0+b)],'stem_x':sx} for a,b in zip(starts,ends) if b-a>=.14*space]
    return direction,int(beams),flags,float(tip),evidence


def accidental_for(head,accidentals,space):
    # The centroid of a flat is above its sounding step; sharp/natural centred.
    candidates=[]
    for acc in accidentals:
        dx=(head['xy'][0]-acc['xy'][0])/space;dy=(acc['xy'][1]-head['xy'][1])/space
        target=-.75 if acc['class']==9 else 0.
        if .65<dx<3.3 and abs(dy-target)<(.55 if acc['class']==9 else .35):
            candidates.append((abs(dy-target)+.15*dx,acc))
    return min(candidates,key=lambda p:p[0])[1] if candidates else None


def make_groups(nodes,stems,links,prob,thresholds,staff):
    space=staff['space_px'];heads={h['id']:h for h in nodes if h['class']<2}
    stem_heads=defaultdict(list);unlinked=set(heads)
    for edge in links:stem_heads[edge['stem']].append(heads[edge['head']]);unlinked.discard(edge['head'])
    groups=[]
    for stem_id,hs in stem_heads.items():
        stem=stems[stem_id];direction,beams,flags,tip,evidence=beam_relations(stem,hs,prob,thresholds,space)
        hollow=any(h['class']==1 for h in hs);duration=960 if hollow else 480//(2**max(beams,flags))
        groups.append({'heads':hs,'stem':stem_id,'direction':direction,'beams':beams,'beam_relations':evidence,'flags':flags,'tip':tip,'duration':duration,'x':float(np.median([h['xy'][0] for h in hs])),'y':float(np.mean([h['xy'][1] for h in hs]))})
    for head_id in unlinked:
        head=heads[head_id];groups.append({'heads':[head],'stem':None,'direction':None,'beams':0,'flags':0,'tip':None,'duration':1920 if head['class']==1 else 480,'x':head['xy'][0],'y':head['xy'][1]})
    for rest in nodes:
        if 2<=rest['class']<=7:
            groups.append({'heads':[],'rest':rest,'direction':None,'duration':[1920,960,480,240,120,60][rest['class']-2],'x':rest['xy'][0],'y':rest['xy'][1]})
    dots=[n for n in nodes if n['class']==11]
    # Exclusive dot ownership avoids applying one dot to consecutive notes.
    candidates=[]
    for gi,g in enumerate(groups):
        for di,dot in enumerate(dots):
            for head in g.get('heads') or [g['rest']]:
                dx=(dot['xy'][0]-head['xy'][0])/space;dy=abs(dot['xy'][1]-head['xy'][1])/space
                if .65<dx<1.7 and dy<.65:candidates.append((dy+.2*dx,gi,di))
    used_g=set();used_d=set()
    for _,gi,di in sorted(candidates):
        if gi in used_g or di in used_d:continue
        used_g.add(gi);used_d.add(di);groups[gi]['duration']=groups[gi]['duration']*3//2;groups[gi]['dotted']=True
    return sorted(groups,key=lambda g:g['x'])


def voices(groups,space):
    # Opposed stems or a rest+note at one column provide polyphony evidence.
    poly=False
    for i,a in enumerate(groups):
        for b in groups[i+1:]:
            if b['x']-a['x']>.6*space:break
            if (a.get('direction') and b.get('direction') and a['direction']!=b['direction']) or bool(a.get('rest'))!=bool(b.get('rest')):poly=True
    if not poly:return {1:groups},False
    lanes={1:[],2:[]};ys=[g['y'] for g in groups];middle=float(np.median(ys))
    for g in groups:
        lane=1 if g.get('direction')=='up' else 2 if g.get('direction')=='down' else (1 if g['y']<middle else 2)
        lanes[lane].append(g)
    return {k:v for k,v in lanes.items() if v},True


def voice_offsets(lanes,space):
    """Use shared visual columns to align a later-starting voice.

    This permits an explicit forward gap instead of inventing a rest. When
    several shared columns disagree, retain offset zero and report mismatch.
    """
    if not lanes:return {},[]
    base=max(lanes,key=lambda k:len(lanes[k]));anchors=[];cursor=0
    for group in lanes[base]:anchors.append((group['x'],cursor));cursor+=group['duration']
    offsets={base:0};issues=[]
    for lane,sequence in lanes.items():
        if lane==base:continue
        proposed=[];cursor=0
        for group in sequence:
            nearby=[(abs(x-group['x']),t) for x,t in anchors if abs(x-group['x'])<=.6*space]
            if nearby:proposed.append(min(nearby)[1]-cursor)
            cursor+=group['duration']
        if proposed and len(set(proposed))==1 and proposed[0]>=0:offsets[lane]=proposed[0]
        else:
            offsets[lane]=0
            if proposed and any(p!=0 for p in proposed):issues.append({'voice':lane,'reason':'Shared time columns disagree','candidate_offsets':proposed})
    return offsets,issues


def decode_staff(nodes,stems,links,prob,thresholds,staff,bounds,ctx,number,offset):
    groups=make_groups(nodes,stems,links,prob,thresholds,staff);space=staff['space_px']
    clef=ctx['clefs'].get(number,'treble');key=ctx['key'];meter=ctx['meter'];measures=[];diagnostics=[]
    accidentals=[n for n in nodes if 8<=n['class']<=10]
    for mi,(left,right) in enumerate(zip(bounds,bounds[1:])):
        inside=[g for g in groups if left<g['x']<right];lanes,poly=voices(inside,space);events=[];carry={}
        for sequence in lanes.values():
            if len(sequence)==1 and sequence[0].get('rest') and sequence[0]['rest']['class']==2:sequence[0]['duration']=theory.measure_ticks(meter)
        offsets,alignment_issues=voice_offsets(lanes,space)
        diagnostics.extend(dict(issue,measure=offset+mi+1,staff=number) for issue in alignment_issues)
        # Pitch accidental state follows horizontal time order across voices.
        for g in inside:
            for head in g['heads']:
                step=staff_step(staff,*head['xy'])['nearest_step'];letter,octave=theory.step_to_pitch(step,clef)
                acc=accidental_for(head,accidentals,space);explicit={8:'sharp',9:'flat',10:'natural'}.get(acc['class']) if acc else None
                alter=theory.resolve_accidental(letter,explicit,key,carry,octave)
                if explicit:carry[(letter,octave)]=alter
                head['pitch']=(letter,octave,alter,explicit)
        for lane,sequence in lanes.items():
            cursor=offsets[lane];evs=[]
            for gi,g in enumerate(sequence):
                duration=g['duration']
                # A whole rest alone in its lane represents a measure rest.
                if g.get('rest') and g['rest']['class']==2 and len(sequence)==1:duration=theory.measure_ticks(meter)
                if g.get('rest'):
                    n=g['rest'];evs.append(Event(type='rest',onset_ticks=cursor,duration_ticks=duration,confidence=n['confidence'],bbox=BBox(n['xy'][0]-.5*space,n['xy'][1]-space,space,2*space)))
                else:
                    for head in g['heads']:
                        letter,octave,alter,explicit=head['pitch'];name=theory.pitch_name(letter,octave,alter)
                        evs.append(Event(type='note',onset_ticks=cursor,duration_ticks=duration,pitch=theory.midi_pitch(letter,octave,alter),written_name=name,sounding_name=name,accidental=explicit,stem=g['direction'],visual_tail_beams=g['beams'],visual_tail_flags=g['flags'],visual_stem_tip_y=g['tip'],chord_id=gi+1 if len(g['heads'])>1 else None,confidence=head['confidence'],bbox=BBox(head['xy'][0]-.6*space,head['xy'][1]-.4*space,1.2*space,.8*space)))
                cursor+=duration
            events.append(Voice(lane,evs));expected=theory.measure_ticks(meter)
            if cursor!=expected:diagnostics.append({'measure':offset+mi+1,'staff':number,'voice':lane,'actual_ticks':cursor,'expected_ticks':expected,'polyphonic':poly})
        measures.append(Measure(offset+mi+1,time_signature=meter,voices=events or [Voice(1,[])],key_fifths=key,clef=clef))
    return measures,diagnostics,groups


def recognize(photo,geometry,headers,symbol_model,symbol_scores,stroke_model,stroke_scores,edge_model,edge_scores,output,backend='torch',staff_scale=None,classical_edge_gate=False,export_xml=True):
    start=time.perf_counter();raw=cv2.imread(str(photo),0)
    if raw is None:raise ValueError('Unreadable photo')
    payload=json.loads(geometry.read_text(encoding='utf8'))
    if payload['image_size_px']!=[raw.shape[1],raw.shape[0]]:raise ValueError('Staff geometry coordinate mismatch')
    import hashlib
    if payload['image_sha256']!=hashlib.sha256(photo.read_bytes()).hexdigest():raise ValueError('Staff geometry source mismatch')
    image=cv2.resize(raw,None,fx=sn.FACTOR,fy=sn.FACTOR,interpolation=cv2.INTER_AREA);staffs=resize_staves(payload['staves'],sn.FACTOR,sn.FACTOR)
    if backend=='onnx':
        from .runtime_inference import OnnxModel
        symbol=OnnxModel(symbol_model);stroke=OnnxModel(stroke_model);edge=OnnxModel(edge_model)
    else:
        symbol,_=sn.open_net(symbol_model);stroke,_=rn.open_net(stroke_model);edge=rn.edge_net();edge.load_state_dict(torch.load(edge_model,map_location='cpu',weights_only=True));edge.eval()
    sth=[v['threshold'] for v in json.loads(symbol_scores.read_text(encoding='utf8'))['reports']['validation']['classes'].values()]
    rth=[v['threshold'] for v in json.loads(stroke_scores.read_text(encoding='utf8'))['reports']['validation'].values()]
    eth=json.loads(edge_scores.read_text(encoding='utf8'))['threshold']
    inference_profile={'mode':'fixed','scale':sn.FACTOR,'tile_shape':[sn.W,sn.H],'tile_overlap_px':32,'sequential_tiles':True}
    if staff_scale is None:
        predictions=sn.predict(symbol,image);prob=rn.predict(stroke,image)
    else:
        from .staff_scale_inference import infer
        predictions,prob,inference_profile=infer(symbol,stroke,raw,payload['staves'],staff_scale)
    from .testset3_stroke_group_cache import strokes as grouped_strokes
    stems=grouped_strokes(prob[0],rth[0]);bars=grouped_strokes(prob[2],rth[2],minimum=2.8)
    edge_gate_report={}
    if classical_edge_gate:
        nodes,links=assign_graph(predictions,staffs,sth,stems,edge,eth,gray=image,gate_report=edge_gate_report)
    else:
        nodes,links=assign_graph(predictions,staffs,sth,stems,edge,eth)
    system_ids=sorted(set(s['system'] for s in staffs));context=contexts(headers,len(system_ids));all_measures=defaultdict(list);graph=[];issues=[];offset=0
    for si,system in enumerate(system_ids):
        indices=[i for i,s in enumerate(staffs) if s['system']==system];bounds=bar_bounds([staffs[i] for i in indices],bars,prob,image,[n for i in indices for n in nodes[i]])
        for slot,index in enumerate(indices,1):
            measures,diagnostics,groups=decode_staff(nodes[index],stems,links[index],prob,rth,staffs[index],bounds,context[si],slot,offset)
            all_measures[slot].extend(measures);issues.extend(diagnostics)
            graph.append({'system':system,'staff':slot,'bounds':bounds,'nodes':nodes[index],'head_stem_edges':links[index],'groups':groups})
        offset+=len(bounds)-1
    output.parent.mkdir(parents=True,exist_ok=True)
    if export_xml:
        score=Score(title='Learned structural graph trial',parts=[Part(name='Recognized score',staves=[Staff(number=k,measures=v,clef=context[0]['clefs'].get(k,'treble'),key_fifths=context[0]['key']) for k,v in sorted(all_measures.items())])])
        output.write_text(export_score(score),encoding='utf8')
    graph_payload={'stems':stems,'bars':bars,'systems':graph,'rhythm_issues':issues,'head_proposals':predictions[predictions[:,0]==0].tolist(),'image_sha256':payload['image_sha256'],'reference_supplied_to_recognizer':False,'coordinate_space':'photograph pixels scaled by 16/19.8','inference_profile':inference_profile}
    if classical_edge_gate:
        gate_trace=output.with_suffix('.edge_gate.json')
        rn.save(gate_trace,edge_gate_report)
        # Keep diagnostic proof out of the repeatedly decoded/copied work graph.
        gate_summary={key:edge_gate_report[key] for key in ('enabled','statistics','geometry_s','resolving_s')}
        gate_summary.update(trace=str(gate_trace),trace_sha256=hashlib.sha256(gate_trace.read_bytes()).hexdigest())
        graph_payload['classical_edge_gate']=gate_summary
    rn.save(output.with_suffix('.graph.json'),graph_payload)
    report={'systems':len(system_ids),'staves':len(staffs),'measures':offset,'symbol_nodes':sum(map(len,nodes)),'stems':len(stems),'bar_strokes':len(bars),'head_stem_edges':sum(map(len,links)),'rhythm_issues':len(issues),'elapsed_s':time.perf_counter()-start,'reference_supplied_to_recognizer':False,'scaffold':'Frozen image-derived staff polylines and predicted first header of each system only','limitations':['No trained voice continuity model; voice inference requires local opposed-stem or rest-note evidence.','Inline clef/key/meter changes, tuplets, grace notes, ties and cross-staff beams not yet fully decoded.','No ARM64 or FPGA verification.']}
    report['inference_profile']=inference_profile
    if classical_edge_gate:report['classical_edge_gate']=gate_summary
    rn.save(output.with_suffix('.run.json'),report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('photo',type=Path);p.add_argument('geometry',type=Path);p.add_argument('headers',type=Path);p.add_argument('output',type=Path);p.add_argument('--root',type=Path,default=Path('build/relation_decoder_20261010'));a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    recognize(a.photo,a.geometry,a.headers,Path('build/staff_removal_20261010/model_raw_v2/best.pt'),Path('build/staff_removal_20261010/scores_raw_v2/scores.json'),a.root/'model/best.pt',a.root/'scores/scores.json',a.root/'model/edge.pt',a.root/'model/edge_training.json',a.output)


if __name__=='__main__':main()
