"""Grace/context/continuity decoding on the learned photographed-symbol graph."""
import argparse,copy,json,time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .runtime_inference import torch,save
from experimental_omr import theory
from experimental_omr.model import Event,BBox,Voice,Measure,Score,Part,Staff
from . import relation_decode as rd,relation_context as rc,relation_grace as rg,relation_voices as rv
from .relation_export import export_score
from .staff_preprocess import staff_step
from .testset3_staff_resize_fast import resize_staves
from .symbol_cnn import FACTOR


@dataclass
class RelationEvent(Event):
    grace:bool=False
    written_duration:int=240
    slash:bool=False
    beam_chain:int|None=None
    time_modification:tuple[int,int]|None=None


def decode(groups,nodes,staff,bounds,ctx,number,offset,changes,state,use_voices=True,attach_graces=False,voice_continuity=False):
    space=staff['space_px'];current=copy.deepcopy(ctx);measures=[];issues=[]
    all_accs=[n for n in nodes if 8<=n['class']<=10]
    for mi,(left,right) in enumerate(zip(bounds,bounds[1:])):
        inside=[g for g in groups if left<g['x']<right];local=[c for c in changes if c['measure']==mi]
        first=min((g['x'] for g in inside),default=right)
        start=[c for c in local if c['x']<first]
        for c in start:rc.update(current,c)
        # A glyph belonging to a bar-aligned key change must not attach to the
        # first nearby note as a written accidental. The new key state already
        # supplies its pitch effect through resolve_accidental below.
        signature_ids=rc.key_signature_nodes(nodes,changes,mi,first,space,staff,current['clefs'].get(number,ctx['clefs'][1]))
        accs=[n for n in all_accs if n['id'] not in signature_ids]
        meter=current['meter'];key=current['key'];initial=copy.deepcopy(current['clefs'].get(number,ctx['clefs'][1]));carry={}
        expected=theory.measure_ticks(meter)
        for g in inside:
            for c in local:
                if c not in start and c['x']<g['x']:rc.update(current,c)
            for h in g.get('heads',[]):
                step=staff_step(staff,*h['xy'])['nearest_step'];letter,octave=rc.pitch_for_step(step,current['clefs'].get(number,initial))
                acc=rd.accidental_for(h,accs,space);explicit={8:'sharp',9:'flat',10:'natural'}.get(acc['class']) if acc else None
                alter=theory.resolve_accidental(letter,explicit,current['key'],carry,octave)
                if explicit:carry[(letter,octave)]=alter
                h['pitch']=(letter,octave,alter,explicit)
        for c in local:rc.update(current,c)
        if use_voices:lanes,poly=rv.decode_voices(inside,space,expected,state,attach_graces,voice_continuity)
        else:
            ordinary=[g for g in inside if not g.get('grace')];lanes,poly=rd.voices(ordinary,space)
            for g in inside:
                if g.get('grace'):
                    near=[(abs(h['x']-g['x']),k) for k,seq in lanes.items() for h in seq if h['x']>=g['x']]
                    lane=min(near)[1] if near else next(iter(lanes),1);lanes.setdefault(lane,[]).append(g);lanes[lane].sort(key=lambda g:g['x'])
        for seq in lanes.values():
            if len(seq)==1 and seq[0].get('rest') and seq[0]['rest']['class']==2:seq[0]['duration']=expected
        offsets,alignment=rv.offsets(lanes,space);issues.extend(dict(x,measure=offset+mi+1,staff=number) for x in alignment);voices=[]
        for lane,seq in lanes.items():
            cursor=offsets[lane];events=[]
            for gi,g in enumerate(seq):
                if 'locked_onset' in g:cursor=g['locked_onset']
                grace=g.get('grace',False);duration=rv.duration(g);g['onset']=cursor;g['voice']=lane
                if g.get('rest'):
                    n=g['rest'];events.append(RelationEvent(type='rest',onset_ticks=cursor,duration_ticks=duration,confidence=n['confidence'],bbox=BBox(n['xy'][0]-.5*space,n['xy'][1]-space,space,2*space)))
                else:
                    for h in g['heads']:
                        letter,octave,alter,explicit=h['pitch'];name=theory.pitch_name(letter,octave,alter)
                        events.append(RelationEvent(type='note',onset_ticks=cursor,duration_ticks=duration,pitch=theory.midi_pitch(letter,octave,alter),written_name=name,sounding_name=name,accidental=explicit,stem=g.get('direction'),visual_tail_beams=g.get('beams',0),visual_tail_flags=g.get('flags',0),beam_chain=g.get('beam_chain') if g.get('beams') else None,chord_id=gi+1 if len(g['heads'])>1 else None,confidence=h['confidence'],bbox=BBox(h['xy'][0]-.6*space,h['xy'][1]-.4*space,1.2*space,.8*space),grace=grace,written_duration=g.get('written_duration',g['duration']),tuplet=g.get('tuplet'),time_modification=g.get('tuplet_ratio')))
                cursor+=duration
            voices.append(Voice(lane,events))
            if cursor!=expected:issues.append({'measure':offset+mi+1,'staff':number,'voice':lane,'actual_ticks':cursor,'expected_ticks':expected,'polyphonic':poly})
        measure=Measure(offset+mi+1,time_signature=meter,voices=voices or [Voice(1,[])],key_fifths=key,clef=initial['name']);measure.relation_export=True;measure.initial_clef=initial;measure.context_events=[]
        for c in local:
            if c in start or number not in c.get('clefs',{}):continue
            following=[g for g in inside if g['x']>c['x']]
            onset=min((g['onset'] for g in following),default=max((e.onset_ticks+e.duration_ticks for v in voices for e in v.events),default=0))
            measure.context_events.append({'onset':onset,'clef':c['clefs'][number],'evidence':c['evidence']})
        measures.append(measure)
    return measures,issues


def redecode(photo,geometry,headers,cache,output,mode='all',root=rg.OUT,refinement=None,backend='torch'):
    start=time.perf_counter();raw=cv2.imread(str(photo),0);payload=json.loads(geometry.read_text(encoding='utf8'))
    import hashlib
    if payload['image_size_px']!=[raw.shape[1],raw.shape[0]] or payload['image_sha256']!=hashlib.sha256(photo.read_bytes()).hexdigest():raise ValueError('Photo/geometry mismatch')
    image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA);staffs=resize_staves(payload['staves'],FACTOR,FACTOR)
    graph=json.loads(cache.read_text(encoding='utf8'));systems=sorted(set(g['system'] for g in graph['systems']))
    if graph.get('image_sha256') and graph['image_sha256']!=payload['image_sha256']:raise ValueError('Cached graph belongs to another photograph')
    contexts=rc.contexts(headers,len(systems))
    if mode in ('grace','all'):
        if backend=='onnx':
            from .runtime_inference import OnnxModel
            model=OnnxModel(root/'grace.onnx');threshold=json.loads((root/'grace_training.json').read_text())['threshold']
        else:model,threshold=rg.load(root)
    else:model=None
    proposal_rows=None;rescue_changes=[];staff_proposals=defaultdict(list)
    if refinement and refinement.get('rescue'):
        from .grace_rescue import read_proposals
        proposal_rows=read_proposals(refinement['rescue'],photo)
        for proposal in proposal_rows:
            _,x,y,_=proposal
            eligible=[i for i,s in enumerate(staffs) if s['lines'][0]['start'][0]<x<s['lines'][0]['end'][0]]
            if eligible:staff_proposals[min(eligible,key=lambda i:rd.staff_distance(staffs[i],x,y))].append(proposal)
    all_measures=defaultdict(list);states=defaultdict(dict);issues=[];offset=0;change_count=0;graces=0
    for si,system in enumerate(systems):
        rows=[g for g in graph['systems'] if g['system']==system]
        for row in rows:
            index=next(i for i,s in enumerate(staffs) if s['system']==system and sum(1 for t in staffs[:i+1] if t['system']==system)==row['staff'])
            staff=staffs[index];groups=row['groups'];ctx=copy.deepcopy(contexts[si]);number=row['staff']
            row['space_px']=staff['space_px']
            if refinement and refinement.get('identity_repairs'):
                from .relation_chord_coverage import attach_seconds
                row['chord_attachment_repairs']=attach_seconds(row,{s['id']:s for s in graph['stems']},image)
            if mode in ('context','all'):changes=rc.visual_changes(image,staff,row['bounds'],ctx,number)
            else:
                # Isolate grace/voice changes from the original context behavior.
                old=rd.contexts(headers,len(systems))[si];ctx.update({'meter':old['meter'],'key':old['key'],'clefs':{k:{'sign':{'treble':'G','bass':'F','alto':'C','tenor':'C'}[v],'line':{'treble':2,'bass':4,'alto':3,'tenor':4}[v],'octave':0,'name':v} for k,v in old['clefs'].items()}});changes=[]
            if model is not None:rg.classify_groups(groups,image,staff,model,threshold)
            if proposal_rows is not None:
                from .grace_rescue import rescue
                # A low-threshold head is owned once across the page. Passing
                # every proposal to each nearby staff duplicates ledger graces
                # on the other piano staff despite detecting the right pixels.
                owned=staff_proposals[index]
                rescue_changes.extend(rescue(groups,row['nodes'],image,staff,row['bounds'],owned,model,threshold))
            if refinement and refinement.get('join_grace_chords'):
                from .grace_rescue import join_grace_chords
                rescue_changes.extend(join_grace_chords(groups,graph['stems'],staff['space_px']))
            if mode in ('voices','all'):rv.beam_chains(groups,image,staff['space_px'])
            if refinement and refinement.get('visual_tuplets'):
                from .relation_tuplets import repair as repair_tuplets
                row['tuplet_changes']=repair_tuplets(groups,image,staff['space_px'],row['bounds'],ctx,changes,local_spans=bool(refinement.get('local_tuplet_spans')),recover_crossings=bool(refinement.get('tuplet_crossings')),expanded_rhythm=bool(refinement.get('expanded_rhythm')))
            ms,diagnostics=decode(groups,row['nodes'],staff,row['bounds'],ctx,number,offset,changes,states[number],mode in ('voices','all'),bool(refinement and refinement.get('attach_graces')),bool(refinement and refinement.get('voice_continuity')))
            all_measures[number].extend(ms);issues.extend(diagnostics);row['context_changes']=changes;change_count+=len(changes);graces+=sum(len(g['heads']) for g in groups if g.get('grace'))
        offset+=len(rows[0]['bounds'])-1
    score=Score(title='Relation context trial',parts=[Part(name='Recognized score',staves=[Staff(number=k,measures=v,clef=v[0].clef,key_fifths=v[0].key_fifths or 0) for k,v in sorted(all_measures.items())])])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(export_score(score),encoding='utf8')
    graph['rhythm_issues']=issues
    graph['image_sha256']=payload['image_sha256'];graph['coordinate_scale']=FACTOR
    save(output.with_suffix('.graph.json'),graph)
    report={'mode':mode,'measures':offset,'grace_heads':graces,'context_changes':change_count,'rhythm_issues':len(issues),'elapsed_s':time.perf_counter()-start,'reference_supplied_to_recognizer':False,'inputs':'Cached image-predicted nodes/strokes/groups; SHA-guarded original photo and image staff polylines; predicted XML attributes. No reference labels or sequence.','limits':['Cue-sized notes untested; inline bass clefs visually verified, other predicted clefs remain a dependency.','At most two voice lanes and 32 search states; no full tuplet/tie/cross-staff model.','Desktop verification only, no ARM64/physical FPGA verification.']}
    save(output.with_suffix('.run.json'),report);print(json.dumps(report),flush=True)
    if refinement:save(output.with_suffix('.refinement.json'),{'grace_rescue':rescue_changes,'grace_principal_attachment':bool(refinement.get('attach_graces')),'reference_supplied_to_recognizer':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('photo',type=Path);p.add_argument('geometry',type=Path);p.add_argument('headers',type=Path);p.add_argument('cache',type=Path);p.add_argument('output',type=Path);p.add_argument('--mode',choices=['control','grace','context','voices','all'],default='all');p.add_argument('--proposals',type=Path);p.add_argument('--refine-voices',action='store_true');p.add_argument('--attach-graces',action='store_true');a=p.parse_args()
    refinement={'rescue':a.proposals,'voice_continuity':a.refine_voices,'attach_graces':a.attach_graces} if a.proposals or a.refine_voices or a.attach_graces else None
    if a.proposals and a.mode not in ('grace','all'):p.error('Grace rescue requires a grace-classifying mode')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);redecode(a.photo,a.geometry,a.headers,a.cache,a.output,a.mode,refinement=refinement)
