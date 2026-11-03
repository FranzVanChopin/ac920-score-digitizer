"""Conservative, reference-free arbitration of two photographed OMR routes.

The source score is accepted only by the separate trial scorer. Arbitration
uses candidate XML, original ink, predicted graph and physical staff geometry.
"""
import argparse,copy,json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as E
import cv2
import numpy as np
from .staff_preprocess import resize_staves
from .symbol_cnn import FACTOR
from .runtime_inference import save

ROOT=Path(__file__).resolve().parents[2]

OUT=ROOT/'build/relation_context_20261010'


def flow_health(root):
    result=[];divisions=1;meter=(4,4)
    for measure in root.find('part').findall('measure'):
        cursor=Fraction(0);previous=cursor;groups=defaultdict(list)
        for n in measure:
            if n.tag=='attributes':
                divisions=int(n.findtext('divisions',str(divisions)))
                if n.find('time') is not None:meter=(int(n.findtext('time/beats')),int(n.findtext('time/beat-type')))
            elif n.tag in ('backup','forward'):cursor+=Fraction(int(n.findtext('duration','0')),divisions)*(1 if n.tag=='forward' else -1)
            elif n.tag=='note':
                chord=n.find('chord') is not None;grace=n.find('grace') is not None;onset=previous if chord else cursor;dur=Fraction(int(n.findtext('duration','0')),divisions)
                if not chord:
                    previous=onset
                    if not grace:cursor+=dur
                if not chord and not grace:groups[(n.findtext('staff','1'),n.findtext('voice','1'))].append((onset,dur))
        expected=Fraction(meter[0]*4,meter[1]);penalty=Fraction(0);full=bool(groups);overlap=Fraction(0);overflow=Fraction(0)
        for events in groups.values():
            end=Fraction(0)
            for onset,dur in sorted(events):
                penalty+=abs(onset-end);overlap+=max(Fraction(0),end-onset);end=onset+dur
                overflow+=max(Fraction(0),end-expected)
            penalty+=abs(end-expected);full=full and end==expected
        result.append({'penalty':float(penalty),'full':full,'meter':meter,'overlap':float(overlap),'overflow':float(overflow)})
    return result


def voice_shapes(root):
    from collections import Counter
    divisions=1;result=[]
    for measure in root.find('part').findall('measure'):
        voices=defaultdict(Counter)
        for n in measure:
            if n.tag=='attributes':divisions=int(n.findtext('divisions',str(divisions)))
            elif n.tag=='note' and n.find('grace') is None:
                staff=n.findtext('staff','1');voice=n.findtext('voice','1');duration=Fraction(int(n.findtext('duration','0')),divisions);p=n.find('pitch')
                written=(p.findtext('step'),p.findtext('octave'),p.findtext('alter','0')) if p is not None else ('rest','','')
                voices[(staff,voice)][(written,duration)]+=1
        # Ignore arbitrary voice-number spelling, retain each staff's symbol
        # membership. A meter heuristic must not merge two real partial voices.
        result.append(sorted((staff,tuple(sorted(counter.items()))) for (staff,voice),counter in voices.items()))
    return result


def physical_bar_support(photo,geometry,graph):
    import hashlib
    raw=cv2.imread(str(photo),0);payload=json.loads(geometry.read_text(encoding='utf8'))
    if payload['image_size_px']!=[raw.shape[1],raw.shape[0]] or payload['image_sha256']!=hashlib.sha256(photo.read_bytes()).hexdigest():raise ValueError('Arbitration photo/geometry mismatch')
    image=cv2.resize(raw,None,fx=FACTOR,fy=FACTOR,interpolation=cv2.INTER_AREA)
    staffs=resize_staves(payload['staves'],FACTOR,FACTOR);proof=[]
    for system in sorted(set(g['system'] for g in graph['systems'])):
        selected=[s for s in staffs if s['system']==system];rows=[g for g in graph['systems'] if g['system']==system]
        if len(selected)<2:return {'accepted':False,'reason':'Structural arbitration requires two independent staff crossings'}
        for x in rows[0]['bounds'][1:-1]:
            supported=[]
            for staff in selected:
                s=staff['space_px'];top=np.interp(x,*np.array(staff['lines'][0]['polyline']).T);bottom=np.interp(x,*np.array(staff['lines'][-1]['polyline']).T)
                ys=np.linspace(top+.15*s,bottom-.15*s,64).astype('float32')[:,None]
                xs=np.linspace(x-.6*s,x+.6*s,round(1.2*s)+1).astype('float32')[None]
                q=cv2.remap((image<155).astype('float32'),np.broadcast_to(xs,(64,len(xs[0]))).copy(),np.broadcast_to(ys,(64,len(xs[0]))).copy(),cv2.INTER_NEAREST)
                supported.append(float(q.mean(0).max()))
            proof.append(min(supported))
        # Empty intervals on both staves cannot certify an extra measure.
        for a,b in zip(rows[0]['bounds'],rows[0]['bounds'][1:]):
            if not any(a<g['x']<b for row in rows for g in row['groups']):return {'accepted':False,'reason':'No music symbol inside proposed measure'}
    return {'accepted':bool(proof and min(proof)>=.85),'minimum_crossing_coverage':min(proof,default=0),'physical_bars':len(proof)}


def arbitrate(production,relations,graph_path,photo,geometry,output,timing_repair=False):
    base=E.parse(production).getroot();candidate=E.parse(relations).getroot();bm=base.find('part').findall('measure');cm=candidate.find('part').findall('measure');changes=[]
    bh=flow_health(base);ch=flow_health(candidate);proof=None
    if len(bm)!=len(cm):
        graph=json.loads(graph_path.read_text(encoding='utf8'));proof=physical_bar_support(photo,geometry,graph)
        fraction=sum(h['full'] for h in ch)/max(1,len(ch))
        # Strong cross-staff bar evidence and well-formed rhythmic contents are
        # required together. Single-staff ambiguous sparse pages abstain.
        # For weaker crossings require 85% full measures. Near-complete ink
        # crossings on every staff can support structural recovery with 75%.
        # Contents still retain their measured recognition errors in scoring.
        required=.75 if proof.get('minimum_crossing_coverage',0)>=.95 else .85
        if proof['accepted'] and fraction>=required:
            base=candidate;changes.append({'method':'cross_staff_physical_structure','old_measures':len(bm),'new_measures':len(cm),'complete_measure_fraction':fraction,'proof':proof})
    elif timing_repair:
        from compare_musicxml_events import read_measures
        be=read_measures(production);ce=read_measures(relations)
        bs=voice_shapes(base);cs=voice_shapes(candidate)
        for i,(a,b) in enumerate(zip(be,ce)):
            def shape(m):
                from collections import Counter
                notes=Counter();rests=Counter()
                for (staff,onset,pitch,duration),count in m['notes'].items():notes[(staff,pitch,duration)]+=count
                for (staff,onset,duration),count in m['rests'].items():rests[(staff,duration)]+=count
                return notes,rests
            if shape(a)!=shape(b) or a['grace']!=b['grace'] or tuple(bh[i]['meter'])!=tuple(ch[i]['meter']):continue
            if bs[i]!=cs[i]:continue
            if ch[i]['penalty']>=bh[i]['penalty'] or ch[i]['penalty']!=0:continue
            # An explicit leading forward or a short secondary voice is legal;
            # meter completion alone cannot justify changing those onsets.
            if bh[i]['overlap']==0 and bh[i]['overflow']==0:continue
            if a['notes']==b['notes'] and a['rests']==b['rests']:continue
            # Rebuild only time streams. Preserve production attributes and
            # unrelated notation where possible; candidate notes carry its
            # proven pitch/duration multiset and corrected voice/onset graph.
            target=bm[i]
            for n in list(target):
                if n.tag in ('note','backup','forward'):target.remove(n)
            for n in cm[i]:
                if n.tag in ('note','backup','forward'):target.append(copy.deepcopy(n))
            changes.append({'method':'same_symbols_complete_voice_flow','measure':i+1,'old_flow_penalty':bh[i]['penalty'],'new_flow_penalty':ch[i]['penalty']})
    output.parent.mkdir(parents=True,exist_ok=True);E.indent(base,space='  ');E.ElementTree(base).write(output,encoding='utf-8',xml_declaration=True)
    report={'changes':changes,'structural_proof':proof,'ordinary_timing_repair_enabled':timing_repair,'reference_supplied_to_arbitrator':False,'scope':'Image-supported count recovery and notation consensus; physical ARM64 acceptance remains separate.'};save(output.with_suffix('.arbitration.json'),report)
    return report


def main():
    from .pilot import evaluate
    from .run_contexts import semantic_score
    p=argparse.ArgumentParser();p.add_argument('--relations',default='all_v5');p.add_argument('--name',default='candidate_v5');p.add_argument('--timing-repair',action='store_true')
    for option in ('production','relation-score','graph','photo','geometry','output'):p.add_argument('--'+option,type=Path)
    a=p.parse_args()
    if a.photo:
        if not all((a.production,a.relation_score,a.graph,a.geometry,a.output)):p.error('Single-photo arbitration requires all six input/output paths')
        print(json.dumps(arbitrate(a.production,a.relation_score,a.graph,a.photo,a.geometry,a.output,a.timing_repair)),flush=True);return
    manifest=json.loads((ROOT/'build/relation_decoder_20261010/manifest.json').read_text(encoding='utf8'));inventory=json.loads((ROOT/'build/testset_20261010/inventory.json').read_text(encoding='utf8'));rows=[]
    for row in manifest['rows']:
        sid,pid=row['id'],row['page'];photo=ROOT/row['photo'];production=ROOT/'build/testset_round2_20261010/full_final'/sid/(pid+'.musicxml');relations=OUT/a.relations/sid/(pid+'.musicxml');target=OUT/a.name/sid/(pid+'.musicxml');geometry=ROOT/'build/staff_removal_20261010/geometries'/(sid+'_'+pid+'.json')
        report=arbitrate(production,relations,relations.with_suffix('.graph.json'),photo,geometry,target,a.timing_repair)
        page=json.loads((photo.parent/'page.json').read_text(encoding='utf8'));source=ROOT/inventory[int(sid[5:])]['path']
        result={'id':sid,'page':pid,**evaluate(source,target,page['start_measure'],page['measure_count']),**semantic_score(source,target,page['start_measure'],page['measure_count']),'arbitration':report};rows.append(result)
        totals={k:sum(r[k] for r in rows) for k in ['reference_notes','reference_rests','exact_note_events','exact_rest_events','reference_graces','candidate_graces','exact_grace_pitch_onset','exact_notes_with_consistent_voice']}
        totals['strict_note_recall']=totals['exact_note_events']/max(1,totals['reference_notes']);totals['strict_rest_recall']=totals['exact_rest_events']/max(1,totals['reference_rests']);totals['invalid_pages']=sum(bool(r.get('error') or r.get('invalid_alignment')) for r in rows)
        save(OUT/a.name/'summary.json',{'totals':totals,'rows':rows,'reference_supplied_to_arbitrator':False})
        print(sid,pid,'changes',len(report['changes']),'notes',result['exact_note_events'],flush=True)
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
