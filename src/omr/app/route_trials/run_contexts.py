"""Fixed-denominator context/voice ablations; references are scorer-only."""
import argparse,itertools,json
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import xml.etree.ElementTree as E
from .pilot import ROOT,evaluate,save
from .relation_v2 import redecode

OUT=ROOT/'build/relation_context_20261010'
BASE=ROOT/'build/relation_decoder_20261010'


def semantic(path):
    root=E.parse(path).getroot();parts=root.findall('part')
    if len(parts)!=1:raise ValueError('Semantic scorer requires one part')
    divisions=1;result=[]
    for m in parts[0].findall('measure'):
        cursor=Fraction(0);previous=cursor;graces=Counter();voices=defaultdict(Counter)
        for n in m:
            if n.tag=='attributes':divisions=int(n.findtext('divisions',str(divisions)))
            elif n.tag in ('backup','forward'):cursor+=Fraction(int(n.findtext('duration','0')),divisions)*(1 if n.tag=='forward' else -1)
            elif n.tag=='note':
                onset=previous if n.find('chord') is not None else cursor
                duration=Fraction(int(n.findtext('duration','0')),divisions);grace=n.find('grace') is not None
                if n.find('chord') is None:
                    previous=onset
                    if not grace:cursor+=duration
                if n.get('print-object')=='no' or n.find('pitch') is None:continue
                staff=int(n.findtext('staff','1'));voice=n.findtext('voice','1');pitch=n.find('pitch');midi=(int(pitch.findtext('octave'))+1)*12+{'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[pitch.findtext('step')]+int(pitch.findtext('alter','0'))
                if grace:graces[(staff,onset,midi)]+=1
                else:voices[(staff,voice)][(staff,onset,midi,duration)]+=1
        result.append({'graces':graces,'voices':dict(voices)})
    return result


def semantic_score(source,candidate,start,count):
    ref=semantic(source)[start-1:start-1+count];candidate=semantic(candidate)
    total_graces=sum(sum(m['graces'].values()) for m in ref);result={'reference_graces':total_graces,'candidate_graces':sum(sum(m['graces'].values()) for m in candidate),'exact_grace_pitch_onset':0,'exact_notes_with_consistent_voice':0}
    if len(ref)!=len(candidate):return result
    result['exact_grace_pitch_onset']=sum(sum((r['graces']&c['graces']).values()) for r,c in zip(ref,candidate))
    # Use a single voice-name permutation for the whole page/staff, rather than
    # relabelling each measure to hide broken continuity.
    staves=sorted({s for m in ref for s,v in m['voices']})
    for staff in staves:
        rv=sorted({v for m in ref for s,v in m['voices'] if s==staff});cv=sorted({v for m in candidate for s,v in m['voices'] if s==staff})
        matrix=[[sum(sum((r['voices'].get((staff,a),Counter())&c['voices'].get((staff,b),Counter())).values()) for r,c in zip(ref,candidate)) for a in rv] for b in cv]
        # Current decoder has <=2 lanes. Limit scorer permutations for unusual
        # references with many voices without assigning labels during decoding.
        if len(rv)<=8 and len(cv)<=8:
            if len(cv)<=len(rv):
                best=max((sum(matrix[i][j] for i,j in enumerate(p)) for p in itertools.permutations(range(len(rv)),len(cv))),default=0)
            else:
                best=max((sum(matrix[i][j] for j,i in enumerate(p)) for p in itertools.permutations(range(len(cv)),len(rv))),default=0)
            result['exact_notes_with_consistent_voice']+=best
    return result


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['control','grace','context','voices','all'],default='all');p.add_argument('--name');p.add_argument('--scores',nargs='*',type=int);a=p.parse_args();name=a.name or a.mode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf8'));inventory=json.loads((ROOT/'build/testset_20261010/inventory.json').read_text(encoding='utf8'));rows=[]
    for row in manifest['rows']:
        sid,pid=row['id'],row['page'];i=int(sid[5:])
        if a.scores and i not in a.scores:continue
        photo=ROOT/row['photo'];geometry=ROOT/'build/staff_removal_20261010/geometries'/(sid+'_'+pid+'.json');headers=ROOT/'build/testset_round2_20261010/full_final'/sid/(pid+'.musicxml');cache=BASE/'full_final'/sid/(pid+'.graph.json');target=OUT/name/sid/(pid+'.musicxml')
        redecode(photo,geometry,headers,cache,target,a.mode)
        page=json.loads((photo.parent/'page.json').read_text(encoding='utf8'));source=ROOT/inventory[i]['path']
        result={'id':sid,'page':pid,'split':row['split'],**evaluate(source,target,page['start_measure'],page['measure_count']),**semantic_score(source,target,page['start_measure'],page['measure_count'])}
        result['previous_graph']=evaluate(source,BASE/'full_final'/sid/(pid+'.musicxml'),page['start_measure'],page['measure_count'],False)
        result['production']=evaluate(source,headers,page['start_measure'],page['measure_count'],False);rows.append(result)
        keys=['reference_notes','reference_rests','exact_note_events','exact_rest_events','candidate_notes','candidate_rests','reference_graces','candidate_graces','exact_grace_pitch_onset','exact_notes_with_consistent_voice']
        totals={k:sum(r[k] for r in rows) for k in keys};totals['strict_note_recall']=totals['exact_note_events']/max(1,totals['reference_notes']);totals['strict_rest_recall']=totals['exact_rest_events']/max(1,totals['reference_rests']);totals['invalid_pages']=sum(bool(r.get('error') or r.get('invalid_alignment')) for r in rows)
        baseline={key:sum(r['previous_graph'][key] for r in rows) for key in ('exact_note_events','exact_rest_events')};production={key:sum(r['production'][key] for r in rows) for key in ('exact_note_events','exact_rest_events')}
        save(OUT/name/'summary.json',{'mode':a.mode,'totals':totals,'previous_graph':baseline,'production':production,'rows':rows,'reference_supplied_to_recognizer':False,'scope':'All material previously used in development; not independent acceptance. Cached photographed graph ablations.'})
        print(json.dumps({'id':sid,'page':pid,'notes':result['exact_note_events'],'reference':result['reference_notes'],'previous':result['previous_graph']['exact_note_events'],'production':result['production']['exact_note_events'],'graces':result['exact_grace_pitch_onset'],'invalid':result.get('invalid_alignment',False)}),flush=True)
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
