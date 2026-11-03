"""Sequential, bounded route pilots with strict scores and sampled process RSS."""
import argparse,os,json,time,subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from compare_musicxml_events import read_measures,compare
from compare_two_part_musicxml_events import read_two_parts

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'build/route_trials_20261010'
OLD=ROOT/'build/tools/homr/venv/Scripts/python.exe';NEW=OUT/'homr_venv/Scripts/python.exe';TORCH=ROOT/'build/alternative_omr/transcoda_venv/Scripts/python.exe'
CASES=[(0,1),(6,1),(8,1),(11,1),(22,3),(23,2),(34,1),(35,1)]
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def execute(command,log,extra_path='',timeout=600):
    import psutil
    env=os.environ.copy();env.update({'PYTHONUTF8':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','PYTHONPATH':str(ROOT/'software')+(os.pathsep+extra_path if extra_path else ''),'TEMP':str(OUT/'tmp'),'TMP':str(OUT/'tmp')});Path(env['TEMP']).mkdir(exist_ok=True)
    start=time.monotonic();peak=0;reason=None
    with log.open('w',encoding='utf8') as f:
        process=subprocess.Popen(list(map(str,command)),stdout=f,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
        while process.poll() is None:
            try:
                parent=psutil.Process(process.pid);family=[parent]+parent.children(recursive=True)
                rss=sum(p.memory_info().rss for p in family if p.is_running());peak=max(peak,rss)
                if rss>2300*1024**2 or time.monotonic()-start>timeout:
                    reason='RSS cap 2300 MiB' if rss>2300*1024**2 else 'per-case time cap'
                    for p in family[::-1]:
                        try:p.terminate()
                        except psutil.NoSuchProcess:pass
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.04)
        code=process.wait(timeout=30)
    return {'exit_code':code,'elapsed_s':time.monotonic()-start,'sampled_peak_tree_rss_mib':peak/1024**2,'stopped_for':reason,'rss_sample_interval_s':.04}

def evaluate(source,candidate,start,count,write_report=True):
    ref=read_measures(source);target=ref[start-1:start-1+count]
    row={'reference_notes':sum(sum(m['notes'].values()) for m in target),'reference_rests':sum(sum(m['rests'].values()) for m in target),'exact_note_events':0,'exact_rest_events':0,'candidate_notes':0,'candidate_rests':0,'reference_measures':count}
    if not candidate.exists():row['error']='no MusicXML generated';return row
    try:
        parts=ET.parse(candidate).getroot().findall('part');row['candidate_parts']=len(parts)
        cand=read_measures(candidate) if len(parts)==1 else read_two_parts(candidate)
        row.update({'candidate_notes':sum(sum(m['notes'].values()) for m in cand),'candidate_rests':sum(sum(m['rests'].values()) for m in cand),'candidate_measures':len(cand)})
        if len(cand)!=count:row['invalid_alignment']=True
        else:
            report=compare(ref,cand,start);row.update(report['totals'])
            if write_report:save(candidate.with_suffix('.comparison.json'),report)
    except Exception as e:row['error']=str(e)
    return row

def summarize(rows,route):
    summed={k:sum(r.get(k,0) for r in rows) for k in ['reference_notes','reference_rests','exact_note_events','exact_rest_events','candidate_notes','candidate_rests']}
    summed['strict_note_recall']=summed['exact_note_events']/max(1,summed['reference_notes']);summed['strict_note_precision']=summed['exact_note_events']/max(1,summed['candidate_notes']);summed['invalid_cases']=sum(bool(r.get('error') or r.get('invalid_alignment') or r.get('exit_code')) for r in rows)
    save(OUT/route/'summary.json',{'route':route,'totals':summed,'rows':rows,'limitations':['Small diagnostic pilot, not whole-set acceptance.','Same previously used score material, one font and simulated clear-lit photographs.','No ARM64 or physical FPGA deployment validated.','Invalid structural alignment retained with zero strict matches.']})
    return summed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('route',choices=['cnn_hybrid','homr396_raw','homr465_raw','smt','homr396_smtcrop','homr465_smtcrop']);a=parser.parse_args();inventory=json.loads((ROOT/'build/testset_20261010/inventory.json').read_text(encoding='utf8'));rows=[]
    for i,page in CASES:
        sid=f'score{i:02d}';pid=f'page{page:03d}';photo=ROOT/f'build/testset_20261010/render/{sid}/{pid}/photo.jpg';frozen=ROOT/f'build/testset_round2_20261010/full_final/{sid}/{pid}';target=OUT/a.route/sid/(pid+'.musicxml');target.parent.mkdir(parents=True,exist_ok=True)
        start=json.loads((photo.parent/'page.json').read_text(encoding='utf8'))['start_measure'];count=json.loads((photo.parent/'page.json').read_text(encoding='utf8'))['measure_count']
        source=ROOT/inventory[i]['path']
        if a.route in ('cnn_hybrid','smt') or a.route.endswith('_smtcrop'):
            native=OUT/'native'/sid/(pid+'.png');native.parent.mkdir(parents=True,exist_ok=True)
            if not native.exists():
                meta=execute([OLD,'-m','route_trials.prep_native',photo,Path(str(frozen)+'.geometry.json'),native],native.with_suffix('.log'))
                if meta['exit_code']:raise RuntimeError('native preprocessing failed: '+str(native))
        extra=''
        if a.route=='cnn_hybrid':
            command=[TORCH,'-m','route_trials.cnn_graph',native,OUT/'symbol_model_balanced/best.pt',OUT/'symbol_scores_balanced/scores.json',target,'--geometry',Path(str(frozen)+'.geometry.json'),'--headers',Path(str(frozen)+'.musicxml')]
        elif a.route=='smt' or a.route.endswith('_smtcrop'):
            crops=json.loads(native.with_suffix('.crops.json').read_text(encoding='utf8'))
            if not crops:continue
            crop=Path(crops[0]['path']);extra=str(OUT/'smt_compatible_source')
            command=[TORCH,'-m','route_trials.smt_run',crop,OUT/'smt_model',target]
            if a.route.endswith('_smtcrop'):
                command=[OLD if a.route.startswith('homr396') else NEW,'-m','route_trials.native_homr',crop,target]
                extra='' if a.route.startswith('homr396') else str(OUT/'homr_main/source')
            # Truth used here exclusively by scorer, not crop generator/recognizer.
            svg=ET.parse(photo.parent/'clean.svg').getroot();system=next(e for e in svg.iter() if e.get('class')=='system')
            numbers=[int(e.get('id')[4:]) for e in system.iter() if e.get('class')=='measure' and e.get('id','').startswith('omrM')];start=min(numbers);count=len(numbers)
        else:
            command=[OLD if a.route=='homr396_raw' else NEW,'-m','route_trials.native_homr',photo,target]
            if a.route=='homr465_raw':extra=str(OUT/'homr_main/source')
        result=execute(command,target.with_suffix('.log'),extra)
        row={'id':sid,'page':page,'scope':'first image-derived grand-staff system' if a.route=='smt' or a.route.endswith('_smtcrop') else 'full page',**result,**evaluate(source,target,start,count)}
        baseline=Path(str(frozen)+'.evaluation.json')
        if a.route!='smt' and not a.route.endswith('_smtcrop') and baseline.exists():row['production_baseline']=json.loads(baseline.read_text(encoding='utf8'))['exact_note_events']
        rows.append(row);total=summarize(rows,a.route);print(json.dumps({'route':a.route,'id':sid,'page':page,'exact':row['exact_note_events'],'reference':row['reference_notes'],'candidate_measures':row.get('candidate_measures'),'peak_mib':result['sampled_peak_tree_rss_mib'],'exit':result['exit_code'],'error':row.get('error'),'elapsed_s':result['elapsed_s']}),flush=True)
    print(json.dumps(summarize(rows,a.route),indent=2),flush=True)
if __name__=='__main__':main()
