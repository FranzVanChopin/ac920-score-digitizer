"""Read-only Vivado report inspection; no project loading or hardware access."""
import argparse,json,re
from pathlib import Path

def timing(text):
    lines=text.splitlines()
    header=next((i for i,s in enumerate(lines) if 'WNS(ns)' in s and 'TPWS Total Endpoints' in s),None)
    if header is None:raise ValueError('Missing complete timing summary; do not infer PASS')
    v=next((s.split() for s in lines[header+1:header+12] if len(s.split())==12 and all(re.fullmatch(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)',x) for x in s.split())),None)
    if v is None:raise ValueError('Missing numeric timing row; do not infer PASS')
    try:
        return dict(wns_ns=float(v[0]),tns_ns=float(v[1]),setup_failures=int(v[2]),whs_ns=float(v[4]),ths_ns=float(v[5]),hold_failures=int(v[6]),wpws_ns=float(v[8]),pulse_failures=int(v[10]))
    except ValueError as e:raise ValueError('Unsupported timing summary') from e

def drc(text):
    rows=[]
    for line in text.splitlines():
        m=re.match(r'\s*\|\s*([\w-]+)\s*\|\s*(Error|Critical Warning|Warning|Advisory)\s*\|.*?\|\s*(\d+)\s*\|',line)
        if m:rows.append(dict(rule=m[1],severity=m[2],count=int(m[3])))
    if not rows:
        if re.search(r'(?:Checks found|Violations found)\s*:\s*0',text):return dict(known=True,rules=[],counts={})
        return dict(known=False,rules=[],counts={})
    counts={s:sum(r['count'] for r in rows if r['severity']==s) for s in {r['severity'] for r in rows}}
    return dict(known=True,rules=rows,counts=counts)

def review(folder):
    folder=Path(folder);out={}
    for stage in ['synthesis','implementation']:
        tp=folder/f'{stage}_timing.rpt';dp=folder/f'{stage}_drc.rpt';up=folder/f'{stage}_utilization.rpt'
        t=timing(tp.read_text(errors='replace'));d=drc(dp.read_text(errors='replace'))
        resources={}
        for key in ['CLB LUTs','CLB Registers','Block RAM Tile','URAM','DSPs']:
            row=next((l for l in up.read_text(errors='replace').splitlines() if '| '+key in l),None)
            if row:
                c=[x.strip() for x in row.split('|') if x.strip()]
                resources[key]=dict(used=c[1],available=c[4],percent=c[5])
        blocked=any(r['rule']=='INBB-1' and r['count'] for r in d['rules'])
        out[stage]=dict(timing=t,drc=d,resources=resources,unresolved_blackboxes=blocked,
            timing_constraints_met=all(t[k]>=0 for k in ['wns_ns','whs_ns','wpws_ns']) and not any(t[k] for k in ['setup_failures','hold_failures','pulse_failures']))
    out['final_review_pass']=bool(out['implementation']['timing_constraints_met'] and out['implementation']['drc']['known'] and not any(out['implementation']['drc']['counts'].get(s,0) for s in ['Error','Critical Warning']) and not out['implementation']['unresolved_blackboxes'] and not out['synthesis']['unresolved_blackboxes'])
    out['scope']='Declared constraints only; resource figures and warnings remain separate. Not a physical-board test.'
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    try:r=review(a.folder)
    except (OSError,ValueError) as e:print(json.dumps({'valid':False,'error':str(e)}));raise SystemExit(2)
    print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['final_review_pass'] else 1)
