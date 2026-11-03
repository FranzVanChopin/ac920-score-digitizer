"""Rebuild two complete staff streams only with matching photographic columns.

Preserves every pitch, duration, voice, chord, grace time and notation. This
fixes a global XML cursor advanced by interleaved independent staff streams.
Meter completion alone is insufficient; both staffs need ink-backed bindings
and multiple independent coincident columns with the same proposed time.
"""
from collections import Counter,defaultdict
from fractions import Fraction
import copy
from omr_event_records import records
from repair_grace_timing import _rebuild


def plans(root):
    result=[];division=1;meter=None
    for i,(m,rs) in enumerate(zip(root.findall('part/measure'),records(root))):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            if a.find('time') is not None:
                b=a.findtext('time/beats');u=a.findtext('time/beat-type')
                meter=Fraction(int(b)*4,int(u)) if b and b.isdecimal() and u and u.isdecimal() and int(u)>0 else None
        if not meter or not 0<meter<=16 or not 0<division<=9600 or m.get('implicit')=='yes' or not 1<=len(rs)<=128 or {r['staff'] for r in rs}!={1,2}:continue
        labels=defaultdict(set)
        for r in rs:labels[r['staff']].add(r['voice'])
        if any(len(v)!=1 for v in labels.values()) or labels[1]&labels[2]:continue
        if not any(r['onset']+r['duration']>meter for r in rs if not r['grace']):continue
        if any(r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None for r in rs):continue
        indices=[j for j,n in enumerate(m) if n.tag=='note']
        if not indices or any(n.tag in ('backup','forward') for n in list(m)[:indices[0]]) or any(n.tag not in ('note','backup','forward') for n in list(m)[indices[0]:indices[-1]+1]) or any(n.tag!='barline' for n in list(m)[indices[-1]+1:]):continue
        totals=defaultdict(Fraction);previous={};desired={};groups=defaultdict(list);valid=True
        for r in rs:
            sn=r['staff']
            if r['chord']:
                old=previous.get(sn)
                if old is None or old['grace']!=r['grace'] or old['duration']!=r['duration'] or r['onset']!=old['onset']:valid=False;break
                desired[r['node']]=desired[old['node']]
                if not r['grace']:groups[sn][-1].append(r)
            else:
                desired[r['node']]=r['onset'] if r['grace'] else totals[sn]
                if not r['grace']:
                    if r['duration']<=0:valid=False;break
                    groups[sn].append([r]);totals[sn]+=r['duration']
            previous[sn]=r
            # Grace positions are already independently accepted; this phase
            # cannot move any grace to an assumed principal onset.
            if r['grace'] and desired[r['node']]!=r['onset']:valid=False;break
        if not valid or set(totals.values())!={meter} or any(not 3<=len(g)<=64 or any(len(c)>8 for c in g) for g in groups.values()):continue
        if any((onset*division).denominator!=1 for onset in desired.values()):continue
        result.append({'index':i,'groups':groups,'desired':desired,'division':division,'meter':meter})
    return result


def token(group):
    return Counter(None if r['midi'] is None else (r['node'].findtext('pitch/step'),int(r['node'].findtext('pitch/octave'))) for r in group)


def rest_ink(rest,image,space):
    """A rest centre can be white; certify a compact connected ink body."""
    import cv2,numpy as np
    x,y=rest['xy'];s=space;l,r=round(x-2*s),round(x+2*s);top,bottom=round(y-2*s),round(y+2*s)
    if rest.get('confidence',0)<.6 or min(l,top)<0 or r>=image.shape[1] or bottom>=image.shape[0]:return False
    ink=(image[top:bottom,l:r]<140).astype('uint8')
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,round(2.5*s)),np.uint8))
    clean=cv2.morphologyEx(cv2.subtract(ink,horizontal),cv2.MORPH_CLOSE,np.ones((5,3),np.uint8))
    _,_,stats,centres=cv2.connectedComponentsWithStats(clean,8)
    hits=[]
    for (a,b,w,h,n),(cx,cy) in zip(stats[1:],centres[1:]):
        if a==0 or b==0 or a+w>=clean.shape[1] or b+h>=clean.shape[0]:continue
        if .2*s<w<1.4*s and s<h<3.3*s and .2*s*s<n<2.8*s*s and abs(cx-(x-l))<.7*s and abs(cy-(y-top))<.7*s:hits.append(n)
    return len(hits)==1


def fuse(root,graph,photo,geometry):
    jobs=plans(root)
    if not jobs or len(jobs)>128:return []
    from .relation_coverage import inputs
    from .relation_measure_joint import context_rows,head_witness
    from .relation_measure_realign import hollow_on_line,traced_stem
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None or len(graph.get('stems',[]))>4096:return []
    image,tracks=loaded;ms=root.findall('part/measure');ctx=context_rows(root);stems={s['id']:s for s in graph['stems']};locations={};offset=0
    for sy in sorted({r['system'] for r in graph['systems']}):
        rows=[r for r in graph['systems'] if r['system']==sy];ts=[t for t in tracks if t['system']==sy]
        for j in range(len(rows[0]['bounds'])-1):locations[offset+j]=(rows,ts,j)
        offset+=len(rows[0]['bounds'])-1
    changes=[]
    for job in jobs:
        i=job['index'];rows,ts,mi=locations[i];bound={};valid=True
        if len(rows)!=2 or len(ts)!=2:continue
        for row,t in zip(rows,ts):
            sn=row['staff'];s=row['space_px'];left,right=row['bounds'][mi:mi+2];gs=[g for g in row['groups'] if left<g['x']<right and not g.get('grace')]
            source=job['groups'][sn]
            if len(gs)!=len(source) or any(b['x']-a['x']<.65*s for a,b in zip(gs,gs[1:])):valid=False;break
            gt=[Counter([None]) if g.get('rest') else Counter(tuple(h['pitch'][:2]) for h in g.get('heads',[])) for g in gs]
            if gt!=[token(group) for group in source]:valid=False;break
            for group,g in zip(source,gs):
                if g.get('rest'):
                    rest=g['rest']
                    if not rest_ink(rest,image,s) or len(group)!=1 or g['duration']!=group[0]['duration']*480:valid=False;break
                elif not traced_stem(g,row,stems,image) or any(not (head_witness(h,row,t,ctx[i][1].get(sn,{}),image) or hollow_on_line(h,row,t,ctx[i][1].get(sn,{}),image)) for h in g.get('heads',[])):valid=False;break
            if not valid:break
            bound[sn]=(row,source,gs)
        if not valid:continue
        a,ag=bound[1][1:];b,bg=bound[2][1:];s=min(bound[1][0]['space_px'],bound[2][0]['space_px']);anchors=[]
        for u,gu in zip(a,ag):
            hits=[(v,gv) for v,gv in zip(b,bg) if abs(gu['x']-gv['x'])<=.35*s]
            if not hits:continue
            if len(hits)!=1 or job['desired'][u[0]['node']]!=job['desired'][hits[0][0][0]['node']]:valid=False;break
            anchors.append(job['desired'][u[0]['node']])
        if not valid or len(set(anchors))<2 or min(anchors)!=0:continue
        def identity_of(rs):return Counter((r['staff'],r['voice'],r['midi'],r['duration'],r['grace'],r['onset'] if r['grace'] else None) for r in rs)
        m=ms[i];saved=copy.deepcopy(list(m));identity=identity_of(records(root)[i]);desired={n:int(t*job['division']) for n,t in job['desired'].items()}
        if not _rebuild(m,desired):m[:]=saved;continue
        after=records(root)[i]
        if identity!=identity_of(after):raise ValueError('Complete-clock recovery changed event identity')
        changes.append({'measure':i+1,'method':'two_complete_staff_streams_with_ink_and_coincident_columns','coincident_columns':len(set(anchors)),'pitch_duration_voice_grace_cardinality_preserved':True})
    return changes
