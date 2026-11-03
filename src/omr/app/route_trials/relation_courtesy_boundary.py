"""Experimental removal of a glyph-only courtesy-signature pseudo-measure.

Requires two photographed key signatures, a classified meter after a double
bar, no segmented or graph noteheads in that strip, and the same meter at the
next system. All retained measure events and inherited attributes survive.
"""
import copy,hashlib
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as E
import cv2,numpy as np
from omr_event_records import records
from omr_bar_geometry import measure_systems
from repair_prefix_symbols import detect
from .staff_preprocess import staff_step
from .testset3_staff_resize_fast import resize_staves
from .relation_context import pitch_for_step
from .relation_measure_joint import context_rows
from .symbol_cnn import FACTOR


def fuse(root,graph,photo,geometry,homr):
    if len(root.findall('part'))!=1 or len(graph.get('stems',[]))>4096:return []
    systems=measure_systems(root);ids=sorted({r['system'] for r in graph['systems']})
    if not 2<=len(systems)==len(ids)==len(homr.get('systems',[]))<=64:return []
    if any(len(r.get('groups',[]))>2048 or len(r.get('nodes',[]))>2048 or len(r['bounds'])>129 for r in graph['systems']):return []
    if graph.get('image_sha256')!=geometry.get('image_sha256') or geometry['image_sha256']!=hashlib.sha256(Path(photo).read_bytes()).hexdigest():raise ValueError('Courtesy photograph mismatch')
    # Cheap structural candidate gate before reading or processing a raster.
    eligible=[i for i,ms in enumerate(systems[:-1]) if len(ms)>=2 and ms[-1].find('attributes/key') is not None and ms[-1].find('attributes/time') is not None]
    if not eligible:return []
    raw=cv2.imread(str(photo))
    if raw is None or raw.shape[0]*raw.shape[1]>16000000 or max(raw.shape[:2])>8192:return []
    from homr.autocrop import autocrop
    from homr.resize import resize_image
    from homr.color_adjust import apply_clahe
    image=apply_clahe(resize_image(autocrop(raw)));image=cv2.resize(image,tuple(homr['preprocessed_size']))
    tracks=resize_staves(geometry['staves'],FACTOR,FACTOR);ctx=context_rows(root)
    part=root.find('part');ms_all=part.findall('measure');before=records(root)
    sig=lambda rs:Counter((r['key'],r['voice'],r['grace']) for r in rs)
    preserved={id(m):sig(rs) for m,rs in zip(ms_all,before)};offsets=[];offset=0
    for ms in systems:offsets.append(offset);offset+=len(ms)
    cw,ch=geometry['crop_size_px'];pw,ph=homr['preprocessed_size'];ox,oy=geometry['crop_xy'];changes=[]
    for i in eligible:
        ms=systems[i];tail=ms[-1];following=systems[i+1][0];sy=homr['systems'][i]
        if len(sy['staffs'])!=1 or len(sy['staffs'][0]['grid'][0]['y'])!=10:continue
        st=sy['staffs'][0];rr=sorted([r for r in graph['systems'] if r['system']==ids[i]],key=lambda r:r['staff']);ts=[t for t in tracks if t['system']==ids[i]]
        if len(rr)!=2 or len(ts)!=2 or rr[0]['bounds']!=rr[1]['bounds'] or len(rr[0]['bounds'])!=len(ms)+1:continue
        attrs=tail.findall('attributes')
        if not attrs or any(c.tag not in ('key','time') for a in attrs for c in a):continue
        if any(n.tag not in ('attributes','note','backup','forward','barline') for n in tail) or len(tail.findall('note'))>8:continue
        keytext=tail.findtext('attributes/key/fifths')
        if keytext is None:continue
        key=int(keytext)
        if not 1<=abs(key)<=7:continue
        left,right=rr[0]['bounds'][-2:];s=rr[0]['space_px']
        if not 3*s<right-left<12*s:continue
        x0=((left+.5)/FACTOR-ox)*pw/cw-.5;x1=((right+.5)/FACTOR-ox)*pw/cw-.5
        if any(h['x']>x0-.4*s*pw/cw/FACTOR for h in st.get('notes',[])):continue
        if any(g['x']>left for r in rr for g in r['groups']):continue
        # The tail starts at two visible full-height strokes, not at an
        # artificial XML boundary or a note's isolated stem.
        ys=np.array([np.interp(x0,[p['x'] for p in st['grid']],[p['y'][k] for p in st['grid']]) for k in (0,9)])
        space=float(np.median(np.diff(st['grid'][0]['y'][:5])))
        l,r=round(x0-space),round(x0+space);top,bottom=map(round,ys)
        if min(l,top)<0 or r>=image.shape[1] or bottom>=image.shape[0]:continue
        q=(image[top:bottom,l:r+1]<140).mean(0);columns=np.flatnonzero(q>=.9)+l;runs=[]
        for x in columns:
            if not runs or x-runs[-1][-1]>1:runs.append([])
            runs[-1].append(int(x))
        if len(runs)!=2 or any(len(run)>.4*space for run in runs) or not .15*space<=runs[1][0]-runs[0][-1]<=space:continue
        signs=[];valid=True
        for row,track in zip(rr,ts):
            nodes=[n for n in row['nodes'] if n['xy'][0]>left+.2*s and n.get('confidence',0)>=.8]
            if len(nodes)!=abs(key) or any(n['class']!=(8 if key>0 else 9) for n in nodes):valid=False;break
            nodes.sort(key=lambda n:n['xy'][0]);clef=ctx[offsets[i]][1].get(row['staff'])
            if clef is None:valid=False;break
            for n,letter in zip(nodes,('FCGDAEB' if key>0 else 'BEADGCF')[:abs(key)]):
                pos=staff_step(track,n['xy'][0],n['xy'][1]+(.75*s if key<0 else 0))
                if abs(pos['residual'])>.4 or pitch_for_step(pos['nearest_step'],clef)[0]!=letter:valid=False;break
            signs.append(nodes)
        if not valid or len(signs)!=2 or any(abs(a['xy'][0]-b['xy'][0])>.4*s for a,b in zip(*signs)):continue
        ev=detect({'systems':[sy]},image,full_width=True)[0]['meters']
        meters=[v for v in ev if x0+space<v['x']<x1+space and v['confidence']>=.95]
        if not meters or len({(v['beats'],v['beat_type']) for v in meters})!=1:continue
        meter=(str(meters[0]['beats']),str(meters[0]['beat_type']))
        next_time=following.find('attributes/time')
        if next_time is None or (next_time.findtext('beats'),next_time.findtext('beat-type'))!=meter:continue
        last_sign=max(n['xy'][0] for ns in signs for n in ns)
        meter_x=((meters[0]['x']+.5)*cw/pw+ox)*FACTOR-.5
        if not .5*s<meter_x-last_sign<3*s or right-meter_x>1.8*s:continue
        # A repeated courtesy signature carries no time. Materialize missing
        # attributes at the following system; explicit new headers win.
        target=following.find('attributes')
        if target is None:target=E.Element('attributes');following.insert(0,target)
        for a in attrs:
            for child in a:
                if following.find('attributes/'+child.tag) is None:
                    at=sum(c.tag in ('divisions',) for c in target) if child.tag=='key' else sum(c.tag in ('divisions','key') for c in target)
                    target.insert(at,copy.deepcopy(child))
        part.remove(tail)
        changes.append(dict(system=ids[i],removed_measure=offsets[i]+len(ms),discarded_unbacked_events=len(tail.findall('note')),meter=list(map(int,meter)),key=key,method='two_staff_courtesy_after_double_bar_no_physical_heads',retained_measure_events_preserved=True))
    if changes:
        for m,rs in zip(part.findall('measure'),records(root)):
            if preserved[id(m)]!=sig(rs):raise ValueError('Courtesy boundary changed a retained event')
        for n,m in enumerate(part.findall('measure'),1):m.set('number',str(n))
    return changes
