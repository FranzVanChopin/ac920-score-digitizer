"""Experimental addition of complete bars beyond the original image crop.

This consumes two independently recognized hypotheses. It never replaces an
existing measure. The extended view must certify full barlines and a bijection
between every exported pitch and every segmented head in each added interval.
No reference score, expected measure count or work identity is accepted.
"""
import copy
from collections import Counter
import xml.etree.ElementTree as E
from omr_bar_geometry import measure_systems,bar_bounds
from repair_augmentation_dots import _head_map
from align_homr_geometry import _position
from omr_event_records import records
from .relation_coverage import core,hollow_core
from .relation_hollow_evidence import enclosed_hollow


def contexts(root):
    division=1;state={};out=[]
    for m in root.findall('part/measure'):
        for a in m.findall('attributes'):
            division=int(a.findtext('divisions',str(division)))
            for key in ('key','time','clef','transpose'):
                for node in a.findall(key):state[key,node.get('number','1')]=E.tostring(node,encoding='unicode').strip()
        out.append((division,dict(state)))
    return out


def fuse(base,extended,base_geometry,extended_geometry,image):
    if len(base.findall('part'))!=1 or len(extended.findall('part'))!=1 or image.ndim!=2 or image.size>16000000:return []
    old_systems=measure_systems(base);new_systems=measure_systems(extended)
    if not 1<=len(old_systems)==len(new_systems)==len(base_geometry.get('systems',[]))==len(extended_geometry.get('systems',[]))<=64:return []
    if base_geometry['preprocessed_size']!=extended_geometry['preprocessed_size']:return []
    old_events=records(base);new_events=records(extended);old_context=contexts(base);new_context=contexts(extended)
    if len(old_events)>1024 or len(new_events)>1024:return []
    mapped=_head_map(extended,extended_geometry);jobs=[];a=b=0
    for si,(ms,ns,og,ng) in enumerate(zip(old_systems,new_systems,base_geometry['systems'],extended_geometry['systems'])):
        if not len(ms)<len(ns)<=len(ms)+4:a+=len(ms);b+=len(ns);continue
        if len(og['staffs'])!=1 or len(ng['staffs'])!=1:a+=len(ms);b+=len(ns);continue
        old,new=og['staffs'][0],ng['staffs'][0];bounds=bar_bounds(new,image)
        if bounds is None or len(bounds)!=len(ns)+1:a+=len(ms);b+=len(ns);continue
        space=new['notes'][0]['staff_space'] if new.get('notes') else 0
        if not space or bounds[len(ms)]<old['x_range'][1]+2*space:a+=len(ms);b+=len(ns);continue
        # Attribute and divisions equality makes a copied bar independent of
        # any new interpretation of earlier bars. No rescaling side effects.
        prefix_ok=old_context[a+len(ms)-1]==new_context[b+len(ms)-1]
        accepted=[]
        for j in range(len(ms),len(ns)):
            m=ns[j];rs=new_events[b+j];lo,hi=bounds[j:j+2]
            if not prefix_ok or m.findall('attributes') or m.find('print') is not None or m.get('implicit')=='yes' or not 2<=len(rs)<=128:break
            if any(r['midi'] is None or r['grace'] or r['node'].find('tie') is not None or r['node'].find('time-modification') is not None or r['node'].get('print-object')=='no' for r in rs):break
            hs=[mapped.get(r['node']) for r in rs]
            if any(h is None or not lo+space/2<h['x']<hi-space/2 or _position(r['node'])[1]!=h['position'] for r,h in zip(rs,hs)):break
            if any(not ((hollow_core(dict(xy=[h['x'],h['y']]),image,h['staff_space']) or enclosed_hollow(dict(xy=[h['x'],h['y']]),image,h['staff_space'])) if r['node'].findtext('type') in ('whole','half') else core(dict(xy=[h['x'],h['y']]),image,h['staff_space'])) for r,h in zip(rs,hs)):break
            observed=[h for h in new['notes'] if lo+space/2<h['x']<hi-space/2]
            if Counter(id(h) for h in hs)!=Counter(id(h) for h in observed):break
            if any(r['duration']<=0 or r['onset']<0 or r['onset']+r['duration']>16 for r in rs):break
            accepted.append((m,len(rs),[lo,hi]))
        if accepted and len(accepted)==len(ns)-len(ms):jobs.append((si,ms[-1],accepted))
        a+=len(ms);b+=len(ns)
    if not jobs:return []
    # Store the original children, including attributes and all voice/timing
    # tokens. The only allowed edit is insertion after a retained measure.
    part=base.find('part');original={id(m):E.tostring(m) for m in part.findall('measure')};changes=[]
    for si,anchor,accepted in jobs:
        at=list(part).index(anchor)+1
        for m,count,bounds in accepted:
            part.insert(at,copy.deepcopy(m));at+=1
            changes.append(dict(system=si+1,added_notes=count,physical_interval=bounds,original_measures_preserved=True,method='append_complete_image_certified_bar_outside_original_grid'))
    assert all(E.tostring(m)==original[id(m)] for m in part.findall('measure') if id(m) in original)
    return changes
