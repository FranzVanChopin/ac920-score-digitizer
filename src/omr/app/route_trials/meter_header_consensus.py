"""Two-staff printed system-prefix meters, with canonical glyph aspect ratio.

Signature metadata only. Existing notes, rests, voices and clocks are retained.
The two candidate XMLs must still have the image's recognized system layout.
"""
import hashlib
from pathlib import Path
import cv2,numpy as np
import xml.etree.ElementTree as E
from omr_bar_geometry import measure_systems
from repair_prefix_symbols import detect
from .testset3_staff_resize_fast import resize_staves
from .relation_context import flatten
from .relation_decode import staff_distance
from .symbol_cnn import FACTOR


def homr_head_rows(homr,geometry):
    """Conservative head guards from the same image's original segmentation."""
    if not geometry.get('staves') or not homr.get('systems'):
        return {'systems':[],'image_sha256':geometry.get('image_sha256')}
    result=[];cw,ch=geometry['crop_size_px'];pw,ph=homr['preprocessed_size'];left,top=geometry['crop_xy']
    for si,system in enumerate(homr['systems'],1):
        tracks=[s for s in geometry['staves'] if s['system']==si];heads=[[] for _ in tracks]
        for physical in system['staffs']:
            for n in physical['notes']:
                x=(n['x']+.5)*cw/pw-.5+left;y=(n['y']+.5)*ch/ph-.5+top
                if tracks:
                    index=min(range(len(tracks)),key=lambda i:staff_distance(tracks[i],x,y))
                    heads[index].append({'xy':[(x+.5)*FACTOR-.5,(y+.5)*FACTOR-.5],'geometry_guard':True})
        result.extend({'system':si,'staff':i+1,'groups':[{'heads':h}]} for i,h in enumerate(heads))
    return {'systems':result,'image_sha256':geometry['image_sha256']}


def repair(base,headers,graph,photo,geometry):
    groups=measure_systems(base);other=measure_systems(headers)
    ids=sorted({s['system'] for s in geometry.get('staves',[])})
    if len(ids)>64 or len(groups)!=len(ids) or len(other)!=len(ids):return []
    digest=hashlib.sha256(Path(photo).read_bytes()).hexdigest()
    if geometry.get('image_sha256')!=digest or graph.get('image_sha256')!=digest:raise ValueError('Meter consensus photo mismatch')
    raw=cv2.imread(str(photo),0)
    if raw is None or raw.size>16000000:return []
    spaces=[s['space_px'] for s in geometry['staves']]
    factor=16/float(np.median(spaces))
    if not .25<=factor<=2:return []
    size=(round(raw.shape[1]*factor),round(raw.shape[0]*factor))
    if size[0]*size[1]>16000000 or max(size)>8192:return []
    image=cv2.resize(raw,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    staffs=resize_staves(geometry['staves'],factor,factor);changes=[]
    offset=0
    for si,system in enumerate(ids):
        selected=[s for s in staffs if s['system']==system]
        rows=[r for r in graph['systems'] if r['system']==system]
        if len(selected)!=2 or len(rows)!=2:offset+=len(groups[si]);continue
        hits=[]
        for number,staff in enumerate(selected,1):
            flat,left=flatten(image,staff)
            # Canonical x and y spacing: flatten alone normalizes y only.
            sx=16/staff['space_px'];flat=cv2.resize(flat,(round(flat.shape[1]*sx),192),interpolation=cv2.INTER_AREA if sx<1 else cv2.INTER_LINEAR)
            geo={'systems':[{'staffs':[{'x_range':[0,flat.shape[1]-1],'grid':[{'x':0,'y':[64,80,96,112,128]}]}]}]}
            values=[v for v in detect(geo,flat[:,:min(flat.shape[1],240)])[0]['meters'] if v['confidence']>=.91 and 3*16<v['x']<12*16]
            if not values or len({(v['beats'],v['beat_type']) for v in values})!=1 or max(v['x'] for v in values)-min(v['x'] for v in values)>.35*16:break
            best=max(values,key=lambda v:v['confidence']);x=left+best['x']/sx
            row=next(r for r in rows if r['staff']==number)
            heads=[h for g in row['groups'] for h in g.get('heads',[]) if h.get('geometry_guard') or h.get('confidence',0)>=.9]
            # A header-shaped glyph after a real musical head is not certified
            # as a system opening meter. Clef bodies occupy the first 3 spaces.
            if any(left+3*staff['space_px']<(h['xy'][0]+.5)*factor/FACTOR-.5<x for h in heads):break
            if not any((h['xy'][0]+.5)*factor/FACTOR-.5>x+staff['space_px'] for h in heads):break
            hits.append((number,best,x))
        if len(hits)==2 and (hits[0][1]['beats'],hits[0][1]['beat_type'])==(hits[1][1]['beats'],hits[1][1]['beat_type']) and abs(hits[0][2]-hits[1][2])<=.35*16:
            meter=(str(hits[0][1]['beats']),str(hits[0][1]['beat_type']))
            targets=(groups[si][0],other[si][0])
            if any(len({(t.findtext('beats'),t.findtext('beat-type')) for t in m.findall('attributes/time')})>1 for m in targets):
                offset+=len(groups[si]);continue
            edited=False
            for target in targets:
                existing=target.findall('attributes/time')
                if existing and all((t.findtext('beats'),t.findtext('beat-type'))==meter for t in existing):continue
                attr=target.find('attributes')
                if attr is None:attr=E.Element('attributes');target.insert(0,attr)
                if not existing:
                    time=E.Element('time');attr.insert(sum(n.tag in ('divisions','key') for n in attr),time)
                    E.SubElement(time,'beats');E.SubElement(time,'beat-type');existing=[time]
                for time in existing:
                    time.find('beats').text=meter[0];time.find('beat-type').text=meter[1]
                    time.attrib.pop('symbol',None)
                    if hits[0][1].get('symbol')==hits[1][1].get('symbol') and hits[0][1].get('symbol'):
                        time.set('symbol',hits[0][1]['symbol'])
                edited=True
            if edited:changes.append({'system':system,'measure':offset+1,'meter':list(map(int,meter)),'confidence':min(h[1]['confidence'] for h in hits),'proof':'two_staff_meter_prefix_at_canonical_aspect_before_music','ordinary_rest_clocks_preserved':True})
        offset+=len(groups[si])
    return changes
