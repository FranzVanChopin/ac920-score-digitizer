"""Read clear numeric meters and octave clefs in bounded image prefixes.

Generic SMuFL templates are generated independently of evaluation scores.
Uncertain shapes, competing pairs or inconsistent staves cause abstention.
No neural model or reference MusicXML is required.
"""
from functools import lru_cache
from pathlib import Path
from statistics import median
import base64
import json
import xml.etree.ElementTree as ET


@lru_cache(maxsize=1)
def _templates():
    import numpy as np
    rows=json.loads(Path(__file__).with_name('omr_prefix_templates.json').read_text(encoding='utf8'))['glyphs']
    for r in rows:
        r['mask']=np.unpackbits(np.frombuffer(base64.b64decode(r['bits']),dtype=np.uint8)).reshape(64,48)
    return rows


def _classify(mask, codes):
    import cv2
    import numpy as np
    ratio=mask.shape[1]/mask.shape[0]
    small=cv2.resize(mask,(48,64),interpolation=cv2.INTER_NEAREST)>0
    expanded=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
    scores={}
    for r in _templates():
        if r['code'] not in codes:continue
        ref=r['mask']>0
        dil=cv2.dilate(r['mask'],np.ones((3,3),np.uint8))>0
        precision=(small&dil).sum()/max(1,small.sum())
        recall=(ref&expanded).sum()/max(1,ref.sum())
        score=(precision+recall)/2 - 0.35*abs(ratio-r['ratio'])
        n=r['code']; scores[n]=max(scores.get(n,0),float(score))
    ranking=sorted(scores.items(),key=lambda r:r[1],reverse=True)
    if ranking[0][1]<0.87 or ranking[0][1]-ranking[1][1]<0.04:return None
    return ranking[0]


def _digit(mask):
    result=_classify(mask,[f'E08{n}' for n in range(1,10)])
    return (int(result[0][-1]),result[1]) if result else None


def detect(geometry,image,debug=None,full_width=False):
    import cv2
    import numpy as np
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    results=[]
    for si,system in enumerate(geometry.get('systems',[])):
        if len(system.get('staffs',[]))!=1:continue
        staff=system['staffs'][0];grid=staff.get('grid',[])
        if not grid or len(grid[0]['y']) not in (5,10):continue
        point=min(grid,key=lambda p:abs(p['x']-(staff['x_range'][0]+60)))
        staves=len(point['y'])//5
        row={'system':si,'meters':[],'octave_clefs':[]}
        for sn in range(1,staves+1):
            lines=point['y'][(sn-1)*5:sn*5];space=median(b-a for a,b in zip(lines,lines[1:]))
            if not 5<=space<=40:continue
            left=max(0,round(staff['x_range'][0]-space))
            right=min(image.shape[1],round(staff['x_range'][1]+3*space) if full_width else round(left+15*space))
            top=max(0,round(lines[0]-3.5*space));bottom=min(image.shape[0],round(lines[-1]+3.5*space))
            crop=image[top:bottom,left:right]
            ink=cv2.threshold(crop,140,255,cv2.THRESH_BINARY_INV)[1]
            horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(round(space*3),1)))
            clean=cv2.subtract(ink,horizontal)
            clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(1,3)))
            if debug:
                folder=Path(debug);folder.mkdir(parents=True,exist_ok=True)
                cv2.imwrite(str(folder/f'system{si:02}_staff{sn}.png'),image[top:bottom,left:right])
                cv2.imwrite(str(folder/f'system{si:02}_staff{sn}.mask.png'),clean)
            digits=[]
            for half in (0,1):
                a=max(0,round(lines[half*2]-top));b=min(clean.shape[0],round(lines[half*2+2]-top)+1)
                count,labels,stats,centers=cv2.connectedComponentsWithStats(clean[a:b])
                for i in range(1,count):
                    x,y,w,h,area=stats[i]
                    if not (0.5*space<=w<=1.9*space and 1.65*space<=h<=2.3*space):continue
                    if abs((top+a+y+h/2)-(lines[half*2]+lines[half*2+2])/2)>0.35*space:continue
                    digit=_digit((labels[y:y+h,x:x+w]==i).astype('uint8'))
                    if digit:digits.append({'digit':digit[0],'confidence':digit[1],'x':left+x+w/2,'half':half})
            pairs=[(a,b) for a in digits for b in digits if a['half']==0 and b['half']==1 and b['digit'] in (2,4,8) and abs(a['x']-b['x'])<0.3*space]
            if full_width or len(pairs)==1:
                for a,b in pairs:
                    row['meters'].append({'staff':sn,'beats':a['digit'],'beat_type':b['digit'],'x':float((a['x']+b['x'])/2),'confidence':min(a['confidence'],b['confidence'])})
            count,labels,stats,_=cv2.connectedComponentsWithStats(clean)
            for i in range(1,count):
                x,y,w,h,_=stats[i]
                cx=left+x+w/2
                local=min(grid,key=lambda p:abs(p['x']-cx))['y'][(sn-1)*5:sn*5]
                # The two numerals can touch at the middle staff line. Split
                # the whole component rather than cutting either digit at a
                # skewed global row. Each half must independently match.
                if (.5*space<=w<=2.1*space and 3.65*space<=h<=4.4*space
                        and abs(top+y+h/2-local[2])<.35*space):
                    block=(labels[y:y+h,x:x+w]==i).astype('uint8');values=[]
                    for sub in (block[:h//2],block[h//2:]):
                        yy,xx=np.where(sub)
                        values.append(_digit(sub[yy.min():yy.max()+1,xx.min():xx.max()+1]) if len(xx) else None)
                    if all(values) and values[1][0] in (2,4,8):
                        row['meters'].append({'staff':sn,'beats':values[0][0],'beat_type':values[1][0],
                                              'x':float(cx),'confidence':min(v[1] for v in values)})
                if (left+x < staff['x_range'][0]+3*space or not 1.6*space<=h<=3.4*space
                        or not 1.0*space<=w<=2.1*space or abs(top+y+h/2-local[2])>0.3*space):continue
                result=_classify((labels[y:y+h,x:x+w]==i).astype('uint8'),['E08A','E08B'])
                if result and result[1]>=0.90:
                    common=result[0]=='E08A'
                    row['meters'].append({'staff':sn,'beats':4 if common else 2,'beat_type':4 if common else 2,
                                          'x':float(left+x+w/2),'confidence':result[1],'symbol':'common' if common else 'cut'})
            # The small 8 must have two enclosed counters and sit directly
            # beneath the clef, outside all five staff lines.
            contours,hierarchy=cv2.findContours(ink,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            eights=[]
            if hierarchy is not None:
                counters=[]
                for i,c in enumerate(contours):
                    x,y,w,h=cv2.boundingRect(c)
                    if hierarchy[0,i,3]<0:continue
                    area=cv2.contourArea(c)
                    if (0.15*space<=w<=0.7*space and 0.15*space<=h<=0.65*space and 1.3*space<=x<=4*space and 1.25*space<=top+y-lines[-1]<=3.1*space and area>=1.5):
                        counters.append((x+w/2,y+h/2,w,h,area))
                pairs=[(a,b) for a in counters for b in counters if 0.25*space<b[1]-a[1]<0.8*space and abs(a[0]-b[0])<0.25*space and min(a[4],b[4])/max(a[4],b[4])>0.35]
                if len(pairs)==1:
                    a,b=pairs[0];eights.append({'staff':sn,'octave':-1,'x':left+(a[0]+b[0])/2,'y':top+(a[1]+b[1])/2})
            if len(eights)==1:row['octave_clefs'].extend(eights)
        results.append(row)
    return results


def repair(root,geometry,image):
    """Amend only image-confirmed signatures and verified single G clefs.

    Interior signatures require independent bar or neighboring-head anchors.
    Octave corrections require repeated clef and per-note geometric evidence.
    """
    evidence=detect(geometry,image)
    changes=[]
    parts=root.findall('part')
    if len(parts)!=1 or not evidence:return changes
    first=parts[0].find('measure')
    meters=evidence[0]['meters']
    if meters and len({(r['beats'],r['beat_type']) for r in meters})==1:
        time=first.find('attributes/time')
        if time is not None:
            a,b=meters[0]['beats'],meters[0]['beat_type']
            old=(time.findtext('beats'),time.findtext('beat-type'))
            if old!=(str(a),str(b)):
                time.find('beats').text=str(a);time.find('beat-type').text=str(b)
                changes.append({'method':'printed_numeric_meter','measure':1,'old':old,'meter':[a,b]})
    # Read interior signatures only when visible bar strokes independently
    # confirm every measure boundary in that system. This cannot repair a
    # missing/merged measure by borrowing positions from a reference score.
    from omr_bar_geometry import measure_systems,bar_bounds,single_bar_bounds
    systems_of_measures=measure_systems(root)
    full=detect(geometry,image,full_width=True)
    from repair_augmentation_dots import _head_map
    fallback_map=None
    if len(systems_of_measures)==len(geometry.get('systems',[])):
        measure_numbers={m:i for i,m in enumerate(parts[0].findall('measure'),1)}
        for ms,sy,ev in zip(systems_of_measures,geometry['systems'],full):
            if len(sy['staffs'])!=1:continue
            staff=sy['staffs'][0];nlines=len(staff['grid'][0]['y'])
            bounds=single_bar_bounds(staff,image,len(ms),[e['x'] for e in ev['meters']]) if nlines==5 else bar_bounds(staff,image)
            space=median(b-a for a,b in zip(staff['grid'][0]['y'][:5],staff['grid'][0]['y'][1:5]))
            if bounds is None or len(bounds)!=len(ms)+1:
                if nlines!=5:continue
                if fallback_map is None:fallback_map=_head_map(root,geometry)
                # A lone full-bar rest has no head anchor. Use its immediately
                # adjacent, completely mapped pitched measures and a visible
                # bar stroke before the signature. Competing gaps abstain.
                replacements=[]
                for ev_meter in ev['meters']:
                    x=ev_meter['x'];gaps=[]
                    for j in range(1,len(ms)-1):
                        notes=ms[j].findall('note')
                        if (len(notes)!=1 or notes[0].find('rest') is None or
                                not (notes[0].findtext('type')=='whole' or notes[0].find('rest').get('measure')=='yes')):continue
                        sides=[[n for n in m.findall('note') if n.find('pitch') is not None] for m in (ms[j-1],ms[j+1])]
                        if not all(ns and all(n in fallback_map for n in ns) for ns in sides):continue
                        if max(fallback_map[n]['x'] for n in sides[0])+.5*space<x<min(fallback_map[n]['x'] for n in sides[1])-.5*space:
                            gaps.append(ms[j])
                    if len(gaps)!=1:continue
                    import numpy as np
                    import cv2
                    gray=image if image.ndim==2 else cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    strokes=[]
                    for bx in range(max(0,round(x-2.5*space)),round(x-.7*space)):
                        point=min(staff['grid'],key=lambda p:abs(p['x']-bx))['y']
                        ys=np.rint(np.linspace(point[0],point[-1],48)).astype(int)
                        if ys.min()<0 or ys.max()>=gray.shape[0]:continue
                        if (gray[ys,bx]<140).mean()>=.9:strokes.append(bx)
                    if not strokes or max(strokes)-min(strokes)>.6*space:continue
                    replacements.append((gaps[0],ev_meter))
                for m,record in replacements:
                    if len({(e['beats'],e['beat_type']) for target,e in replacements if target is m})!=1:continue
                    attributes=m.find('attributes')
                    if attributes is None:attributes=ET.Element('attributes');m.insert(0,attributes)
                    time=attributes.find('time');new=(str(record['beats']),str(record['beat_type']))
                    if time is not None and (time.findtext('beats'),time.findtext('beat-type'))==new:continue
                    if time is None:time=ET.SubElement(attributes,'time');ET.SubElement(time,'beats');ET.SubElement(time,'beat-type')
                    old=(time.findtext('beats'),time.findtext('beat-type'));time.find('beats').text=new[0];time.find('beat-type').text=new[1]
                    changes.append({'method':'printed_meter_at_anchored_empty_bar','measure':measure_numbers[m], 'old':old,'meter':list(new),'x':record['x']})
                continue
            for j,(m,a,b) in enumerate(zip(ms,bounds,bounds[1:])):
                candidates=[e for e in ev['meters'] if a<e['x']<min(b,a+(10 if j==0 else 6)*space)]
                if not candidates or len({(e['beats'],e['beat_type']) for e in candidates})!=1:continue
                if nlines==10 and {e['staff'] for e in candidates}!={1,2}:continue
                x=max(e['x'] for e in candidates)
                if nlines==5 and any(a<h['x']<x for h in staff['notes']):continue
                time=m.find('attributes/time');new=(str(candidates[0]['beats']),str(candidates[0]['beat_type']))
                if time is not None and (time.findtext('beats'),time.findtext('beat-type'))==new:continue
                attributes=m.find('attributes')
                if attributes is None:attributes=ET.Element('attributes');m.insert(0,attributes)
                if time is None:time=ET.SubElement(attributes,'time');ET.SubElement(time,'beats');ET.SubElement(time,'beat-type')
                old=(time.findtext('beats'),time.findtext('beat-type'))
                time.find('beats').text=new[0];time.find('beat-type').text=new[1]
                changes.append({'method':'printed_meter_in_verified_bar','measure':measure_numbers[m],
                                'old':old,'meter':list(new),'x':x})
    # Only single-staff G clefs with repeated printed evidence are supported.
    # Verify the decoder's current pitches against the visible head positions
    # before shifting; already corrected notes are not shifted twice.
    from align_homr_geometry import _xml_systems, _position
    from repair_augmentation_dots import _head_map
    proof={r['system'] for r in evidence if len(r['octave_clefs'])==1 and r['octave_clefs'][0]['staff']==1}
    if len(proof)<2:
        return changes
    systems=_xml_systems(root)
    mapped=_head_map(root,geometry)
    clefs=root.findall('part/measure/attributes/clef')
    if (len(proof)>=2 and len(systems)==len(geometry.get('systems',[]))
            and all(len(s['staffs'])==1 and len(s['staffs'][0]['grid'][0]['y'])==5 for s in geometry['systems'])
            and clefs and all(c.findtext('sign')=='G' and c.findtext('line')=='2'
                             and c.findtext('clef-octave-change','0')=='0' for c in clefs)):
        shifted=0
        for si,notes in enumerate(systems):
            for n in notes:
                h=mapped.get(n)
                if h is None or _position(n)[1]!=h['position']:continue
                octv=n.find('pitch/octave');octv.text=str(int(octv.text)-1);shifted+=1
        for clef in clefs:
            ET.SubElement(clef,'clef-octave-change').text='-1'
        if shifted:
            changes.append({'method':'printed_low_octave_clef','shifted_notes':shifted,'verified_systems':sorted(proof)})
    return changes
