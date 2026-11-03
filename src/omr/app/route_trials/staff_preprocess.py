"""Image-only staff polylines and conservative crossing-preserving removal.

Experimental host implementation. Retain the original image and geometry;
the no-staff image alone cannot provide a pitch reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from experimental_omr.staff import group_staves, vertical_run_length

def detect(gray):
    h, w = gray.shape
    threshold = min(210, max(100, cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]))
    ink = gray < threshold
    segments = cv2.HoughLinesP(ink.astype('uint8')*255, 1, np.pi/1800,
        threshold=max(40,w//10), minLineLength=max(30,w//4), maxLineGap=3)
    if segments is None: return [], ink
    candidates = []
    for x0,y0,x1,y1 in np.asarray(segments).reshape(-1,4):
        if x1<x0: x0,y0,x1,y1=x1,y1,x0,y0
        dx=x1-x0
        if dx < w/4: continue
        slope=(y1-y0)/dx
        if abs(slope)>.07: continue
        candidates.append({'y':float(y0+slope*(w/2-x0)), 'slope':float(slope),
                           'x0':int(x0),'x1':int(x1),'length':int(dx)})
    clusters=[]
    for item in sorted(candidates,key=lambda p:p['y']):
        if clusters and abs(item['y']-np.median([p['y'] for p in clusters[-1]]))<2.7:
            clusters[-1].append(item)
        else: clusters.append([item])
    lines=[]
    for cluster in clusters:
        lines.append({'y':float(np.median([p['y'] for p in cluster])),
                      'slope':float(np.median([p['slope'] for p in cluster])),
                      'x0':min(p['x0'] for p in cluster),'x1':max(p['x1'] for p in cluster)})
    groups=group_staves([p['y'] for p in lines],spacing_tol=.20)
    result=[]
    for group in groups:
        selected=[min(lines,key=lambda p:abs(p['y']-y)) for y in group.line_ys]
        space=float(np.median(np.diff(group.line_ys)))
        if space<5 or space>h/10: continue
        left=max(p['x0'] for p in selected);right=min(p['x1'] for p in selected)
        if right-left<w/4: continue
        records=[]
        for number,line in enumerate(selected,1):
            xs=np.arange(left,right+1,max(8,round(2*space))).astype(int)
            if xs[-1]!=right: xs=np.r_[xs,right]
            ys=[];radius=max(2,round(.18*space));window=max(6,round(1.5*space))
            for x in xs:
                prediction=line['y']+line['slope']*(x-w/2)
                a,b=max(0,round(prediction)-radius),min(h,round(prediction)+radius+1)
                projection=ink[a:b,max(left,x-window):min(right+1,x+window+1)].mean(axis=1)
                if len(projection) and projection.max()>.35:
                    choices=np.flatnonzero(projection>=projection.max()-.04)+a
                    ys.append(float(choices[np.argmin(abs(choices-prediction))]))
                else: ys.append(float(prediction))
            smooth=[float(np.median(ys[max(0,i-1):i+2])) for i in range(len(ys))]
            points=[[int(x),y] for x,y in zip(xs,smooth)]
            records.append({'number':number,'start':points[0],'end':points[-1],'polyline':points})
        result.append({'index':len(result),'space_px':space,'lines':records})
    return result,ink

def remove(gray,staves=None):
    if staves is None: staves,ink=detect(gray)
    else: ink=gray<min(210,max(100,cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]))
    h,w=gray.shape
    # Work on each staff's small bounding strip, never a repeated full-page
    # int32 run-length map. All removal decisions use the original ink.
    erased=np.zeros_like(ink);protected=np.zeros_like(ink)
    for staff in staves:
        space=staff['space_px'];lines=staff['lines']
        top=max(0,int(min(y for line in lines for _,y in line['polyline'])-space))
        bottom=min(h,int(max(y for line in lines for _,y in line['polyline'])+space+1))
        left=max(0,int(np.floor(min(line['start'][0] for line in lines))));right=min(w-1,int(np.ceil(max(line['end'][0] for line in lines))))
        if right<=left or bottom<=top: continue
        local=ink[top:bottom,left:right+1];runs=vertical_run_length(local)
        masks=[];thickness=[]
        for line in lines:
            xs=np.arange(max(left,int(np.ceil(line['start'][0]))),min(right,int(np.floor(line['end'][0])))+1,dtype=int)
            ys=np.rint(np.interp(xs,*np.array(line['polyline']).T)).astype(int)-top
            valid=(ys>=0)&(ys<local.shape[0]);xs,ys=xs[valid],ys[valid]
            values=runs[ys,xs-left];values=values[(values>0)&(values<.35*space)]
            thick=float(np.percentile(values,25)) if len(values) else 1.
            thick=min(max(1.,thick),max(1.,.18*space));thickness.append(thick)
            radius=max(1,int(np.ceil(thick*.7)))
            mask=np.zeros_like(local)
            for offset in range(-radius,radius+1):
                yy=ys+offset;ok=(yy>=0)&(yy<local.shape[0]);mask[yy[ok],xs[ok]-left]=True
            masks.append(mask)
        band=np.logical_or.reduce(masks)
        # Off-line ink protects diagonal, tangent and vertical strokes around
        # crossings. Ambiguous overlap is retained; no invented gap filling.
        radius=max(1,int(np.ceil(max(thickness)*.7)))
        off=local&~band
        # A hollow head tangent to a line may extend several pixels sideways
        # before reaching off-line ink. Protect that local neighbourhood too.
        side=max(1,int(np.ceil(.5*space)))
        near=cv2.dilate(off.astype('uint8'),np.ones((2*radius+5,2*side+1),np.uint8))>0
        keep=near|(runs>max(thickness)+1)
        kill=local&band&~keep
        erased[top:bottom,left:right+1]|=kill
        protected[top:bottom,left:right+1]|=local&band&keep
        staff['line_thickness_px']=thickness
    output=gray.copy();output[erased]=255
    stats={'staves':len(staves),'lines':5*len(staves),'erased_ink_pixels':int(erased.sum()),
           'protected_band_pixels':int(protected.sum()),'total_ink_pixels':int(ink.sum()),
           'original_geometry_preserved':True,'reference_used':False}
    return output,staves,erased,protected,stats

def staff_step(staff,x,y):
    """Diatonic half-space offset above bottom line, before clef/key handling."""
    from .relation_geometry_index import indexed_array,query_index
    index=query_index(x,y)
    ys=[float(np.interp(x,*indexed_array(line['polyline'],index).T)) for line in staff['lines']]
    space=float(np.median(np.diff(ys)))
    if space<=0: raise ValueError('staff line order invalid')
    value=(ys[-1]-y)/(space/2)
    return {'half_space_steps':value,'nearest_step':round(value),'residual':value-round(value),
            'local_space_px':space,'bottom_line_y':ys[-1]}

def resize_staves(staves,sx,sy):
    import copy
    result=copy.deepcopy(staves)
    for staff in result:
        staff['space_px']*=sy
        for line in staff['lines']:
            for field in ['polyline']:
                line[field]=[[(x+.5)*sx-.5,(y+.5)*sy-.5] for x,y in line[field]]
            line['start']=line['polyline'][0];line['end']=line['polyline'][-1]
    return result

def homr_staves(geometry):
    staves=[]
    for si,system in enumerate(geometry['systems']):
        for physical in system['staffs']:
            grid=physical['grid']
            for index in range(len(grid[0]['y'])//5):
                lines=[]
                for k in range(5):
                    points=[[p['x'],p['y'][5*index+k]] for p in grid]
                    lines.append({'number':k+1,'start':points[0],'end':points[-1],'polyline':points})
                space=float(np.median([np.diff(p['y'][5*index:5*index+5]) for p in grid]))
                staves.append({'index':len(staves),'system':si+1,'space_px':space,'lines':lines})
    return staves

def save_images(image,out,geometry=None):
    gray=cv2.imread(str(image),0)
    if gray is None: raise ValueError('image unreadable')
    provided=json.loads(geometry.read_text(encoding='utf8')) if geometry else None
    if provided and 'image_size_px' in provided:
        if provided['image_size_px']!=[gray.shape[1],gray.shape[0]]:raise ValueError('geometry/image size mismatch')
        if provided.get('image_sha256') and provided['image_sha256']!=hashlib.sha256(image.read_bytes()).hexdigest():raise ValueError('geometry belongs to another photograph')
    if provided and 'preprocessed_size' in provided:
        assert provided['preprocessed_size']==[gray.shape[1],gray.shape[0]]
        provided={'staves':homr_staves(provided),'geometry_source':'Frozen Homr segmentation in this preprocessed image frame'}
    clean,staves,erased,protected,stats=remove(gray,provided['staves'] if provided else None)
    out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/'original.png'),gray);cv2.imwrite(str(out/'staffless.png'),clean)
    cv2.imwrite(str(out/'removed_mask.png'),erased.astype('uint8')*255)
    overlay=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for staff in staves:
        for line in staff['lines']:
            pts=np.rint(line['polyline']).astype('int32')
            cv2.polylines(overlay,[pts],False,(0,180,0),1)
            cv2.circle(overlay,tuple(pts[0]),4,(255,0,0),-1)
            cv2.circle(overlay,tuple(pts[-1]),4,(0,0,255),-1)
    cv2.imwrite(str(out/'staff_endpoints.png'),overlay)
    comparison=np.concatenate([cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR),cv2.cvtColor(clean,cv2.COLOR_GRAY2BGR)],axis=1)
    cv2.imwrite(str(out/'comparison.png'),comparison)
    data={'image':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
          'image_size_px':[gray.shape[1],gray.shape[0]],'coordinate_space':'input image pixels',
          'line_numbering':'top=1,bottom=5','staves':staves,'statistics':stats,'algorithm':'conservative_v2_crossing_tangent',
          'geometry_source':provided.get('geometry_source') if provided else 'experimental Hough detector',
          'limitations':['Conservative image-only estimate, not all intersections guaranteed.',
                         'Original image retained; clef, key and accidentals still needed for pitch.']}
    (out/'staff_geometry.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print(json.dumps(stats),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--geometry',type=Path,required=True,help='Recorded image-derived staff geometry in the same image coordinate frame')
    a=p.parse_args();save_images(a.image,a.output,a.geometry)
