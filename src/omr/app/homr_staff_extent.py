"""Experimental recovery of truncated staff grids before grand-staff merging.

Homr intersects the x coordinates of paired staves. One prematurely ended
grid can therefore crop an otherwise visible measure from both staves.
Extend only to another detected grid's endpoint, with all five raster lines
continuing throughout every bounded window. No measure count is consulted.
"""
import numpy as np
import cv2


def continued_grid(grid,target,image):
    if not 12<=len(grid)<=1024 or len(grid[0]['y']) not in (5,10):return None
    count=len(grid[0]['y']);xs=np.array([p['x'] for p in grid],float)
    ys=np.array([p['y'] for p in grid],float)
    if ys.shape!=(len(grid),count) or not np.isfinite(ys).all():return None
    spacing=np.median(np.concatenate([np.diff(ys[:,i:i+5],axis=1).ravel() for i in range(0,count,5)]))
    step=float(np.median(np.diff(xs)))
    if not 5<=spacing<=40 or not 1<=step<=spacing or not 3*spacing<target-xs[-1]<=64*spacing:return None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if image.size>16000000 or max(image.shape)>8192 or target>=image.shape[1]:return None
    fits=[]
    for i in range(count):
        # The premature endpoint can itself be several pixels off the real
        # line. Fit long raster runs around the existing grid instead of
        # extrapolating that erroneous endpoint into empty space.
        center=float(np.median(ys[:,i]));radius=round(.6*spacing)
        top,bottom=round(center)-radius,round(center)+radius+1
        if top<0 or bottom>image.shape[0]:return None
        window=round(10*spacing);samples=[]
        for left in range(round(xs[0]+2*spacing),round(target)-window//2,window):
            right=min(round(target),left+window)
            q=(image[top:bottom,left:right]<150).mean(1)
            peak=float(q.max())
            if peak<.6:continue
            choices=np.flatnonzero(q>=max(.6,peak-.08));groups=[]
            for value in choices:
                if not groups or value-groups[-1][-1]>1:groups.append([])
                groups[-1].append(int(value))
            bands=[g for g in groups if len(g)<=max(3,round(.25*spacing))]
            if len(bands)!=1:continue
            samples.append(((left+right-1)/2,top+float(np.mean(bands[0]))))
        if len(samples)<5:return None
        sx,sy=np.array(samples).T;slope,intercept=np.polyfit(sx,sy,1)
        keep=abs(sy-(slope*sx+intercept))<=.15*spacing
        if keep.sum()<.8*len(samples):return None
        slope,intercept=np.polyfit(sx[keep],sy[keep],1)
        if abs(slope)>.03 or np.median(abs(ys[:,i]-(slope*xs+intercept)))>.6*spacing:return None
        fits.append((slope,intercept))
    xx=np.arange(round(xs[-1]-2*spacing),round(target)+1)
    offsets=np.arange(-max(1,round(.15*spacing)),max(1,round(.15*spacing))+1)
    block=max(16,round(4*spacing));supports=[]
    for slope,intercept in fits:
        yy=np.rint(slope*xx+intercept).astype(int)
        if yy.min()+offsets.min()<0 or yy.max()+offsets.max()>=image.shape[0]:return None
        hits=(image[yy[:,None]+offsets,xx[:,None]]<150).any(1)
        scores=[float(hits[k:k+block].mean()) for k in range(0,len(hits),block)]
        if min(scores)<.9:return None
        supports.append(min(scores))
    extension=[]
    for x in np.arange(xs[-1]+step,target+step/2,step):
        values=[float(a*x+b) for a,b in fits]
        if any(abs(b-a-spacing)>.2*spacing for i in range(0,count,5) for a,b in zip(values[i:i+4],values[i+1:i+5])):return None
        extension.append(dict(x=float(x),y=values))
    if not extension or abs(extension[-1]['x']-target)>.5:return None
    return extension,dict(from_x=float(xs[-1]),to_x=float(target),added_points=len(extension),minimum_line_support=min(supports),method='five_continuous_raster_lines_before_grandstaff_intersection')


def install_staff_extent(*,apply_extensions=False):
    import homr.main as main
    from homr.model import Staff,StaffPoint
    original=main.detect_staff;reports=[]
    def detect(debug,mask,*args):
        staffs=original(debug,mask,*args)
        if not 2<=len(staffs)<=128:return staffs
        endpoints=sorted({float(s.max_x) for s in staffs},reverse=True);result=[]
        for index,staff in enumerate(staffs):
            grid=[dict(x=float(p.x),y=list(map(float,p.y))) for p in staff.grid]
            replacement=None
            for target in endpoints:
                if target<=staff.max_x:continue
                extension=continued_grid(grid,target,debug.original_image)
                if extension is None:continue
                points,report=extension
                replacement=Staff(list(staff.grid)+[StaffPoint(p['x'],p['y'],staff.grid[-1].angle) for p in points])
                replacement.symbols=list(staff.symbols)
                reports.append(dict(report,staff_index=index));break
            result.append(replacement if apply_extensions and replacement is not None else staff)
        return result
    main.detect_staff=detect
    return reports
