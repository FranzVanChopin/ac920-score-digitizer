"""Unpromoted raster-certified staff continuation in either direction.

The formal tail implementation remains untouched. Endpoint trials can stop up
to two grid steps inside another detected endpoint, because that endpoint may
include a small blank margin. Every actual new sample still needs five-line
support; support thresholds, slope and extension-length bounds are unchanged.
"""
import math
import numpy as np
import cv2
from homr_staff_extent import continued_grid


def extend_grid(grid,starts,ends,image):
    if not grid or len(grid)>1024:return grid,[]
    step=float(np.median(np.diff([p['x'] for p in grid])))
    if not 1<=step<=40:return grid,[]
    result=list(grid);reports=[];width=image.shape[1]-1
    for direction,targets in [('right',sorted(ends,reverse=True)),('left',sorted(starts))]:
        mirror=direction=='left'
        work=[dict(x=width-p['x'],y=p['y']) for p in result[::-1]] if mirror else result
        raster=image[:,::-1] if mirror else image
        accepted=None
        for target in targets:
            target=width-target if mirror else target
            if target<=work[-1]['x']:continue
            aligned=work[-1]['x']+math.floor((target-work[-1]['x'])/step)*step
            for inward in range(3):
                trial=continued_grid(work,aligned-inward*step,raster)
                if trial is not None and len(result)+len(trial[0])<=1024:
                    accepted=trial;break
            if accepted is not None:break
        if accepted is None:continue
        points,report=accepted
        if mirror:
            points=[dict(x=width-p['x'],y=p['y']) for p in points[::-1]]
            report=dict(report,from_x=width-report['from_x'],to_x=width-report['to_x'])
            result=points+result
        else:result=result+points
        reports.append(dict(report,direction=direction))
    return result,reports


def install_staff_extent(*,apply_extensions=False):
    import homr.main as main
    from homr.model import Staff,StaffPoint
    original=main.detect_staff;reports=[]
    def detect(debug,mask,*args):
        staffs=original(debug,mask,*args)
        if not 2<=len(staffs)<=128:return staffs
        image=debug.original_image
        if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if image.size>16000000 or max(image.shape)>8192:return staffs
        starts={float(s.min_x) for s in staffs};ends={float(s.max_x) for s in staffs};result=[]
        for index,staff in enumerate(staffs):
            grid=[dict(x=float(p.x),y=list(map(float,p.y))) for p in staff.grid]
            extended,changes=extend_grid(grid,starts,ends,image)
            if changes and apply_extensions:
                replacement=Staff([StaffPoint(p['x'],p['y'],staff.grid[0].angle) for p in extended]);replacement.symbols=list(staff.symbols);result.append(replacement)
            else:result.append(staff)
            reports.extend(dict(r,staff_index=index) for r in changes)
        return result
    main.detect_staff=detect
    return reports
