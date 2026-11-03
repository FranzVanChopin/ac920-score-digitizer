"""Read full-height piano barlines without trusting decoder measure tokens.

Missing grid data, repeats/empty intervals and unclear boundaries cause
abstention. This is an image consistency check, not an accuracy estimator.
"""

from statistics import median


def measure_systems(root):
    parts = root.findall("part")
    if len(parts) != 1:
        return []
    result = [[]]
    for index, measure in enumerate(parts[0].findall("measure")):
        p = measure.find("print")
        if index and p is not None and p.get("new-system") == "yes":
            result.append([])
        result[-1].append(measure)
    return result


def bar_bounds(staff, image):
    import cv2
    grid = staff.get("grid", [])
    if not grid or any(len(p["y"]) != 10 for p in grid):
        return None
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    space = median(b-a for p in grid for a, b in
                   zip(p["y"][:4], p["y"][1:5]))
    top, bottom = map(round, staff["y_range"])
    left = max(0, round(staff["x_range"][0]-space))
    right = min(image.shape[1]-1, round(staff["x_range"][1]+space))
    if space < 4 or top < 0 or bottom > image.shape[0] or bottom-top < 8*space:
        return None
    columns = [x for x in range(left, right+1)
               if (image[top:bottom, x] < 140).mean() >= 0.92]
    groups = []
    for x in columns:
        if not groups or x-groups[-1][-1] > max(3, 0.65*space):
            groups.append([])
        groups[-1].append(x)
    if (not 2 <= len(groups) <= 33
            or any(g[-1]-g[0] > space for g in groups)):
        return None
    bounds = [sum(g)/len(g) for g in groups]
    if (abs(bounds[0]-staff["x_range"][0]) > 1.5*space
            or abs(bounds[-1]-staff["x_range"][1]) > 1.5*space
            or any(b-a < 3*space for a, b in zip(bounds, bounds[1:]))):
        return None
    return bounds


def barline_review(root, geometry, image):
    systems = measure_systems(root)
    if len(systems) != len(geometry.get("systems", [])):
        return [{"reason": "system_count_disagrees", "xml": len(systems),
                 "image": len(geometry.get("systems", []))}]
    result = []
    for index, (measures, system) in enumerate(zip(systems, geometry["systems"]), 1):
        if len(system.get("staffs", [])) != 1:
            continue
        staff = system["staffs"][0]
        bounds = bar_bounds(staff, image)
        # Occlusion can hide a real line, so fewer raster boundaries are not
        # proof of an extra decoder measure. Report only positively visible
        # extra intervals that the decoder has merged.
        if bounds is None or len(bounds)-1 <= len(measures):
            continue
        # A leading repeat/header line or an empty bar lacks note evidence.
        # Do not turn such a raster count into a confident mismatch report.
        mid = sum(staff["y_range"])/2
        if not all(any(a < h["x"] < b and (h["y"] < mid) == upper
                       for h in staff.get("notes", []))
                   for a, b in zip(bounds, bounds[1:]) for upper in (True, False)):
            continue
        result.append({"system": index, "reason": "visible_bar_count_disagrees",
                       "xml_measures": len(measures), "visible_measures": len(bounds)-1,
                       "bar_x": [round(x, 2) for x in bounds]})
    return result


def single_bar_bounds(staff,image,expected_count,trailing_meter_x=()):
    """Require the exact decoded count of full-height five-line bar strokes."""
    import cv2
    import numpy as np
    grid=staff.get('grid',[])
    if not grid or any(len(p['y'])!=5 for p in grid) or not 1<=expected_count<=32:return None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    space=median(b-a for p in grid for a,b in zip(p['y'],p['y'][1:]))
    left,right=map(round,staff['x_range'])
    xs=np.arange(max(0,left),min(image.shape[1],right+round(space)+1))
    points=sorted({p['x']:p['y'] for p in grid}.items());gx=[x for x,_ in points]
    top=np.interp(xs,gx,[y[0] for _,y in points]);bottom=np.interp(xs,gx,[y[-1] for _,y in points])
    ys=np.rint(top[:,None]+(bottom-top)[:,None]*np.linspace(0,1,48)).astype(int)
    if ys.min()<0 or ys.max()>=image.shape[0]:return None
    columns=xs[(image[ys,xs[:,None]]<140).mean(axis=1)>=0.92]
    groups=[]
    for x in columns:
        if x<left+3*space:continue
        if not groups or x-groups[-1][-1]>max(3,0.65*space):groups.append([])
        groups[-1].append(int(x))
    if len(groups)!=expected_count or any(g[-1]-g[0]>space for g in groups):return None
    ends=[sum(g)/len(g) for g in groups]
    right_ok=(abs(ends[-1]-right)<=1.5*space or
              0<right-ends[-1]<=6*space and (
                  any(ends[-1]<x<right+3*space for x in trailing_meter_x) or
                  groups[-1][-1]-groups[-1][0]>=.45*space and
                  not any(h['x']>ends[-1]+.5*space for h in staff.get('notes',[]))))
    if not right_ok or any(b-a<3*space for a,b in zip([left]+ends,ends)):
        return None
    return [float(left)]+ends


def anchored_bar_bounds(staff, image):
    """Raster/grid bounds for five-line crops, excluding note-attached stems.

    Small horizontal dilation tolerates photo slant and line-fit noise. A
    nearby segmented head makes a five-line vertical stroke ambiguous.
    The caller must still require its independent decoded measure count.
    """
    import cv2
    import numpy as np
    grid=staff.get('grid',[])
    if not grid or any(len(p['y'])!=5 for p in grid):return None
    if image.ndim==3:image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    space=median(b-a for p in grid for a,b in zip(p['y'],p['y'][1:]))
    if not 5<=space<=40:return None
    left,right=map(round,staff['x_range'])
    xs=np.arange(max(0,left),min(image.shape[1],right+round(space)+1))
    points=sorted({p['x']:p['y'] for p in grid}.items())
    top=np.interp(xs,[x for x,_ in points],[y[0] for _,y in points])
    bottom=np.interp(xs,[x for x,_ in points],[y[-1] for _,y in points])
    ys=np.rint(top[:,None]+(bottom-top)[:,None]*np.linspace(.05,.95,48)).astype(int)
    if ys.min()<0 or ys.max()>=image.shape[0]:return None
    ink=cv2.dilate((image<140).astype('uint8'),np.ones((1,3),dtype='uint8'))
    columns=xs[ink[ys,xs[:,None]].mean(axis=1)>=.94]
    groups=[]
    for x in columns:
        if x<left+3*space:continue
        if not groups or x-groups[-1][-1]>max(3,.65*space):groups.append([])
        groups[-1].append(int(x))
    ends=[]
    for group in groups:
        x=sum(group)/len(group)
        if group[-1]-group[0]>space:return None
        if any(abs(h['x']-x)<.85*space for h in staff.get('notes',[])):continue
        ends.append(x)
    if (not 1<=len(ends)<=32 or abs(ends[-1]-right)>1.5*space
        or any(b-a<3*space for a,b in zip([left]+ends,ends))):return None
    return [float(left)]+ends
