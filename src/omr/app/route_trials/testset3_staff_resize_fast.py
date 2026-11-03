"""Exact staff scaling without copying coordinates that are replaced immediately.

Only the ordinary, non-shared plain geometry schema takes the fast path.
Unknown metadata, shared containers, cycles and custom objects use the original
deepcopy implementation, preserving its alias graph and exceptional behavior.
No model, reference, cache across calls, or rounding change is introduced.
"""
from .staff_preprocess import resize_staves as _original_resize

MAXIMUM_STAVES=128
MAXIMUM_LINES=640
MAXIMUM_POINTS=500_000
_counts={'fast_calls':0,'fallback_calls':0}


def _ordinary(staves,sx,sy):
    if type(staves) is not list or type(sx) not in (int,float) or type(sy) not in (int,float):return False
    if len(staves)>MAXIMUM_STAVES:return False
    seen={id(staves)};lines_count=points_count=0
    def unique(value):
        identity=id(value)
        if identity in seen:return False
        seen.add(identity);return True
    def point(value):
        return type(value) is list and len(value)==2 and all(type(v) in (int,float) for v in value)
    for staff in staves:
        if type(staff) is not dict or set(staff)!={'index','system','space_px','lines'} or not unique(staff):return False
        if any(type(staff[k]) not in (int,float) for k in ('index','system','space_px')):return False
        lines=staff['lines']
        if type(lines) is not list or not unique(lines):return False
        lines_count+=len(lines)
        if lines_count>MAXIMUM_LINES:return False
        for line in lines:
            if type(line) is not dict or set(line)!={'number','polyline','start','end'} or not unique(line):return False
            if type(line['number']) not in (int,float) or not point(line['start']) or not point(line['end']):return False
            polyline=line['polyline']
            if type(polyline) is not list or not polyline or not unique(polyline):return False
            points_count+=len(polyline)
            if points_count>MAXIMUM_POINTS:return False
            for xy in polyline:
                if not point(xy) or not unique(xy):return False
    return True


def resize_staves(staves,sx,sy):
    """Same API, formulas, field order and endpoint aliases as frozen helper."""
    if not _ordinary(staves,sx,sy):
        _counts['fallback_calls']+=1
        return _original_resize(staves,sx,sy)
    _counts['fast_calls']+=1;result=[]
    for staff in staves:
        new_staff=staff.copy();new_staff['space_px']*=sy;new_lines=[]
        for line in staff['lines']:
            new_line=line.copy()
            new_line['polyline']=[[(x+.5)*sx-.5,(y+.5)*sy-.5] for x,y in line['polyline']]
            new_line['start']=new_line['polyline'][0];new_line['end']=new_line['polyline'][-1]
            new_lines.append(new_line)
        new_staff['lines']=new_lines;result.append(new_staff)
    return result


def statistics():return dict(_counts)


def reset_statistics():
    _counts.update(fast_calls=0,fallback_calls=0)
