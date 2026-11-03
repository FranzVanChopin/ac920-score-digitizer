"""Whole-chain ratio hypothesis certified by ink, a number and exact meter."""
from fractions import Fraction
from itertools import product
from .relation_joint_lane_clock import proposal as strict_proposal,fuse as lane_fuse
from .relation_tuplets import geometry_match
from repair_tuplets import continuous_beams,printed_number
import numpy as np


def graph_end(h,ink,up):
    g=h.get('group');s=h['staff_space']
    if not g or g.get('direction')!=('up' if up else 'down') or not g.get('beam_relations'):return None
    rs=g['beam_relations'];r=(min if up else max)(rs,key=lambda a:sum(a['y_range']))
    x=r['stem_x'];y=sum(r['y_range'])/2
    if abs(x-h['x'])>s or not 2*s<abs(y-h['y'])<9*s:return None
    return x,y


def graph_beam(a,b,ink,up,count):
    ends=[graph_end(h,ink,up) for h in (a,b)];s=a['staff_space']
    if any(e is None for e in ends) or any(h['group'].get('beams')!=count for h in (a,b)):return False
    (xa,ya),(xb,yb)=ends
    if not .7*s<xb-xa<8*s or abs(ya-yb)>2*s:return False
    for f in (.125,.25,.375,.5,.625,.75,.875):
        x=round(xa+(xb-xa)*f);y=ya+(yb-ya)*f;top=max(0,round(y-1.2*s));bottom=min(ink.shape[0],round(y+1.2*s))
        if x<0 or x>=ink.shape[1]:return False
        bits=ink[top:bottom,x]>0;edges=np.diff(np.r_[False,bits,False].astype('int8'));spans=[(top+l,top+r) for l,r in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if .2*s<=r-l<=.8*s]
        candidates=[(l,r) for l,r in spans if y-.45*s<=(l+r-1)/2<=y+.45*s or (up and y<=(l+r-1)/2<=y+1.1*s) or (not up and y-1.1*s<=(l+r-1)/2<=y)]
        if len(candidates)!=count:return False
        outer=(min if up else max)((l+r-1)/2 for l,r in candidates)
        if abs(outer-y)>.4*s:return False
    return True


def proposal(groups,bound,row,track,image,ink,meter):
    old=strict_proposal(groups,bound,row,track,image,ink,meter)
    if old:return old
    if not 6<=len(groups)<=24 or len(groups)!=len(bound) or not geometry_match(groups,bound,row,track,image):return None
    if any(r['grace'] or r['midi'] is None or r['node'].get('print-object')=='no' or r['node'].find('tie') is not None or r['node'].find('notations/tied') is not None or r['node'].findall('dot') for g in groups for r in g):return None
    kinds={r['node'].findtext('type') for g in groups for r in g}
    if len(kinds)!=1 or next(iter(kinds)) not in ('eighth','16th'):return None
    kind=next(iter(kinds));unit=Fraction(1,2) if kind=='eighth' else Fraction(1,4);beams=1 if kind=='eighth' else 2;choices=[(a,n) for a,n in ((3,2),(4,3)) if len(groups)%a==0 and len(groups)*unit*n/a==meter]
    if len(choices)!=1:return None
    a,n=choices[0];dur=unit*n/a
    if any(r['duration'] not in (unit,dur) or r['node'].find('time-modification') is not None and (r['node'].findtext('time-modification/actual-notes'),r['node'].findtext('time-modification/normal-notes'))!=(str(a),str(n)) for g in groups for r in g):return None
    proof=[]
    for up in (True,False):
        edge=min if up else max;heads=[edge(g['heads'],key=lambda h:h['xy'][1]) for g in bound];hs=[dict(x=h['xy'][0],y=h['xy'][1],staff_space=row['space_px'],group=g) for h,g in zip(heads,bound)]
        if not all(graph_beam(x,y,ink,up,beams) for x,y in zip(hs,hs[1:])):continue
        witnessed=[i for i in range(0,len(hs),a) if printed_number(hs[i:i+a],ink,a,recover_crossings=True,flexible_beams=True,beam_count=beams,beam_proof_fn=lambda x,y,im,up:graph_beam(x,y,im,up,beams),stem_end_fn=graph_end)]
        if witnessed:proof.append(witnessed)
    if len(proof)!=1:return segmented(groups,bound,row,track,image,ink,meter,kind,unit,beams)
    return {'count':a,'normal':n,'duration':dur,'type':kind,'beams':beams,'whole_chain_meter_proof':True}


def segmented(groups,bound,row,track,image,ink,meter,kind,unit,beams):
    plans=[]
    for up in (True,False):
        edge=min if up else max;hs=[dict(x=(h:=edge(g['heads'],key=lambda h:h['xy'][1]))['xy'][0],y=h['xy'][1],staff_space=row['space_px']) for g in bound]
        cuts=[0]+[i for i in range(1,len(hs)) if not continuous_beams(hs[i-1],hs[i],ink,up,beams)]+[len(hs)]
        if not 2<=len(cuts)-1<=8:continue
        cells=[];numbers=[]
        for lo,hi in zip(cuts,cuts[1:]):
            count=hi-lo
            if count not in (3,4):break
            seen=printed_number(hs[lo:hi],ink,count,recover_crossings=True,flexible_beams=True,beam_count=beams)
            numbers.append(seen);cells.append((lo,hi,count))
        else:
            if not any(numbers):continue
            choices=[]
            for (lo,hi,count),seen in zip(cells,numbers):
                opts=[(count,2 if count==3 else 3)] if seen else [(1,1),(count,2 if count==3 else 3)]
                # An unnumbered cell can inherit only an observed ratio of
                # the same beam-chain cardinality elsewhere in this lane.
                if not seen and not any(num and cell[2]==count for cell,num in zip(cells,numbers)):opts=[(1,1)]
                choices.append(opts)
            solutions=[]
            for ratios in product(*choices):
                ds=[unit*n/a for (lo,hi,count),(a,n) in zip(cells,ratios) for i in range(lo,hi)]
                if sum(ds,Fraction())!=meter:continue
                if any(r['duration'] not in (unit,d) for group,d in zip(groups,ds) for r in group):continue
                solutions.append((ratios,ds))
            if len(solutions)!=1:continue
            ratios,ds=solutions[0];plans.append({'count':cells[0][2],'normal':ratios[0][1],'duration':ds[0],'durations':ds,'cells':[{'start':lo,'end':hi,'count':a,'normal':n} for (lo,hi,count),(a,n) in zip(cells,ratios)],'type':kind,'beams':beams,'segmented_meter_proof':True})
    return plans[0] if len(plans)==1 else None


def fuse(root,graph,photo,geometry):
    changes=lane_fuse(root,graph,photo,geometry,proposal_fn=proposal)
    return changes,{'method':'complete_continuous_chain_one_or_more_number_witnesses_and_exact_meter'}
