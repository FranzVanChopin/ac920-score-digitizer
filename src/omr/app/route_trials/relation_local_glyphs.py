"""Independent connected glyph proposals in bounded chord-side windows."""
import copy
import cv2,numpy as np
from .relation_joint_candidates import component
from .relation_coverage import inputs
from .staff_preprocess import staff_step


def enrich(root,graph,photo,geometry):
    loaded=inputs(root,graph,photo,geometry)
    if loaded is None:return graph,[]
    image,tracks=loaded;result=copy.deepcopy(graph);changes=[];next_id=max((n['id'] for row in graph['systems'] for n in row['nodes']),default=0)+1
    for row in result['systems']:
        candidates=[t for t in tracks if t['system']==row['system']]
        track=candidates[row['staff']-1] if 0<row['staff']<=len(candidates) else None
        s=row['space_px'];proposals=[];seen=set()
        for g in row['groups']:
            hs=g.get('heads',[])
            if not hs or g.get('grace') or len(hs)>8:continue
            for h in hs:
                x,y=h['xy'];l,r=round(x-5*s),round(x-.55*s);t,b=round(y-2.5*s),round(y+2.5*s)
                if min(l,t)<0 or r>=image.shape[1] or b>=image.shape[0]:continue
                ink=(image[t:b,l:r]<140).astype('uint8');horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(3,round(3*s))),np.uint8));clean=cv2.morphologyEx(cv2.subtract(ink,horizontal),cv2.MORPH_CLOSE,np.ones((3,1),np.uint8));count,lab,stats,_=cv2.connectedComponentsWithStats(clean)
                if count>64:continue
                for xx,yy,w,hh,area in stats[1:]:
                    if not .35*s<=w<=1.35*s or not 2*s<=hh<=3.6*s:continue
                    ax,ay=l+xx+(w-1)/2,t+yy+(hh-1)/2;key=(round(ax),round(ay))
                    if key in seen:continue
                    seen.add(key);valid=[]
                    for cls in (8,9,10):
                        n={'class':cls,'xy':[ax,ay],'confidence':0.,'independent_shape_proposal':True}
                        if component(n,image,s) is not None:valid.append(n)
                    if len(valid)!=1:continue
                    n=valid[0]
                    if not any(.65<(a['xy'][0]-ax)/s<4.5 and abs((ay-a['xy'][1])/s-(-.75 if n['class']==9 else 0))<.3 for a in hs):continue
                    nearby=[a for a in row['nodes'] if a['class']==n['class'] and np.linalg.norm(np.array(a['xy'])-[ax,ay])<.6*s]
                    if len(nearby)==1:
                        old=nearby[0]['xy']
                        # A sharp's slanted crossbars make its connected-box
                        # centre asymmetric. Do not move a recognized glyph
                        # to a different diatonic position merely to centre
                        # that box; that can steal it from the preceding chord.
                        anchor=.75*s if n['class']==9 else 0.
                        if track is None or staff_step(track,old[0],old[1]+anchor)['nearest_step']!=staff_step(track,ax,ay+anchor)['nearest_step']:continue
                        nearby[0]['xy']=n['xy'];nearby[0]['independent_shape_proposal']=True;changes.append({'node':nearby[0]['id'],'old_xy':old,'xy':n['xy'],'method':'isolated_glyph_bbox_center'})
                    elif not nearby and len(proposals)<128:
                        n['id']=next_id;next_id+=1;proposals.append(n);changes.append({'node':n['id'],'xy':n['xy'],'class':n['class'],'method':'isolated_glyph_local_proposal'})
        if len(row['nodes'])+len(proposals)<=2048:row['nodes']+=proposals
    return result,changes


def fuse(root,graph,photo,geometry):
    from .relation_measure_joint import fuse as joint
    richer,changes=enrich(root,graph,photo,geometry)
    repairs=joint(root,richer,photo,geometry,expanded_candidates=True)
    return repairs,{'glyph_proposals':len(changes),'details':changes}
