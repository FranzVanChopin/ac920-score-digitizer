"""Exact group-statistics reuse for the frozen stroke Hough decoder.

The original function remains an independent baseline. Hough, mask, order,
comparisons and final polyfit are unchanged. A group's original np.mean/min/
max formulas are refreshed after each append; no incremental float sum is
used. Only statistics for currently unchanged groups are reused.
"""
import cv2
import numpy as np

from .relation_decode import strokes as baseline_strokes


def strokes(prob, threshold, space=16., minimum=1.6):
    mask = (prob >= round(threshold*255)).astype('uint8')
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,1),np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((round(minimum*space),1),np.uint8))
    cv2.setRNGSeed(7123)
    lines = cv2.HoughLinesP(mask, 1, np.pi/720, threshold=round(minimum*space*.65),
                          minLineLength=round(minimum*space), maxLineGap=round(.3*space))
    groups, statistics = [], []
    for x1,y1,x2,y2 in sorted(lines.reshape(-1,4).tolist() if lines is not None else [],
                            key=lambda p: -abs(p[3]-p[1])):
        if abs(y2-y1) < minimum*space or abs(x2-x1) > max(2,.08*abs(y2-y1)):
            continue
        if y1 > y2:
            x1,y1,x2,y2 = x2,y2,x1,y1
        x = (x1+x2)/2
        selected = None
        for index, group in enumerate(groups):
            gx,lo,hi = statistics[index]
            if abs(x-gx) < .35*space and y1 < hi+.4*space and y2 > lo-.4*space:
                selected = index
                break
        if selected is None:
            selected = len(groups)
            groups.append([[x1,y1,x2,y2]])
            statistics.append(None)
        else:
            groups[selected].append([x1,y1,x2,y2])
        group = groups[selected]
        statistics[selected] = (np.mean([(p[0]+p[2])/2 for p in group]),
                                min(p[1] for p in group), max(p[3] for p in group))
    result = []
    for group in groups:
        xy = np.array(group).reshape(-1,2)
        lo,hi = int(xy[:,1].min()), int(xy[:,1].max())
        slope,intercept = np.polyfit(xy[:,1],xy[:,0],1)
        a,b = float(slope*lo+intercept), float(slope*hi+intercept)
        yy = np.arange(lo,hi+1)
        xx = np.clip(np.rint(slope*yy+intercept).astype(int),0,prob.shape[1]-1)
        result.append({'id':len(result), 'xy':[[a,float(lo)],[b,float(hi)]],
                       'confidence':float(prob[yy,xx].mean()/255),
                       'bbox':[int(min(a,b)),lo,max(1,int(abs(b-a))+1),hi-lo+1]})
    return result
