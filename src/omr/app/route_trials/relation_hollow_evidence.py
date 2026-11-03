"""Bounded enclosed-white-area proof for line-crossed hollow noteheads.

The caller must separately bind the candidate to a segmented head and verify
its staff/pitch position. No staff line is removed or painted over. The proof
uses the same enclosed-area thresholds as relation_measure_realign's existing
hollow_on_line classifier, without requiring its graph coordinate frame.
"""
import cv2


def enclosed_hollow(head,image,space):
    if image.ndim!=2 or not 5<=space<=80:return False
    x,y=map(round,head['xy']);rx=max(2,round(.8*space));ry=max(2,round(.7*space))
    if min(x-rx,y-ry)<0 or x+rx>=image.shape[1] or y+ry>=image.shape[0]:return False
    crop=image[y-ry:y+ry+1,x-rx:x+rx+1]
    _,_,stats,centres=cv2.connectedComponentsWithStats((crop>=140).astype('uint8'),8)
    area=0
    for (cx,cy),(lx,ly,w,h,n) in zip(centres[1:],stats[1:]):
        if lx==0 or ly==0 or lx+w>=crop.shape[1] or ly+h>=crop.shape[0]:continue
        if n>=6 and abs(cx-rx)<.45*space and abs(cy-ry)<.45*space:area+=int(n)
    return .15*space*space<=area<=.85*space*space
