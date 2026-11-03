"""Canonical staff-spacing inference, returned in the legacy graph frame.

Only image-derived staff spacing selects the scale. This remains opt-in until
the full event regression gate passes; it does not consult score identities.
"""
import cv2
import numpy as np
from . import symbol_cnn as sn,relation_net as rn


def infer(symbol,stroke,raw,staffs,target_space=16.):
    spaces=np.asarray([s['space_px'] for s in staffs],dtype='float64')
    if not len(spaces) or not np.isfinite(spaces).all() or np.any(spaces<=0):
        raise ValueError('Invalid image-derived staff spacing')
    median=float(np.median(spaces));factor=target_space/median
    if not 8<=target_space<=24 or not .25<=factor<=2:
        raise ValueError('Staff normalization exceeds bounded scale range')
    h,w=raw.shape;size=(round(w*factor),round(h*factor))
    if min(size)<32 or size[0]*size[1]>16000000 or max(size)>8192:
        raise ValueError('Normalized inference exceeds image memory guard')
    scaled=cv2.resize(raw,size,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)
    predictions=sn.predict(symbol,scaled);prob=rn.predict(stroke,scaled)
    legacy=(round(w*sn.FACTOR),round(h*sn.FACTOR))
    # Resize uses half-pixel centres. Separate x/y ratios preserve rounding.
    predictions[:,1]=(predictions[:,1]+.5)*legacy[0]/size[0]-.5
    predictions[:,2]=(predictions[:,2]+.5)*legacy[1]/size[1]-.5
    projected=np.empty((6,legacy[1],legacy[0]),dtype=np.uint8)
    for k in range(6):projected[k]=cv2.resize(prob[k],legacy,interpolation=cv2.INTER_NEAREST)
    report={'mode':'staff','target_space_px':target_space,'measured_median_space_px':median,
            'scale':factor,'inference_size':list(size),'graph_size':list(legacy),
            'tile_shape':[sn.W,sn.H],'tile_overlap_px':32,'sequential_tiles':True,
            'coordinate_space':'photograph pixels scaled by 16/19.8',
            'reference_supplied_to_recognizer':False}
    return predictions,projected,report
