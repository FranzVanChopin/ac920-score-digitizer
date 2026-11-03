"""Bound image decoding and Homr's fixed-width resize before model loading.

This read-only check uses the installed Homr autocrop and size calculation,
which import NumPy/Pillow/OpenCV only. It never calls resize or inference and
does not alter the input photograph. The returned payload contains metadata,
not decoded image arrays.
"""
from pathlib import Path

MAXIMUM_PIXELS=16000000
MAXIMUM_DIMENSION=8192


class HomrImageLimitError(ValueError):
    def __init__(self,reason,message,details=None):
        self.reason=reason;self.details=dict(details or {})
        super().__init__(message)


def _check_size(width,height,stage,details):
    if width<=0 or height<=0:
        raise HomrImageLimitError(stage+'_empty',f'{stage} image has no pixels',details)
    if width*height>MAXIMUM_PIXELS or max(width,height)>MAXIMUM_DIMENSION:
        raise HomrImageLimitError(stage+'_bounds',f'{stage} image {width}x{height} exceeds {MAXIMUM_PIXELS} pixels or {MAXIMUM_DIMENSION} per dimension',details)


def validate_homr_image(source):
    """Return crop/target dimensions, or reject before any unbounded resize.

    Call immediately before Homr's detector starts, on the same immutable
    staged image path. Both the encoded header and actual autocrop result are
    checked. The arrays are released before returning or raising.
    """
    from PIL import Image
    path=Path(source);details={'source':str(path.resolve())}
    try:
        with Image.open(path) as header:
            width,height=map(int,header.size);details['format']=header.format
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        raise HomrImageLimitError('header_unreadable','Cannot read image metadata: '+str(exc),details) from exc
    details['header_size']=[width,height]
    _check_size(width,height,'header',details)
    import cv2
    # These modules do not import Homr's model loader or inference sessions.
    from homr.autocrop import autocrop
    from homr.resize import calc_target_image_size
    raw=cropped=None
    try:
        raw=cv2.imread(str(path))
        if raw is None:raise HomrImageLimitError('decode_unreadable','OpenCV cannot decode image pixels',details)
        rh,rw=raw.shape[:2];details['decoded_size']=[rw,rh]
        _check_size(rw,rh,'decoded',details)
        cropped=autocrop(raw);ch,cw=cropped.shape[:2];details['autocrop_size']=[cw,ch]
        _check_size(cw,ch,'autocrop',details)
        offset=int(cropped.__array_interface__['data'][0]-raw.__array_interface__['data'][0]);top,left_bytes=divmod(offset,raw.strides[0])
        details['autocrop_xy']=[int(left_bytes//raw.strides[1]),int(top)]
        tw,th=calc_target_image_size(cw,ch);details['homr_target_size']=[int(tw),int(th)]
        _check_size(tw,th,'homr_target',details)
        details.update(accepted=True,maximum_pixels=MAXIMUM_PIXELS,maximum_dimension=MAXIMUM_DIMENSION,resized=False,input_modified=False)
        return details
    except HomrImageLimitError:raise
    except (cv2.error,ValueError,OverflowError) as exc:
        raise HomrImageLimitError('preprocessing_failed','Cannot certify bounded Homr preprocessing: '+str(exc),details) from exc
    finally:
        # A crop is usually a view holding the original color buffer alive.
        # Release both references before the caller loads model sessions.
        cropped=None;raw=None
