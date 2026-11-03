#include "dp_pixels.h"
int dp_pixels_from_bgr(const uint8_t *src, unsigned w, unsigned h,
                       size_t ss, uint8_t *dst, size_t ds)
{
    unsigned x,y;
    if(!src || !dst || !w || !h || w>8192u || h>8192u ||
       ss<(size_t)w*3u || ds<(size_t)w*4u || ss>SIZE_MAX/h || ds>SIZE_MAX/h) return -1;
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        const uint8_t *p=src+(size_t)y*ss+3u*x;
        uint8_t *q=dst+(size_t)y*ds+4u*x;
        q[0]=p[2];q[1]=p[1];q[2]=p[0];q[3]=255u;
    }
    return 0;
}
