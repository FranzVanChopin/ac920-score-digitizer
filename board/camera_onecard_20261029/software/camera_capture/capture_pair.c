#include "capture_pair.h"

static int row_valid(const capture_pair_frame *f, uint32_t y, uint8_t *out,
                     size_t length, unsigned channels)
{
    return f && f->bgr && out && f->width && f->width<=8192u &&
           f->height && f->height<=8192u && y<f->height &&
           f->stride>=(size_t)f->width*3u &&
           f->stride<=SIZE_MAX/f->height && length==(size_t)f->width*channels;
}

int capture_pair_rgb_row(void *context, uint32_t y, uint8_t *out, size_t length)
{
    const capture_pair_frame *f=context;
    const uint8_t *p;
    unsigned x;
    if(!row_valid(f,y,out,length,3u)) return CAPTURE_ROW_ERROR;
    p=f->bgr+(size_t)y*f->stride;
    for(x=0;x<f->width;++x) {
        out[3u*x]=p[3u*x+2u];out[3u*x+1u]=p[3u*x+1u];out[3u*x+2u]=p[3u*x];
    }
    if(f->service) f->service(f->service_context);
    return CAPTURE_OK;
}

int capture_pair_gray_row(void *context, uint32_t y, uint8_t *out, size_t length)
{
    const capture_pair_frame *f=context;
    const uint8_t *p;
    unsigned x;
    if(!row_valid(f,y,out,length,1u)) return CAPTURE_ROW_ERROR;
    p=f->bgr+(size_t)y*f->stride;
    for(x=0;x<f->width;++x)
        out[x]=(uint8_t)((4899u*p[3u*x+2u]+9617u*p[3u*x+1u]+1868u*p[3u*x]+8192u)>>14);
    if(f->service) f->service(f->service_context);
    return CAPTURE_OK;
}
