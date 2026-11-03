#include "capture_denoise.h"
#include <string.h>

#define RADIUS 4u
#define AREA 81
#define NOISE_VARIANCE 25
static int rounded(int value,int divisor)
{ return value>=0?(value+divisor/2)/divisor:-((-value+divisor/2)/divisor); }
static uint8_t clip(int v) { return (uint8_t)(v<0?0:(v>255?255:v)); }
static int luma(const uint8_t *p) { return (77*p[2]+150*p[1]+29*p[0]+128)/256; }
static unsigned bounded(int coordinate,unsigned size)
{ return coordinate<0?0u:((unsigned)coordinate>=size?size-1u:(unsigned)coordinate); }

static void load_row(capture_denoise_workspace *w,unsigned slot,const uint8_t *row,unsigned width,int replace)
{
    unsigned x,c;
    for(x=0;x<width;++x) {
        int y=luma(row+3u*x), values[2]={row[3u*x]-y,row[3u*x+2u]-y};
        for(c=0;c<2u;++c) {
            int old=replace?w->rows[slot][x][c]:0,v=values[c];
            w->columns[x][c]+=v-old;
            w->columns[x][c+2u]+=v*v-old*old;
            w->rows[slot][x][c]=(int16_t)v;
        }
    }
}

int capture_denoise_bgr(uint8_t *pixels,unsigned width,unsigned height,size_t stride,
    capture_denoise_workspace *w,void (*service)(void *),void *context)
{
    unsigned x,y,c,slot;
    size_t bytes;
    uintptr_t p=(uintptr_t)pixels,a=(uintptr_t)w;
    if(!pixels || !w || !width || !height || width>CAPTURE_DENOISE_MAX_WIDTH || height>4096u ||
       stride<(size_t)width*3u || (size_t)(height-1u)>(SIZE_MAX-width*3u)/stride) return -1;
    bytes=(size_t)(height-1u)*stride+width*3u;
    if(p>UINTPTR_MAX-bytes || a>UINTPTR_MAX-sizeof(*w) ||
       (p<a+sizeof(*w) && a<p+bytes)) return -1;
    memset(w->columns,0,sizeof(w->columns));
    for(slot=0;slot<9u;++slot)
        load_row(w,slot,pixels+(size_t)bounded((int)slot-4,height)*stride,width,0);
    for(y=0;y<height;++y) {
        int sums[4]={0,0,0,0};
        uint8_t *row=pixels+(size_t)y*stride;
        for(slot=0;slot<9u;++slot) for(c=0;c<4u;++c)
            sums[c]+=w->columns[bounded((int)slot-4,width)][c];
        for(x=0;x<width;++x) {
            uint8_t *pixel=row+3u*x;
            int yy=luma(pixel), chroma[2]={pixel[0]-yy,pixel[2]-yy};
            int original_cb=chroma[0],original_cr=chroma[1];
            for(c=0;c<2u;++c) {
                int variance=(sums[c+2u]*AREA-sums[c]*sums[c])/(AREA*AREA);
                int mean=rounded(sums[c],AREA);
                int divisor=variance>NOISE_VARIANCE?variance:NOISE_VARIANCE;
                chroma[c]+=rounded((mean-chroma[c])*NOISE_VARIANCE,divisor);
            }
            if(chroma[0]!=original_cb || chroma[1]!=original_cr) {
                pixel[0]=clip(yy+chroma[0]);pixel[2]=clip(yy+chroma[1]);
                pixel[1]=clip(rounded(256*yy-77*pixel[2]-29*pixel[0],150));
            }
            for(c=0;c<4u;++c)
                sums[c]+=w->columns[bounded((int)x+5,width)][c]-w->columns[bounded((int)x-4,width)][c];
        }
        if(y+1u<height)
            load_row(w,y%9u,pixels+(size_t)bounded((int)y+5,height)*stride,width,1);
        if(service && ((y&15u)==15u || y+1u==height)) service(context);
    }
    return 0;
}
