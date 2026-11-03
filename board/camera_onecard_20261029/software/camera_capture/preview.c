/* SPDX-License-Identifier: GPL-2.0-only */
#include "preview.h"
#include <string.h>

int preview_letterbox(const uint8_t *source, unsigned sw, unsigned sh,
                      size_t ss, uint8_t *dest, unsigned dw, unsigned dh,
                      size_t ds, void (*service)(void *), void *context)
{
    unsigned w, h, left, top, x, y;
    if (!source || !dest || !sw || !sh || !dw || !dh ||
        sw > 8192 || sh > 8192 || dw > 8192 || dh > 8192 ||
        ss < (size_t)sw*3 || ds < (size_t)dw*3) return -1;
    w=dw; h=(unsigned)((uint64_t)dw*sh/sw);
    if (h>dh) { h=dh; w=(unsigned)((uint64_t)dh*sw/sh); }
    if (!w) w=1;
    if (!h) h=1;
    left=(dw-w)/2; top=(dh-h)/2;
    for(y=0;y<dh;y++) {
        uint8_t *row=dest+(size_t)y*ds;
        memset(row,0,(size_t)dw*3);
        if (y>=top && y<top+h) {
            const uint8_t *src=source+(size_t)(((uint64_t)(y-top)*sh)/h)*ss;
            /* Fixed-point phase avoids a division for every output pixel. */
            uint64_t phase=0, step=((uint64_t)sw<<32)/w;
            for(x=0;x<w;x++,phase+=step) {
                unsigned sx=(unsigned)(phase>>32);
                memcpy(row+(size_t)(left+x)*3,src+(size_t)sx*3,3);
            }
        }
        if(service && (y&31u)==0) service(context);
    }
    return 0;
}

static const uint8_t *glyph(char c)
{
    /* 5x7 row bitmaps, sufficient for the capture status sidebar. */
    static const uint8_t digits[10][7]={
        {14,17,19,21,25,17,14},{4,12,4,4,4,4,14},
        {14,17,1,2,4,8,31},{30,1,1,14,1,1,30},
        {2,6,10,18,31,2,2},{31,16,16,30,1,1,30},
        {14,16,16,30,17,17,14},{31,1,2,4,8,8,8},
        {14,17,17,14,17,17,14},{14,17,17,15,1,1,14}};
    static const uint8_t letters[26][7]={
        {14,17,17,31,17,17,17},{30,17,17,30,17,17,30},
        {14,17,16,16,16,17,14},{30,17,17,17,17,17,30},
        {31,16,16,30,16,16,31},{31,16,16,30,16,16,16},
        {14,17,16,23,17,17,15},{17,17,17,31,17,17,17},
        {14,4,4,4,4,4,14},{7,2,2,2,2,18,12},
        {17,18,20,24,20,18,17},{16,16,16,16,16,16,31},
        {17,27,21,21,17,17,17},{17,25,21,19,17,17,17},
        {14,17,17,17,17,17,14},{30,17,17,30,16,16,16},
        {14,17,17,17,21,18,13},{30,17,17,30,20,18,17},
        {15,16,16,14,1,1,30},{31,4,4,4,4,4,4},
        {17,17,17,17,17,17,14},{17,17,17,17,17,10,4},
        {17,17,17,21,21,21,10},{17,17,10,4,10,17,17},
        {17,17,10,4,4,4,4},{31,1,2,4,8,16,31}};
    static const uint8_t dash[7]={0,0,0,31,0,0,0};
    static const uint8_t dot[7]={0,0,0,0,0,6,6};
    static const uint8_t blank[7]={0,0,0,0,0,0,0};
    if(c>='0'&&c<='9') return digits[c-'0'];
    if(c>='a'&&c<='z') c=(char)(c-'a'+'A');
    if(c>='A'&&c<='Z') return letters[c-'A'];
    if(c=='.') return dot;
    return c=='-'?dash:blank;
}

void preview_text(uint8_t *dst,unsigned w,unsigned h,size_t stride,
                  unsigned x,unsigned y,const char *text,unsigned scale,uint32_t rgb)
{
    unsigned r,c,yy,xx;
    if(!dst||!text||!scale||scale>16||stride<(size_t)w*3) return;
    for(;*text && x<w;text++,x+=6*scale) {
        const uint8_t *g=glyph(*text);
        for(r=0;r<7;r++) for(c=0;c<5;c++) if(g[r]&(16u>>c))
            for(yy=0;yy<scale;yy++) for(xx=0;xx<scale;xx++) {
                unsigned px=x+c*scale+xx, py=y+r*scale+yy;
                if(px<w&&py<h) {
                    uint8_t *p=dst+(size_t)py*stride+(size_t)px*3;
                    p[0]=(uint8_t)rgb; p[1]=(uint8_t)(rgb>>8); p[2]=(uint8_t)(rgb>>16);
                }
            }
    }
}

unsigned preview_brightness(const uint8_t *src,unsigned w,unsigned h,size_t stride)
{
    unsigned bins[256]={0}, count=0, cumulative=0, x,y,i;
    if(!src||!w||!h||stride<(size_t)w*3) return 0;
    for(y=0;y<h;y+=32) for(x=0;x<w;x+=32) {
        const uint8_t *p=src+(size_t)y*stride+(size_t)x*3;
        unsigned luma=(29u*p[0]+150u*p[1]+77u*p[2])>>8;
        bins[luma]++; count++;
    }
    for(i=0;i<256;i++) { cumulative+=bins[i]; if(cumulative*10>=count*9) return i; }
    return 255;
}
