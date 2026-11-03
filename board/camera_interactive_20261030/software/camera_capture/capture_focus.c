/* SPDX-License-Identifier: MIT */
#include "capture_focus.h"
#include <string.h>

const char *capture_focus_region_name(unsigned region)
{
    static const char *const names[]={"C","TL","TR","BL","BR"};
    return region<5u?names[region]:"INVALID";
}
int capture_focus_region(unsigned w,unsigned h,unsigned region,unsigned bw,unsigned bh,capture_focus_rect *r)
{
    unsigned cx,cy;
    if(!r || !w || !h || w>8192u || h>8192u || region>=5u || !bw || !bh || bw>w || bh>h) return -1;
    cx=region?((region==2u || region==4u)?5u*w/6u:w/6u):w/2u;
    cy=region?((region>=3u)?5u*h/6u:h/6u):h/2u;
    r->x=cx>bw/2u?cx-bw/2u:0u;r->y=cy>bh/2u?cy-bh/2u:0u;
    if(r->x>w-bw) r->x=w-bw;
    if(r->y>h-bh) r->y=h-bh;
    r->width=bw;r->height=bh;return 0;
}
/* NumPy's linear quantile convention: fractional rank (count-1)*percent/100. */
static double percentile(const uint32_t *hist,unsigned bins,unsigned count,unsigned percent)
{
    unsigned rank=(count-1u)*percent,lo=rank/100u,hi=lo+(rank%100u!=0u),sum=0,i,a=0,b=0;
    int have=0;
    for(i=0;i<bins;++i) {
        sum+=hist[i];
        if(!have && sum>lo) { a=i;have=1; }
        if(sum>hi) { b=i;break; }
    }
    return a+(b-a)*(rank%100u)/100.0;
}
static void edge_widths(capture_focus_workspace *s,unsigned axis,unsigned *count,unsigned *median,
                         void (*service)(void *),void *context)
{
    const unsigned n=CAPTURE_FOCUS_SIZE;
    unsigned col,i;
    *count=0;*median=0;memset(s->edge_hist,0,sizeof(s->edge_hist));
    for(col=0;col<n;col+=7u) {
        uint32_t hist[1021]={0};
        double lo,hi,t10,t90;
        for(i=0;i<n;++i) {
            unsigned before=i?i-1u:1u,after=i+1u<n?i+1u:n-2u;
            unsigned a=axis?s->gray[col*n+before]:s->gray[before*n+col];
            unsigned b=axis?s->gray[col*n+i]:s->gray[i*n+col];
            unsigned c=axis?s->gray[col*n+after]:s->gray[after*n+col];
            s->profile[i]=(uint16_t)(a+2u*b+c);++hist[s->profile[i]];
        }
        lo=percentile(hist,1021u,n,3u);hi=percentile(hist,1021u,n,90u);
        if(hi-lo>=160.0) {
            t10=lo+0.1*(hi-lo);t90=lo+0.9*(hi-lo);
            for(i=1u;i<n-16u;++i) if(s->profile[i]>=t90 && t90>s->profile[i+1u]) {
                unsigned j=i+1u;
                while(j<n-1u && s->profile[j]>t10 && j-i<16u) ++j;
                if(s->profile[j]<=t10 && j-i<16u) {
                    double a=i+(s->profile[i]-t90)/(s->profile[i]-s->profile[i+1u]);
                    double b=j-1u+(s->profile[j-1u]-t10)/(s->profile[j-1u]-s->profile[j]);
                    unsigned width=(unsigned)((b-a)*1000.0+0.5);
                    if(width<=16000u) { ++s->edge_hist[width];++*count; }
                }
            }
        }
        if(service && col%28u==0u) service(context);
    }
    if(*count) *median=(unsigned)(percentile(s->edge_hist,16001u,*count,50u)+0.5);
}
int capture_focus_measure(const uint8_t *source,unsigned w,unsigned h,size_t stride,
                           unsigned region,int edges,capture_focus_workspace *s,
                           capture_focus_result *r,void (*service)(void *),void *context)
{
    uint32_t hist[256]={0};
    const unsigned n=CAPTURE_FOCUS_SIZE;
    unsigned x,y,clipped=0,contrast;
    uint64_t sum=0,denominator;
    size_t bytes;
    uintptr_t src=(uintptr_t)source,work=(uintptr_t)s;
    if(!r) return -1;
    memset(r,0,sizeof(*r));
    if(!source || !s || stride<(size_t)w*3u ||
       capture_focus_region(w,h,region,n,n,&r->roi) ||
       (size_t)(h-1u)>(SIZE_MAX-(size_t)w*3u)/stride) return -1;
    bytes=(size_t)(h-1u)*stride+(size_t)w*3u;
    if(src>UINTPTR_MAX-bytes || work>UINTPTR_MAX-sizeof(*s) ||
       (src<work+sizeof(*s) && work<src+bytes)) return -1;
    for(y=0;y<n;++y) {
        const uint8_t *p=source+(size_t)(y+r->roi.y)*stride+3u*r->roi.x+1u;
        for(x=0;x<n;++x) {
            unsigned v=p[3u*x];s->gray[y*n+x]=(uint8_t)v;++hist[v];clipped+=v>=254u;
        }
        if(service && (y&15u)==15u) service(context);
    }
    r->p5=(unsigned)(percentile(hist,256u,n*n,5u)+0.5);
    r->p95=(unsigned)(percentile(hist,256u,n*n,95u)+0.5);
    r->clip_permille=(unsigned)((uint64_t)clipped*1000u/(n*n));
    contrast=r->p95-r->p5;
    for(y=1u;y<n-1u;++y) {
        const uint8_t *a=s->gray+(y-1u)*n,*b=a+n,*c=b+n;
        for(x=1u;x<n-1u;++x) {
            int gx=(int)a[x+1u]+2*(int)b[x+1u]+c[x+1u]-a[x-1u]-2*(int)b[x-1u]-c[x-1u];
            int gy=(int)c[x-1u]+2*(int)c[x]+c[x+1u]-a[x-1u]-2*(int)a[x]-a[x+1u];
            unsigned energy=(unsigned)(gx*gx+gy*gy);
            sum+=energy;r->edge_pixels+=energy>=1024u;
        }
        if(service && (y&15u)==15u) service(context);
    }
    if(contrast<16u || r->edge_pixels<64u) r->flags|=CAPTURE_FOCUS_LOW_TEXTURE;
    if(r->clip_permille>=50u) r->flags|=CAPTURE_FOCUS_CLIPPED;
    if(!r->flags) {
        denominator=(uint64_t)(n-2u)*(n-2u)*contrast*contrast;
        r->tenengrad_x10000=(uint32_t)((sum*10000u+denominator/2u)/denominator);
    }
    if(edges) {
        edge_widths(s,0u,&r->edge_h_count,&r->edge_h_milli,service,context);
        edge_widths(s,1u,&r->edge_v_count,&r->edge_v_milli,service,context);
        r->edges_computed=1u;
    }
    if(service) service(context);
    return 0;
}
void capture_focus_history_reset(capture_focus_history *s) { if(s) memset(s,0,sizeof(*s)); }
unsigned capture_focus_uneven(const capture_focus_result five[5])
{
    unsigned axis,i,flags=0;
    if(!five) return 0;
    for(axis=0;axis<2u;++axis) {
        uint64_t v[5]={0},a,b;
        int valid=1;
        for(i=1;i<5u;++i) {
            unsigned count=axis?five[i].edge_v_count:five[i].edge_h_count;
            v[i]=axis?five[i].edge_v_milli:five[i].edge_h_milli;
            if(five[i].flags || count<CAPTURE_FOCUS_MIN_EDGES || !v[i]) valid=0;
        }
        if(!valid) continue;
        a=v[1]+v[2];b=v[3]+v[4];
        if(a*10u>b*13u || b*10u>a*13u) flags|=1u;
        a=v[1]+v[3];b=v[2]+v[4];
        if(a*10u>b*13u || b*10u>a*13u) flags|=2u;
    }
    return flags;
}
void capture_focus_history_push(capture_focus_history *s,const capture_focus_result *r)
{
    if(!s || !r) return;
    if(r->flags) { s->total=0;s->index=s->count=s->valid=0;s->current=0;return; }
    if(s->count==3u) s->total-=s->values[s->index]; else ++s->count;
    s->values[s->index]=r->tenengrad_x10000;s->total+=r->tenengrad_x10000;
    s->index=(s->index+1u)%3u;s->current=(uint32_t)((s->total+s->count/2u)/s->count);
    s->valid=1u;if(s->current>s->peak) s->peak=s->current;
}
