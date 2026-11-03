#include "capture_denoise.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static capture_denoise_workspace work;
static int yy(const uint8_t *p) { return (77*p[2]+150*p[1]+29*p[0]+128)/256; }
static int rounded(int v,int d) { return v>=0?(v+d/2)/d:-((-v+d/2)/d); }
static unsigned bound(int n,unsigned size) { return n<0?0u:((unsigned)n>=size?size-1u:(unsigned)n); }
static uint8_t clip(int v) { return (uint8_t)(v<0?0:(v>255?255:v)); }
static void reference(const uint8_t *in,uint8_t *out,unsigned w,unsigned h,size_t stride)
{
    unsigned x,y,c;
    memcpy(out,in,stride*h);
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        const uint8_t *p=in+y*stride+3*x;
        uint8_t *q=out+y*stride+3*x;
        int base=yy(p),result[2],a,b;
        for(c=0;c<2;++c) {
            int sum=0,square=0,mean,var,original=p[2*c]-base;
            for(a=-4;a<=4;++a) for(b=-4;b<=4;++b) {
                const uint8_t *neighbor=in+bound((int)y+a,h)*stride+3*bound((int)x+b,w);
                int chroma=neighbor[2*c]-yy(neighbor);
                sum+=chroma;square+=chroma*chroma;
            }
            mean=rounded(sum,81);var=(square*81-sum*sum)/6561;
            result[c]=original+rounded((mean-original)*25,var>25?var:25);
        }
        if(result[0]!=p[0]-base || result[1]!=p[2]-base) {
            q[0]=clip(base+result[0]);q[2]=clip(base+result[1]);
            q[1]=clip(rounded(256*base-77*q[2]-29*q[0],150));
        }
    }
}
static void service(void *ctx) { ++*(unsigned *)ctx; }
int main(void)
{
    const unsigned dims[]={1,2,4,9,15,31};
    unsigned i,j,k,seed=37;
    for(i=0;i<6;++i) for(j=0;j<6;++j) {
        unsigned w=dims[i],h=dims[j],calls=0;
        size_t stride=w*3u+7u,bytes=stride*h;
        uint8_t *a=malloc(bytes+16),*b=malloc(bytes+16),*c=malloc(bytes+16);
        assert(a && b && c);memset(a,0xab,bytes+16);
        for(k=0;k<bytes;++k) { seed=seed*1664525u+1013904223u;a[k]=(uint8_t)(seed>>24); }
        memcpy(b,a,bytes+16);memcpy(c,a,bytes+16);reference(a,c,w,h,stride);
        assert(capture_denoise_bgr(b,w,h,stride,&work,service,&calls)==0);
        assert(!memcmp(b,c,bytes+16) && calls==(h+15u)/16u);
        free(a);free(b);free(c);
    }
    {
        unsigned w=4056,h=2868,x,y;
        size_t bytes=(size_t)w*h*3;
        uint8_t *a=malloc(bytes),*b=malloc(bytes);
        double before=0,after=0;
        assert(a && b);
        for(y=0;y<h;++y) for(x=0;x<w;++x) {
            size_t n=((size_t)y*w+x)*3;
            int noise;
            seed=seed*1664525u+1013904223u;noise=(int)((seed>>24)%13)-6;
            a[n]=(uint8_t)(128+noise);a[n+1]=128;a[n+2]=(uint8_t)(128-noise);
            /* A one-pixel staff line retains its luminance and position. */
            if(y%60u==0 || x==2000) a[n]=a[n+1]=a[n+2]=20;
        }
        memcpy(b,a,bytes);assert(capture_denoise_bgr(b,w,h,w*3u,&work,NULL,NULL)==0);
        for(y=0;y<h;++y) for(x=0;x<w;++x) {
            size_t n=((size_t)y*w+x)*3;
            int original_y=yy(a+n),filtered_y=yy(b+n);
            assert(original_y==filtered_y);
            if(y%60u==0 || x==2000) assert(filtered_y==20);
            before+=(a[n+2]-original_y)*(a[n+2]-original_y);
            after+=(b[n+2]-filtered_y)*(b[n+2]-filtered_y);
        }
        assert(after<before*0.25);free(a);free(b);
    }
    {
        uint8_t solid[9*9*3],before[sizeof(solid)];
        for(i=0;i<9*9;++i) { solid[3*i]=17;solid[3*i+1]=94;solid[3*i+2]=201; }
        memcpy(before,solid,sizeof(solid));assert(!capture_denoise_bgr(solid,9,9,27,&work,NULL,NULL));
        assert(!memcmp(solid,before,sizeof(solid)));
        assert(capture_denoise_bgr(solid,0,9,27,&work,NULL,NULL)==-1);
        assert(capture_denoise_bgr(solid,9,9,26,&work,NULL,NULL)==-1);
        assert(capture_denoise_bgr((uint8_t *)&work,9,9,27,&work,NULL,NULL)==-1);
    }
    puts("PASS chroma denoise: independent 9x9 reference, borders/stride/guards, native A4 luma/staff preservation, noise reduction, flat color");
    return 0;
}
