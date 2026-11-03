/* SPDX-License-Identifier: MIT */
#include "capture_focus.h"
#include "preview.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N CAPTURE_FOCUS_SIZE
#define STRIDE (N*3u+16u)
static uint8_t source[N*STRIDE], saved[N*STRIDE];
static capture_focus_workspace work;
static unsigned services;
static void service(void *ctx) { ++*(unsigned *)ctx; }
static void fixture(double sigma,int vertical,unsigned low,unsigned high)
{
    unsigned x,y;
    memset(source,0xa7,sizeof(source));
    /* One falling edge, with analytical Gaussian convolution at pixel centers.
     * Each half is large enough for the 3/90 and 5/95 percentile plateaus. */
    for(y=0;y<N;++y) for(x=0;x<N;++x) {
        double d=((double)(vertical?x:y)-249.3)/(sqrt(2.0)*sigma);
        source[y*STRIDE+x*3u]=3u;
        source[y*STRIDE+x*3u+1u]=(uint8_t)(low+(high-low)*0.5*(1.0-erf(d))+0.5);
        source[y*STRIDE+x*3u+2u]=251u;
    }
}
static capture_focus_result measure(int edges)
{
    capture_focus_result r;
    memcpy(saved,source,sizeof(source));
    assert(!capture_focus_measure(source,N,N,STRIDE,0,edges,&work,&r,service,&services));
    assert(!memcmp(saved,source,sizeof(source)));
    assert(services>0);
    return r;
}
static void test_geometry_and_errors(void)
{
    capture_focus_rect r;
    capture_focus_result result;
    const unsigned xy[5][2]={{1778,1270},{426,256},{3130,256},{426,2283},{3130,2283}};
    unsigned i,z;
    for(i=0;i<5u;++i) {
        assert(!capture_focus_region(4056,3040,i,500,500,&r));
        assert(r.x==xy[i][0] && r.y==xy[i][1] && r.width==500u && r.height==500u);
        for(z=1;z<=2;++z) {
            assert(!capture_focus_region(4056,3040,i,640u/z,432u/z,&r));
            assert(r.x+r.width<=4056u && r.y+r.height<=3040u);
        }
        assert(!capture_focus_region(500,500,i,500,500,&r));assert(!r.x && !r.y);
    }
    assert(capture_focus_region(4056,3040,5,500,500,&r));
    assert(capture_focus_region(499,500,0,500,500,&r));
    assert(capture_focus_region(8193,500,0,500,500,&r));
    assert(capture_focus_region(500,500,0,500,500,NULL));
    assert(capture_focus_measure(NULL,N,N,STRIDE,0,0,&work,&result,NULL,NULL));
    assert(capture_focus_measure(source,N,N,STRIDE,5,0,&work,&result,NULL,NULL));
    assert(capture_focus_measure(source,N,N,1499,0,0,&work,&result,NULL,NULL));
    assert(capture_focus_measure(source,N,N,SIZE_MAX,0,0,&work,&result,NULL,NULL));
    assert(capture_focus_measure(work.gray,N,N,1500,0,0,&work,&result,NULL,NULL));
    assert(!strcmp(capture_focus_region_name(4),"BR"));
}
static void test_blur_and_reference_score(void)
{
    unsigned j,x,y,previous_width=0,previous_t=UINT32_MAX;
    const double sigmas[]={0.5,1.0,2.0,4.0};
    for(j=0;j<4u;++j) {
        capture_focus_result r,transposed;
        double energy=0.0,expected;
        fixture(sigmas[j],0,30,230);r=measure(1);
        assert(!r.flags && r.edge_h_count==72u && !r.edge_v_count);
        assert(r.edge_h_milli>previous_width && r.tenengrad_x10000<previous_t);
        previous_width=r.edge_h_milli;previous_t=r.tenengrad_x10000;
        for(y=1;y<N-1u;++y) for(x=1;x<N-1u;++x) {
            /* Independent kernel sum in floating point on immutable input. */
            const int kx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
            const int ky[3][3]={{-1,-2,-1},{0,0,0},{1,2,1}};
            int a,b;double gx=0,gy=0;
            for(a=-1;a<=1;++a) for(b=-1;b<=1;++b) {
                double p=source[(size_t)((int)y+a)*STRIDE+3u*(unsigned)((int)x+b)+1u];
                gx+=kx[a+1][b+1]*p;gy+=ky[a+1][b+1]*p;
            }
            energy+=gx*gx+gy*gy;
        }
        expected=energy*10000.0/((N-2u)*(N-2u)*40000.0);
        assert(fabs(r.tenengrad_x10000-expected)<=0.5);
        fixture(sigmas[j],1,30,230);transposed=measure(1);
        assert(!transposed.flags && !transposed.edge_h_count && transposed.edge_v_count==72u);
        assert(transposed.edge_v_milli==r.edge_h_milli);
        assert(transposed.tenengrad_x10000==r.tenengrad_x10000);
        printf("FOCUS fixture sigma=%.1f T=%u width_milli=%u count=%u\n",sigmas[j],r.tenengrad_x10000,r.edge_h_milli,r.edge_h_count);
    }
    fixture(2.0,0,30,230);
    { capture_focus_result a=measure(0),b;
      fixture(2.0,0,70,170);b=measure(0);
      assert(fabs((double)a.tenengrad_x10000/b.tenengrad_x10000-1.0)<0.04);
    }
}
static void test_invalid_and_large_accumulation(void)
{
    capture_focus_result r;
    unsigned x,y;
    for(y=0;y<N;++y) for(x=0;x<N;++x) source[y*STRIDE+x*3u+1u]=220;
    r=measure(1);assert(r.flags==CAPTURE_FOCUS_LOW_TEXTURE && !r.tenengrad_x10000 && !r.edge_h_count);
    for(x=0;x<N;++x) source[250u*STRIDE+x*3u+1u]=30;
    r=measure(1);assert(r.flags&CAPTURE_FOCUS_LOW_TEXTURE); /* Sparse ink P95=P5. */
    fixture(1.0,0,30,255);r=measure(1);assert(r.flags&CAPTURE_FOCUS_CLIPPED);
    assert(!r.tenengrad_x10000);
    for(y=0;y<N;++y) for(x=0;x<N;++x) source[y*STRIDE+x*3u+1u]=(x%4u<2u)?230:30;
    r=measure(0);assert(!r.flags && r.tenengrad_x10000==160000u);
    /* Energy sum > 2^32; a 32-bit accumulator cannot pass this case. */
}
static void test_history(void)
{
    capture_focus_history h;
    capture_focus_result r={0};
    capture_focus_history_reset(&h);
    r.tenengrad_x10000=100;capture_focus_history_push(&h,&r);assert(h.current==100 && h.peak==100);
    r.tenengrad_x10000=200;capture_focus_history_push(&h,&r);assert(h.current==150);
    r.tenengrad_x10000=300;capture_focus_history_push(&h,&r);assert(h.current==200);
    r.tenengrad_x10000=400;capture_focus_history_push(&h,&r);assert(h.current==300 && h.peak==300);
    r.flags=CAPTURE_FOCUS_LOW_TEXTURE;capture_focus_history_push(&h,&r);assert(!h.valid && !h.current && h.peak==300);
    r.flags=0;r.tenengrad_x10000=50;capture_focus_history_push(&h,&r);assert(h.current==50 && h.peak==300);
    capture_focus_history_reset(&h);assert(!h.peak && !h.count && !h.valid);
    r.tenengrad_x10000=UINT32_MAX;capture_focus_history_push(&h,&r);capture_focus_history_push(&h,&r);
    capture_focus_history_push(&h,&r);assert(h.current==UINT32_MAX);
}
static void test_uneven(void)
{
    capture_focus_result five[5]={0};
    unsigned i;
    for(i=0;i<5u;++i) { five[i].edge_h_count=10;five[i].edge_h_milli=2000; }
    assert(!capture_focus_uneven(five));
    five[3].edge_h_milli=five[4].edge_h_milli=2600;assert(!capture_focus_uneven(five));
    five[3].edge_h_milli=five[4].edge_h_milli=2601;assert(capture_focus_uneven(five)==1u);
    five[4].edge_h_count=9;assert(!capture_focus_uneven(five));
    five[4].edge_h_count=10;five[4].flags=CAPTURE_FOCUS_CLIPPED;assert(!capture_focus_uneven(five));
    five[4].flags=0;five[3].edge_h_milli=2000;five[2].edge_h_milli=2601;
    assert(capture_focus_uneven(five)==2u);
}
static void test_native_viewports(void)
{
    size_t stride=12224u,bytes=stride*3040u,output=640u*480u*3u;
    uint8_t *frame=malloc(bytes),*display=malloc(output+32u);
    unsigned region,z,x,y,c;
    assert(frame && display);
    for(y=0;y<3040u;++y) for(x=0;x<4056u;++x) for(c=0;c<3u;++c)
        frame[y*stride+3u*x+c]=(uint8_t)(17u*x+23u*y+39u*c);
    for(region=0;region<5u;++region) for(z=1;z<=2u;++z) {
        capture_focus_rect r;
        assert(!capture_focus_region(4056,3040,region,640u/z,432u/z,&r));
        memset(display,0xa5,output+32u);
        assert(!preview_letterbox(frame+r.y*stride+3u*r.x,r.width,r.height,stride,
            display+48u*1920u,640,432,1920,NULL,NULL));
        for(y=0;y<432u;++y) for(x=0;x<640u;++x) for(c=0;c<3u;++c)
            assert(display[(y+48u)*1920u+3u*x+c]==frame[(r.y+y/z)*stride+3u*(r.x+x/z)+c]);
        for(x=0;x<48u*1920u;++x) assert(display[x]==0xa5);
        for(x=0;x<32u;++x) assert(display[output+x]==0xa5);
    }
    free(frame);free(display);
}
int main(void)
{
    test_geometry_and_errors();test_blur_and_reference_score();
    test_invalid_and_large_accumulation();test_history();test_native_viewports();test_uneven();
    puts("PASS capture_focus: native five-region geometry, immutable G, Gaussian blur ordering, independent Sobel reference, low-texture/clipping, 64-bit accumulation, peak history and exact1X/2X viewports");
    return 0;
}
