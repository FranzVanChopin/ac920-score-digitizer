/* SPDX-License-Identifier: MIT */
#include "capture_color.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static capture_color color;
/* Independent double-precision reference, without implementation tables. */
static const double matrix[3][3] = {
    {2.2286541768,-0.9614188773,-0.2672352996},
    {-0.3468200582,2.1199418325,-0.7731189761},
    {-0.1613435660,-0.4447762013,1.6061161664}
};
static double inverse[256];
static unsigned reachable[256];

static unsigned hardware_encode(unsigned raw10)
{
    double v = raw10 <= 64u ? 0.0 : (raw10 - 64.0) / 959.0;
    return ((unsigned)floor(1023.0 * pow(v, 1.0 / 2.2) + 0.5)) / 4u;
}

static void reference_init(void)
{
    unsigned raw, code;
    double sums[256] = {0.0};
    for(raw=0;raw<1024u;++raw) {
        code=hardware_encode(raw);
        ++reachable[code];
        sums[code]+=raw>64u?(raw-64.0)/959.0:0.0;
    }
    for(code=0;code<256u;++code)
        if(reachable[code]) inverse[code]=sums[code]/reachable[code];
    for(code=1;code<255u;++code) if(!reachable[code]) {
        unsigned low=code,high=code;
        while(!reachable[low]) --low;
        while(!reachable[high]) ++high;
        inverse[code]=inverse[low]+(inverse[high]-inverse[low])*
            (double)(code-low)/(double)(high-low);
    }
}

static void reference_pixel(const unsigned char in[3],unsigned char out[3])
{
    unsigned r,c;
    double linear[3]={inverse[in[2]],inverse[in[1]],inverse[in[0]]};
    for(r=0;r<3;++r) {
        double sum=0.0;
        for(c=0;c<3;++c) sum+=matrix[r][c]*linear[c];
        if(sum<0.0) sum=0.0;
        if(sum>1.0) sum=1.0;
        out[2u-r]=(unsigned char)floor(255.0*pow(sum,1.0/2.2)+0.5);
    }
}

static unsigned difference(unsigned a,unsigned b)
{
    return a>b?a-b:b-a;
}

static void test_inverse(void)
{
    unsigned i,count=0;
    double maximum_error=0.0;
    assert(color.decode[0]==0.0);
    for(i=0;i<256u;++i) {
        if(reachable[i]) ++count;
        assert(fabs(color.decode[i]-inverse[i])<1e-12);
        assert(isfinite(color.decode[i]));
        if(i) assert(color.decode[i]>=color.decode[i-1u]);
    }
    assert(count==232u);
    assert(reachable[255]==8u);
    assert(hardware_encode(1015u)==254u);
    assert(hardware_encode(1016u)==255u);
    for(i=65;i<1024u;++i) {
        double e=fabs(color.decode[hardware_encode(i)]-(i-64.0)/959.0);
        if(e>maximum_error) maximum_error=e;
    }
    assert(maximum_error<=3.5/959.0+1e-12);
    /* Gamma LUT is nondecreasing, including its endpoints. */
    assert(color.encode_q8[0]==0u && color.encode_q8[4096]==65280u);
    for(i=1;i<=4096u;++i) assert(color.encode_q8[i]>=color.encode_q8[i-1u]);
}

static void test_pixel_accuracy(void)
{
    unsigned i,j,max_error=0;
    uint32_t state=UINT32_C(0x94e20137);
    unsigned char input[3],actual[3],expected[3],in_place[3];
    for(i=0;i<200000u;++i) {
        for(j=0;j<3u;++j) {
            state=state*UINT32_C(1664525)+UINT32_C(1013904223);
            input[j]=(unsigned char)(state>>24);
        }
        reference_pixel(input,expected);
        capture_color_apply_bgr(&color,input,actual);
        memcpy(in_place,input,3);
        capture_color_apply_bgr(&color,in_place,in_place);
        assert(!memcmp(actual,in_place,3));
        for(j=0;j<3u;++j) {
            unsigned e=difference(actual[j],expected[j]);
            if(e>max_error) max_error=e;
            if(e>2u) {
                fprintf(stderr,"reference mismatch BGR=%u,%u,%u channel=%u got=%u expected=%u\n",
                    input[0],input[1],input[2],j,actual[j],expected[j]);
                assert(e<=2u);
            }
        }
    }
    /* Negative matrix terms must clamp to zero without unsigned wrap. */
    input[0]=0;input[1]=0;input[2]=230;
    capture_color_apply_bgr(&color,input,actual);
    assert(actual[0]==0u && actual[1]==0u && actual[2]==255u);
    input[0]=230;input[1]=0;input[2]=0;
    capture_color_apply_bgr(&color,input,actual);
    assert(actual[0]==255u && actual[1]==0u && actual[2]==0u);
    memset(input,0,3);
    capture_color_apply_bgr(&color,input,actual);
    assert(!memcmp(input,actual,3));
    /* A neutral input stays neutral: there is no second software WB. */
    memset(input,160,3);
    capture_color_apply_bgr(&color,input,actual);
    assert(difference(actual[0],actual[1])<=1u && difference(actual[1],actual[2])<=1u);
    printf("capture_color: 200000 float-reference pixels, maximum encoding error %u/255\n",max_error);
}

static void increment_service(void *context)
{
    ++*(unsigned *)context;
}

/* Test-only nearest-neighbor sampler. Matches preview's documented fixed-
 * point sampling positions, without linking or calling preview.c. */
static void sample_letterbox(const unsigned char *source,unsigned sw,unsigned sh,
                             size_t source_stride,unsigned char *dest,
                             unsigned dw,unsigned dh,size_t dest_stride)
{
    unsigned w=dw,h=(unsigned)((uint64_t)dw*sh/sw),left,top,x,y;
    uint64_t step;
    if(h>dh) { h=dh; w=(unsigned)((uint64_t)dh*sw/sh); }
    if(!w) w=1;
    if(!h) h=1;
    left=(dw-w)/2; top=(dh-h)/2;
    step=((uint64_t)sw<<32)/w;
    for(y=0;y<dh;++y) {
        unsigned char *row=dest+(size_t)y*dest_stride;
        memset(row,0,(size_t)dw*3u);
        if(y>=top && y<top+h) for(x=0;x<w;++x) {
            unsigned sy=(unsigned)((uint64_t)(y-top)*sh/h);
            unsigned sx=(unsigned)(((uint64_t)x*step)>>32);
            memcpy(row+(size_t)(left+x)*3u,source+(size_t)sy*source_stride+(size_t)sx*3u,3u);
        }
    }
}

static void test_row_rgb(void)
{
    static const unsigned widths[]={1u,7u,4056u,CAPTURE_COLOR_MAX_DIMENSION};
    unsigned wi,x,i;
    for(wi=0;wi<sizeof(widths)/sizeof(widths[0]);++wi) {
        unsigned w=widths[wi];
        size_t bytes=(size_t)w*3u;
        unsigned char *source=(unsigned char *)malloc(bytes+64u);
        unsigned char *source_copy=(unsigned char *)malloc(bytes+64u);
        unsigned char *dest=(unsigned char *)malloc(bytes+64u);
        unsigned char *old_pipeline=(unsigned char *)malloc(bytes);
        assert(source && source_copy && dest && old_pipeline);
        memset(source,0xa5,bytes+64u);
        memset(dest,0x5a,bytes+64u);
        for(x=0;x<w;++x) {
            source[32u+3u*x]=(unsigned char)(x*17u+53u);
            source[32u+3u*x+1u]=(unsigned char)(x*29u+7u);
            source[32u+3u*x+2u]=(unsigned char)(x*11u+199u);
        }
        memcpy(source_copy,source,bytes+64u);
        memcpy(old_pipeline,source+32u,bytes);
        assert(capture_color_process_bgr(&color,old_pipeline,w,1u,bytes,NULL,NULL)==CAPTURE_COLOR_OK);
        assert(capture_color_row_rgb(&color,source+32u,dest+32u,w)==CAPTURE_COLOR_OK);
        for(x=0;x<w;++x) {
            /* Exactly the old full-frame CCM followed by main.photo_row's
             * BGR-to-RGB swap, including saturation and rounding. */
            assert(dest[32u+3u*x]==old_pipeline[3u*x+2u]);
            assert(dest[32u+3u*x+1u]==old_pipeline[3u*x+1u]);
            assert(dest[32u+3u*x+2u]==old_pipeline[3u*x]);
        }
        assert(!memcmp(source,source_copy,bytes+64u));
        for(i=0;i<32u;++i) {
            assert(dest[i]==0x5au);
            assert(dest[32u+bytes+i]==0x5au);
        }
        free(source);free(source_copy);free(dest);free(old_pipeline);
    }
    {
        capture_color uninitialized={0};
        unsigned char buffer[32],before[32];
        memset(buffer,0x37,sizeof(buffer));
        memcpy(before,buffer,sizeof(buffer));
        assert(capture_color_row_rgb(NULL,buffer,buffer+16,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&uninitialized,buffer,buffer+16,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,NULL,buffer+16,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,NULL,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,buffer+16,0)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,buffer+16,CAPTURE_COLOR_MAX_DIMENSION+1u)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,buffer,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,buffer+1,2)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer+1,buffer,2)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,(const uint8_t *)(UINTPTR_MAX-1u),buffer,1)==CAPTURE_COLOR_INVALID);
        assert(capture_color_row_rgb(&color,buffer,(uint8_t *)(UINTPTR_MAX-1u),1)==CAPTURE_COLOR_INVALID);
        assert(!memcmp(buffer,before,sizeof(buffer)));
        /* Adjacent, non-overlapping slices are valid. */
        assert(capture_color_row_rgb(&color,buffer,buffer+3u,1)==CAPTURE_COLOR_OK);
        assert(!memcmp(buffer,before,3u));
    }
    puts("capture_color: row RGB bit-exact to previous photo path; source/guards and invalid-input handling passed");
}

static void test_small_scale_commutation(void)
{
    static const unsigned shapes[][4]={{7,5,19,17},{19,17,7,5},{3,29,17,7},{29,3,7,17},{1,1,3,9}};
    unsigned k;
    for(k=0;k<sizeof(shapes)/sizeof(shapes[0]);++k) {
        unsigned sw=shapes[k][0],sh=shapes[k][1],dw=shapes[k][2],dh=shapes[k][3],x,y;
        size_t ss=(size_t)sw*3u+7u,ds=(size_t)dw*3u+11u;
        unsigned char *source=(unsigned char *)malloc(ss*sh);
        unsigned char *full=(unsigned char *)malloc(ss*sh);
        unsigned char *a=(unsigned char *)malloc(ds*dh),*b=(unsigned char *)malloc(ds*dh);
        assert(source && full && a && b);
        memset(source,0xa5,ss*sh);memset(a,0x63,ds*dh);memset(b,0x63,ds*dh);
        for(y=0;y<sh;++y) for(x=0;x<sw;++x) {
            source[(size_t)y*ss+3u*x]=(unsigned char)(x*71u+y*53u);
            source[(size_t)y*ss+3u*x+1u]=(unsigned char)(x*19u+y*31u);
            source[(size_t)y*ss+3u*x+2u]=(unsigned char)(x*13u+y*43u);
        }
        memcpy(full,source,ss*sh);
        sample_letterbox(source,sw,sh,ss,b,dw,dh,ds);
        assert(capture_color_process_bgr(&color,full,sw,sh,ss,NULL,NULL)==CAPTURE_COLOR_OK);
        sample_letterbox(full,sw,sh,ss,a,dw,dh,ds);
        assert(capture_color_process_bgr(&color,b,dw,dh,ds,NULL,NULL)==CAPTURE_COLOR_OK);
        assert(!memcmp(a,b,ds*dh));
        free(source);free(full);free(a);free(b);
    }
}

static void test_full_frame(void)
{
    const unsigned w=4056u,h=3040u;
    const size_t stride=12224u,bytes=stride*h;
    unsigned char palette[32][3],expected[32][3];
    unsigned char *allocation=(unsigned char *)malloc(bytes+64u),*pixels;
    const size_t preview_stride=640u*3u+16u,preview_bytes=preview_stride*432u;
    unsigned char *preview_before=(unsigned char *)malloc(preview_bytes);
    unsigned char *preview_after=(unsigned char *)malloc(preview_bytes);
    unsigned x,y,i,services=0;
    assert(allocation && preview_before && preview_after);
    memset(allocation,0xa5,bytes+64u);
    pixels=allocation+32u;
    for(i=0;i<32u;++i) {
        palette[i][0]=(unsigned char)(7u*i);
        palette[i][1]=(unsigned char)(255u-5u*i);
        palette[i][2]=(unsigned char)(19u*i);
        reference_pixel(palette[i],expected[i]);
    }
    for(y=0;y<h;++y) for(x=0;x<w;++x)
        memcpy(pixels+(size_t)y*stride+3u*x,palette[(x+3u*y)&31u],3u);
    memset(preview_before,0x69,preview_bytes);
    memset(preview_after,0x69,preview_bytes);
    sample_letterbox(pixels,w,h,stride,preview_before,640u,432u,preview_stride);
    assert(capture_color_process_bgr(&color,pixels,w,h,stride,increment_service,&services)==CAPTURE_COLOR_OK);
    sample_letterbox(pixels,w,h,stride,preview_after,640u,432u,preview_stride);
    assert(capture_color_process_bgr(&color,preview_before,640u,432u,preview_stride,NULL,NULL)==CAPTURE_COLOR_OK);
    assert(!memcmp(preview_before,preview_after,preview_bytes));
    assert(services==(h+15u)/16u);
    for(y=0;y<h;++y) {
        for(x=0;x<w;++x) {
            const unsigned char *p=pixels+(size_t)y*stride+3u*x;
            for(i=0;i<3u;++i) assert(difference(p[i],expected[(x+3u*y)&31u][i])<=2u);
        }
        for(i=3u*w;i<stride;++i) assert(pixels[(size_t)y*stride+i]==0xa5u);
    }
    for(i=0;i<32u;++i) {
        assert(allocation[i]==0xa5u);
        assert(allocation[32u+bytes+i]==0xa5u);
    }
    free(allocation);
    free(preview_before);free(preview_after);
    puts("capture_color: 4056x3040 in-place conversion, all pixels/padding/guards and 190 callbacks passed");
    puts("capture_color: nearest-neighbor before/after CCM is bit-exact, including 4056x3040 -> 640x432 letterbox");
}

static void fill_constant(unsigned char *p,unsigned w,unsigned h,size_t stride,
                          unsigned b,unsigned g,unsigned r)
{
    unsigned x,y;
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        p[(size_t)y*stride+3u*x]=(unsigned char)b;
        p[(size_t)y*stride+3u*x+1u]=(unsigned char)g;
        p[(size_t)y*stride+3u*x+2u]=(unsigned char)r;
    }
}

static void test_statistics(void)
{
    unsigned char pixels[12*16],before[sizeof(pixels)],expected[3];
    capture_color_stats s;
    unsigned char input[3]={255,80,120};
    fill_constant(pixels,4,12,16,input[0],input[1],input[2]);
    memcpy(before,pixels,sizeof(pixels));
    reference_pixel(input,expected);
    assert(capture_color_measure(&color,pixels,4,12,16,1,&s)==CAPTURE_COLOR_OK);
    assert(s.samples==48u);
    assert(s.raw_p99_rgb[0]==120u && s.raw_p99_rgb[1]==80u && s.raw_p99_rgb[2]==255u);
    assert(s.raw_saturated_permille_rgb[0]==0u && s.raw_saturated_permille_rgb[1]==0u &&
           s.raw_saturated_permille_rgb[2]==1000u);
    assert(s.corrected_max_p99==255u && s.corrected_saturated_permille==1000u &&
           s.corrected_high_clip_permille==1000u);
    assert(difference(s.corrected_luma_p90,(29u*expected[0]+150u*expected[1]+77u*expected[2])>>8)<=2u);
    assert(!memcmp(before,pixels,sizeof(pixels)));
    assert(capture_color_measure(&color,pixels,4,12,16,8192,&s)==CAPTURE_COLOR_OK && s.samples==1u);
    fill_constant(pixels,4,12,16,0,0,0);
    assert(capture_color_measure(&color,pixels,4,12,16,1,&s)==CAPTURE_COLOR_OK);
    assert(s.corrected_max_p99==0u && s.corrected_high_clip_permille==0u && s.corrected_luma_p90==0u);
}

static void test_calibration(void)
{
    unsigned char pixels[32*104],before[sizeof(pixels)];
    unsigned current[3]={830u,256u,357u};
    capture_color_white_result result;
    capture_color saved=color;
    unsigned i;
    memset(pixels,0xa5,sizeof(pixels));
    fill_constant(pixels,32,32,104,160,160,160);
    memcpy(before,pixels,sizeof(pixels));
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,4,4,24,24,current,&result)==CAPTURE_COLOR_OK);
    for(i=0;i<3u;++i) assert(result.proposed_sensor_gain_q8[i]==current[i]);
    assert(result.samples==576u);
    assert(!memcmp(&saved,&color,sizeof(color)) && !memcmp(before,pixels,sizeof(pixels)));
    fill_constant(pixels,32,32,104,140,170,150);
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_OK);
    assert(result.proposed_sensor_gain_q8[0]==(unsigned)floor(current[0]*inverse[170]/inverse[150]+0.5));
    assert(result.proposed_sensor_gain_q8[1]==256u);
    assert(result.proposed_sensor_gain_q8[2]==(unsigned)floor(current[2]*inverse[170]/inverse[140]+0.5));
    fill_constant(pixels,32,32,104,160,160,160);
    pixels[0]=248u;
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_SATURATED);
    assert(result.proposed_sensor_gain_q8[0]==0u);
    fill_constant(pixels,32,32,104,16,16,16);
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_DARK);
    fill_constant(pixels,32,32,104,100,220,100);
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_GAIN_RANGE);
    fill_constant(pixels,32,32,104,80,80,80);
    fill_constant(pixels+16*104,32,16,104,210,210,210);
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_NONUNIFORM);
    current[1]=257u;
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,32,32,current,&result)==CAPTURE_COLOR_INVALID);
    current[1]=256u;
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,0,0,2,2,current,&result)==CAPTURE_COLOR_INSUFFICIENT);
    assert(capture_color_calibrate_white(&color,pixels,32,32,104,31,31,2,2,current,&result)==CAPTURE_COLOR_INVALID);
    assert(!memcmp(&saved,&color,sizeof(color)));
}

static void test_invalid(void)
{
    unsigned char pixel[3]={0,0,0};
    capture_color bad={0};
    capture_color_stats stats;
    assert(capture_color_init(NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&bad,pixel,1,1,3,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,NULL,1,1,3,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,pixel,0,1,3,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,pixel,1,1,2,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,pixel,8193,1,8193u*3u,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,pixel,8192,8192,8192u*3u,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_process_bgr(&color,pixel,1,3,SIZE_MAX,NULL,NULL)==CAPTURE_COLOR_INVALID);
    assert(capture_color_measure(&color,pixel,1,1,3,0,&stats)==CAPTURE_COLOR_INVALID);
    assert(capture_color_measure(&color,pixel,1,1,3,1,NULL)==CAPTURE_COLOR_INVALID);
    assert(strcmp(capture_color_result_string(CAPTURE_COLOR_SATURATED),"unknown color error"));
}

int main(void)
{
    reference_init();
    assert(capture_color_init(&color)==CAPTURE_COLOR_OK);
    test_inverse();
    test_pixel_accuracy();
    test_statistics();
    test_calibration();
    test_invalid();
    test_row_rgb();
    test_small_scale_commutation();
    test_full_frame();
    puts("capture_color: all tests passed (sensor WB only, linear CCM, gamma, stats and white-target validation)");
    return 0;
}
