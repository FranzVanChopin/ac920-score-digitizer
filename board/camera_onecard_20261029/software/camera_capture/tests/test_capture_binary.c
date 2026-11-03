/* SPDX-License-Identifier: MIT */
#include "capture_binary.h"
#include "capture_focus.h"
#include "preview.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static capture_color color;
static uint8_t *workspace_allocation,*workspace;
static unsigned services;
static void service(void *context) { ++*(unsigned *)context; }
static unsigned gray(unsigned b,unsigned g,unsigned r)
{
    uint8_t in[3]={(uint8_t)b,(uint8_t)g,(uint8_t)r},out[3];
    capture_color_apply_bgr(&color,in,out);
    return (1868u*out[0]+9617u*out[1]+4899u*out[2]+8192u)>>14;
}
static void set_gray(uint8_t *p,unsigned v) { p[0]=p[1]=p[2]=(uint8_t)v; }
static void guards(void)
{
    unsigned i;
    for(i=0;i<32u;++i) {
        assert(workspace_allocation[i]==0x65u);
        assert(workspace[CAPTURE_BINARY_WORKSPACE_BYTES+i]==0x65u);
    }
}
static void binary_only(const capture_binary_result *r)
{
    size_t i;
    assert(r->stride==r->width);
    for(i=0;i<(size_t)r->width*r->height;++i) assert(r->binary[i]==0u || r->binary[i]==255u);
    guards();
}
static unsigned clamp(int x,unsigned n) { return x<0?0u:(x>=(int)n?n-1u:(unsigned)x); }
static void check_naive_mean(const capture_binary_config *c,const capture_binary_result *r)
{
    unsigned x,y,k,j,rad=c->adaptive_block_size/2u,area=c->adaptive_block_size*c->adaptive_block_size;
    for(y=0;y<r->height;++y) for(x=0;x<r->width;++x) {
        unsigned sum=0,expected;
        for(k=0;k<c->adaptive_block_size;++k) for(j=0;j<c->adaptive_block_size;++j)
            sum+=r->clahe[(size_t)clamp((int)y+(int)k-(int)rad,r->height)*r->width+
                         clamp((int)x+(int)j-(int)rad,r->width)];
        expected=(int)r->clahe[(size_t)y*r->width+x]>(int)((sum+area/2u)/area)-c->adaptive_c?255u:0u;
        assert(r->binary[(size_t)y*r->width+x]==expected);
    }
}
static void test_area_native_and_mean(void)
{
    enum {W=21,H=15,S=W*3+9};
    uint8_t source[S*H+64],before[sizeof(source)];
    capture_binary_config c;
    capture_binary_result r;
    unsigned x,y,ox,oy,k,j;
    memset(source,0xa6,sizeof(source));
    for(y=0;y<H;++y) for(x=0;x<W;++x) {
        uint8_t *p=source+32+y*S+3*x;
        p[0]=(uint8_t)(17*x+3*y);p[1]=(uint8_t)(7*x+13*y);p[2]=(uint8_t)(11*x+19*y);
    }
    memcpy(before,source,sizeof(source));
    capture_binary_default_config(&c);c.mode=CAPTURE_BINARY_FULL;c.target_width=7;c.adaptive_block_size=11;
    assert(capture_binary_run(&color,source+32,W,H,S,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services)==0);
    assert(r.width==7u && r.height==5u && r.interpolation==CAPTURE_BINARY_AREA && !r.has_paper_quad);
    for(oy=0;oy<5u;++oy) for(ox=0;ox<7u;++ox) {
        unsigned sum=0;
        for(k=0;k<3u;++k) for(j=0;j<3u;++j) {
            const uint8_t *p=source+32+(oy*3u+k)*S+(ox*3u+j)*3u;
            sum+=gray(p[0],p[1],p[2]);
        }
        assert(r.gray_resized[oy*7u+ox]==(sum+4u)/9u);
    }
    assert(!memcmp(source,before,sizeof(source)));binary_only(&r);check_naive_mean(&c,&r);
    /* Noninteger footprint reference: overlap in source-pixel edge units. */
    c.target_width=8;
    assert(capture_binary_run(&color,source+32,W,H,S,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==0);
    assert(r.height==6u);
    for(oy=0;oy<r.height;++oy) for(ox=0;ox<r.width;++ox) {
        double sum=0.0,x0=(double)ox*W/r.width,x1=(double)(ox+1u)*W/r.width;
        double y0=(double)oy*H/r.height,y1=(double)(oy+1u)*H/r.height;
        for(y=0;y<H;++y) for(x=0;x<W;++x) {
            double wx=fmax(0.0,fmin(x1,x+1.0)-fmax(x0,x));
            double wy=fmax(0.0,fmin(y1,y+1.0)-fmax(y0,y));
            const uint8_t *p=source+32+y*S+3*x;
            sum+=gray(p[0],p[1],p[2])*wx*wy;
        }
        assert(abs((int)r.gray_resized[oy*r.width+ox]-(int)floor(sum/((x1-x0)*(y1-y0))+0.5))<=1);
    }
    c.target_width=W;c.mode=CAPTURE_BINARY_ROI;c.roi_xywh[0]=2;c.roi_xywh[1]=3;c.roi_xywh[2]=7;c.roi_xywh[3]=5;
    assert(capture_binary_run(&color,source+32,W,H,S,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==0);
    assert(r.width==21u && r.height==15u && r.interpolation==CAPTURE_BINARY_CUBIC);
    assert(r.crop_xywh[0]==2u && r.crop_xywh[1]==3u);binary_only(&r);check_naive_mean(&c,&r);
    assert(!memcmp(source,before,sizeof(source)));
    puts("capture_binary: exact area reduction, native ROI, cubic selection, source/padding guards, and replicated mean reference PASS");
}
static void test_constant_and_lines(void)
{
    const unsigned w=800u,h=600u;
    uint8_t *source=(uint8_t *)malloc((size_t)w*h*3u);
    capture_binary_config c;
    capture_binary_result r;
    unsigned x,y,lines=0;
    assert(source);
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        unsigned v=140u+x*90u/w;
        if((y>=180u && y<=260u && y%20u==0u) || x==401u) v=15u;
        set_gray(source+((size_t)y*w+x)*3u,v);
    }
    capture_binary_default_config(&c);c.mode=CAPTURE_BINARY_FULL;c.target_width=400u;
    assert(capture_binary_run(&color,source,w,h,w*3u,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services)==0);
    assert(r.height==300u);
    for(y=90u;y<=130u;y+=10u) {
        unsigned black=0;
        for(x=40u;x<360u;++x) if(r.binary[(size_t)y*r.width+x]==0u) ++black;
        assert(black>300u);++lines;
    }
    /* NN sampling x=400 would miss source x=401; area retains this thin line. */
    assert(r.gray_resized[40u*r.width+200u]+30u<r.gray_resized[40u*r.width+199u]);
    assert(r.binary[40u*r.width+200u]==0u);
    assert(r.binary[40u*r.width+40u]==255u && lines==5u);binary_only(&r);
    for(y=0;y<h;++y) for(x=0;x<w;++x) set_gray(source+((size_t)y*w+x)*3u,175u);
    c.target_width=1000u;
    assert(capture_binary_run(&color,source,w,h,w*3u,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==0);
    assert(r.interpolation==CAPTURE_BINARY_CUBIC && r.height==750u);
    for(y=0;y<r.height;++y) for(x=0;x<r.width;++x) {
        assert(r.gray_resized[(size_t)y*r.width+x]==gray(175,175,175));
        assert(r.binary[(size_t)y*r.width+x]==255u);
    }
    binary_only(&r);free(source);
    puts("capture_binary: shaded background and 1-pixel staff/vertical lines survive area reduction; constant cubic image PASS");
}

static void page(uint8_t *source,unsigned w,unsigned h,unsigned stride,
                 double x0,double y0,double ux,double uy,double vx,double vy,int print)
{
    unsigned x,y;double determinant=ux*vy-uy*vx;
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        double dx=x-x0,dy=y-y0,u=(dx*vy-dy*vx)/determinant,v=(ux*dy-uy*dx)/determinant;
        if(u>=0.0 && u<=1.0 && v>=0.0 && v<=1.0) {
            unsigned value=(unsigned)(145.0+90.0*u);
            if(print && u>0.10 && u<0.90 && v>0.18 && v<0.85) {
                double local=fmod(v*700.0,130.0);
                unsigned line=(unsigned)local;
                if((line>=10u && line<12u) || (line>=24u && line<26u) || (line>=38u && line<40u) ||
                   (line>=52u && line<54u) || (line>=66u && line<68u)) value=15u;
                if(fmod(u*900.0,70.0)<4.0 && local>15.0 && local<63.0) value=15u;
            }
            set_gray(source+(size_t)y*stride+3u*x,value);
        }
    }
}
static void fill(uint8_t *source,unsigned w,unsigned h,unsigned stride,unsigned value)
{
    unsigned x,y;
    for(y=0;y<h;++y) for(x=0;x<w;++x) set_gray(source+(size_t)y*stride+3u*x,value);
}
static void save_pgm(const char *directory,const char *name,const uint8_t *image,unsigned w,unsigned h)
{
    char path[1024];FILE *file;
    int n=snprintf(path,sizeof(path),"%s/%s.pgm",directory,name);
    assert(n>0 && (size_t)n<sizeof(path));file=fopen(path,"wb");assert(file);
    assert(fprintf(file,"P5\n%u %u\n255\n",w,h)>0);
    assert(fwrite(image,1,(size_t)w*h,file)==(size_t)w*h);assert(fclose(file)==0);
}
static void test_detection(const char *output_directory)
{
    const unsigned w=960u,h=720u,stride=w*3u+16u;
    size_t bytes=(size_t)stride*h;
    uint8_t *source=(uint8_t *)malloc(bytes+64u),*before=(uint8_t *)malloc(bytes+64u);
    capture_binary_config c;
    capture_binary_result r;
    unsigned x,y;int status;
    assert(source && before);memset(source,0x91,bytes+64u);
    fill(source+32,w,h,stride,30u);
    page(source+32,w,h,stride,210,80,590,45,-45,515,1);
    memcpy(before,source,bytes+64u);
    capture_binary_default_config(&c);c.mode=CAPTURE_BINARY_AUTO;c.target_width=640u;
    status=capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services);
    if(status) fprintf(stderr,"tilted-page detection: %d (%s), candidates=%u\n",status,capture_binary_result_string(status),r.candidates);
    assert(status==0 && r.has_paper_quad && r.width==640u);
    assert(r.crop_xywh[0]<=165u && r.crop_xywh[1]<=80u);
    assert(r.crop_xywh[0]+r.crop_xywh[2]>800u && r.crop_xywh[1]+r.crop_xywh[3]>640u);
    assert(r.area_fraction>0.4 && r.rectangularity>0.85 && r.whitened_pixels>0u);
    assert(r.gray_resized[0]==255u);binary_only(&r);
    assert(!memcmp(source,before,bytes+64u));
    {
        uint8_t *detect_workspace=(uint8_t *)malloc(CAPTURE_PAPER_WORKSPACE_BYTES+64u);
        capture_binary_result detected;
        unsigned k;
        assert(detect_workspace); memset(detect_workspace,0x96,CAPTURE_PAPER_WORKSPACE_BYTES+64u);
        assert(capture_paper_detect(&color,source+32,w,h,stride,detect_workspace+32,
            CAPTURE_PAPER_WORKSPACE_BYTES,&detected,service,&services)==CAPTURE_BINARY_OK);
        assert(!memcmp(detected.crop_xywh,r.crop_xywh,sizeof(r.crop_xywh)));
        assert(!memcmp(detected.paper_quad_xy,r.paper_quad_xy,sizeof(r.paper_quad_xy)));
        assert(!detected.gray_resized && !detected.clahe && !detected.binary);
        assert(detected.source_width==w && detected.source_height==h);
        for(k=0;k<32u;++k) {
            assert(detect_workspace[k]==0x96u);
            assert(detect_workspace[32u+CAPTURE_PAPER_WORKSPACE_BYTES+k]==0x96u);
        }
        assert(!memcmp(source,before,bytes+64u)); free(detect_workspace);
        puts("capture_paper_detect: preview/color geometry matches binary crop, 3MiB guards and immutable source PASS");
    }
    if(output_directory) {
        uint8_t *g=(uint8_t *)malloc((size_t)w*h);assert(g);
        for(y=0;y<h;++y) for(x=0;x<w;++x) g[(size_t)y*w+x]=source[32+(size_t)y*stride+x*3u];
        save_pgm(output_directory,"tilted-source",g,w,h);free(g);
        save_pgm(output_directory,"tilted-gray",r.gray_resized,r.width,r.height);
        save_pgm(output_directory,"tilted-clahe",r.clahe,r.width,r.height);
        save_pgm(output_directory,"tilted-binary",r.binary,r.width,r.height);
    }
    /* Entire white view has no independently visible boundary. */
    fill(source+32,w,h,stride,220u);
    assert(capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_NOT_FOUND);
    assert(!r.binary);
    assert(capture_paper_detect(&color,source+32,w,h,stride,workspace,
        CAPTURE_PAPER_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_NOT_FOUND);
    assert(!r.has_paper_quad && !r.binary);
    /* White desk with internal rectangular staff frames, including thick
     * black rules that fool tests sampling only immediately outside a box. */
    {
        static const unsigned thickness[]={2u,5u,10u,20u,40u};unsigned ti;
        for(ti=0;ti<sizeof(thickness)/sizeof(thickness[0]);++ti) {
            unsigned t=thickness[ti];fill(source+32,w,h,stride,220u);
            for(y=120;y<600u;++y) for(x=160;x<800u;++x)
                if(x<160u+t || x>=800u-t || y<120u+t || y>=600u-t ||
                   (y>200u && y<500u && y%25u<2u))
                    set_gray(source+32+(size_t)y*stride+3u*x,15u);
            status=capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL);
            if(status!=CAPTURE_BINARY_NOT_FOUND) fprintf(stderr,"printed frame thickness=%u returned %d\n",t,status);
            assert(status==CAPTURE_BINARY_NOT_FOUND);
        }
    }
    /* No safe boundary on near-white background. */
    fill(source+32,w,h,stride,220u);page(source+32,w,h,stride,170,90,630,0,0,540,0);
    assert(capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_NOT_FOUND);
    /* Cropped paper is rejected, even when most of the page is visible. */
    fill(source+32,w,h,stride,30u);page(source+32,w,h,stride,-25,80,800,0,0,570,1);
    assert(capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_NOT_FOUND);
    /* Two separate plausible pages, each >18% frame area, cannot be guessed. */
    fill(source+32,w,h,stride,30u);
    page(source+32,w,h,stride,45,100,380,0,0,520,1);
    page(source+32,w,h,stride,535,100,380,0,0,520,1);
    status=capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL);
    if(status!=CAPTURE_BINARY_AMBIGUOUS) fprintf(stderr,"two-page result=%d candidates=%u\n",status,r.candidates);
    assert(status==CAPTURE_BINARY_AMBIGUOUS && !r.binary);
    free(source);free(before);guards();
    puts("capture_binary: rotated/shadowed paper succeeds; white view, printed frame, low contrast, clipped paper and ambiguous pages abstain PASS");
}

static void test_full_resolution(const char *output_directory)
{
    const unsigned w=4056u,h=3040u,stride=12224u;
    size_t bytes=(size_t)stride*h;
    uint8_t *source=(uint8_t *)malloc(bytes+64u);
    capture_binary_config c;capture_binary_result r;
    unsigned x,y,i,before_services=services;
    uint64_t hash_before=0,hash_after=0;
    assert(source);memset(source,0x83,bytes+64u);fill(source+32,w,h,stride,24u);
    page(source+32,w,h,stride,780,350,2600,120,-120,2200,1);
    for(i=0;i<bytes+64u;++i) hash_before=hash_before*131u+source[i];
    capture_binary_default_config(&c);
    c.mode=CAPTURE_BINARY_AUTO; /* Retain optional detector regression explicitly. */
    assert(capture_binary_run(&color,source+32,w,h,stride,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services)==0);
    assert(r.width==1920u && r.height>1500u && r.height<1900u);
    assert(r.interpolation==CAPTURE_BINARY_AREA);
    assert(r.crop_xywh[0]<=660u && r.crop_xywh[1]<=350u && r.crop_xywh[0]+r.crop_xywh[2]>3380u &&
           r.crop_xywh[1]+r.crop_xywh[3]>2670u);
    assert(fabs(r.scale_xy[0]-1920.0/r.crop_xywh[2])<1e-12);
    assert(services-before_services>300u);
    for(i=0;i<bytes+64u;++i) hash_after=hash_after*131u+source[i];
    assert(hash_before==hash_after);binary_only(&r);
    for(y=0;y<h;++y) for(x=w*3u;x<stride;++x) assert(source[32+(size_t)y*stride+x]==0x83u);
    if(output_directory) {
        save_pgm(output_directory,"full-gray",r.gray_resized,r.width,r.height);
        save_pgm(output_directory,"full-clahe",r.clahe,r.width,r.height);
        save_pgm(output_directory,"full-binary",r.binary,r.width,r.height);
    }
    printf("capture_binary: native 4056x3040 -> ROI [%u,%u,%u,%u] -> %ux%u, source/padding unchanged, %u service calls PASS\n",
           r.crop_xywh[0],r.crop_xywh[1],r.crop_xywh[2],r.crop_xywh[3],r.width,r.height,services-before_services);
    free(source);
}

static void test_fixed_a4_geometry(void)
{
    unsigned roi[4],small[4],i,z;
    const unsigned native[4]={0,86,4056,2868};
    const unsigned preview[4]={0,43,2028,1434};
    capture_focus_rect focus;
    assert(!capture_a4_crop(4056,3040,roi) && !memcmp(roi,native,sizeof(roi)));
    assert(!capture_a4_crop(2028,1520,small) && !memcmp(small,preview,sizeof(small)));
    assert(roi[0]==2u*small[0] && roi[1]==2u*small[1]);
    assert(roi[2]==2u*small[2] && roi[3]==2u*small[3]);
    assert(fabs((double)roi[2]*210.0/297.0-roi[3])<0.5);
    assert(roi[0]==0u && roi[2]==4056u && roi[1]*2u+roi[3]==3040u);
    for(i=0;i<5u;++i) {
        assert(!capture_focus_region(roi[2],roi[3],i,500,500,&focus));
        assert(focus.x+500u<=roi[2] && focus.y+500u<=roi[3]);
        for(z=1;z<=2u;++z) {
            assert(!capture_focus_region(roi[2],roi[3],i,640u/z,432u/z,&focus));
            assert(focus.x+focus.width<=roi[2] && focus.y+focus.height<=roi[3]);
        }
    }
    assert(capture_a4_crop(0,3040,roi)==CAPTURE_BINARY_INVALID);
    for(i=0;i<4u;++i) assert(!roi[i]);
    assert(capture_a4_crop(4056,0,roi)==CAPTURE_BINARY_INVALID);
    assert(capture_a4_crop(4056,2867,roi)==CAPTURE_BINARY_INVALID);
    assert(capture_a4_crop(8193,99,roi)==CAPTURE_BINARY_INVALID);
    assert(capture_a4_crop(8192,8192,roi)==CAPTURE_BINARY_INVALID);
    assert(capture_a4_crop(4056,3040,NULL)==CAPTURE_BINARY_INVALID);
    assert(!capture_a4_crop(99,70,roi) && roi[2]==99u && roi[3]==70u);
    assert(!capture_a4_crop(4056,2868,roi) && !roi[0] && !roi[1]);
    puts("capture_A4: all4056 columns retained, centered2868 short edge, exact2:1 preview and all five focus regions inside saved crop PASS");
}

static void test_fixed_a4_any_scene(void)
{
    const unsigned w=420u,h=396u,stride=w*3u+8u;
    const size_t bytes=(size_t)stride*h;
    uint8_t *source=malloc(bytes),*before=malloc(bytes);
    capture_binary_config c;capture_binary_result r;
    unsigned scene,x,y,expected[4];
    assert(source && before);assert(!capture_a4_crop(w,h,expected));
    capture_binary_default_config(&c);
    assert(c.mode==CAPTURE_BINARY_A4);
    c.target_width=w; /* Same no-resize policy as the board application. */
    for(scene=0;scene<5u;++scene) {
        memset(source,0xb4,bytes);
        /* White view, dark view, no-paper checker, clipped off-centre page,
         * and two separate white regions must all produce the SAME rectangle. */
        fill(source,w,h,stride,scene==0?255u:0u);
        if(scene==2) for(y=0;y<h;++y) for(x=0;x<w;++x)
            set_gray(source+y*stride+3u*x,((x/20u+y/20u)%2u)?200u:30u);
        if(scene==3) page(source,w,h,stride,-50,30,300,0,0,300,1);
        if(scene==4) {
            page(source,w,h,stride,5,40,130,0,0,300,0);
            page(source,w,h,stride,280,40,130,0,0,300,0);
        }
        memcpy(before,source,bytes);
        assert(capture_binary_run(&color,source,w,h,stride,&c,workspace,
            CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_OK);
        assert(r.width==420u && r.height==297u && !r.has_paper_quad && !r.whitened_pixels);
        assert(r.interpolation==CAPTURE_BINARY_NONE);
        assert(!r.detection_width && !r.detection_height && !r.candidates);
        assert(!memcmp(r.crop_xywh,expected,sizeof(expected)));
        assert(!memcmp(source,before,bytes));binary_only(&r);
        if(scene<2u) for(y=0;y<r.height;++y) for(x=0;x<r.width;++x)
            assert(r.gray_resized[y*r.width+x]==gray(scene?0u:255u,scene?0u:255u,scene?0u:255u));
    }
    free(source);free(before);
    puts("capture_A4: white, black, no paper, partial page and multiple pages all save-ready; fixed crop, no detector/whitening/source mutation PASS");
}

static void test_fixed_a4_native(void)
{
    const unsigned w=4056u,h=3040u,stride=12224u;
    const size_t bytes=(size_t)stride*h;
    uint8_t *source=malloc(bytes+64u),*display=malloc(640u*480u*3u+32u);
    capture_binary_config c;capture_binary_result r;
    unsigned roi[4],x,y,i;
    uint64_t before=0,after=0;
    capture_focus_result focus;
    capture_focus_workspace *fw=malloc(sizeof(*fw));
    const uint8_t *crop;
    assert(source && display && fw);
    assert(!capture_a4_crop(w,h,roi));
    memset(source,0xc3,bytes+64u);fill(source+32,w,h,stride,220u);
    crop=source+32+(size_t)roi[1]*stride+3u*roi[0];
    /* A uniform view has no detectable page. The board configuration still
     * succeeds at real4056x3040 input and native4056x2868 Grey/Binary output. */
    capture_binary_default_config(&c);
    c.target_width=w;c.whiten_outside_paper=0u;
    assert(!capture_binary_run(&color,source+32,w,h,stride,&c,workspace,
        CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services));
    assert(r.width==4056u && r.height==2868u && r.interpolation==CAPTURE_BINARY_NONE);
    assert(!r.has_paper_quad && !r.whitened_pixels && !memcmp(roi,r.crop_xywh,sizeof(roi)));
    for(i=0;i<r.width*r.height;++i) assert(r.gray_resized[i]==gray(220,220,220) && r.binary[i]==255u);
    /* Coloured ramp identifies every native source pixel, crop origin, row
     * stride and edge. No geometric resampling in the native size path. */
    for(y=0;y<h;++y) for(x=0;x<w;++x) {
        uint8_t *p=source+32+(size_t)y*stride+3u*x;
        p[0]=(uint8_t)(x*3u+y);p[1]=(uint8_t)(x+y*5u);p[2]=(uint8_t)(x*7u+y*11u);
    }
    for(i=0;i<bytes+64u;++i) before=before*131u+source[i];
    assert(!capture_binary_run(&color,source+32,w,h,stride,&c,workspace,
        CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL));
    assert(r.width==4056u && r.height==2868u && r.interpolation==CAPTURE_BINARY_NONE);
    for(y=0;y<r.height;++y) for(x=0;x<r.width;++x) {
        const uint8_t *p=crop+(size_t)y*stride+3u*x;
        assert(r.gray_resized[y*r.width+x]==gray(p[0],p[1],p[2]));
    }
    /* Preview letterbox must show this sensor-oriented crop, with black bars outside
     * and the preserved48-pixel UI area. Origin-aligned NN mapping is checked. */
    memset(display,0x96,640u*480u*3u+32u);
    assert(!preview_letterbox(crop,roi[2],roi[3],stride,display+48u*1920u,640,432,1920,NULL,NULL));
    for(y=0;y<432u;++y) for(x=0;x<640u;++x) {
        const uint8_t *p=display+(y+48u)*1920u+3u*x;
        if(x<15u || x>=625u) assert(!p[0] && !p[1] && !p[2]);
        else {
            unsigned sx=(x-15u)*4056u/610u,sy=y*2868u/432u;
            if(memcmp(p,crop+(size_t)sy*stride+3u*sx,3u)) {
                /* Truncated Q32 phase can land one pixel below an exact
                 * rational boundary; it cannot skip arbitrary source pixels. */
                assert(sx && (x-15u)*4056u==sx*610u);
                assert(!memcmp(p,crop+(size_t)sy*stride+3u*(sx-1u),3u));
            }
        }
    }
    for(i=0;i<48u*1920u;++i) assert(display[i]==0x96);
    for(i=0;i<32u;++i) assert(display[640u*480u*3u+i]==0x96);
    for(i=0;i<5u;++i) {
        unsigned xx,yy;
        assert(!capture_focus_measure(crop,4056,2868,stride,i,0,fw,&focus,NULL,NULL));
        for(yy=0;yy<500u;++yy) for(xx=0;xx<500u;++xx)
            assert(fw->gray[yy*500u+xx]==crop[(size_t)(focus.roi.y+yy)*stride+3u*(focus.roi.x+xx)+1u]);
    }
    for(i=0;i<bytes+64u;++i) after=after*131u+source[i];
    assert(before==after);binary_only(&r);free(source);free(display);free(fw);
    puts("capture_A4: full-resolution blank scene ->4056x2868; every native pixel including both long-edge endpoints; A4 preview, five focus G ROIs, padding and guards PASS");
}

static void test_invalid(void)
{
    uint8_t source[32]={0};capture_binary_config c;capture_binary_result r;capture_color bad={0};
    capture_binary_default_config(&c);c.mode=CAPTURE_BINARY_FULL;c.target_width=1u;
    assert(capture_paper_detect(&color,source,1,1,3,workspace,CAPTURE_PAPER_WORKSPACE_BYTES-1u,&r,NULL,NULL)==CAPTURE_BINARY_WORKSPACE);
    assert(capture_paper_detect(&color,source,1,1,3,workspace+1,CAPTURE_PAPER_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_WORKSPACE);
    assert(capture_paper_detect(&bad,source,1,1,3,workspace,CAPTURE_PAPER_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_paper_detect(&color,workspace,1,1,3,workspace,CAPTURE_PAPER_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_paper_detect(&color,source,1,3,SIZE_MAX,workspace,CAPTURE_PAPER_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(NULL,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&bad,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,NULL,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,source,0,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,source,1,1,2,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,source,1,3,SIZE_MAX,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,source,8192,8192,8192u*3u,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    assert(capture_binary_run(&color,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES-1u,&r,NULL,NULL)==CAPTURE_BINARY_WORKSPACE);
    assert(capture_binary_run(&color,source,1,1,3,&c,workspace+1,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_WORKSPACE);
    assert(capture_binary_run(&color,workspace,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    c.adaptive_block_size=50;
    assert(capture_binary_run(&color,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    c.adaptive_block_size=51;c.clahe_clip_limit=NAN;
    assert(capture_binary_run(&color,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    c.clahe_clip_limit=2.0;c.mode=CAPTURE_BINARY_ROI;c.roi_xywh[2]=2;c.roi_xywh[3]=1;
    assert(capture_binary_run(&color,source,1,1,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_INVALID);
    c.mode=CAPTURE_BINARY_FULL;c.target_width=8192;
    assert(capture_binary_run(&color,source,1,2,3,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,NULL,NULL)==CAPTURE_BINARY_OUTPUT_BOUNDS);
    guards();
}

static void test_workspace_limit(void)
{
    const unsigned w=4000u,h=4000u;
    const size_t bytes=(size_t)w*h*3u,pixels=(size_t)w*h;
    uint8_t *source=(uint8_t *)malloc(bytes);
    capture_binary_config c;capture_binary_result r;
    size_t i;unsigned before=services;
    assert(source);memset(source,175,bytes);
    capture_binary_default_config(&c);c.mode=CAPTURE_BINARY_FULL;c.target_width=w;
    c.clahe_tiles_x=c.clahe_tiles_y=16u;c.adaptive_block_size=255u;
    assert(capture_binary_run(&color,source,w,h,w*3u,&c,workspace,CAPTURE_BINARY_WORKSPACE_BYTES,&r,service,&services)==0);
    assert(r.width==w && r.height==h && r.interpolation==CAPTURE_BINARY_NONE);
    assert(r.binary+pixels<=workspace+CAPTURE_BINARY_WORKSPACE_BYTES);
    assert(r.clahe==r.gray_resized+pixels && r.binary==r.clahe+pixels);
    for(i=0;i<pixels;++i) {
        assert(r.gray_resized[i]==gray(175,175,175));
        assert(r.clahe[i]==r.clahe[0]);assert(r.binary[i]==255u);
    }
    for(i=0;i<bytes;++i) assert(source[i]==175u);
    assert(services-before>500u);guards();
    printf("capture_binary: exact 16MP limit, all 48MB planes, 255x255 mean and 16x16 CLAHE, workspace used %lu of %u bytes PASS\n",
           (unsigned long)((r.binary+pixels)-workspace),CAPTURE_BINARY_WORKSPACE_BYTES);
    free(source);
}
int main(int argc,char **argv)
{
    const char *directory=argc>1?argv[1]:NULL;
    workspace_allocation=(uint8_t *)malloc(CAPTURE_BINARY_WORKSPACE_BYTES+64u);assert(workspace_allocation);
    memset(workspace_allocation,0x65,CAPTURE_BINARY_WORKSPACE_BYTES+64u);workspace=workspace_allocation+32u;
    assert(((uintptr_t)workspace&7u)==0u);assert(capture_color_init(&color)==CAPTURE_COLOR_OK);
    test_invalid();test_area_native_and_mean();test_constant_and_lines();test_detection(directory);test_full_resolution(directory);
    test_workspace_limit();
    test_fixed_a4_geometry();test_fixed_a4_any_scene();test_fixed_a4_native();
    free(workspace_allocation);
    puts("capture_binary: all tests PASS");return 0;
}
