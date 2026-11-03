/* Offscreen UI layout check using the production overlay, not a DP link test. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../preview.h"
#include "../studio_font.h"
#include "../capture_store.h"
#include "../capture_focus.h"
#define DISP_W 1280u
#define DISP_H 720u
#define DISP_STRIDE (DISP_W*3u)
#define PREVIEW_TOP 100u
#define PREVIEW_X 248u
#define PREVIEW_W 784u
#define PREVIEW_H 552u
static int sd_ready=1,focus_active,auto_exposure=1,denoise_enabled=1,focus_grid;
static unsigned focus_region,preview_fps_x10=148;
static const char *ui_status="LIVE";
static capture_store stores[3];
static capture_focus_history focus_history;
static capture_focus_result focus_result;
struct test_mode_info {unsigned line_length_clocks,timing_pixel_rate_hz;};
static const struct test_mode_info test_mode={12000u,840000000u};
static struct {unsigned exposure_lines,analogue_gain;const struct test_mode_info *mode;} sensor={1960u,0u,&test_mode};
int capture_store_current_name(const capture_store *s,char name[CAPTURE_NAME_CAPACITY])
{(void)s;strcpy(name,"01-01.bmp");return CAPTURE_OK;}
const char *capture_focus_region_name(unsigned r) {(void)r;return "CENTER";}
static const char *focus_quality(const capture_focus_result *r) {(void)r;return "READY";}
static int focus_frame_eligible(void) {return 1;}
static void focus_edge_text(char *s,size_t cap,unsigned value,unsigned count)
{(void)count;snprintf(s,cap,"%u.%03u",value/1000,value%1000);}
#include "../../../studio_overlay.c.inc"
static struct {uint8_t first[64],frame[DISP_H*DISP_STRIDE],last[64];} guarded;
static int pixel_is(unsigned x,unsigned y,uint32_t rgb)
{
    const uint8_t *p=guarded.frame+y*DISP_STRIDE+x*3u;
    return p[0]==(uint8_t)rgb && p[1]==(uint8_t)(rgb>>8) && p[2]==(uint8_t)(rgb>>16);
}
int main(int argc,char **argv)
{
    FILE *input,*output;unsigned x,y;
    if(argc!=3 && argc!=4)return 2;
    if(argc==4){focus_active=1;focus_history.current=3750;focus_history.peak=4200;
        focus_result.edge_h_milli=2506;focus_result.edge_v_milli=3307;
        focus_result.edge_h_count=focus_result.edge_v_count=72;}
    memset(&guarded,0,sizeof(guarded));memset(guarded.first,0xa5,64);memset(guarded.last,0xa5,64);
    input=fopen(argv[1],"rb");if(!input)return 3;
    for(y=0;y<PREVIEW_H;y++)assert(fread(guarded.frame+(PREVIEW_TOP+y)*DISP_STRIDE+PREVIEW_X*3u,1,PREVIEW_W*3u,input)==PREVIEW_W*3u);
    fclose(input);studio_overlay(guarded.frame);
    for(x=0;x<64;x++)assert(guarded.first[x]==0xa5 && guarded.last[x]==0xa5);
    if(!focus_active) {
        const unsigned right=PREVIEW_X+PREVIEW_W,mid=PREVIEW_TOP+PREVIEW_H/2u;
        for(y=PREVIEW_TOP+40u;y<PREVIEW_TOP+PREVIEW_H-40u;++y)
            for(x=right-6u;x<right;++x)assert(pixel_is(x,y,0x007aff));
        /* The triangle has a flat left edge and a rightward tip at mid-height. */
        for(y=0;y<17u;++y) {
            unsigned width=(y<9u?y:16u-y)+1u;
            for(x=0;x<width;++x)assert(pixel_is(right-36u+x,mid-8u+y,0x007aff));
            assert(pixel_is(right-36u+width,mid-8u+y,0xffffff));
        }
    }
    output=fopen(argv[2],"wb");if(!output)return 4;
    fprintf(output,"P6\n1280 720\n255\n");
    for(y=0;y<DISP_H;y++)for(x=0;x<DISP_W;x++){
        uint8_t *p=guarded.frame+y*DISP_STRIDE+x*3u;
        fputc(p[2],output);fputc(p[1],output);fputc(p[0],output);
    }
    fclose(output);puts("PASS production capture overlay:1280x720, buffer guards intact");return 0;
}
