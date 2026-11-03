#include "../preview.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned services;
static void service(void *ctx) { (void)ctx; services++; }

static void test_capture_viewport(void)
{
    const size_t source_size=(size_t)12224*3040;
    const size_t output_size=(size_t)1920*480;
    unsigned char *source=malloc(source_size);
    unsigned char *output=malloc(output_size+64);
    unsigned x,y;
    assert(source && output);
    memset(source,0x7f,source_size);
    memset(output,0xa5,output_size+64);
    assert(preview_letterbox(source,4056,3040,12224,output+48*1920,
                             640,432,1920,NULL,NULL)==0);
    /* The reserved status rows and trailing guard must not be overwritten.
     * The complete 4:3 sensor field fills 576x432, with 32-pixel side bars. */
    for(y=0;y<48;y++) for(x=0;x<1920;x++) assert(output[y*1920+x]==0xa5);
    for(y=48;y<480;y++) for(x=0;x<640;x++) {
        unsigned char expected=(x<32 || x>=608)?0:0x7f;
        assert(output[y*1920+3*x]==expected);
        assert(output[y*1920+3*x+1]==expected);
        assert(output[y*1920+3*x+2]==expected);
    }
    for(x=0;x<64;x++) assert(output[output_size+x]==0xa5);
    free(source);
    free(output);
}
int main(void)
{
    unsigned x,y;
    unsigned char input[4][32], output[8][64];
    memset(input,0xee,sizeof(input));
    memset(output,0xa5,sizeof(output));
    for(y=0;y<4;y++) for(x=0;x<8;x++) {
        input[y][3*x]=(unsigned char)(10+x);
        input[y][3*x+1]=(unsigned char)(20+y);
        input[y][3*x+2]=200;
    }
    assert(preview_letterbox(&input[0][0],8,4,32,&output[0][0],16,8,64,service,NULL)==0);
    assert(services>0);
    for(y=0;y<8;y++) for(x=0;x<16;x++) {
        assert(output[y][3*x]==10+x/2);
        assert(output[y][3*x+1]==20+y/2);
        assert(output[y][3*x+2]==200);
    }
    for(y=0;y<8;y++) for(x=48;x<64;x++) assert(output[y][x]==0xa5);
    /* A cropped page retains the original sensor stride and source offset.
     * 4x2 crop at (2,1), not a tightly packed copy, displayed as 8x4. */
    assert(preview_letterbox(&input[1][6],4,2,32,&output[0][0],8,4,64,NULL,NULL)==0);
    for(y=0;y<4;y++) for(x=0;x<8;x++) {
        assert(output[y][3*x]==12+x/2);
        assert(output[y][3*x+1]==21+y/2);
        assert(output[y][3*x+2]==200);
    }
    memset(output,0xa5,sizeof(output));
    assert(preview_letterbox(&input[0][0],8,4,32,&output[0][0],8,8,64,NULL,NULL)==0);
    for(y=0;y<8;y++) for(x=0;x<8;x++) {
        if(y<2||y>=6) assert(output[y][3*x]==0&&output[y][3*x+1]==0&&output[y][3*x+2]==0);
        else assert(output[y][3*x]==10+x&&output[y][3*x+1]==20+y-2);
    }
    assert(preview_letterbox(&input[0][0],8,4,23,&output[0][0],8,8,64,NULL,NULL)==-1);
    memset(output,0,sizeof(output));
    preview_text(&output[0][0],8,8,64,6,6,"999",4,0x123456);
    for(y=0;y<8;y++) for(x=24;x<64;x++) assert(output[y][x]==0);
    memset(input,255,sizeof(input));
    assert(preview_brightness(&input[0][0],8,4,32)==255);
    memset(input,0,sizeof(input));
    assert(preview_brightness(&input[0][0],8,4,32)==0);
    test_capture_viewport();
    puts("PASS preview: BGR colour order, scale, full-resolution viewport, padding, clipping, brightness");
    return 0;
}
