#include "capture_pair.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

static void service(void *context) { ++*(unsigned *)context; }

int main(void)
{
    unsigned x,y,calls=0;
    uint8_t source[3][256*3+16],before[sizeof(source)],rgb[256*3],gray[256];
    capture_pair_frame frame={source[0],256,3,sizeof(source[0]),service,&calls};
    memset(source,0xCD,sizeof(source));
    for(y=0;y<3;++y) for(x=0;x<256;++x) {
        source[y][3*x]=(uint8_t)(y?x*7+3:x);
        source[y][3*x+1]=(uint8_t)(y?x*13+1:x);
        source[y][3*x+2]=(uint8_t)(y?x*23+9:x);
    }
    memcpy(before,source,sizeof(source));
    for(y=0;y<3;++y) {
        assert(capture_pair_rgb_row(&frame,y,rgb,sizeof(rgb))==CAPTURE_OK);
        assert(capture_pair_gray_row(&frame,y,gray,sizeof(gray))==CAPTURE_OK);
        for(x=0;x<256;++x) {
            assert(rgb[3*x]==source[y][3*x+2] && rgb[3*x+1]==source[y][3*x+1] && rgb[3*x+2]==source[y][3*x]);
            assert(gray[x]==(uint8_t)((4899u*rgb[3*x]+9617u*rgb[3*x+1]+1868u*rgb[3*x+2]+8192u)>>14));
            if(!y) assert(gray[x]==x);
        }
    }
    assert(calls==6 && memcmp(before,source,sizeof(source))==0);
    assert(capture_pair_gray_row(&frame,3,gray,sizeof(gray))==CAPTURE_ROW_ERROR);
    assert(capture_pair_rgb_row(&frame,0,rgb,sizeof(rgb)-1)==CAPTURE_ROW_ERROR);
    assert(capture_pair_gray_row(&frame,0,NULL,sizeof(gray))==CAPTURE_ROW_ERROR);
    frame.stride=1;
    assert(capture_pair_gray_row(&frame,0,gray,sizeof(gray))==CAPTURE_ROW_ERROR);
    puts("capture_pair: exact saved-RGB to gray, all gray levels, row bounds, service and immutable source PASS");
    return 0;
}
