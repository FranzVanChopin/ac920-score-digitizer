#include "dp_pixels.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void)
{
    const uint8_t src[24]={0,0,255,0,255,0,255,0,0,7,8,9,
                          255,255,255,0,0,0,33,22,11,7,8,9};
    const uint8_t expected[32]={255,0,0,255,0,255,0,255,0,0,255,255,0xA5,0xA5,0xA5,0xA5,
                               255,255,255,255,0,0,0,255,11,22,33,255,0xA5,0xA5,0xA5,0xA5};
    uint8_t out[34];memset(out,0xA5,sizeof(out));
    assert(dp_pixels_from_bgr(src,3,2,12,out+1,16)==0);
    assert(out[0]==0xA5 && out[33]==0xA5 && !memcmp(out+1,expected,32));
    assert(dp_pixels_from_bgr(NULL,3,2,12,out,16)==-1);
    assert(dp_pixels_from_bgr(src,3,2,12,NULL,16)==-1);
    assert(dp_pixels_from_bgr(src,0,2,12,out,16)==-1);
    assert(dp_pixels_from_bgr(src,3,2,8,out,16)==-1);
    assert(dp_pixels_from_bgr(src,3,2,12,out,11)==-1);
    assert(dp_pixels_from_bgr(src,3,2,SIZE_MAX,out,16)==-1);
    puts("PASS DP pixels: primary colors, alpha, padded rows, guards and invalid dimensions/strides");
    return 0;
}
