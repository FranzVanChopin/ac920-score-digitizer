/* Verify clipping/address guards and proportional mixed-case text rendering. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "../studio_font.h"
#define WIDTH 96u
#define HEIGHT 48u
#define STRIDE (WIDTH*3u+7u)
static struct {uint8_t before[32],image[STRIDE*HEIGHT],after[32];} surface;
static uint8_t reference[sizeof(surface.image)];

static void reset(void)
{
    unsigned y;
    memset(&surface,0xa5,sizeof(surface));
    for(y=0;y<HEIGHT;++y) memset(surface.image+y*STRIDE,255,WIDTH*3u);
}
static void guards(void)
{
    unsigned i,y;
    for(i=0;i<32;++i) assert(surface.before[i]==0xa5 && surface.after[i]==0xa5);
    for(y=0;y<HEIGHT;++y)for(i=WIDTH*3u;i<STRIDE;++i)assert(surface.image[y*STRIDE+i]==0xa5);
}
int main(void)
{
    unsigned i,changed=0,antialias=0;
    reset();
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,"AgW9",2,0x0055aau);
    guards();
    for(i=0;i<sizeof(surface.image);++i) {
        if(surface.image[i]!=255 && surface.image[i]!=0xa5) ++changed;
        if(surface.image[i]>0 && surface.image[i]<255 && surface.image[i]!=0x55 &&
           surface.image[i]!=0xaa && surface.image[i]!=0xa5) ++antialias;
    }
    assert(changed>100u && antialias>100u);
    reset();studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,WIDTH-2,HEIGHT-2,"jg_~",4,0);
    guards();
    reset();memcpy(reference,surface.image,sizeof(reference));
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,UINT_MAX,0,"x",2,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,UINT_MAX,"x",2,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,"x",0,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,"x",5,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,WIDTH*3u-1u,0,0,"x",2,0);
    studio_font_text(surface.image,UINT_MAX,HEIGHT,STRIDE,0,0,"x",2,0);
    studio_font_text(surface.image,WIDTH,UINT_MAX,STRIDE,0,0,"x",2,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,SIZE_MAX,0,0,"x",2,0);
    studio_font_text((uint8_t *)(UINTPTR_MAX-1u),1u,1u,3u,0u,0u,"x",1u,0u);
    studio_font_text(NULL,WIDTH,HEIGHT,STRIDE,0,0,"x",2,0);
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,NULL,2,0);
    assert(memcmp(reference,surface.image,sizeof(reference))==0);guards();
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,"A\xe4\xb8\xad" "B",2,0);
    memcpy(reference,surface.image,sizeof(reference));reset();
    studio_font_text(surface.image,WIDTH,HEIGHT,STRIDE,0,0,"AB",2,0);
    assert(memcmp(reference,surface.image,sizeof(reference))==0);guards();
    assert(studio_font_measure("A\xe4\xb8\xad" "B",2)==studio_font_measure("AB",2));
    assert(studio_font_measure("WWWW",2)>studio_font_measure("iiii",2));
    assert(studio_font_measure("01-01",4)<=184u);
    assert(studio_font_measure("FIXED CROP",2)<=184u);
    assert(studio_font_measure("_",0)==0 && studio_font_measure(NULL,1)==0);
    printf("PASS studio font: antialias, proportional mixed case, UTF8 skip, clipping, padded rows, pointer/size/coordinate overflow; alpha samples=%u\n",antialias);
    return 0;
}
