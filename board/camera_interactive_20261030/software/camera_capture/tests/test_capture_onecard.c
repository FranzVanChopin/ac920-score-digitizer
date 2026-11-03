#include "capture_onecard.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static unsigned seen,fail_at;
static int step(void *p) {unsigned n=seen++;assert(p==&seen);return n==fail_at?-1:0;}
static void word(uint8_t *p,uint32_t v) {p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8);p[2]=(uint8_t)(v>>16);p[3]=(uint8_t)(v>>24);}
int main(void)
{
    enum capture_onecard_stage stage;unsigned i;
    const capture_onecard_ops ops={{step,step,step,step,step},&seen};
    const char ready[]="AC920_ONECARD_V1\ncapture=0\nlinux=1\nlinux_bytes=1863220\nlinux_crc32=CBF43926\nroot_partuuid=deadbeef-02\n";
    capture_onecard_ready r;uint8_t header[76]={0};uint32_t sum;
    for(i=0;i<ONECARD_STAGE_COUNT;++i) {
        seen=0;fail_at=i;
        assert(capture_onecard_switch(&ops,&stage)==-1);
        assert((unsigned)stage==i && seen==i+1u); /* no later reset/register write */
    }
    seen=0;fail_at=99;assert(capture_onecard_switch(&ops,&stage)==0 && seen==5);
    assert(capture_onecard_parse_ready(ready,strlen(ready),&r)==0 && r.linux_bytes==1863220 && r.linux_crc32==0xcbf43926u);
    assert(capture_onecard_parse_ready(ready,strlen(ready)-30,&r)==-1);
    assert(capture_onecard_parse_ready("bad",3,&r)==-1);
    assert(capture_onecard_parse_ready(NULL,0,&r)==-1);
    assert((capture_onecard_crc32(0xffffffffu,(const uint8_t *)"123456789",9)^0xffffffffu)==0xcbf43926u);
    sum=0xaa995566u+0x584c4e58u;
    word(header+0x20,0xaa995566u);word(header+0x24,0x584c4e58u);word(header+0x48,~sum);
    assert(capture_onecard_boot_header(header,sizeof(header))==0);
    header[0x30]=1;assert(capture_onecard_boot_header(header,sizeof(header))==-1);
    puts("PASS onecard: all staged failure cutoffs, CRC32, Ready parsing and BOOT header checks");
    return 0;
}
