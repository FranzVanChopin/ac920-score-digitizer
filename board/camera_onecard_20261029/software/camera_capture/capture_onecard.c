#include "capture_onecard.h"
#include <string.h>
#include <stdlib.h>
#include <errno.h>
static uint32_t le32(const uint8_t *p)
{return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
int capture_onecard_boot_header(const uint8_t *p,size_t n)
{
    uint32_t sum=0;unsigned i;
    if(!p || n<0x4cu || le32(p+0x20)!=0xaa995566u || le32(p+0x24)!=0x584c4e58u) return -1;
    for(i=0x20;i<=0x48;i+=4) sum+=le32(p+i);
    return sum==0xffffffffu?0:-1;
}
uint32_t capture_onecard_crc32(uint32_t state,const uint8_t *p,size_t n)
{
    size_t i;unsigned b;
    if(!p) return state;
    for(i=0;i<n;++i) {
        state^=p[i];
        for(b=0;b<8;++b) state=(state>>1)^(0xedb88320u&(0u-(state&1u)));
    }
    return state;
}
int capture_onecard_parse_ready(const char *p,size_t n,capture_onecard_ready *out)
{
    static const char prefix[]="AC920_ONECARD_V1\ncapture=0\nlinux=1\nlinux_bytes=";
    char text[1024];const char *crc;char *end;unsigned long value;
    if(!p || !out || n>=sizeof(text) || n<sizeof(prefix) || memchr(p,0,n)) return -1;
    memcpy(text,p,n);text[n]=0;p=text;
    if(strncmp(p,prefix,sizeof(prefix)-1u)) return -1;
    p+=sizeof(prefix)-1u;
    if(*p<'0'||*p>'9') return -1;
    errno=0;value=strtoul(p,&end,10);
    if(errno || value<4096ul || value>0x10000000ul || *end!='\n') return -1;
    out->linux_bytes=(uint32_t)value;
    crc=end+1;
    if(strncmp(crc,"linux_crc32=",12)) return -1;
    crc+=12;
    if(strspn(crc,"0123456789abcdefABCDEF")!=8u || crc[8]!='\n') return -1;
    errno=0;value=strtoul(crc,&end,16);
    if(errno) return -1; /* exactly eight hex digits above already bound it */
    out->linux_crc32=(uint32_t)value;
    /* Additional lines (root PARTUUID/card identity) are for Linux validation. */
    return 0;
}
int capture_onecard_switch(const capture_onecard_ops *ops,enum capture_onecard_stage *stage)
{
    unsigned i;
    if(!ops || !stage) return -1;
    for(i=0;i<ONECARD_STAGE_COUNT;++i) {
        *stage=(enum capture_onecard_stage)i;
        if(!ops->step[i] || ops->step[i](ops->context)) return -1;
    }
    return 0;
}
