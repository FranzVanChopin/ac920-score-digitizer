/* Sparse RAM disk only. Never opens a physical disk or Windows volume. */
#include "ff.h"
#include "diskio.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define SECTORS 4194304u
#define SIZE (1024u*1024u+137u)
static BYTE **media;
static unsigned writes,max_count,fail_write,fail_read,reads;
static FATFS fs;
static BYTE work[4096],buffer[262144];
DSTATUS disk_initialize(BYTE d) { return d==0?0:STA_NOINIT; }
DSTATUS disk_status(BYTE d) { return d==0?0:STA_NOINIT; }
DRESULT disk_ioctl(BYTE d,BYTE command,void *data)
{
    if(d) return RES_PARERR;
    if(command==GET_SECTOR_COUNT) *(LBA_t *)data=SECTORS;
    else if(command==GET_SECTOR_SIZE) *(WORD *)data=512;
    else if(command==GET_BLOCK_SIZE) *(DWORD *)data=1;
    else if(command!=CTRL_SYNC) return RES_PARERR;
    return RES_OK;
}
DRESULT disk_read(BYTE d,BYTE *out,LBA_t sector,UINT count)
{
    unsigned i;
    assert(!d && sector+count<=SECTORS);
    if(++reads==fail_read) return RES_ERROR;
    for(i=0;i<count;++i) {
        if(media[sector+i]) memcpy(out+512u*i,media[sector+i],512);
        else memset(out+512u*i,0,512);
    }
    return RES_OK;
}
DRESULT disk_write(BYTE d,const BYTE *in,LBA_t sector,UINT count)
{
    unsigned i;
    assert(!d && sector+count<=SECTORS);
    if(count>max_count) max_count=count;
    if(++writes==fail_write) return RES_ERROR;
    for(i=0;i<count;++i) {
        if(!media[sector+i]) { media[sector+i]=malloc(512);assert(media[sector+i]); }
        memcpy(media[sector+i],in+512u*i,512);
    }
    return RES_OK;
}
DWORD get_fattime(void) { return (46u<<25)|(10u<<21)|(28u<<16); }
static BYTE pattern(unsigned offset) { return (BYTE)(offset*13u+(offset>>8)); }
static void write_pattern(FIL *f,unsigned size,unsigned first)
{
    unsigned offset=0,i;
    UINT n;
    while(offset<size) {
        unsigned amount=offset==0?first:sizeof(buffer);
        if(amount>size-offset) amount=size-offset;
        for(i=0;i<amount;++i) buffer[i]=pattern(offset+i);
        assert(f_write(f,buffer,amount,&n)==FR_OK && n==amount);
        offset+=amount;
    }
}
static void verify(const char *path,unsigned size,int constant)
{
    FIL f;
    unsigned at=0,i;
    UINT n;
    assert(f_open(&f,path,FA_READ)==FR_OK && f_size(&f)==size);
    while(at<size) {
        unsigned amount=sizeof(buffer);
        if(amount>size-at) amount=size-at;
        assert(f_read(&f,buffer,amount,&n)==FR_OK && n==amount);
        for(i=0;i<amount;++i) assert(buffer[i]==(constant<0?pattern(at+i):(BYTE)constant));
        at+=amount;
    }
    /* Random seeks catch incorrect fp->clust updates at partial cluster ends. */
    for(at=17;at+100<size;at+=8131) {
        assert(f_lseek(&f,at)==FR_OK && f_read(&f,buffer,100,&n)==FR_OK && n==100);
        for(i=0;i<100;++i) assert(buffer[i]==(constant<0?pattern(at+i):(BYTE)constant));
    }
    assert(f_close(&f)==FR_OK);
}
int main(void)
{
    FIL a,b;
    MKFS_PARM format={FM_FAT32|FM_SFD,2,0,0,16384};
    UINT n;
    unsigned i,continuous_calls,fragment_calls;
    media=calloc(SECTORS,sizeof(*media));assert(media);
    assert(f_mkfs("0:",&format,work,sizeof(work))==FR_OK);
    assert(f_mount(&fs,"0:",1)==FR_OK && fs.fs_type==FS_FAT32 && fs.csize==32);
    assert(f_open(&a,"0:/full.bmp",FA_WRITE|FA_CREATE_NEW)==FR_OK);
    assert(f_lseek(&a,SIZE)==FR_OK && f_tell(&a)==SIZE && f_lseek(&a,0)==FR_OK);
    writes=max_count=0;
    write_pattern(&a,SIZE,54);
    continuous_calls=writes;
#ifndef ORIGINAL_FATFS
    assert(max_count>32 && max_count<=512);
#else
    assert(max_count<=32);
#endif
    assert(f_close(&a)==FR_OK);
    verify("0:/full.bmp",SIZE,-1);
    /* All64 clusters alternate between two files: no illegal coalescing. */
    assert(f_open(&a,"0:/frag.bmp",FA_WRITE|FA_CREATE_NEW)==FR_OK);
    assert(f_open(&b,"0:/guard.bmp",FA_WRITE|FA_CREATE_NEW)==FR_OK);
    memset(buffer,0x5a,16384);
    for(i=0;i<64;++i) {
        assert(f_lseek(&a,(i+1u)*16384u)==FR_OK);
        assert(f_write(&b,buffer,16384,&n)==FR_OK && n==16384);
    }
    assert(f_close(&b)==FR_OK && f_lseek(&a,0)==FR_OK);
    writes=max_count=0;
    write_pattern(&a,1048576,262144);
    fragment_calls=writes;
    assert(max_count<=32 && f_close(&a)==FR_OK);
    verify("0:/frag.bmp",1048576,-1);
    verify("0:/guard.bmp",1048576,0x5a);
    /* Growing files, no preallocation; unaligned tails; remount and verify. */
    assert(f_open(&a,"0:/grow.bmp",FA_WRITE|FA_CREATE_NEW)==FR_OK);
    write_pattern(&a,SIZE,1);
    assert(f_close(&a)==FR_OK);
    assert(f_mount(NULL,"0:",0)==FR_OK && f_mount(&fs,"0:",1)==FR_OK);
    verify("0:/grow.bmp",SIZE,-1);
    verify("0:/full.bmp",SIZE,-1);
    /* A failed multi-sector data transfer must remain a filesystem error. */
    assert(f_open(&a,"0:/fail.bmp",FA_WRITE|FA_CREATE_NEW)==FR_OK);
    assert(f_lseek(&a,SIZE)==FR_OK && f_lseek(&a,0)==FR_OK);
    fail_write=writes+1;
    assert(f_write(&a,buffer,sizeof(buffer),&n)==FR_DISK_ERR && n==0);
    fail_write=0;(void)f_close(&a);
    verify("0:/guard.bmp",1048576,0x5a);
    assert(f_mount(NULL,"0:",0)==FR_OK);
    for(i=0;i<SECTORS;++i) free(media[i]);
    free(media);
    printf("PASS actual FatFs: contiguous calls=%u fragmented calls=%u; byte-exact reopen/seek/remount; growth, unaligned, guard-file and I/O-failure cases\n",continuous_calls,fragment_calls);
    return 0;
}
