#include "capture_fatfs.h"
#include "xtime_l.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static capture_fatfs fs;
static uint8_t disk[2*CAPTURE_SD_BUFFER_BYTES+100],data[sizeof(disk)];
static unsigned writes,syncs,closes,seek_calls,fail_write,short_write,fail_sync,fail_close,short_seek,fail_seek;
static const capture_fs_ops *ops=&capture_fatfs_operations;
void XTime_GetTime(XTime *t) { static XTime n;*t=n;n+=1000; }
FRESULT f_mount(FATFS *v,const char *s,unsigned n) { (void)s;(void)n;v->csize=64;return FR_OK; }
FRESULT f_mkdir(const char *s) { (void)s;return FR_OK; }
FRESULT f_opendir(DIR *d,const char *s) { (void)d;(void)s;return FR_OK; }
FRESULT f_readdir(DIR *d,FILINFO *i) { (void)d;i->fname[0]=0;return FR_OK; }
FRESULT f_closedir(DIR *d) { (void)d;return FR_OK; }
FRESULT f_open(FIL *f,const char *s,unsigned mode) { (void)s;assert(mode==6);memset(f,0,sizeof(*f));return FR_OK; }
FRESULT f_read(FIL *f,void *p,UINT n,UINT *read) { (void)f;(void)p;(void)n;*read=0;return FR_OK; }
FRESULT f_write(FIL *f,const void *p,UINT n,UINT *written)
{
    ++writes;*written=0;
    if(writes==fail_write) return FR_DISK_ERR;
    assert(!((uintptr_t)p&63u));
    if(writes==short_write) --n;
    assert(f->position+n<=sizeof(disk));memcpy(disk+f->position,p,n);f->position+=n;
    if(f->position>f->size) f->size=f->position;
    *written=n;return FR_OK;
}
FRESULT f_sync(FIL *f) { (void)f;++syncs;return fail_sync?FR_DISK_ERR:FR_OK; }
FRESULT f_close(FIL *f) { (void)f;++closes;return fail_close?FR_DISK_ERR:FR_OK; }
FRESULT f_lseek(FIL *f,FSIZE_t n)
{
    ++seek_calls;if(fail_seek==seek_calls) return FR_DISK_ERR;
    f->position=short_seek && n?n-1:n;
    if(f->position>f->size) f->size=f->position;
    return FR_OK;
}
FRESULT f_rename(const char *a,const char *b) { (void)a;(void)b;return FR_OK; }
FRESULT f_unlink(const char *s) { (void)s;return FR_OK; }
static void reset(void)
{
    writes=syncs=closes=seek_calls=fail_write=short_write=fail_sync=fail_close=short_seek=fail_seek=0;
    memset(disk,0xac,sizeof(disk));assert(capture_fatfs_mount(&fs,"0:")==CAPTURE_OK);
}
int main(void)
{
    void *file;size_t written,pos;unsigned i;
    for(i=0;i<sizeof(data);++i) data[i]=(uint8_t)(i*17u);
    reset();assert(ops->create_exclusive(&fs,"0:/PHOTO.TMP",&file)==0);
    assert(ops->reserve(&fs,file,sizeof(data))==0 && f_tell((FIL *)file)==0 && seek_calls==2);
    assert(capture_fatfs_reset_stats(&fs)==CAPTURE_INVALID);
    /* An unaligned BMP header followed by irregular rows becomes aligned blocks. */
    assert(ops->write(&fs,file,data,54,&written)==0 && written==54);
    for(pos=54;pos<sizeof(data);) {
        size_t n=12168;if(n>sizeof(data)-pos)n=sizeof(data)-pos;
        assert(ops->write(&fs,file,data+pos,n,&written)==0 && written==n);pos+=n;
    }
    assert(writes==2 && fs.buffered==100);
    assert(ops->sync(&fs,file)==0 && writes==3 && syncs==1 && !memcmp(data,disk,sizeof(data)));
    assert(ops->close(&fs,file)==0 && writes==3 && closes==1 && !fs.output_open);
    assert(fs.stats.bytes==sizeof(data) && fs.stats.writes==3 && fs.stats.write_ticks==3000);
    assert(capture_fatfs_reset_stats(&fs)==0 && !fs.stats.bytes);
    for(i=0;i<8;++i) {
        reset();assert(ops->create_exclusive(&fs,"TMP",&file)==0);
        if(i<2) {
            if(i==0) fail_write=1;else short_write=1;
            assert(ops->write(&fs,file,data,CAPTURE_SD_BUFFER_BYTES,&written)<0);
            assert(ops->sync(&fs,file)<0 && writes==1);
            assert(ops->close(&fs,file)<0 && writes==1 && closes==1 && !fs.output_open);
        } else if(i==2) {
            assert(ops->write(&fs,file,data,54,&written)==0);short_write=1;
            assert(ops->sync(&fs,file)==CAPTURE_FS_NO_SPACE);
            assert(ops->close(&fs,file)<0 && writes==1);
        } else if(i<5) {
            if(i==3) short_seek=1;else fail_seek=2;
            assert(ops->reserve(&fs,file,1000)<0);
            assert(ops->write(&fs,file,data,54,&written)<0 && !written && !writes);
            assert(ops->close(&fs,file)<0 && !fs.output_open);
        } else if(i==5) {
            assert(ops->write(&fs,file,data,54,&written)==0);fail_sync=1;
            assert(ops->sync(&fs,file)<0 && writes==1);
            assert(ops->close(&fs,file)<0 && writes==1 && !fs.output_open);
        } else if(i==6) {
            fail_close=1;assert(ops->close(&fs,file)<0 && fs.output_open);
            assert(ops->create_exclusive(&fs,"OTHER",&file)<0);
        } else {
            assert(ops->write(&fs,file,data,54,&written)==0);
            assert(ops->close(&fs,file)==0 && writes==1 && !memcmp(disk,data,54));
        }
    }
    puts("PASS FatFs buffering: aligned blocks, exact tail, preallocation rewind/full/failure, no retries, sync/close errors and locks");
    return 0;
}
