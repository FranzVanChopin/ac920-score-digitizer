#include "capture_fatfs.h"

#include <limits.h>
#include <string.h>
#include "xtime_l.h"

static uint64_t ticks(void) { XTime t; XTime_GetTime(&t); return (uint64_t)t; }

int capture_fatfs_reset_stats(capture_fatfs *fs)
{
    if(!fs || fs->output_open) return CAPTURE_INVALID;
    memset(&fs->stats,0,sizeof(fs->stats));return CAPTURE_OK;
}

static int map_result(FRESULT result)
{
    switch (result) {
    case FR_OK: return CAPTURE_OK;
    case FR_NO_FILE:
    case FR_NO_PATH: return CAPTURE_FS_NOENT;
    case FR_EXIST: return CAPTURE_FS_EXISTS;
    default: return CAPTURE_FS_IO;
    }
}

int capture_fatfs_mount(capture_fatfs *context, const char *volume)
{
    if (!context || !volume)
        return CAPTURE_INVALID;
    memset(context, 0, sizeof(*context));
    return map_result(f_mount(&context->volume, volume, 1));
}

static int make_directory(void *context, const char *path)
{
    (void)context;
    return map_result(f_mkdir(path));
}

static int list_directory(void *context, const char *path,
                          capture_list_visitor visitor, void *visitor_context)
{
    DIR directory;
    FILINFO info;
    FRESULT result, close_result;
    int visit_result = CAPTURE_OK;
    (void)context;
    result = f_opendir(&directory, path);
    if (result != FR_OK)
        return map_result(result);
    for (;;) {
        result = f_readdir(&directory, &info);
        if (result != FR_OK || info.fname[0] == '\0')
            break;
        visit_result = visitor(visitor_context, info.fname, (info.fattrib & AM_DIR) != 0);
        if (visit_result != CAPTURE_OK)
            break;
    }
    close_result = f_closedir(&directory);
    if (visit_result != CAPTURE_OK)
        return visit_result;
    return map_result(result != FR_OK ? result : close_result);
}

static int read_small(void *context, const char *path, uint8_t *data,
                      size_t capacity, size_t *length)
{
    FIL file;
    FRESULT result, close_result;
    UINT bytes_read = 0;
    (void)context;
    *length = 0;
    result = f_open(&file, path, FA_READ);
    if (result != FR_OK)
        return map_result(result);
    if (f_size(&file) > capacity || capacity > UINT_MAX) {
        f_close(&file);
        return CAPTURE_STATE_CORRUPT;
    }
    result = f_read(&file, data, (UINT)capacity, &bytes_read);
    close_result = f_close(&file);
    *length = bytes_read;
    return map_result(result != FR_OK ? result : close_result);
}

static int create_file(void *context, const char *path, void **file)
{
    capture_fatfs *fs = (capture_fatfs *)context;
    FRESULT result;
    if (fs->output_open)
        return CAPTURE_FS_IO;
    result = f_open(&fs->output, path, FA_WRITE | FA_CREATE_NEW);
    if (result != FR_OK)
        return map_result(result);
    fs->output_open = 1;
    fs->buffered=0; fs->failure=CAPTURE_OK;
    *file = &fs->output;
    return CAPTURE_OK;
}

static int flush_buffer(capture_fatfs *fs)
{
    FRESULT result;
    UINT bytes_written = 0;
    uint64_t start, elapsed;
    unsigned ms;
    if(fs->failure!=CAPTURE_OK) return fs->failure;
    if(!fs->buffered) return CAPTURE_OK;
    start=ticks();
    result=f_write(&fs->output,fs->buffer,(UINT)fs->buffered,&bytes_written);
    elapsed=ticks()-start;
    fs->stats.write_ticks+=elapsed;fs->stats.writes++;fs->stats.bytes+=bytes_written;
    ms=(unsigned)(elapsed/(COUNTS_PER_SECOND/1000u));
    if(ms>fs->stats.max_write_ms) fs->stats.max_write_ms=ms;
    fs->failure=map_result(result);
    if(result==FR_OK && bytes_written!=fs->buffered) fs->failure=CAPTURE_FS_NO_SPACE;
    /* Never retry partial bytes after failure: caller discards this TMP. */
    fs->buffered=0;return fs->failure;
}

static int write_file(void *context, void *file, const uint8_t *data,
                      size_t length, size_t *written)
{
    capture_fatfs *fs=context;
    *written=0;
    if(!fs->output_open || file!=&fs->output || (!data && length)) return CAPTURE_INVALID;
    if(fs->failure!=CAPTURE_OK) return fs->failure;
    fs->stats.logical_writes++;
    while(*written<length) {
        size_t amount=length-*written;
        if(amount>CAPTURE_SD_BUFFER_BYTES-fs->buffered) amount=CAPTURE_SD_BUFFER_BYTES-fs->buffered;
        memcpy(fs->buffer+fs->buffered,data+*written,amount);
        fs->buffered+=(unsigned)amount;*written+=amount;
        if(fs->buffered==CAPTURE_SD_BUFFER_BYTES && flush_buffer(fs)!=CAPTURE_OK) return fs->failure;
    }
    return CAPTURE_OK;
}

static int sync_file(void *context, void *file)
{
    capture_fatfs *fs=context;
    uint64_t start;
    if(!fs->output_open || file!=&fs->output) return CAPTURE_INVALID;
    if(flush_buffer(fs)!=CAPTURE_OK) return fs->failure;
    start=ticks();fs->failure=map_result(f_sync((FIL *)file));
    fs->stats.sync_ticks+=ticks()-start;return fs->failure;
}

static int close_file(void *context, void *file)
{
    capture_fatfs *fs = (capture_fatfs *)context;
    int result, pending;
    uint64_t start;
    if(!fs->output_open || file!=&fs->output) return CAPTURE_INVALID;
    pending=flush_buffer(fs);
    start=ticks();result=map_result(f_close((FIL *)file));
    fs->stats.close_ticks+=ticks()-start;
    /* A failed close can leave a FatFs sharing lock alive. Require an explicit
     * remount instead of overwriting the still-live FIL on the next attempt. */
    if (result == CAPTURE_OK)
        fs->output_open = 0;
    return pending!=CAPTURE_OK?pending:result;
}

static int reserve_file(void *context, void *file, uint32_t bytes)
{
    capture_fatfs *fs=context;
    FRESULT rc;
    uint64_t start;
    if(!fs->output_open || file!=&fs->output || fs->buffered || f_tell(&fs->output)) return CAPTURE_INVALID;
    start=ticks();
    rc=f_lseek(&fs->output,(FSIZE_t)bytes);
    fs->failure=map_result(rc);
    /* FatFs can return FR_OK with a shortened seek when the volume is full. */
    if(rc==FR_OK && f_tell(&fs->output)!=(FSIZE_t)bytes) fs->failure=CAPTURE_FS_NO_SPACE;
    if(fs->failure==CAPTURE_OK) fs->failure=map_result(f_lseek(&fs->output,0));
    fs->stats.reserve_ticks+=ticks()-start;
    return fs->failure;
}

static int rename_file(void *context, const char *from, const char *to)
{
    (void)context;
    /* FatFs f_rename rejects an existing destination and syncs filesystem
     * metadata. Both names are on the same mounted SD volume. */
    return map_result(f_rename(from, to));
}

static int remove_file(void *context, const char *path)
{
    (void)context;
    return map_result(f_unlink(path));
}

const capture_fs_ops capture_fatfs_operations = {
    make_directory, list_directory, read_small, create_file, write_file,
    sync_file, close_file, rename_file, remove_file, reserve_file
};
