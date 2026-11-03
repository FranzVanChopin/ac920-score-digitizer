#ifndef CAPTURE_FATFS_H
#define CAPTURE_FATFS_H

#include "capture_store.h"
#include "ff.h"

#define CAPTURE_SD_BUFFER_BYTES (256u*1024u)
typedef struct {
    uint64_t bytes, write_ticks, reserve_ticks, sync_ticks, close_ticks;
    uint32_t logical_writes, writes, max_write_ms;
} capture_sd_stats;

typedef struct {
    FATFS volume;
    FIL output;
    unsigned output_open;
    unsigned buffered;
    int failure;
    capture_sd_stats stats;
    uint8_t buffer[CAPTURE_SD_BUFFER_BYTES] __attribute__((aligned(64)));
} capture_fatfs;

/* volume is the logical SD drive assigned by the generated xilffs BSP, e.g.
 * "0:". The adapter never formats the card. Mount before capture_store_init. */
int capture_fatfs_mount(capture_fatfs *context, const char *volume);
int capture_fatfs_reset_stats(capture_fatfs *context);
extern const capture_fs_ops capture_fatfs_operations;

#endif
