#ifndef CAPTURE_ONECARD_H
#define CAPTURE_ONECARD_H
#include <stddef.h>
#include <stdint.h>
enum capture_onecard_stage {ONECARD_PREFLIGHT,ONECARD_QUIESCE,ONECARD_CLOSE_STORAGE,
                           ONECARD_SELECT_LINUX,ONECARD_RESET,ONECARD_STAGE_COUNT};
typedef struct {uint32_t linux_bytes,linux_crc32;} capture_onecard_ready;
typedef struct {int (*step[ONECARD_STAGE_COUNT])(void *);void *context;} capture_onecard_ops;
int capture_onecard_parse_ready(const char *,size_t,capture_onecard_ready *);
uint32_t capture_onecard_crc32(uint32_t,const uint8_t *,size_t);
int capture_onecard_boot_header(const uint8_t *,size_t);
/* Stops on the first failed step. Never selects Linux before storage closes. */
int capture_onecard_switch(const capture_onecard_ops *,enum capture_onecard_stage *);
#endif
