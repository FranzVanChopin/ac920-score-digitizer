#ifndef CAPTURE_DENOISE_H
#define CAPTURE_DENOISE_H
#include <stddef.h>
#include <stdint.h>
#define CAPTURE_DENOISE_MAX_WIDTH 4096u
typedef struct {
    int16_t rows[9][CAPTURE_DENOISE_MAX_WIDTH][2];
    int32_t columns[CAPTURE_DENOISE_MAX_WIDTH][4];
} capture_denoise_workspace;
/* In-place 9x9 local-variance chroma filter after CCM, before BMP encoding.
 * Y=(77R+150G+29B+128)/256 is not spatially filtered. sigma^2=25 codes^2.
 * Strong chroma edges reduce filtering; output clipping/rounding can change Y.
 * Original rows are retained in a ring, never read back from filtered output.
 * Do NOT use on focus metrics or on a DMA-owned frame. Returns 0 or -1. */
int capture_denoise_bgr(uint8_t *pixels,unsigned width,unsigned height,size_t stride,
    capture_denoise_workspace *work,void (*service)(void *),void *context);
#endif
