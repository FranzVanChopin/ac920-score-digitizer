#ifndef DP_PIXELS_H
#define DP_PIXELS_H
#include <stddef.h>
#include <stdint.h>
/* DPDMA RGBA8888: memory bytes R,G,B,A, as in AMD's video example.
 * Distinct buffers; neither source nor padding is modified/read as pixels. */
int dp_pixels_from_bgr(const uint8_t *src, unsigned width, unsigned height,
                       size_t source_stride, uint8_t *dst, size_t dest_stride);
#endif
