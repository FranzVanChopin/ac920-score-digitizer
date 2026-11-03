/* SPDX-License-Identifier: GPL-2.0-only */
#ifndef CAMERA_PREVIEW_H
#define CAMERA_PREVIEW_H
#include <stddef.h>
#include <stdint.h>
/* All buffers contain B,G,R bytes. Source is a complete, immutable frame.
 * Destination must be a non-displayed buffer until the caller flips at VSYNC. */
int preview_letterbox(const uint8_t *source, unsigned sw, unsigned sh,
                      size_t source_stride, uint8_t *dest, unsigned dw,
                      unsigned dh, size_t dest_stride,
                      void (*service)(void *), void *context);
void preview_text(uint8_t *dest, unsigned width, unsigned height, size_t stride,
                  unsigned x, unsigned y, const char *text, unsigned scale,
                  uint32_t rgb);
/* Approximate 90th-percentile brightness, bounded sampling work. */
unsigned preview_brightness(const uint8_t *source, unsigned width,
                             unsigned height, size_t stride);
#endif
