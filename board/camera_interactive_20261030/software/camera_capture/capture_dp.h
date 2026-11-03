#ifndef CAPTURE_DP_H
#define CAPTURE_DP_H
#include <stddef.h>
#include <stdint.h>
#define CAPTURE_DP_WIDTH 1280u
#define CAPTURE_DP_HEIGHT 720u
int capture_dp_init(void (*service)(void *),void *context);
/* Copy the completed BGR canvas into a privately owned RGBA framebuffer.
 * No HPD/link is a recoverable display-only condition, not a camera failure. */
int capture_dp_present(const uint8_t *bgr,size_t stride);
void capture_dp_status(void);
/* Drain and disable all DP DMA reads before a whole-system handoff. */
int capture_dp_shutdown(void);
#endif
