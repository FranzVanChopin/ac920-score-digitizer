#ifndef CAPTURE_PAIR_H
#define CAPTURE_PAIR_H
#include "capture_store.h"

/* Both callbacks read the SAME immutable, final color-corrected BGR crop.
 * Gray is BT.601 encoded luma, with no resize, CLAHE or threshold. */
typedef struct {
    const uint8_t *bgr;
    unsigned width, height;
    size_t stride;
    void (*service)(void *);
    void *service_context;
} capture_pair_frame;
int capture_pair_rgb_row(void *context, uint32_t y, uint8_t *output, size_t length);
int capture_pair_gray_row(void *context, uint32_t y, uint8_t *output, size_t length);
#endif
