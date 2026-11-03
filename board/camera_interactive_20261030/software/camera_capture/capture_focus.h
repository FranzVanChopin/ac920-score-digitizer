/* SPDX-License-Identifier: MIT */
#ifndef CAPTURE_FOCUS_H
#define CAPTURE_FOCUS_H
#include <stdint.h>
#include <stddef.h>

#define CAPTURE_FOCUS_SIZE 500u
#define CAPTURE_FOCUS_LOW_TEXTURE 1u
#define CAPTURE_FOCUS_CLIPPED 2u
#define CAPTURE_FOCUS_MIN_EDGES 10u

typedef struct { unsigned x,y,width,height; } capture_focus_rect;
typedef struct {
    capture_focus_rect roi;
    unsigned p5,p95,clip_permille,edge_pixels,flags;
    uint32_t tenengrad_x10000;
    unsigned edge_h_milli,edge_v_milli,edge_h_count,edge_v_count;
    unsigned edges_computed;
} capture_focus_result;
typedef struct {
    uint8_t gray[CAPTURE_FOCUS_SIZE*CAPTURE_FOCUS_SIZE];
    uint16_t profile[CAPTURE_FOCUS_SIZE]; /* exact [1,2,1]/4 in Q2 */
    uint32_t edge_hist[16001]; /* width quantized to .001 px, bounded median */
} capture_focus_workspace;
typedef struct {
    uint32_t values[3], current, peak;
    uint64_t total;
    unsigned count,index,valid;
} capture_focus_history;

/* 0=C,1=TL,2=TR,3=BL,4=BR. Centers match floor(w/6),floor(5*w/6).
 * Requested box is centered and clamped inside frame; no source reads. */
int capture_focus_region(unsigned width,unsigned height,unsigned region,
                          unsigned box_width,unsigned box_height,capture_focus_rect *roi);
/* Complete immutable BGR frame BEFORE PS CCM. Measure G on a fixed500x500 ROI
 * independent of display zoom. No allocation, no sensor or filesystem access.
 * T=mean(Sobel Gx^2+Gy^2)/(P95-P5)^2, scaled10000. A valid score requires
 * contrast>=16, >=64 pixels with Sobel magnitude>=32, and <5% G>=254.
 * Edges use script-compatible Q2 binomial smoothing, linear percentiles3/90,
 * 7px profile step and <16px falling edges. This measures the processed input,
 * not an absolute optical PSF; do not compare with post-CCM BMP unqualified.
 * Returns0 on a valid call even if LOW_TEXTURE/CLIPPED; flags govern the score. */
int capture_focus_measure(const uint8_t *bgr,unsigned width,unsigned height,size_t stride,
                           unsigned region,int edges,capture_focus_workspace *workspace,
                           capture_focus_result *result,void (*service)(void *),void *context);
void capture_focus_history_reset(capture_focus_history *history);
void capture_focus_history_push(capture_focus_history *history,const capture_focus_result *result);
const char *capture_focus_region_name(unsigned region);
/* Advisory only: bit0 top/bottom, bit1 left/right. Require valid metrics and
 * >=10 edges in every corner in the tested orientation. Content may differ. */
unsigned capture_focus_uneven(const capture_focus_result five[5]);
#endif
