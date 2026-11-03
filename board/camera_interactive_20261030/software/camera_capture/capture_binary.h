/* SPDX-License-Identifier: MIT */
#ifndef CAPTURE_BINARY_H
#define CAPTURE_BINARY_H

#include "capture_color.h"

/* Caller-owned, 8-byte-aligned workspace. No allocation or full-size RGB copy.
 * Includes three <=16 MP grayscale planes and all detector/filter scratch. */
#define CAPTURE_BINARY_WORKSPACE_BYTES (52u * 1024u * 1024u)
#define CAPTURE_PAPER_WORKSPACE_BYTES (3u * 1024u * 1024u)
#define CAPTURE_BINARY_MAX_DIMENSION 8192u
#define CAPTURE_BINARY_MAX_PIXELS 16000000u

enum {
    CAPTURE_BINARY_OK = 0,
    CAPTURE_BINARY_INVALID = -1,
    CAPTURE_BINARY_WORKSPACE = -2,
    CAPTURE_BINARY_NOT_FOUND = -3,
    CAPTURE_BINARY_AMBIGUOUS = -4,
    CAPTURE_BINARY_OUTPUT_BOUNDS = -5
};
enum { CAPTURE_BINARY_AUTO = 0, CAPTURE_BINARY_ROI = 1, CAPTURE_BINARY_FULL = 2,
       CAPTURE_BINARY_A4 = 3 };
enum { CAPTURE_BINARY_NONE = 0, CAPTURE_BINARY_AREA = 1, CAPTURE_BINARY_CUBIC = 2 };

typedef struct {
    unsigned target_width;             /* default 1920 */
    double clahe_clip_limit;            /* default 2.0, finite and >0 */
    unsigned clahe_tiles_x, clahe_tiles_y; /* default 8,8; range 1..16 */
    unsigned adaptive_block_size;      /* default 51; odd, 3..255 */
    int adaptive_c;                    /* default 10; range -255..255 */
    unsigned mode;                     /* fixed sensor-long-edge A4 unless overridden */
    unsigned roi_xywh[4];              /* source coordinates, only for ROI */
    unsigned whiten_outside_paper;     /* default 1; AUTO only */
} capture_binary_config;

typedef struct {
    unsigned width, height;
    size_t stride;                     /* contiguous 1-byte rows: == width */
    const uint8_t *gray_resized, *clahe, *binary;
    unsigned source_width, source_height;
    unsigned crop_xywh[4];             /* crop from ORIGINAL pixels */
    double paper_quad_xy[8];           /* pixel centers, TL then clockwise */
    unsigned has_paper_quad;
    unsigned detection_width, detection_height, candidates;
    double area_fraction, rectangularity, edge_contrast[4];
    unsigned crop_padding_px, whiten_margin_px;
    uint64_t whitened_pixels;
    double scale_xy[2];
    unsigned interpolation;            /* enum above; no perspective warp */
} capture_binary_result;

void capture_binary_default_config(capture_binary_config *config);
/* Content-independent A4 crop with the paper LONG edge along the sensor X axis.
 * Keep all source columns; short edge=round(width*210/297), centered vertically.
 * At4056x3040 => (0,86,4056,2868), at2028x1520 => (0,43,2028,1434).
 * These modes have exactly matching2:1 crop coordinates. Pixel rounding causes
 * a <0.5px short-edge aspect approximation; no long-edge resampling/trimming.
 * Reject an input too short to fit, rather than silently trimming the long edge.
 * Returns0 on success; clears xywh on failure. No detection or perspective warp. */
int capture_a4_crop(unsigned width,unsigned height,unsigned xywh[4]);
/* Detection only: no resize/CLAHE/threshold and no image planes are produced.
 * Coordinates refer to this exact immutable input frame. Suitable for preview
 * localization and full-resolution color crops. Same detector as binary_run.
 * Result is cleared on entry; never use its ROI after a nonzero return. */
int capture_paper_detect(const capture_color *color,
                         const uint8_t *source_bgr, unsigned width, unsigned height,
                         size_t source_stride, void *workspace, size_t workspace_bytes,
                         capture_binary_result *result,
                         void (*service)(void *), void *context);
/* Returns zero only on success. Default A4 always uses the fixed crop, including
 * blank/dark/clipped/no-paper scenes. AUTO is an optional explicitly selected
 * library mode that abstains on weak/clipped/ambiguous paper;
 * it NEVER silently substitutes the whole frame. Source is BGR BEFORE PS CCM,
 * already sensor-WB balanced. Source/workspace must not overlap. Color, source,
 * and config remain unchanged. Result planes live in workspace until next run.
 * Service is called during all long stages (also after each <=16 output rows).
 * Native-size crop, conservative polygon whitening, area reduction / cubic
 * enlargement, CLAHE, and replicated-border adaptive mean are applied in order.
 * Parameters match the Python pipeline; integer rounding/detection are not a
 * promise of byte-for-byte OpenCV identity. No perspective/lens correction. */
int capture_binary_run(const capture_color *color,
                       const uint8_t *source_bgr, unsigned width, unsigned height,
                       size_t source_stride, const capture_binary_config *config,
                       void *workspace, size_t workspace_bytes,
                       capture_binary_result *result,
                       void (*service)(void *), void *context);
const char *capture_binary_result_string(int result);

#endif
