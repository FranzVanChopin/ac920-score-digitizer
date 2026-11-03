/* SPDX-License-Identifier: MIT */
#ifndef CAPTURE_COLOR_H
#define CAPTURE_COLOR_H

#include <stddef.h>
#include <stdint.h>

#define CAPTURE_COLOR_MAX_DIMENSION 8192u
#define CAPTURE_COLOR_MAX_PIXELS 16000000u
#define CAPTURE_COLOR_SENSOR_GAIN_MIN 256u
#define CAPTURE_COLOR_SENSOR_GAIN_MAX 2048u
#define CAPTURE_COLOR_LINEAR_ONE 1048576

enum {
    CAPTURE_COLOR_OK = 0,
    CAPTURE_COLOR_INVALID = -1,
    CAPTURE_COLOR_DARK = -2,
    CAPTURE_COLOR_SATURATED = -3,
    CAPTURE_COLOR_NONUNIFORM = -4,
    CAPTURE_COLOR_GAIN_RANGE = -5,
    CAPTURE_COLOR_INSUFFICIENT = -6
};

/* RGB row-major: output_rgb = CCM * input_linear_rgb. Sensor R/G/B gains
 * are applied by IMX477; this module deliberately contains NO software WB. */
extern const double capture_color_ccm_rgb[9];

typedef struct {
    /* Approximate inverse of the actual PL quantizer, after black removal. */
    double decode[256];
    /* [input RGB channel][encoded value][output RGB channel], signed Q20. */
    int32_t contribution[3][256][3];
    /* Gamma 2.2 encoding at linear i/4096, encoded value in unsigned Q8. */
    uint16_t encode_q8[4097];
    uint32_t initialized;
} capture_color;

typedef struct {
    unsigned samples;
    unsigned raw_p99_rgb[3];
    /* Near-saturation is encoded input >=254, not a recovered RAW clipping
     * measurement. All ratios are integers 0..1000. */
    unsigned raw_saturated_permille_rgb[3];
    unsigned corrected_luma_p90;
    unsigned corrected_max_p99;
    unsigned corrected_saturated_permille;
    /* Any CCM output >1 before clamping. Negative gamut excursions are not
     * counted here: they should not force AE to reduce exposure. */
    unsigned corrected_high_clip_permille;
} capture_color_stats;

typedef struct {
    unsigned proposed_sensor_gain_q8[3]; /* RGB; G always 256 on success. */
    double linear_mean_rgb[3];
    double coefficient_variation_rgb[3];
    unsigned samples;
} capture_color_white_result;

/* Initialization is the only processing path using pow(). */
int capture_color_init(capture_color *color);
/* Valid initialized color is required; input/output can be the same pixel. */
void capture_color_apply_bgr(const capture_color *color,
                             const uint8_t input[3], uint8_t output[3]);
/* Converts one original sensor BGR row to corrected RGB for the BMP row
 * callback. Both buffers contain at least width*3 bytes and must not overlap.
 * Source bytes are never modified; width must be 1..MAX_DIMENSION. */
int capture_color_row_rgb(const capture_color *color,
                          const uint8_t *source_bgr, uint8_t *dest_rgb,
                          unsigned width);
/* Processes only w*3 bytes per row; padding and sensor-frame dimensions stay
 * unchanged. Service runs after each 16 rows and the final partial group. */
int capture_color_process_bgr(const capture_color *color, uint8_t *pixels,
                             unsigned width, unsigned height, size_t stride,
                             void (*service)(void *), void *context);
/* Call on the ORIGINAL gamma-encoded sensor frame before process_bgr. */
int capture_color_measure(const capture_color *color, const uint8_t *pixels,
                          unsigned width, unsigned height, size_t stride,
                          unsigned sample_step, capture_color_stats *result);
/* Explicit user-selected neutral white/gray target only; never gray-world
 * AWB on arbitrary scenes. ROI is in original frame coordinates. Rejects any
 * sampled channel >=248, low signal, CV>0.18, or proposed gains outside 1..8x.
 * Computes sensor gains only; does not modify color, the image, or hardware.
 * Input frame must precede PS CCM. Sensor G must currently equal unity. */
int capture_color_calibrate_white(const capture_color *color,
                                 const uint8_t *pixels, unsigned width,
                                 unsigned height, size_t stride,
                                 unsigned x, unsigned y, unsigned roi_width,
                                 unsigned roi_height,
                                 const unsigned current_q8_rgb[3],
                                 capture_color_white_result *result);
const char *capture_color_result_string(int result);

#endif
