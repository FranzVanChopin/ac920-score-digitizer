/* SPDX-License-Identifier: MIT */
#include "capture_color.h"
#include <math.h>
#include <string.h>

#define COLOR_MAGIC UINT32_C(0x43434d31)
#define ENCODE_FRACTION_BITS 8u /* Q20 input -> Q12 table position. */

/* Raspberry Pi libcamera vc4 imx477.json, commit
 * 6c1dd9d55573010f710c9e190a73e7e76f0d9432 (BSD-2-Clause tuning data).
 * 6500 K interpolation between its 5881 K and 7600 K CCMs.
 * See COLOR_SOURCE for provenance. This is a starting profile, not a measured
 * calibration of this lens/illuminant. WB is applied only in the sensor. */
const double capture_color_ccm_rgb[9] = {
     2.2286541768, -0.9614188773, -0.2672352996,
    -0.3468200582,  2.1199418325, -0.7731189761,
    -0.1613435660, -0.4447762013,  1.6061161664
};

static int valid_frame(const capture_color *c, const void *p, unsigned w,
                       unsigned h, size_t stride)
{
    if (!c || c->initialized != COLOR_MAGIC || !p || !w || !h ||
        w > CAPTURE_COLOR_MAX_DIMENSION || h > CAPTURE_COLOR_MAX_DIMENSION ||
        (uint64_t)w * h > CAPTURE_COLOR_MAX_PIXELS || stride < (size_t)w * 3u)
        return 0;
    if ((size_t)(h - 1u) > (SIZE_MAX - (size_t)w * 3u) / stride)
        return 0;
    return 1;
}

int capture_color_init(capture_color *c)
{
    int first[256], last[256];
    unsigned i, input, output;
    if (!c) return CAPTURE_COLOR_INVALID;
    memset(c, 0, sizeof(*c));
    for (i = 0; i < 256; ++i) first[i] = last[i] = -1;
    /* Match main.c's PL LUT exactly, including 10-bit rounding then >>2.
     * Do not subtract the black pedestal a second time after this inverse. */
    for (i = 0; i < 1024; ++i) {
        double linear = i > 64u ? (double)(i - 64u) / 959.0 : 0.0;
        unsigned q10 = (unsigned)(1023.0 * pow(linear, 1.0 / 2.2) + 0.5);
        unsigned q8 = q10 >> 2;
        if (first[q8] < 0) first[q8] = (int)i;
        last[q8] = (int)i;
    }
    c->decode[0] = 0.0;
    for (i = 1; i < 256; ++i) {
        if (first[i] >= 0)
            c->decode[i] = ((double)(first[i] + last[i]) * 0.5 - 64.0) / 959.0;
    }
    /* Some low 8-bit codes cannot be produced by the PL quantizer. Give
     * arbitrary/test pixels a monotonic interpolation between valid bins. */
    for (i = 1; i < 255; ++i) {
        if (first[i] < 0) {
            unsigned lo = i - 1u, hi = i + 1u;
            while (lo && first[lo] < 0) --lo;
            while (hi < 255u && first[hi] < 0) ++hi;
            c->decode[i] = c->decode[lo] +
                (c->decode[hi] - c->decode[lo]) * (double)(i - lo) / (double)(hi - lo);
        }
    }
    for (input = 0; input < 3; ++input) {
        for (i = 0; i < 256; ++i) {
            for (output = 0; output < 3; ++output) {
                double v = capture_color_ccm_rgb[3u * output + input] *
                           c->decode[i] * (double)CAPTURE_COLOR_LINEAR_ONE;
                c->contribution[input][i][output] = (int32_t)(v >= 0.0 ? v + 0.5 : v - 0.5);
            }
        }
    }
    for (i = 0; i <= 4096; ++i)
        c->encode_q8[i] = (uint16_t)(65280.0 * pow((double)i / 4096.0, 1.0 / 2.2) + 0.5);
    c->initialized = COLOR_MAGIC;
    return CAPTURE_COLOR_OK;
}

static uint8_t encode(const capture_color *c, int32_t linear)
{
    unsigned index, fraction;
    uint32_t interpolated;
    if (linear <= 0) return 0;
    if (linear >= CAPTURE_COLOR_LINEAR_ONE) return 255;
    index = (unsigned)linear >> ENCODE_FRACTION_BITS;
    fraction = (unsigned)linear & 255u;
    /* Interpolate the Q8 output, then round once to an 8-bit pixel. */
    interpolated = (uint32_t)c->encode_q8[index] * (256u - fraction) +
                   (uint32_t)c->encode_q8[index + 1u] * fraction;
    return (uint8_t)((interpolated + 32768u) >> 16);
}

static unsigned transform(const capture_color *c, const uint8_t in[3], uint8_t out[3])
{
    unsigned channel, high_clip = 0;
    /* Save source values first so in-place conversion is safe. */
    const int32_t *r = c->contribution[0][in[2]];
    const int32_t *g = c->contribution[1][in[1]];
    const int32_t *b = c->contribution[2][in[0]];
    for (channel = 0; channel < 3; ++channel) {
        int32_t linear = r[channel] + g[channel] + b[channel];
        if (linear > CAPTURE_COLOR_LINEAR_ONE) high_clip = 1;
        out[2u - channel] = encode(c, linear);
    }
    return high_clip;
}

void capture_color_apply_bgr(const capture_color *c, const uint8_t input[3], uint8_t output[3])
{
    (void)transform(c, input, output);
}

int capture_color_row_rgb(const capture_color *c, const uint8_t *source,
                          uint8_t *dest, unsigned width)
{
    size_t bytes;
    uintptr_t source_address, dest_address;
    unsigned x;
    if (!c || c->initialized != COLOR_MAGIC || !source || !dest ||
        !width || width > CAPTURE_COLOR_MAX_DIMENSION)
        return CAPTURE_COLOR_INVALID;
    bytes = (size_t)width * 3u;
    source_address = (uintptr_t)source;
    dest_address = (uintptr_t)dest;
    if (source_address > UINTPTR_MAX - bytes || dest_address > UINTPTR_MAX - bytes)
        return CAPTURE_COLOR_INVALID;
    if (source_address < dest_address + bytes && dest_address < source_address + bytes)
        return CAPTURE_COLOR_INVALID;
    for (x = 0; x < width; ++x) {
        uint8_t bgr[3];
        (void)transform(c, source + (size_t)x * 3u, bgr);
        dest[(size_t)x * 3u] = bgr[2];
        dest[(size_t)x * 3u + 1u] = bgr[1];
        dest[(size_t)x * 3u + 2u] = bgr[0];
    }
    return CAPTURE_COLOR_OK;
}

int capture_color_process_bgr(const capture_color *c, uint8_t *pixels,
                             unsigned w, unsigned h, size_t stride,
                             void (*service)(void *), void *context)
{
    unsigned x, y;
    if (!valid_frame(c, pixels, w, h, stride)) return CAPTURE_COLOR_INVALID;
    for (y = 0; y < h; ++y) {
        uint8_t *row = pixels + (size_t)y * stride;
        for (x = 0; x < w; ++x) (void)transform(c, row + (size_t)x * 3u, row + (size_t)x * 3u);
        if (service && ((y & 15u) == 15u || y + 1u == h)) service(context);
    }
    return CAPTURE_COLOR_OK;
}

static unsigned percentile(const unsigned histogram[256], unsigned total, unsigned percent)
{
    unsigned i, accumulated = 0;
    unsigned target = (unsigned)(((uint64_t)total * percent + 99u) / 100u);
    for (i = 0; i < 256; ++i) {
        accumulated += histogram[i];
        if (accumulated >= target) return i;
    }
    return 255;
}

static unsigned permille(unsigned count, unsigned total)
{
    return (unsigned)(((uint64_t)count * 1000u + total / 2u) / total);
}

int capture_color_measure(const capture_color *c, const uint8_t *pixels,
                          unsigned w, unsigned h, size_t stride,
                          unsigned step, capture_color_stats *result)
{
    unsigned histogram[5][256] = {{0}};
    unsigned saturated[3] = {0}, corrected_saturated = 0, high_clip = 0;
    unsigned x, y, channel, samples = 0, start_x, start_y;
    if (!result) return CAPTURE_COLOR_INVALID;
    memset(result, 0, sizeof(*result));
    if (!valid_frame(c, pixels, w, h, stride) || !step || step > CAPTURE_COLOR_MAX_DIMENSION)
        return CAPTURE_COLOR_INVALID;
    start_x = step / 2u < w ? step / 2u : w / 2u;
    start_y = step / 2u < h ? step / 2u : h / 2u;
    for (y = start_y; y < h; y += step) {
        const uint8_t *row = pixels + (size_t)y * stride;
        for (x = start_x; x < w; x += step) {
            const uint8_t *p = row + (size_t)x * 3u;
            uint8_t corrected[3];
            unsigned maximum, luma;
            for (channel = 0; channel < 3; ++channel) {
                unsigned v = p[2u - channel];
                ++histogram[channel][v];
                if (v >= 254u) ++saturated[channel];
            }
            high_clip += transform(c, p, corrected);
            maximum = corrected[0] > corrected[1] ? corrected[0] : corrected[1];
            if (corrected[2] > maximum) maximum = corrected[2];
            luma = (29u * corrected[0] + 150u * corrected[1] + 77u * corrected[2]) >> 8;
            ++histogram[3][luma];
            ++histogram[4][maximum];
            if (maximum >= 254u) ++corrected_saturated;
            ++samples;
        }
    }
    result->samples = samples;
    for (channel = 0; channel < 3; ++channel) {
        result->raw_p99_rgb[channel] = percentile(histogram[channel], samples, 99);
        result->raw_saturated_permille_rgb[channel] = permille(saturated[channel], samples);
    }
    result->corrected_luma_p90 = percentile(histogram[3], samples, 90);
    result->corrected_max_p99 = percentile(histogram[4], samples, 99);
    result->corrected_saturated_permille = permille(corrected_saturated, samples);
    result->corrected_high_clip_permille = permille(high_clip, samples);
    return CAPTURE_COLOR_OK;
}

int capture_color_calibrate_white(const capture_color *c, const uint8_t *pixels,
                                 unsigned w, unsigned h, size_t stride,
                                 unsigned x, unsigned y, unsigned rw, unsigned rh,
                                 const unsigned current[3], capture_color_white_result *result)
{
    double sum[3] = {0.0}, sum_sq[3] = {0.0}, proposed[3];
    unsigned channel, px, py, step, samples = 0, saturated = 0;
    if (!result) return CAPTURE_COLOR_INVALID;
    memset(result, 0, sizeof(*result));
    if (!valid_frame(c, pixels, w, h, stride) || !current || !rw || !rh ||
        x >= w || y >= h || rw > w - x || rh > h - y || current[1] != 256u)
        return CAPTURE_COLOR_INVALID;
    for (channel = 0; channel < 3; ++channel)
        if (current[channel] < CAPTURE_COLOR_SENSOR_GAIN_MIN || current[channel] > CAPTURE_COLOR_SENSOR_GAIN_MAX)
            return CAPTURE_COLOR_INVALID;
    step = ((rw > rh ? rw : rh) + 255u) / 256u;
    for (py = 0; py < rh; py += step) {
        const uint8_t *row = pixels + (size_t)(y + py) * stride;
        for (px = 0; px < rw; px += step) {
            const uint8_t *p = row + (size_t)(x + px) * 3u;
            for (channel = 0; channel < 3; ++channel) {
                unsigned code = p[2u - channel];
                double v = c->decode[code];
                sum[channel] += v;
                sum_sq[channel] += v * v;
                if (code >= 248u) saturated = 1;
            }
            ++samples;
        }
    }
    result->samples = samples;
    if (samples < 64u) return CAPTURE_COLOR_INSUFFICIENT;
    for (channel = 0; channel < 3; ++channel) {
        double mean = sum[channel] / samples;
        double variance = sum_sq[channel] / samples - mean * mean;
        result->linear_mean_rgb[channel] = mean;
        result->coefficient_variation_rgb[channel] = mean > 0.0 ? sqrt(variance > 0.0 ? variance : 0.0) / mean : 0.0;
    }
    if (saturated) return CAPTURE_COLOR_SATURATED;
    for (channel = 0; channel < 3; ++channel)
        if (result->linear_mean_rgb[channel] < 0.02) return CAPTURE_COLOR_DARK;
    if (result->linear_mean_rgb[1] < 0.03) return CAPTURE_COLOR_DARK;
    for (channel = 0; channel < 3; ++channel)
        if (result->coefficient_variation_rgb[channel] > 0.18) return CAPTURE_COLOR_NONUNIFORM;
    proposed[0] = current[0] * result->linear_mean_rgb[1] / result->linear_mean_rgb[0];
    proposed[1] = 256.0;
    proposed[2] = current[2] * result->linear_mean_rgb[1] / result->linear_mean_rgb[2];
    for (channel = 0; channel < 3; ++channel)
        if (!isfinite(proposed[channel]) || proposed[channel] < 256.0 || proposed[channel] > 2048.0)
            return CAPTURE_COLOR_GAIN_RANGE;
    for (channel = 0; channel < 3; ++channel)
        result->proposed_sensor_gain_q8[channel] = (unsigned)(proposed[channel] + 0.5);
    return CAPTURE_COLOR_OK;
}

const char *capture_color_result_string(int result)
{
    switch (result) {
    case CAPTURE_COLOR_OK: return "OK";
    case CAPTURE_COLOR_INVALID: return "invalid frame, ROI, or sensor gains";
    case CAPTURE_COLOR_DARK: return "neutral target too dark";
    case CAPTURE_COLOR_SATURATED: return "neutral target saturated; reduce exposure";
    case CAPTURE_COLOR_NONUNIFORM: return "neutral target not uniform";
    case CAPTURE_COLOR_GAIN_RANGE: return "proposed sensor gain outside 1x..8x";
    case CAPTURE_COLOR_INSUFFICIENT: return "neutral ROI too small";
    default: return "unknown color error";
    }
}
