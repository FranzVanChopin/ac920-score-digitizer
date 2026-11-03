/* SPDX-License-Identifier: GPL-2.0-only */
/* Portable Raspberry Pi HQ Camera (IMX477) control. See IMX477_SOURCE.txt. */
#ifndef CAMERA_CAPTURE_IMX477_H
#define CAMERA_CAPTURE_IMX477_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMX477_I2C_ADDRESS_7BIT       0x1au
#define IMX477_CHIP_ID               0x0477u
#define IMX477_XCLK_HZ               24000000u
#define IMX477_LINK_FREQUENCY_HZ     450000000u
#define IMX477_LANE_RATE_MBPS        900u
#define IMX477_NUM_LANES             2u
#define IMX477_CSI2_RAW12_DT          0x2cu
#define IMX477_POWER_UP_DELAY_US      500000u
#define IMX477_EXPOSURE_MIN_LINES     4u
#define IMX477_EXPOSURE_MARGIN_LINES  22u
#define IMX477_ANALOGUE_GAIN_MAX      978u
#define IMX477_DIGITAL_GAIN_UNITY     0x0100u
#define IMX477_COLOUR_GAIN_MAX        0x0800u
#define IMX477_STILL_EXPOSURE_MAX_LINES 35000u /* 1 s at 24000/840MHz. */

enum imx477_result {
    IMX477_OK = 0,
    IMX477_EINVAL = -1,
    IMX477_EIO = -2,
    IMX477_ECHIP_ID = -3,
    IMX477_ESTATE = -4,
    IMX477_EVERIFY = -5
};

enum imx477_mode_id {
    IMX477_MODE_4056X3040_RAW12_10FPS = 0,
    IMX477_MODE_2028X1520_RAW12_30FPS = 1
};

struct imx477_io {
    void *ctx;
    /* All callbacks return zero on success. The platform supplies slave 0x1a.
     * Register addresses are sent MSB first. Reads must use a repeated START.
     * The callbacks must have a bounded timeout; the driver cannot abort them.
     */
    int (*write16addr8)(void *ctx, uint16_t reg, uint8_t value);
    int (*read16addr8)(void *ctx, uint16_t reg, uint8_t *value);
    void (*delay_us)(void *ctx, uint32_t usec);
    /* Optional: send reg_hi, reg_lo, value_hi, value_lo in ONE transaction.
     * Required for changing multi-byte controls during streaming. Without it,
     * the byte-write fallback is only used while the sensor is in standby.
     */
    int (*write16addr16)(void *ctx, uint16_t reg, uint16_t value);
};

struct imx477_mode_info {
    uint16_t width;
    uint16_t height;
    uint16_t line_length_clocks;
    uint16_t frame_length_lines;
    uint32_t timing_pixel_rate_hz;
    /* FPS = timing_pixel_rate_hz / (line_length_clocks*frame_length_lines).
     * Both modes output RGGB Bayer, RAW12, VC0, two 900-Mbps data lanes.
     * The timing pixel clock is NOT the CSI payload pixel throughput.
     */
};

struct imx477 {
    struct imx477_io io;
    const struct imx477_mode_info *mode;
    uint16_t frame_length_lines; /* Actual programmed period, not mode default. */
    uint16_t exposure_lines;
    uint16_t analogue_gain;
    uint16_t digital_gain;
    /* Effective channel gains; valid only while configured. */
    uint16_t red_gain_q8;
    uint16_t green_gain_q8;
    uint16_t blue_gain_q8;
    uint16_t last_failed_register;
    int last_bus_result;
    int last_error;
    uint8_t last_expected_value;
    uint8_t last_read_value;
    uint8_t initialized;
    uint8_t configured;
    uint8_t streaming;
    uint8_t test_pattern;
    uint8_t colour_gains_enabled;
};

const struct imx477_mode_info *imx477_get_mode_info(enum imx477_mode_id mode);

/* Platform: assert the HQ module ENABLE high and provide 3.3 V BEFORE init.
 * The HQ module supplies its own 24-MHz oscillator. init waits 500 ms for its
 * reset supervisor, reads chip ID (bounded retries), then enters standby.
 * Call init again after any power cycle. No physical pins are touched here.
 */
int imx477_init(struct imx477 *dev, const struct imx477_io *io);

/* Stops any active stream, programs common/mode tables and explicit RAW12,
 * timing, gain and orientation. Leaves STANDBY. Start the CSI receiver and
 * frame buffers first, then call imx477_start(). Configure resets exposure to
 * 10 ms, gains to unity, test pattern off. It does not perform AE/AWB.
 */
int imx477_configure(struct imx477 *dev, enum imx477_mode_id mode);

/* Switch a currently configured device, stopping an active stream first.
 * Preserve exposure TIME (nearest line, halves rounded up), analogue gain,
 * individual RGB or global digital gain, and test pattern. Clamp exposure to
 * the new mode's 4..frame_length-22 range. Success leaves STANDBY: the caller
 * must configure/drain the CSI/ISP/DMA path before imx477_start().
 * dev->exposure_lines reports the applied value; rounding/clamping is lossy.
 * To return to an exact previous setup, the caller must save and explicitly
 * restore that setup, rather than round-trip the clamped/rounded value.
 * Any programming error invalidates configuration and prevents start.
 */
int imx477_reconfigure_preserving_controls(struct imx477 *dev,
                                           enum imx477_mode_id mode);
int imx477_start(struct imx477 *dev);
int imx477_stop(struct imx477 *dev);
uint32_t imx477_frame_period_us(const struct imx477 *dev);

/* Native full-resolution STANDBY only. Set analogue gain=1x and extend the
 * frame length as needed for 4..35000 exposure lines (up to1s), with22-line
 * margin. Verify all six written bytes. Keep RGB white-balance gains intact.
 * No long-exposure shift, PLL, link, dimensions or Bayer-order change.
 * Error invalidates configuration; caller must not start an unverified image.
 * Reconfigure restores the selected mode's normal frame length. */
int imx477_set_still_low_gain(struct imx477 *dev, uint16_t lines);

/* Exposure is in line periods; bounds: 4 .. frame_length_lines-22.
 * analogue gain is Sony register code 0..978; gain = 1024/(1024-code).
 * digital gain is Q8 (0x0100 = unity), range 0x0100..0xffff.
 * Live multi-byte updates require io.write16addr16. These calls do not extend
 * the frame period. An I/O failure invalidates configuration: reconfigure.
 */
int imx477_set_exposure_lines(struct imx477 *dev, uint16_t lines);
int imx477_set_analogue_gain(struct imx477 *dev, uint16_t code);
int imx477_set_digital_gain(struct imx477 *dev, uint16_t gain_q8);

/* Sensor-side white balance, Q8 gains in the deliberately restricted 1..8x
 * range. Requires configured STANDBY: stop, set, start. Each written byte is
 * read back, including global/individual mode selection. Any I/O or readback
 * failure invalidates configuration and prevents start until reconfigured.
 * The legacy global digital-gain setter is rejected in individual mode.
 * Register evidence and remaining hardware-validation limits are recorded in
 * IMX477_COLOR_REGISTERS.txt. No group-hold or unverified 0x0214 write is used.
 */
int imx477_set_colour_gains(struct imx477 *dev, uint16_t red_q8,
                          uint16_t green_q8, uint16_t blue_q8);

/* Pattern: 0 off, 1 solid red, 2 colour bars, 3 grey, 4 PN9. */
int imx477_set_test_pattern(struct imx477 *dev, uint8_t pattern);

#ifdef __cplusplus
}
#endif
#endif
