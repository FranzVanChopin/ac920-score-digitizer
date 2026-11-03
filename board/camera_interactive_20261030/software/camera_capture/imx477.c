/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * Portable control based on Raspberry Pi's Linux IMX477 driver.
 * Copyright (C) 2020 Raspberry Pi (Trading) Ltd (register programming).
 * See IMX477_SOURCE.txt for pinned source, timing derivation and license.
 */
#include "imx477.h"
#include <stddef.h>
#include <string.h>

struct imx477_reg { uint16_t address; uint8_t value; };
#include "imx477_registers.h"

#define ARRAY_LENGTH(a) (sizeof(a) / sizeof((a)[0]))
#define REG_MODE_SELECT      0x0100u
#define REG_ORIENTATION      0x0101u
#define REG_EXPOSURE         0x0202u
#define REG_ANALOGUE_GAIN    0x0204u
#define REG_DIGITAL_GAIN     0x020eu
#define REG_RED_GAIN         0x0210u
#define REG_BLUE_GAIN        0x0212u
#define REG_USE_GLOBAL_GAIN  0x3ff9u
#define REG_FRAME_LENGTH     0x0340u
#define REG_LINE_LENGTH      0x0342u
#define REG_TEST_PATTERN     0x0600u

static const struct imx477_mode_info mode_info[] = {
    {4056u, 3040u, 24000u, 3500u, 840000000u},
    {2028u, 1520u, 12000u, 2334u, 840000000u}
};

const struct imx477_mode_info *imx477_get_mode_info(enum imx477_mode_id mode)
{
    if (mode != IMX477_MODE_4056X3040_RAW12_10FPS &&
        mode != IMX477_MODE_2028X1520_RAW12_30FPS)
        return NULL;
    return &mode_info[(unsigned int)mode];
}

static int write8(struct imx477 *dev, uint16_t reg, uint8_t value)
{
    int result = dev->io.write16addr8(dev->io.ctx, reg, value);
    if (result != 0) {
        dev->last_failed_register = reg;
        dev->last_bus_result = result;
        dev->last_error = IMX477_EIO;
        return IMX477_EIO;
    }
    return IMX477_OK;
}

static int read8(struct imx477 *dev, uint16_t reg, uint8_t *value)
{
    int result = dev->io.read16addr8(dev->io.ctx, reg, value);
    if (result != 0) {
        dev->last_failed_register = reg;
        dev->last_bus_result = result;
        dev->last_error = IMX477_EIO;
        return IMX477_EIO;
    }
    return IMX477_OK;
}

static int write16(struct imx477 *dev, uint16_t reg, uint16_t value)
{
    int result;
    if (dev->io.write16addr16 != NULL) {
        result = dev->io.write16addr16(dev->io.ctx, reg, value);
        if (result != 0) {
            dev->last_failed_register = reg;
            dev->last_bus_result = result;
            dev->last_error = IMX477_EIO;
            return IMX477_EIO;
        }
        return IMX477_OK;
    }
    if (dev->streaming)
        return IMX477_ESTATE;
    result = write8(dev, reg, (uint8_t)(value >> 8));
    if (result != IMX477_OK)
        return result;
    return write8(dev, (uint16_t)(reg + 1u), (uint8_t)value);
}

static int write_table(struct imx477 *dev, const struct imx477_reg *table,
                       size_t count)
{
    size_t i;
    int result;
    for (i = 0; i < count; ++i) {
        result = write8(dev, table[i].address, table[i].value);
        if (result != IMX477_OK)
            return result;
    }
    return IMX477_OK;
}

/* Preserve the first failed address/status even if the standby attempt fails.
 * A bus failure cannot guarantee that hardware stopped; configured=0 prevents
 * subsequent start and requires a complete reconfiguration before reuse.
 */
static int invalidate(struct imx477 *dev, int result)
{
    int stopped = dev->io.write16addr8(dev->io.ctx, REG_MODE_SELECT, 0u);
    if (stopped == 0)
        dev->streaming = 0u;
    dev->configured = 0u;
    return result;
}

uint32_t imx477_frame_period_us(const struct imx477 *dev)
{
    uint64_t clocks;
    if (dev == NULL || dev->mode == NULL || !dev->frame_length_lines)
        return 0u;
    clocks = (uint64_t)dev->mode->line_length_clocks *
             dev->frame_length_lines;
    return (uint32_t)((clocks * 1000000u +
                      dev->mode->timing_pixel_rate_hz - 1u) /
                     dev->mode->timing_pixel_rate_hz);
}

static uint32_t frame_wait_us(const struct imx477 *dev)
{
    uint32_t period=imx477_frame_period_us(dev);
    return (period?period:100000u)+1000u;
}

int imx477_init(struct imx477 *dev, const struct imx477_io *io)
{
    struct imx477_io copy;
    uint8_t high = 0u, low = 0u;
    unsigned int attempt;
    int result = IMX477_EIO;
    if (dev == NULL || io == NULL || io->write16addr8 == NULL ||
        io->read16addr8 == NULL || io->delay_us == NULL)
        return IMX477_EINVAL;
    copy = *io; /* Also permits imx477_init(&dev, &dev.io). */
    memset(dev, 0, sizeof(*dev));
    dev->io = copy;
    dev->io.delay_us(dev->io.ctx, IMX477_POWER_UP_DELAY_US);
    for (attempt = 0; attempt < 5u; ++attempt) {
        result = read8(dev, 0x0016u, &high);
        if (result == IMX477_OK)
            result = read8(dev, 0x0017u, &low);
        if (result == IMX477_OK)
            break;
        if (attempt < 4u)
            dev->io.delay_us(dev->io.ctx, 10000u);
    }
    if (result != IMX477_OK)
        return result;
    if ((((uint16_t)high << 8) | low) != IMX477_CHIP_ID)
        return IMX477_ECHIP_ID;
    result = write8(dev, REG_MODE_SELECT, 0u);
    if (result != IMX477_OK)
        return result;
    /* Handles re-initialization when the sensor was already streaming. */
    dev->io.delay_us(dev->io.ctx, 101000u);
    dev->initialized = 1u;
    dev->last_failed_register = 0u;
    dev->last_bus_result = 0;
    dev->last_error = IMX477_OK;
    return IMX477_OK;
}

int imx477_stop(struct imx477 *dev)
{
    int result,need_drain;
    if (dev == NULL || !dev->initialized)
        return IMX477_ESTATE;
    need_drain=dev->streaming || !dev->configured;
    result = write8(dev, REG_MODE_SELECT, 0u);
    if (result != IMX477_OK) {
        dev->configured = 0u;
        return result;
    }
    dev->streaming = 0u;
    /* Keep receiver enabled until the final in-flight frame has drained. */
    if(need_drain) dev->io.delay_us(dev->io.ctx, frame_wait_us(dev));
    return IMX477_OK;
}

int imx477_configure(struct imx477 *dev, enum imx477_mode_id mode)
{
    const struct imx477_mode_info *info = imx477_get_mode_info(mode);
    const struct imx477_reg *table;
    size_t count;
    int result;
    /* These values are the standalone, RAW12, no-flip defaults applied by
     * imx477_start_streaming() and controls in the pinned upstream source.
     * PLL multiplier = 450MHz*2*2*2/24MHz = 150 (0x0096).
     * Requested aggregate link bit rate = 1800 Mbps in unsigned 16.16.
     */
    static const struct imx477_reg configuration[] = {
        {0x030eu, 0x00u}, {0x030fu, 0x96u},
        {0x0820u, 0x07u}, {0x0821u, 0x08u},
        {0x0822u, 0x00u}, {0x0823u, 0x00u},
        {0x0c1au, 0x02u}, {0x0c18u, 0x00u}, {0x0c19u, 0x01u},
        {0x0c12u, 0x01u}, {0x0c14u, 0x00u}, {0x0c15u, 0x00u},
        {0x0c16u, 0x00u}, {0x0c1bu, 0x00u},
        {0x0112u, 0x0cu}, {0x0113u, 0x0cu}, {0x0309u, 0x0cu},
        {0x3f0du, 0x01u}, {0x0b05u, 0x01u}, {0x0b06u, 0x01u},
        {REG_ORIENTATION, 0x00u}, {0x3100u, 0x00u},
        {0x0600u, 0x00u}, {0x0601u, 0x00u},
        {0x0602u, 0x0fu}, {0x0603u, 0xffu},
        {0x0604u, 0x00u}, {0x0605u, 0x00u},
        {0x0606u, 0x00u}, {0x0607u, 0x00u},
        {0x0608u, 0x00u}, {0x0609u, 0x00u},
        {0x4b81u, 0x00u}, {0x3041u, 0x01u},
        {0x3040u, 0x00u}, {0x3f0bu, 0x00u}
    };
    if (dev == NULL || !dev->initialized)
        return IMX477_ESTATE;
    if (info == NULL)
        return IMX477_EINVAL;
    result = imx477_stop(dev);
    if (result != IMX477_OK)
        return result;
    dev->configured = 0u;
    result = write_table(dev, imx477_common_regs,
                         ARRAY_LENGTH(imx477_common_regs));
    if (result != IMX477_OK)
        return invalidate(dev, result);
    if (mode == IMX477_MODE_4056X3040_RAW12_10FPS) {
        table = imx477_4056x3040_regs;
        count = ARRAY_LENGTH(imx477_4056x3040_regs);
    } else {
        table = imx477_2028x1520_regs;
        count = ARRAY_LENGTH(imx477_2028x1520_regs);
    }
    result = write_table(dev, table, count);
    if (result != IMX477_OK)
        return invalidate(dev, result);
    result = write_table(dev, configuration, ARRAY_LENGTH(configuration));
    if (result != IMX477_OK)
        return invalidate(dev, result);
    result = write16(dev, REG_LINE_LENGTH, info->line_length_clocks);
    if (result == IMX477_OK)
        result = write16(dev, REG_FRAME_LENGTH, info->frame_length_lines);
    dev->exposure_lines = (uint16_t)(info->timing_pixel_rate_hz /
                                    (100u * info->line_length_clocks));
    if (result == IMX477_OK)
        result = write16(dev, REG_EXPOSURE, dev->exposure_lines);
    if (result == IMX477_OK)
        result = write16(dev, REG_ANALOGUE_GAIN, 0u);
    if (result == IMX477_OK)
        result = write16(dev, REG_DIGITAL_GAIN, IMX477_DIGITAL_GAIN_UNITY);
    if (result != IMX477_OK)
        return invalidate(dev, result);
    dev->mode = info;
    dev->frame_length_lines = info->frame_length_lines;
    dev->analogue_gain = 0u;
    dev->digital_gain = IMX477_DIGITAL_GAIN_UNITY;
    dev->red_gain_q8 = IMX477_DIGITAL_GAIN_UNITY;
    dev->green_gain_q8 = IMX477_DIGITAL_GAIN_UNITY;
    dev->blue_gain_q8 = IMX477_DIGITAL_GAIN_UNITY;
    dev->colour_gains_enabled = 0u;
    dev->test_pattern = 0u;
    dev->configured = 1u;
    dev->last_failed_register = 0u;
    dev->last_bus_result = 0;
    dev->last_error = IMX477_OK;
    dev->last_expected_value = 0u;
    dev->last_read_value = 0u;
    return IMX477_OK;
}

int imx477_start(struct imx477 *dev)
{
    int result;
    if (dev == NULL || !dev->initialized || !dev->configured)
        return IMX477_ESTATE;
    if (dev->streaming)
        return IMX477_OK;
    result = write8(dev, REG_MODE_SELECT, 1u);
    if (result != IMX477_OK)
        return invalidate(dev, result);
    dev->streaming = 1u;
    return IMX477_OK;
}

static int set_control16(struct imx477 *dev, uint16_t reg, uint16_t value)
{
    int result;
    if (dev == NULL || !dev->initialized || !dev->configured)
        return IMX477_ESTATE;
    result = write16(dev, reg, value);
    if (result == IMX477_EIO)
        return invalidate(dev, result);
    return result;
}

int imx477_set_exposure_lines(struct imx477 *dev, uint16_t lines)
{
    int result;
    if (dev == NULL || !dev->configured || dev->mode == NULL)
        return IMX477_ESTATE;
    if (lines < IMX477_EXPOSURE_MIN_LINES ||
        lines > dev->frame_length_lines - IMX477_EXPOSURE_MARGIN_LINES)
        return IMX477_EINVAL;
    result = set_control16(dev, REG_EXPOSURE, lines);
    if (result == IMX477_OK)
        dev->exposure_lines = lines;
    return result;
}

int imx477_set_analogue_gain(struct imx477 *dev, uint16_t code)
{
    int result;
    if (code > IMX477_ANALOGUE_GAIN_MAX)
        return IMX477_EINVAL;
    result = set_control16(dev, REG_ANALOGUE_GAIN, code);
    if (result == IMX477_OK)
        dev->analogue_gain = code;
    return result;
}

int imx477_set_digital_gain(struct imx477 *dev, uint16_t gain_q8)
{
    int result;
    if (dev != NULL && dev->colour_gains_enabled)
        return IMX477_ESTATE;
    if (gain_q8 < IMX477_DIGITAL_GAIN_UNITY)
        return IMX477_EINVAL;
    result = set_control16(dev, REG_DIGITAL_GAIN, gain_q8);
    if (result == IMX477_OK) {
        dev->digital_gain = gain_q8;
        dev->red_gain_q8 = gain_q8;
        dev->green_gain_q8 = gain_q8;
        dev->blue_gain_q8 = gain_q8;
    }
    return result;
}

static int write_verify8(struct imx477 *dev, uint16_t reg, uint8_t value)
{
    uint8_t actual = 0u;
    int result;
    dev->last_expected_value = value;
    dev->last_read_value = 0u;
    result = write8(dev, reg, value);
    if (result == IMX477_OK)
        result = read8(dev, reg, &actual);
    if (result != IMX477_OK)
        return result;
    dev->last_read_value = actual;
    if (actual != value) {
        dev->last_failed_register = reg;
        dev->last_bus_result = 0;
        dev->last_error = IMX477_EVERIFY;
        return IMX477_EVERIFY;
    }
    return IMX477_OK;
}

int imx477_set_still_low_gain(struct imx477 *dev, uint16_t lines)
{
    uint16_t period;
    unsigned i;
    int result;
    struct imx477_reg updates[6];
    if(!dev || !dev->initialized || !dev->configured || dev->streaming || dev->mode!=&mode_info[0])
        return IMX477_ESTATE;
    if(lines<IMX477_EXPOSURE_MIN_LINES || lines>IMX477_STILL_EXPOSURE_MAX_LINES)
        return IMX477_EINVAL;
    period=(uint16_t)(lines+IMX477_EXPOSURE_MARGIN_LINES);
    if(period<dev->mode->frame_length_lines) period=dev->mode->frame_length_lines;
    updates[0]=(struct imx477_reg){REG_FRAME_LENGTH,(uint8_t)(period>>8)};
    updates[1]=(struct imx477_reg){REG_FRAME_LENGTH+1u,(uint8_t)period};
    updates[2]=(struct imx477_reg){REG_EXPOSURE,(uint8_t)(lines>>8)};
    updates[3]=(struct imx477_reg){REG_EXPOSURE+1u,(uint8_t)lines};
    updates[4]=(struct imx477_reg){REG_ANALOGUE_GAIN,0u};
    updates[5]=(struct imx477_reg){REG_ANALOGUE_GAIN+1u,0u};
    for(i=0;i<6u;++i) {
        result=write_verify8(dev,updates[i].address,updates[i].value);
        if(result!=IMX477_OK) return invalidate(dev,result);
    }
    dev->frame_length_lines=period;dev->exposure_lines=lines;dev->analogue_gain=0u;
    return IMX477_OK;
}

int imx477_set_colour_gains(struct imx477 *dev, uint16_t red_q8,
                          uint16_t green_q8, uint16_t blue_q8)
{
    struct imx477_reg updates[] = {
        {REG_USE_GLOBAL_GAIN, 1u},
        {REG_DIGITAL_GAIN, (uint8_t)(green_q8 >> 8)},
        {REG_DIGITAL_GAIN + 1u, (uint8_t)green_q8},
        {REG_RED_GAIN, (uint8_t)(red_q8 >> 8)},
        {REG_RED_GAIN + 1u, (uint8_t)red_q8},
        {REG_BLUE_GAIN, (uint8_t)(blue_q8 >> 8)},
        {REG_BLUE_GAIN + 1u, (uint8_t)blue_q8},
        {REG_USE_GLOBAL_GAIN, 0u}
    };
    size_t i;
    int result;
    if (dev == NULL || !dev->initialized || !dev->configured || dev->streaming)
        return IMX477_ESTATE;
    if (red_q8 < IMX477_DIGITAL_GAIN_UNITY || red_q8 > IMX477_COLOUR_GAIN_MAX ||
        green_q8 < IMX477_DIGITAL_GAIN_UNITY || green_q8 > IMX477_COLOUR_GAIN_MAX ||
        blue_q8 < IMX477_DIGITAL_GAIN_UNITY || blue_q8 > IMX477_COLOUR_GAIN_MAX)
        return IMX477_EINVAL;
    /* Standby prevents a partially updated frame. Keep the global selector
     * active until all three documented gains have been written and verified.
     * Intentionally use individual byte writes even if atomic16 is available:
     * this identifies the exact failing register and verifies every byte.
     */
    for (i = 0; i < ARRAY_LENGTH(updates); ++i) {
        result = write_verify8(dev, updates[i].address, updates[i].value);
        if (result != IMX477_OK)
            return invalidate(dev, result);
    }
    dev->red_gain_q8 = red_q8;
    dev->green_gain_q8 = green_q8;
    dev->blue_gain_q8 = blue_q8;
    dev->colour_gains_enabled = 1u;
    /* There is no active global multiplier in this mode. */
    dev->digital_gain = IMX477_DIGITAL_GAIN_UNITY;
    dev->last_failed_register = 0u;
    dev->last_bus_result = 0;
    dev->last_error = IMX477_OK;
    dev->last_expected_value = 0u;
    dev->last_read_value = 0u;
    return IMX477_OK;
}

int imx477_set_test_pattern(struct imx477 *dev, uint8_t pattern)
{
    int result;
    if (pattern > 4u)
        return IMX477_EINVAL;
    result = set_control16(dev, REG_TEST_PATTERN, pattern);
    if (result == IMX477_OK)
        dev->test_pattern = pattern;
    return result;
}

int imx477_reconfigure_preserving_controls(struct imx477 *dev,
                                           enum imx477_mode_id mode)
{
    const struct imx477_mode_info *next_mode = imx477_get_mode_info(mode);
    uint64_t numerator, denominator, next_lines;
    uint16_t analogue_gain, digital_gain, red_gain, green_gain, blue_gain;
    uint8_t colour_enabled, pattern;
    int result;
    if (dev == NULL || !dev->initialized || !dev->configured ||
        (dev->mode != &mode_info[0] && dev->mode != &mode_info[1]))
        return IMX477_ESTATE;
    if (next_mode == NULL)
        return IMX477_EINVAL;
    /* Snapshot/validate before configure() resets the public control state.
     * The fixed supported modes bound all products safely inside uint64_t.
     */
    analogue_gain = dev->analogue_gain;
    digital_gain = dev->digital_gain;
    red_gain = dev->red_gain_q8;
    green_gain = dev->green_gain_q8;
    blue_gain = dev->blue_gain_q8;
    colour_enabled = dev->colour_gains_enabled;
    pattern = dev->test_pattern;
    if (dev->frame_length_lines < dev->mode->frame_length_lines ||
        dev->frame_length_lines > 0xffdcu ||
        dev->exposure_lines < IMX477_EXPOSURE_MIN_LINES ||
        dev->exposure_lines > dev->frame_length_lines - IMX477_EXPOSURE_MARGIN_LINES ||
        analogue_gain > IMX477_ANALOGUE_GAIN_MAX || pattern > 4u ||
        (!colour_enabled && digital_gain < IMX477_DIGITAL_GAIN_UNITY) ||
        (colour_enabled &&
         (red_gain < IMX477_DIGITAL_GAIN_UNITY || red_gain > IMX477_COLOUR_GAIN_MAX ||
          green_gain < IMX477_DIGITAL_GAIN_UNITY || green_gain > IMX477_COLOUR_GAIN_MAX ||
          blue_gain < IMX477_DIGITAL_GAIN_UNITY || blue_gain > IMX477_COLOUR_GAIN_MAX)))
        return IMX477_ESTATE;
    numerator = (uint64_t)dev->exposure_lines * dev->mode->line_length_clocks *
                next_mode->timing_pixel_rate_hz;
    denominator = (uint64_t)dev->mode->timing_pixel_rate_hz *
                  next_mode->line_length_clocks;
    next_lines = (numerator + denominator / 2u) / denominator;
    if (next_lines < IMX477_EXPOSURE_MIN_LINES)
        next_lines = IMX477_EXPOSURE_MIN_LINES;
    if (next_lines > next_mode->frame_length_lines - IMX477_EXPOSURE_MARGIN_LINES)
        next_lines = next_mode->frame_length_lines - IMX477_EXPOSURE_MARGIN_LINES;

    result = imx477_configure(dev, mode);
    if (result == IMX477_OK)
        result = imx477_set_exposure_lines(dev, (uint16_t)next_lines);
    if (result == IMX477_OK)
        result = imx477_set_analogue_gain(dev, analogue_gain);
    if (result == IMX477_OK) {
        if (colour_enabled)
            result = imx477_set_colour_gains(dev, red_gain, green_gain, blue_gain);
        else
            result = imx477_set_digital_gain(dev, digital_gain);
    }
    if (result == IMX477_OK && pattern != 0u)
        result = imx477_set_test_pattern(dev, pattern);
    /* configure/control I/O errors already invalidate and preserve their
     * first failing register. Also fail closed if any other restore error
     * occurs after a successful configuration. Never restart automatically.
     */
    if (result != IMX477_OK && dev->configured)
        return invalidate(dev, result);
    return result;
}
