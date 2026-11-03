/* SPDX-License-Identifier: GPL-2.0-only */
/* Host test: compile this with ../imx477.c; no FPGA BSP is required. */
#include "../imx477.h"
#include "../capture_still.h"
#include "../capture_settle.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

struct bus {
    uint8_t regs[65536];
    unsigned int writes;
    unsigned int reads;
    unsigned int write16_calls;
    unsigned int fail_write;
    unsigned int fail_reads;
    unsigned int fail_read_at;
    unsigned int corrupt_read_at;
    unsigned int fail_write_after;
    unsigned int stream_starts;
    uint16_t first_fault_register;
    unsigned int log_count;
    int log_enabled;
    struct { uint16_t reg; uint8_t value; char operation; } log[32];
    uint64_t delayed_us;
};

static struct bus bus;

static void log_io(struct bus *b, char operation, uint16_t reg, uint8_t value)
{
    if (!b->log_enabled)
        return;
    assert(b->log_count < sizeof(b->log) / sizeof(b->log[0]));
    b->log[b->log_count].reg = reg;
    b->log[b->log_count].value = value;
    b->log[b->log_count].operation = operation;
    ++b->log_count;
}

static int write8(void *ctx, uint16_t reg, uint8_t value)
{
    struct bus *b = (struct bus *)ctx;
    ++b->writes;
    log_io(b, 'W', reg, value);
    if (b->fail_write_after != 0u && b->writes >= b->fail_write_after) {
        if (!b->first_fault_register) b->first_fault_register = reg;
        return -94;
    }
    if (b->writes == b->fail_write) {
        if (!b->first_fault_register) b->first_fault_register = reg;
        return -91;
    }
    b->regs[reg] = value;
    if (reg == 0x0100u && value == 1u) ++b->stream_starts;
    return 0;
}

static int read8(void *ctx, uint16_t reg, uint8_t *value)
{
    struct bus *b = (struct bus *)ctx;
    ++b->reads;
    log_io(b, 'R', reg, b->regs[reg]);
    if (b->reads == b->fail_read_at) {
        if (!b->first_fault_register) b->first_fault_register = reg;
        return -92;
    }
    if (b->fail_reads != 0u) {
        --b->fail_reads;
        return -92;
    }
    *value = b->regs[reg];
    if (b->reads == b->corrupt_read_at) {
        if (!b->first_fault_register) b->first_fault_register = reg;
        *value ^= 0x01u;
    }
    return 0;
}

static int write16(void *ctx, uint16_t reg, uint16_t value)
{
    struct bus *b = (struct bus *)ctx;
    ++b->writes;
    ++b->write16_calls;
    if (b->writes == b->fail_write) {
        if (!b->first_fault_register) b->first_fault_register = reg;
        return -93;
    }
    b->regs[reg] = (uint8_t)(value >> 8);
    b->regs[(uint16_t)(reg + 1u)] = (uint8_t)value;
    return 0;
}

static void delay_us(void *ctx, uint32_t usec)
{
    ((struct bus *)ctx)->delayed_us += usec;
}

static uint16_t reg16(uint16_t reg)
{
    return (uint16_t)(((uint16_t)bus.regs[reg] << 8) | bus.regs[reg + 1u]);
}

static struct imx477_io make_io(int atomic16)
{
    struct imx477_io io;
    memset(&bus, 0, sizeof(bus));
    bus.regs[0x0016] = 0x04;
    bus.regs[0x0017] = 0x77;
    io.ctx = &bus;
    io.write16addr8 = write8;
    io.read16addr8 = read8;
    io.delay_us = delay_us;
    io.write16addr16 = atomic16 ? write16 : NULL;
    return io;
}

static void test_probe(void)
{
    struct imx477 dev;
    struct imx477_io io = make_io(0);
    assert(imx477_init(NULL, &io) == IMX477_EINVAL);
    assert(imx477_init(&dev, NULL) == IMX477_EINVAL);
    assert(imx477_get_mode_info((enum imx477_mode_id)99) == NULL);
    bus.fail_reads = 2u;
    assert(imx477_init(&dev, &io) == IMX477_OK);
    assert(bus.reads == 4u);
    assert(bus.delayed_us == 621000u);
    assert(dev.initialized && !dev.configured && !dev.streaming);
    assert(imx477_start(&dev) == IMX477_ESTATE);

    io = make_io(0);
    bus.fail_reads = 100u;
    assert(imx477_init(&dev, &io) == IMX477_EIO);
    assert(bus.reads == 5u && bus.writes == 0u);
    assert(bus.delayed_us == 540000u && !dev.initialized);
    assert(dev.last_failed_register == 0x0016u && dev.last_bus_result == -92);

    io = make_io(0);
    bus.regs[0x0016] = 0x03;
    bus.regs[0x0017] = 0x78;
    assert(imx477_init(&dev, &io) == IMX477_ECHIP_ID);
    assert(bus.writes == 0u && !dev.initialized);
}

static void assert_full_mode(const struct imx477 *dev)
{
    assert(reg16(0x034c) == 4056u && reg16(0x034e) == 3040u);
    assert(reg16(0x0342) == 24000u && reg16(0x0340) == 3500u);
    assert(reg16(0x0202) == 350u); /* 10 ms, not upstream's 1600-line default. */
    assert(reg16(0x0204) == 0u && reg16(0x020e) == 0x0100u);
    assert(bus.regs[0x0112] == 12u && bus.regs[0x0113] == 12u);
    assert(bus.regs[0x0309] == 12u && bus.regs[0x3f0d] == 1u);
    assert(bus.regs[0x0114] == 1u && bus.regs[0x0101] == 0u);
    assert(reg16(0x030e) == 150u && reg16(0x0820) == 1800u);
    assert(reg16(0x0822) == 0u);
    assert(bus.regs[0x0900] == 0u && bus.regs[0x0901] == 0x11u);
    assert(bus.regs[0x0100] == 0u && dev->configured && !dev->streaming);
    assert(bus.regs[0x4b81] == 0u && bus.regs[0x3041] == 1u);
    assert(bus.regs[0x3040] == 0u && bus.regs[0x3f0b] == 0u);
    assert(bus.regs[0x0b05] == 1u && bus.regs[0x0b06] == 1u);
    assert(bus.regs[0x0c1b] == 0u && reg16(0x0600) == 0u);
    assert(dev->mode->width == 4056u && dev->mode->height == 3040u);
    assert(!dev->colour_gains_enabled && bus.regs[0x3ff9] == 1u);
    assert(dev->red_gain_q8 == 256u && dev->green_gain_q8 == 256u &&
           dev->blue_gain_q8 == 256u);
}

static void test_modes_and_controls(void)
{
    struct imx477 dev;
    struct imx477_io io = make_io(0);
    unsigned int writes;
    uint64_t before_stop;
    assert(imx477_init(&dev, &io) == IMX477_OK);
    assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert_full_mode(&dev);
    assert(bus.regs[0x7b75] == 0x0au);
    assert(imx477_set_exposure_lines(&dev, 3u) == IMX477_EINVAL);
    assert(imx477_set_exposure_lines(&dev, 3479u) == IMX477_EINVAL);
    assert(imx477_set_exposure_lines(&dev, 4u) == IMX477_OK);
    assert(imx477_set_exposure_lines(&dev, 3478u) == IMX477_OK);
    assert(reg16(0x0202) == 3478u);
    assert(imx477_set_analogue_gain(&dev, 979u) == IMX477_EINVAL);
    assert(imx477_set_analogue_gain(&dev, 978u) == IMX477_OK);
    assert(reg16(0x0204) == 978u);
    assert(imx477_set_digital_gain(&dev, 255u) == IMX477_EINVAL);
    assert(imx477_set_digital_gain(&dev, 65535u) == IMX477_OK);
    assert(reg16(0x020e) == 65535u);
    assert(imx477_set_test_pattern(&dev, 5u) == IMX477_EINVAL);
    assert(imx477_set_test_pattern(&dev, 2u) == IMX477_OK);
    assert(reg16(0x0600) == 2u);
    assert(imx477_start(&dev) == IMX477_OK);
    assert(bus.regs[0x0100] == 1u && dev.streaming);
    writes = bus.writes;
    assert(imx477_start(&dev) == IMX477_OK && bus.writes == writes);
    assert(imx477_set_exposure_lines(&dev, 350u) == IMX477_ESTATE);
    assert(bus.writes == writes && dev.configured && dev.streaming);
    before_stop = bus.delayed_us;
    assert(imx477_stop(&dev) == IMX477_OK);
    assert(bus.delayed_us - before_stop == 101000u && !dev.streaming);

    assert(imx477_configure(&dev, IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
    assert(reg16(0x034c) == 2028u && reg16(0x034e) == 1520u);
    assert(reg16(0x0342) == 12000u && reg16(0x0340) == 2334u);
    assert(reg16(0x0202) == 700u && dev.exposure_lines == 700u);
    assert(bus.regs[0x0900] == 1u && bus.regs[0x0901] == 0x22u);
    /* Preview omits this mode-specific register, so common table must reload. */
    assert(bus.regs[0x7b75] == 0x0eu);
    assert(reg16(0x0204) == 0u && reg16(0x020e) == 256u);
    assert(reg16(0x0600) == 0u);
    assert(imx477_set_exposure_lines(&dev, 2312u) == IMX477_OK);
    assert(imx477_set_exposure_lines(&dev, 2313u) == IMX477_EINVAL);
    assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert_full_mode(&dev);
}

static void test_live_atomic_controls(void)
{
    struct imx477 dev;
    struct imx477_io io = make_io(1);
    unsigned int before;
    assert(imx477_init(&dev, &io) == IMX477_OK);
    assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert(imx477_start(&dev) == IMX477_OK);
    before = bus.writes;
    assert(imx477_set_exposure_lines(&dev, 700u) == IMX477_OK);
    assert(bus.writes == before + 1u && reg16(0x0202) == 700u);
    assert(imx477_set_analogue_gain(&dev, 512u) == IMX477_OK);
    assert(reg16(0x0204) == 512u);
    bus.fail_write = bus.writes + 1u;
    assert(imx477_set_digital_gain(&dev, 512u) == IMX477_EIO);
    assert(!dev.configured && !dev.streaming && bus.regs[0x0100] == 0u);
    assert(dev.last_failed_register == 0x020eu && dev.last_bus_result == -93);
    assert(imx477_start(&dev) == IMX477_ESTATE);
    assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert_full_mode(&dev);
}

static void test_every_configuration_write_can_fail(void)
{
    struct imx477 dev;
    struct imx477_io io = make_io(0);
    unsigned int first, count, failure;
    assert(imx477_init(&dev, &io) == IMX477_OK);
    first = bus.writes;
    assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    count = bus.writes - first;
    for (failure = 1u; failure <= count; ++failure) {
        io = make_io(0);
        assert(imx477_init(&dev, &io) == IMX477_OK);
        bus.fail_write = bus.writes + failure;
        assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_EIO);
        assert(!dev.configured && bus.regs[0x0100] == 0u);
        assert(dev.last_bus_result == -91);
        assert(imx477_start(&dev) == IMX477_ESTATE);
    }
    printf("Injected failure at each of %u configuration writes: PASS\n", count);
}

static void prepare_colour_test(struct imx477 *dev, int atomic16)
{
    struct imx477_io io = make_io(atomic16);
    assert(imx477_init(dev, &io) == IMX477_OK);
    assert(imx477_configure(dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    bus.regs[0x0214] = 0xa5u;
    bus.regs[0x0215] = 0x5au;
}

static void test_colour_gains(void)
{
    static const uint16_t addresses[] = {
        0x3ff9u, 0x020eu, 0x020fu, 0x0210u,
        0x0211u, 0x0212u, 0x0213u, 0x3ff9u
    };
    static const uint8_t values[] = {1u, 1u, 0u, 3u, 0x3eu, 1u, 0x65u, 0u};
    struct imx477 dev;
    unsigned int writes, reads, i;
    int atomic16;
    memset(&dev, 0, sizeof(dev));
    assert(imx477_set_colour_gains(NULL, 830u, 256u, 357u) == IMX477_ESTATE);
    assert(imx477_set_colour_gains(&dev, 830u, 256u, 357u) == IMX477_ESTATE);
    for (atomic16 = 0; atomic16 <= 1; ++atomic16) {
        prepare_colour_test(&dev, atomic16);
        writes = bus.writes;
        reads = bus.reads;
        assert(imx477_set_colour_gains(&dev, 255u, 256u, 256u) == IMX477_EINVAL);
        assert(imx477_set_colour_gains(&dev, 256u, 255u, 256u) == IMX477_EINVAL);
        assert(imx477_set_colour_gains(&dev, 256u, 256u, 255u) == IMX477_EINVAL);
        assert(imx477_set_colour_gains(&dev, 2049u, 256u, 256u) == IMX477_EINVAL);
        assert(imx477_set_colour_gains(&dev, 256u, 2049u, 256u) == IMX477_EINVAL);
        assert(imx477_set_colour_gains(&dev, 256u, 256u, 2049u) == IMX477_EINVAL);
        assert(bus.writes == writes && bus.reads == reads && dev.configured);
        assert(imx477_start(&dev) == IMX477_OK);
        writes = bus.writes;
        assert(imx477_set_colour_gains(&dev, 830u, 256u, 357u) == IMX477_ESTATE);
        assert(bus.writes == writes && bus.reads == reads && dev.streaming);
        assert(imx477_stop(&dev) == IMX477_OK);
        writes = bus.writes;
        bus.log_enabled = 1;
        assert(imx477_set_colour_gains(&dev, 830u, 256u, 357u) == IMX477_OK);
        bus.log_enabled = 0;
        assert(bus.writes == writes + 8u && bus.reads == reads + 8u);
        assert(bus.log_count == 16u);
        for (i = 0u; i < 8u; ++i) {
            assert(bus.log[i * 2u].operation == 'W');
            assert(bus.log[i * 2u + 1u].operation == 'R');
            assert(bus.log[i * 2u].reg == addresses[i]);
            assert(bus.log[i * 2u + 1u].reg == addresses[i]);
            assert(bus.log[i * 2u].value == values[i]);
        }
        assert(reg16(0x020e) == 256u && reg16(0x0210) == 830u &&
               reg16(0x0212) == 357u && bus.regs[0x3ff9] == 0u);
        assert(reg16(0x0214) == 0xa55au); /* No inferred fourth-channel register. */
        assert(dev.colour_gains_enabled && !dev.streaming && dev.configured);
        assert(dev.red_gain_q8 == 830u && dev.green_gain_q8 == 256u &&
               dev.blue_gain_q8 == 357u && dev.last_error == IMX477_OK);
        writes = bus.writes;
        assert(imx477_set_digital_gain(&dev, 512u) == IMX477_ESTATE);
        assert(bus.writes == writes && reg16(0x020e) == 256u);
        assert(imx477_set_colour_gains(&dev, 2048u, 2048u, 2048u) == IMX477_OK);
        assert(reg16(0x020e) == 2048u && reg16(0x0210) == 2048u &&
               reg16(0x0212) == 2048u);
        assert(imx477_set_colour_gains(&dev, 256u, 256u, 256u) == IMX477_OK);
        assert(imx477_start(&dev) == IMX477_OK);
        assert(imx477_set_digital_gain(&dev, 512u) == IMX477_ESTATE);
        assert(imx477_configure(&dev, IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
        assert_full_mode(&dev);
        assert(imx477_set_digital_gain(&dev, 512u) == IMX477_OK);
        assert(dev.red_gain_q8 == 512u && dev.green_gain_q8 == 512u &&
               dev.blue_gain_q8 == 512u);
    }
}

static void test_every_colour_transaction_can_fail(void)
{
    static const uint16_t addresses[] = {
        0x3ff9u, 0x020eu, 0x020fu, 0x0210u,
        0x0211u, 0x0212u, 0x0213u, 0x3ff9u
    };
    static const uint8_t values[] = {1u, 1u, 0u, 3u, 0x3eu, 1u, 0x65u, 0u};
    struct imx477 dev;
    unsigned int failure, writes;
    int fault, atomic16, cleanup_failure;
    for (atomic16 = 0; atomic16 <= 1; ++atomic16) {
        for (cleanup_failure = 0; cleanup_failure <= 1; ++cleanup_failure) {
            for (fault = 0; fault < 3; ++fault) {
                for (failure = 1u; failure <= 8u; ++failure) {
                    prepare_colour_test(&dev, atomic16);
                    if (fault == 0)
                        bus.fail_write = bus.writes + failure;
                    else if (fault == 1)
                        bus.fail_read_at = bus.reads + failure;
                    else
                        bus.corrupt_read_at = bus.reads + failure;
                    if (cleanup_failure)
                        bus.fail_write_after = bus.writes + failure + 1u;
                    assert(imx477_set_colour_gains(&dev, 830u, 256u, 357u) ==
                           (fault == 2 ? IMX477_EVERIFY : IMX477_EIO));
                    assert(!dev.configured && !dev.streaming);
                    assert(dev.last_failed_register == addresses[failure - 1u]);
                    assert(dev.last_error == (fault == 2 ? IMX477_EVERIFY : IMX477_EIO));
                    assert(dev.last_bus_result == (fault == 0 ? -91 : (fault == 1 ? -92 : 0)));
                    assert(dev.last_expected_value == values[failure - 1u]);
                    if (fault == 2)
                        assert(dev.last_read_value == (uint8_t)(values[failure - 1u] ^ 1u));
                    /* No partially updated public state and no stream restart. */
                    assert(!dev.colour_gains_enabled && dev.red_gain_q8 == 256u &&
                           dev.green_gain_q8 == 256u && dev.blue_gain_q8 == 256u);
                    writes = bus.writes;
                    assert(imx477_start(&dev) == IMX477_ESTATE && bus.writes == writes);
                    assert(bus.regs[0x0100] == 0u && reg16(0x0214) == 0xa55au);
                }
            }
        }
    }
    puts("Colour gains: all 8 writes/reads/mismatches, both buses, cleanup failures: PASS");
}

static void prepare_switch_test(struct imx477 *dev, enum imx477_mode_id mode,
                                int colour, int atomic16)
{
    struct imx477_io io = make_io(atomic16);
    assert(imx477_init(dev, &io) == IMX477_OK);
    assert(imx477_configure(dev, mode) == IMX477_OK);
    assert(imx477_set_exposure_lines(dev, 701u) == IMX477_OK);
    assert(imx477_set_analogue_gain(dev, 512u) == IMX477_OK);
    if (colour)
        assert(imx477_set_colour_gains(dev, 830u, 256u, 357u) == IMX477_OK);
    else
        assert(imx477_set_digital_gain(dev, 384u) == IMX477_OK);
    assert(imx477_set_test_pattern(dev, 2u) == IMX477_OK);
    assert(imx477_start(dev) == IMX477_OK);
}

static void test_mode_switch_preserves_controls(void)
{
    struct imx477 dev;
    int atomic16, colour;
    unsigned int writes, starts;
    uint64_t before;
    memset(&dev, 0, sizeof(dev));
    assert(imx477_reconfigure_preserving_controls(NULL,
           IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_ESTATE);
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_ESTATE);
    for (atomic16 = 0; atomic16 <= 1; ++atomic16) {
        for (colour = 0; colour <= 1; ++colour) {
            prepare_switch_test(&dev, IMX477_MODE_2028X1520_RAW12_30FPS, colour, atomic16);
            writes = bus.writes;
            assert(imx477_reconfigure_preserving_controls(&dev,
                   (enum imx477_mode_id)99) == IMX477_EINVAL);
            assert(bus.writes == writes && dev.configured && dev.streaming);
            before = bus.delayed_us;
            starts = bus.stream_starts;
            assert(imx477_reconfigure_preserving_controls(&dev,
                   IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
            /* Preview 701 half-length lines -> 350.5 full lines -> 351. */
            assert(dev.exposure_lines == 351u && reg16(0x0202) == 351u);
            assert(bus.delayed_us - before == 34343u);
            assert(dev.configured && !dev.streaming && bus.regs[0x0100] == 0u);
            assert(bus.stream_starts == starts);
            assert(reg16(0x034c) == 4056u && reg16(0x034e) == 3040u);
            assert(reg16(0x0342) == 24000u && reg16(0x0340) == 3500u);
            assert(dev.analogue_gain == 512u && reg16(0x0204) == 512u);
            assert(dev.test_pattern == 2u && reg16(0x0600) == 2u);
            if (colour) {
                assert(dev.colour_gains_enabled && bus.regs[0x3ff9] == 0u);
                assert(dev.red_gain_q8 == 830u && reg16(0x0210) == 830u);
                assert(dev.green_gain_q8 == 256u && reg16(0x020e) == 256u);
                assert(dev.blue_gain_q8 == 357u && reg16(0x0212) == 357u);
            } else {
                assert(!dev.colour_gains_enabled && bus.regs[0x3ff9] == 1u);
                assert(dev.digital_gain == 384u && reg16(0x020e) == 384u);
                assert(dev.red_gain_q8 == 384u && dev.green_gain_q8 == 384u &&
                       dev.blue_gain_q8 == 384u);
            }
            assert(imx477_start(&dev) == IMX477_OK);
            assert(imx477_reconfigure_preserving_controls(&dev,
                   IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
            assert(dev.exposure_lines == 702u && reg16(0x0202) == 702u);
            assert(!dev.streaming && dev.configured);
            assert(reg16(0x034c) == 2028u && reg16(0x034e) == 1520u);
            assert(reg16(0x0342) == 12000u && reg16(0x0340) == 2334u);
            assert(dev.analogue_gain == 512u && reg16(0x0204) == 512u);
            assert(dev.colour_gains_enabled == (unsigned)colour);
            /* Caller restores the exact preview snapshot after rounding. */
            assert(imx477_set_exposure_lines(&dev, 701u) == IMX477_OK);
            assert(dev.exposure_lines == 701u);
            /* An even preview exposure makes an exact round trip. */
            assert(imx477_set_exposure_lines(&dev, 700u) == IMX477_OK);
            assert(imx477_reconfigure_preserving_controls(&dev,
                   IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
            assert(dev.exposure_lines == 350u);
            assert(imx477_reconfigure_preserving_controls(&dev,
                   IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
            assert(dev.exposure_lines == 700u);
        }
    }
}

static void test_mode_switch_clamps_time(void)
{
    struct imx477 dev;
    prepare_switch_test(&dev, IMX477_MODE_4056X3040_RAW12_10FPS, 1, 1);
    assert(imx477_set_exposure_lines(&dev, 3478u) == IMX477_OK);
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
    assert(dev.exposure_lines == 2312u && reg16(0x0202) == 2312u);
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert(dev.exposure_lines == 1156u); /* Clamped time cannot be reconstructed. */
    assert(imx477_set_exposure_lines(&dev, 3478u) == IMX477_OK);
    assert(dev.exposure_lines == 3478u); /* Explicit saved-snapshot restore. */
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
    assert(imx477_set_exposure_lines(&dev, 4u) == IMX477_OK);
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_4056X3040_RAW12_10FPS) == IMX477_OK);
    assert(dev.exposure_lines == 4u); /* 2 calculated full lines -> minimum 4. */
    assert(imx477_reconfigure_preserving_controls(&dev,
           IMX477_MODE_2028X1520_RAW12_30FPS) == IMX477_OK);
    assert(dev.exposure_lines == 8u);
}

static void test_mode_switch_faults(void)
{
    struct imx477 dev;
    unsigned int count, first, read_count, failure, starts, writes;
    int direction, atomic16, colour, corrupt;
    for (direction = 0; direction <= 1; ++direction) {
        enum imx477_mode_id old_mode = (enum imx477_mode_id)direction;
        enum imx477_mode_id new_mode = (enum imx477_mode_id)(direction ^ 1);
        for (atomic16 = 0; atomic16 <= 1; ++atomic16) {
            for (colour = 0; colour <= 1; ++colour) {
                prepare_switch_test(&dev, old_mode, colour, atomic16);
                first = bus.writes;
                read_count = bus.reads;
                assert(imx477_reconfigure_preserving_controls(&dev, new_mode) == IMX477_OK);
                count = bus.writes - first;
                read_count = bus.reads - read_count;
                for (failure = 1u; failure <= count; ++failure) {
                    prepare_switch_test(&dev, old_mode, colour, atomic16);
                    bus.fail_write = bus.writes + failure;
                    starts = bus.stream_starts;
                    assert(imx477_reconfigure_preserving_controls(&dev, new_mode) == IMX477_EIO);
                    assert(!dev.configured && dev.last_error == IMX477_EIO);
                    assert(dev.last_failed_register == bus.first_fault_register);
                    assert(dev.last_bus_result == -91 || dev.last_bus_result == -93);
                    writes = bus.writes;
                    assert(imx477_start(&dev) == IMX477_ESTATE && bus.writes == writes);
                    assert(bus.stream_starts == starts);
                }
                for (corrupt = 0; corrupt <= 1; ++corrupt) {
                    for (failure = 1u; failure <= read_count; ++failure) {
                        prepare_switch_test(&dev, old_mode, colour, atomic16);
                        if (corrupt) bus.corrupt_read_at = bus.reads + failure;
                        else bus.fail_read_at = bus.reads + failure;
                        starts = bus.stream_starts;
                        assert(imx477_reconfigure_preserving_controls(&dev, new_mode) ==
                               (corrupt ? IMX477_EVERIFY : IMX477_EIO));
                        assert(!dev.configured);
                        assert(dev.last_failed_register == bus.first_fault_register);
                        assert(dev.last_bus_result == (corrupt ? 0 : -92));
                        writes = bus.writes;
                        assert(imx477_start(&dev) == IMX477_ESTATE && bus.writes == writes);
                        assert(bus.stream_starts == starts);
                    }
                }
            }
        }
    }
    puts("Mode switches: every write/read/mismatch, both directions/gain modes/buses: PASS");
}

static void test_long_still(void)
{
    struct imx477 dev;
    struct imx477_io io;
    unsigned k,kind,atomic16,writes;
    const unsigned lines[]={4u,3478u,3479u,7000u,17500u,35000u};
    for(atomic16=0;atomic16<=1;++atomic16) for(k=0;k<6;++k) {
        unsigned frame=lines[k]+22u;
        uint32_t period;
        uint64_t before;
        io=make_io((int)atomic16);
        assert(imx477_init(&dev,&io)==0 && imx477_configure(&dev,IMX477_MODE_4056X3040_RAW12_10FPS)==0);
        assert(imx477_set_colour_gains(&dev,830,256,357)==0);
        writes=bus.writes;
        assert(imx477_set_still_low_gain(&dev,(uint16_t)lines[k])==0 && bus.writes-writes==6);
        if(frame<3500u) frame=3500u;
        period=(uint32_t)(((uint64_t)frame*24000u*1000000u+839999999u)/840000000u);
        assert(dev.frame_length_lines==frame && reg16(0x0340)==frame && imx477_frame_period_us(&dev)==period);
        assert(dev.exposure_lines==lines[k] && reg16(0x0202)==lines[k] && reg16(0x0204)==0);
        assert(dev.analogue_gain==0 && reg16(0x0210)==830 && reg16(0x0212)==357 && reg16(0x020e)==256);
        assert(reg16(0x034c)==4056 && reg16(0x034e)==3040 && bus.regs[0x3100]==0);
        assert(imx477_start(&dev)==0);writes=bus.writes;
        assert(imx477_set_still_low_gain(&dev,7000)==IMX477_ESTATE && bus.writes==writes);
        before=bus.delayed_us;
        assert(imx477_reconfigure_preserving_controls(&dev,IMX477_MODE_2028X1520_RAW12_30FPS)==0);
        assert(bus.delayed_us-before==period+1000u);
        assert(dev.frame_length_lines==2334 && reg16(0x0340)==2334 && imx477_frame_period_us(&dev)==33343u);
        /* Exact caller restoration, not the lossy converted/clamped shutter. */
        assert(imx477_set_exposure_lines(&dev,1960)==0 && imx477_set_analogue_gain(&dev,768)==0);
        assert(dev.exposure_lines==1960 && dev.analogue_gain==768 && dev.red_gain_q8==830);
    }
    for(kind=0;kind<3;++kind) for(k=1;k<=6;++k) {
        io=make_io(1);assert(imx477_init(&dev,&io)==0 && imx477_configure(&dev,0)==0);
        if(kind==0) bus.fail_write=bus.writes+k;
        if(kind==1) bus.fail_read_at=bus.reads+k;
        if(kind==2) bus.corrupt_read_at=bus.reads+k;
        assert(imx477_set_still_low_gain(&dev,35000)==(kind==2?IMX477_EVERIFY:IMX477_EIO));
        assert(!dev.configured && dev.last_failed_register==bus.first_fault_register);
        writes=bus.writes;assert(imx477_start(&dev)==IMX477_ESTATE && writes==bus.writes);
    }
    io=make_io(1);assert(imx477_init(&dev,&io)==0 && imx477_configure(&dev,0)==0);
    writes=bus.writes;
    assert(imx477_set_still_low_gain(&dev,3)==IMX477_EINVAL);
    assert(imx477_set_still_low_gain(&dev,35001)==IMX477_EINVAL && bus.writes==writes);
    assert(imx477_set_still_low_gain(NULL,100)==IMX477_ESTATE && imx477_frame_period_us(NULL)==0);
    {
        capture_exposure_stats stats={77,80,83,0,0,0};
        capture_exposure_setting current={3478,115},next;
        capture_settle settled={0};unsigned limited;
        assert(capture_still_plan(&stats,&current,35000,&next,&limited)==1 && limited);
        assert(next.lines==35000 && next.gain_code==0);
        assert(imx477_set_still_low_gain(&dev,(uint16_t)next.lines)==0);
        assert(imx477_start(&dev)==0);
        capture_settle_changed_period(&settled,100,imx477_frame_period_us(&dev));
        assert(settled.wait_ms==2052);
        assert(!capture_settle_frame_ready(&settled,450) && !capture_settle_frame_ready(&settled,2151));
        assert(capture_settle_frame_ready(&settled,2152));
    }
    puts("PASS low-ISO sensor: six-byte write/readback faults, 1s frame/drain, exposure limits, WB preservation, preview restore and long-frame settling");
}

int main(void)
{
    test_probe();
    test_modes_and_controls();
    test_live_atomic_controls();
    test_every_configuration_write_can_fail();
    test_colour_gains();
    test_every_colour_transaction_can_fail();
    test_mode_switch_preserves_controls();
    test_mode_switch_clamps_time();
    test_mode_switch_faults();
    test_long_still();
    puts("IMX477 mock-bus tests: PASS");
    return 0;
}
