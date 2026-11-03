/* SPDX-License-Identifier: MIT */
/* Host model using installed real XIicPs/standalone headers. See the
 * platform_i2c_fixture directory; no fake sensor-driver success is involved.
 */
#include "../platform_i2c.h"
#include "xtime_l.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

enum fault {
    NONE, STUCK_IDLE, TX_NACK, RX_NACK, TX_TIMEOUT, RX_TIMEOUT,
    RX_NO_DATA, ARBITRATION, HARDWARE_TIMEOUT, STOP_STUCK, TIMER_STOPPED
};

static struct {
    u32 regs[16];
    uint8_t fifo[4];
    unsigned int fifo_count;
    uint8_t last_tx[4];
    unsigned int last_tx_count;
    unsigned int starts;
    unsigned int read_starts;
    unsigned int aborts;
    int held_after_tx;
    int rx_launched;
    int protocol_error;
    enum fault fault;
    XTime time;
    unsigned int time_reads;
} model;

static XIicPs_Config config = {
#ifndef SDT
    .DeviceId = 0u,
#else
    .Name = "iic0",
#endif
    .BaseAddress = PLATFORM_I2C_PS0_BASEADDR,
    .InputClockHz = 100000000u
};

static unsigned int offset_of(uintptr_t address)
{
    assert(address >= PLATFORM_I2C_PS0_BASEADDR);
    assert(address < PLATFORM_I2C_PS0_BASEADDR + sizeof(model.regs));
    return (unsigned int)(address - PLATFORM_I2C_PS0_BASEADDR);
}

uint32_t camera_test_in32(uintptr_t address)
{
    unsigned int offset = offset_of(address);
    if (offset == XIICPS_DATA_OFFSET) {
        assert((model.regs[XIICPS_SR_OFFSET / 4u] & XIICPS_SR_RXDV_MASK) != 0u);
        model.regs[XIICPS_SR_OFFSET / 4u] &= ~XIICPS_SR_RXDV_MASK;
        return 0x47u;
    }
    return model.regs[offset / 4u];
}

static void receive_after_hold_release(void)
{
    if (model.fault == RX_TIMEOUT)
        return;
    if (model.fault == RX_NACK) {
        model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_NACK_MASK;
        return;
    }
    model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_COMP_MASK;
    model.regs[XIICPS_SR_OFFSET / 4u] =
        model.fault == RX_NO_DATA ? 0u : XIICPS_SR_RXDV_MASK;
    if (model.fault == STOP_STUCK)
        model.regs[XIICPS_SR_OFFSET / 4u] |= XIICPS_SR_BA_MASK;
}

void camera_test_out32(uintptr_t address, uint32_t value)
{
    unsigned int offset = offset_of(address);
    if (offset == XIICPS_ISR_OFFSET) {
        model.regs[offset / 4u] &= ~value; /* W1C */
        return;
    }
    if (offset == XIICPS_DATA_OFFSET) {
        assert(model.fifo_count < sizeof(model.fifo));
        model.fifo[model.fifo_count++] = (uint8_t)value;
        return;
    }
    if (offset == XIICPS_CR_OFFSET) {
        u32 old = model.regs[offset / 4u];
        if ((value & XIICPS_CR_CLR_FIFO_MASK) != 0u) {
            model.fifo_count = 0u;
            model.regs[XIICPS_TRANS_SIZE_OFFSET / 4u] = 0u;
        }
        if ((value & XIICPS_CR_MS_MASK) == 0u) {
            ++model.aborts;
            model.held_after_tx = 0;
            model.rx_launched = 0;
            model.regs[XIICPS_SR_OFFSET / 4u] =
                model.fault == STUCK_IDLE ? XIICPS_SR_BA_MASK : 0u;
        } else if ((old & XIICPS_CR_HOLD_MASK) != 0u &&
                   (value & XIICPS_CR_HOLD_MASK) == 0u) {
            if (model.held_after_tx && !model.rx_launched)
                model.protocol_error = 1;
            if (model.rx_launched)
                receive_after_hold_release();
            model.held_after_tx = 0;
        }
        model.regs[offset / 4u] = value & ~XIICPS_CR_CLR_FIFO_MASK;
        return;
    }
    model.regs[offset / 4u] = value;
    if (offset == XIICPS_ADDR_OFFSET) {
        u32 control = model.regs[XIICPS_CR_OFFSET / 4u];
        assert(value == 0x1au);
        assert((control & XIICPS_CR_MS_MASK) != 0u);
        assert((control & XIICPS_CR_NEA_MASK) != 0u);
        ++model.starts;
        model.regs[XIICPS_SR_OFFSET / 4u] = XIICPS_SR_BA_MASK;
        if ((control & XIICPS_CR_RD_WR_MASK) != 0u) {
            ++model.read_starts;
            if (!model.held_after_tx || (control & XIICPS_CR_HOLD_MASK) == 0u)
                model.protocol_error = 1;
            assert(model.last_tx_count == 2u);
            assert(model.regs[XIICPS_TRANS_SIZE_OFFSET / 4u] == 1u);
            model.rx_launched = 1;
        } else {
            assert(model.fifo_count >= 2u && model.fifo_count <= 4u);
            memcpy(model.last_tx, model.fifo, model.fifo_count);
            model.last_tx_count = model.fifo_count;
            model.fifo_count = 0u;
            model.held_after_tx = (control & XIICPS_CR_HOLD_MASK) != 0u;
            model.rx_launched = 0;
            if (model.fault == TX_NACK)
                model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_NACK_MASK;
            else if (model.fault == ARBITRATION)
                model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_ARB_LOST_MASK;
            else if (model.fault == HARDWARE_TIMEOUT)
                model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_TO_MASK;
            else if (model.fault != TX_TIMEOUT) {
                model.regs[XIICPS_ISR_OFFSET / 4u] = XIICPS_IXR_COMP_MASK;
                if (!model.held_after_tx && model.fault != STOP_STUCK)
                    model.regs[XIICPS_SR_OFFSET / 4u] = 0u;
            }
        }
    }
}

void XTime_GetTime(XTime *value)
{
    ++model.time_reads;
    if (model.fault != TIMER_STOPPED)
        model.time += COUNTS_PER_SECOND / 10000u; /* 100 us per poll. */
    *value = model.time;
}

#ifdef SDT
XIicPs_Config *XIicPs_LookupConfig(u32 base)
{
    return base == PLATFORM_I2C_PS0_BASEADDR ? &config : NULL;
}
#else
XIicPs_Config *XIicPs_LookupConfig(u16 id)
{
    return id == 0u ? &config : NULL;
}
#endif

s32 XIicPs_CfgInitialize(XIicPs *instance, XIicPs_Config *cfg, u32 address)
{
    instance->Config = *cfg;
    instance->Config.BaseAddress = address;
    instance->IsReady = XIL_COMPONENT_IS_READY;
    return XST_SUCCESS;
}

s32 XIicPs_SetSClk(XIicPs *instance, u32 frequency)
{
    (void)instance;
    assert(frequency == 100000u);
    model.regs[XIICPS_CR_OFFSET / 4u] = 0x1400u;
    return XST_SUCCESS;
}

static void setup(XIicPs *instance, struct imx477_io *io)
{
    memset(&model, 0, sizeof(model));
    assert(platform_i2c_init(instance) == PLATFORM_I2C_OK);
    assert(platform_i2c_bind(instance, io) == PLATFORM_I2C_OK);
    assert(io->ctx == instance && io->write16addr16 != NULL);
    assert(model.regs[XIICPS_IDR_OFFSET / 4u] == XIICPS_IXR_ALL_INTR_MASK);
    assert(model.regs[XIICPS_TIME_OUT_OFFSET / 4u] == XIICPS_TO_RESET_VALUE);
}

static void assert_clean(const XIicPs *instance)
{
    assert((model.regs[XIICPS_CR_OFFSET / 4u] & XIICPS_CR_HOLD_MASK) == 0u);
    assert((model.regs[XIICPS_CR_OFFSET / 4u] &
            (XIICPS_CR_DIV_A_MASK | XIICPS_CR_DIV_B_MASK)) == 0x1400u);
    assert(model.fifo_count == 0u);
    assert(model.regs[XIICPS_TRANS_SIZE_OFFSET / 4u] == 0u);
    assert(model.regs[XIICPS_ISR_OFFSET / 4u] == 0u);
    assert(instance->IsRepeatedStart == 0 && instance->SendByteCount == 0 &&
           instance->RecvByteCount == 0 && instance->UpdateTxSize == 0);
}

int main(void)
{
    XIicPs instance;
    struct imx477_io io;
    uint8_t value;
    unsigned int starts, i;
    static const enum fault failures[] = {
        STUCK_IDLE, TX_NACK, RX_NACK, TX_TIMEOUT, RX_TIMEOUT, RX_NO_DATA,
        ARBITRATION, HARDWARE_TIMEOUT, STOP_STUCK
    };
    static const int expected[] = {
        PLATFORM_I2C_EBUSY, PLATFORM_I2C_ENACK, PLATFORM_I2C_ENACK,
        PLATFORM_I2C_ETIMEOUT, PLATFORM_I2C_ETIMEOUT, PLATFORM_I2C_EFIFO,
        PLATFORM_I2C_EARBITRATION, PLATFORM_I2C_ETIMEOUT, PLATFORM_I2C_EBUSY
    };
    setup(&instance, &io);
    assert(io.write16addr8(io.ctx, 0x1234u, 0x56u) == 0);
    assert(model.last_tx_count == 3u && model.last_tx[0] == 0x12u &&
           model.last_tx[1] == 0x34u && model.last_tx[2] == 0x56u);
    assert(io.write16addr16(io.ctx, 0x0202u, 0x0102u) == 0);
    assert(model.last_tx_count == 4u && model.last_tx[0] == 2u &&
           model.last_tx[1] == 2u && model.last_tx[2] == 1u && model.last_tx[3] == 2u);
    starts = model.starts;
    value = 0u;
    assert(io.read16addr8(io.ctx, 0x0016u, &value) == 0 && value == 0x47u);
    assert(model.starts == starts + 2u && model.read_starts == 1u);
    assert(model.last_tx[0] == 0u && model.last_tx[1] == 0x16u);
    assert(!model.protocol_error && !model.held_after_tx);
    assert((model.regs[XIICPS_CR_OFFSET / 4u] & XIICPS_CR_HOLD_MASK) == 0u);
    for (i = 0u; i < sizeof(failures) / sizeof(failures[0]); ++i) {
        unsigned int before, aborts;
        setup(&instance, &io);
        model.fault = failures[i];
        if (model.fault == STUCK_IDLE)
            model.regs[XIICPS_SR_OFFSET / 4u] = XIICPS_SR_BA_MASK;
        before = model.time_reads;
        aborts = model.aborts;
        value = 0xa5u;
        assert(io.read16addr8(io.ctx, 0x0016u, &value) == expected[i]);
        assert(value == 0xa5u); /* No valid data reported after a failed STOP. */
        assert(model.time_reads - before <= 102u);
        assert(model.aborts == aborts + 1u);
        assert_clean(&instance);
        if (model.fault == STUCK_IDLE)
            assert((model.regs[XIICPS_SR_OFFSET / 4u] & XIICPS_SR_BA_MASK) != 0u);
        model.fault = NONE;
        model.regs[XIICPS_SR_OFFSET / 4u] = 0u;
        assert(io.write16addr8(io.ctx, 0x0100u, 0u) == 0); /* recovery */
    }
    memset(&model, 0, sizeof(model));
    model.fault = TIMER_STOPPED;
    assert(platform_i2c_init(&instance) == PLATFORM_I2C_ETIMER);
    assert(model.time_reads <= 10001u);
    assert(platform_i2c_bind(&instance, &io) == PLATFORM_I2C_EINVAL);
    puts("PS IIC0 repeated-START, NACK, timeout, FIFO, STOP and recovery tests: PASS");
    return 0;
}
