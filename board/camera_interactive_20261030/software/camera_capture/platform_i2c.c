/* SPDX-License-Identifier: MIT */
/*
 * Bounded short-message PS IIC0 transport for the IMX477.
 * Register definitions/ordering reviewed against AMD embeddedsw iicps_v3_20
 * (Vitis 2024.2): xiicps_hw.h, xiicps_xfer.c, xiicps_master.c and xiicps.c.
 * Do not replace these loops with XIicPs_MasterRecvPolled: that version's
 * XIicPs_MasterPolledRead can wait forever when RXDV never appears and has no
 * software deadline. No driver transfer routine is used below.
 */
#include "platform_i2c.h"
#include "xparameters.h"
#include "xtime_l.h"
#include <stddef.h>
#include <string.h>

#define DIVIDER_MASK (XIICPS_CR_DIV_A_MASK | XIICPS_CR_DIV_B_MASK)
#define MASTER_CONTROL (XIICPS_CR_MS_MASK | XIICPS_CR_ACKEN_MASK | XIICPS_CR_NEA_MASK)
#define ERROR_INTERRUPTS (XIICPS_IXR_ARB_LOST_MASK | XIICPS_IXR_NACK_MASK | \
                          XIICPS_IXR_TO_MASK | XIICPS_IXR_RX_OVR_MASK | \
                          XIICPS_IXR_RX_UNF_MASK | XIICPS_IXR_TX_OVR_MASK)
/* Secondary finite guard also prevents an endless MMIO loop if the timestamp
 * counter unexpectedly stops after init. Normal expiration always uses time.
 */
#define MAX_TRANSACTION_POLLS 2000000u

struct deadline {
    XTime start;
    XTime ticks;
    unsigned int remaining_polls;
};

static void deadline_start(struct deadline *limit, uint32_t usec)
{
    XTime_GetTime(&limit->start);
    limit->ticks = ((XTime)COUNTS_PER_SECOND * usec + 999999u) / 1000000u;
    limit->remaining_polls = MAX_TRANSACTION_POLLS;
}

static int expired(struct deadline *limit)
{
    XTime now;
    if (limit->remaining_polls == 0u)
        return 1;
    --limit->remaining_polls;
    XTime_GetTime(&now);
    return (now - limit->start) >= limit->ticks;
}

static int valid_instance(const XIicPs *instance)
{
    return instance != NULL && instance->IsReady == XIL_COMPONENT_IS_READY &&
           instance->Config.BaseAddress == PLATFORM_I2C_PS0_BASEADDR;
}

static int interrupt_error(u32 status)
{
    if ((status & XIICPS_IXR_ARB_LOST_MASK) != 0u)
        return PLATFORM_I2C_EARBITRATION;
    if ((status & XIICPS_IXR_NACK_MASK) != 0u)
        return PLATFORM_I2C_ENACK;
    if ((status & XIICPS_IXR_TO_MASK) != 0u)
        return PLATFORM_I2C_ETIMEOUT;
    if ((status & (XIICPS_IXR_RX_OVR_MASK | XIICPS_IXR_RX_UNF_MASK |
                   XIICPS_IXR_TX_OVR_MASK)) != 0u)
        return PLATFORM_I2C_EFIFO;
    return PLATFORM_I2C_OK;
}

static void clear_transfer_state(XIicPs *instance)
{
    instance->SendBufferPtr = NULL;
    instance->RecvBufferPtr = NULL;
    instance->SendByteCount = 0;
    instance->RecvByteCount = 0;
    instance->CurrByteCount = 0;
    instance->UpdateTxSize = 0;
    instance->IsSend = 0;
    instance->IsRepeatedStart = 0;
    instance->Is10BitAddr = 0;
    instance->Options = XIICPS_7_BIT_ADDR_OPTION;
}

static int abort_transfer(XIicPs *instance, int result)
{
    UINTPTR base = instance->Config.BaseAddress;
    u32 dividers = XIicPs_ReadReg(base, XIICPS_CR_OFFSET) & DIVIDER_MASK;
    XIicPs_WriteReg(base, XIICPS_IDR_OFFSET, XIICPS_IXR_ALL_INTR_MASK);
    /* Drop master/HOLD and flush FIFOs, as XIicPs_Abort does, while retaining
     * the previously selected clock dividers. Never write ADDR during cleanup.
     */
    XIicPs_WriteReg(base, XIICPS_CR_OFFSET, dividers | XIICPS_CR_CLR_FIFO_MASK);
    XIicPs_WriteReg(base, XIICPS_TRANS_SIZE_OFFSET, 0u);
    XIicPs_WriteReg(base, XIICPS_ISR_OFFSET, XIICPS_IXR_ALL_INTR_MASK);
    XIicPs_WriteReg(base, XIICPS_CR_OFFSET, dividers | MASTER_CONTROL);
    clear_transfer_state(instance);
    return result;
}

static int wait_idle(XIicPs *instance, struct deadline *limit)
{
    UINTPTR base = instance->Config.BaseAddress;
    while ((XIicPs_ReadReg(base, XIICPS_SR_OFFSET) & XIICPS_SR_BA_MASK) != 0u) {
        if (expired(limit))
            return PLATFORM_I2C_EBUSY;
    }
    return PLATFORM_I2C_OK;
}

static int wait_complete(XIicPs *instance, struct deadline *limit)
{
    UINTPTR base = instance->Config.BaseAddress;
    for (;;) {
        u32 status = XIicPs_ReadReg(base, XIICPS_ISR_OFFSET);
        int result = interrupt_error(status & ERROR_INTERRUPTS);
        if (result != PLATFORM_I2C_OK)
            return result;
        if ((status & XIICPS_IXR_COMP_MASK) != 0u)
            return PLATFORM_I2C_OK;
        if (expired(limit))
            return PLATFORM_I2C_ETIMEOUT;
    }
}

static int send_short(XIicPs *instance, const uint8_t *data, unsigned int count,
                      int repeated_start, struct deadline *limit)
{
    UINTPTR base = instance->Config.BaseAddress;
    u32 control;
    unsigned int i;
    int result;
    if (count == 0u || count > 4u || data == NULL)
        return PLATFORM_I2C_EINVAL;
    result = wait_idle(instance, limit);
    if (result != PLATFORM_I2C_OK)
        return result;
    control = (XIicPs_ReadReg(base, XIICPS_CR_OFFSET) & DIVIDER_MASK) |
              MASTER_CONTROL | XIICPS_CR_CLR_FIFO_MASK;
    if (repeated_start)
        control |= XIICPS_CR_HOLD_MASK;
    XIicPs_WriteReg(base, XIICPS_IDR_OFFSET, XIICPS_IXR_ALL_INTR_MASK);
    XIicPs_WriteReg(base, XIICPS_CR_OFFSET, control);
    XIicPs_WriteReg(base, XIICPS_ISR_OFFSET, XIICPS_IXR_ALL_INTR_MASK);
    for (i = 0; i < count; ++i)
        XIicPs_WriteReg(base, XIICPS_DATA_OFFSET, data[i]);
    XIicPs_WriteReg(base, XIICPS_ADDR_OFFSET, IMX477_I2C_ADDRESS_7BIT);
    result = wait_complete(instance, limit);
    if (result != PLATFORM_I2C_OK)
        return result;
    XIicPs_WriteReg(base, XIICPS_ISR_OFFSET, XIICPS_IXR_COMP_MASK);
    return repeated_start ? PLATFORM_I2C_OK : wait_idle(instance, limit);
}

static int send_register(XIicPs *instance, uint16_t reg, uint16_t value,
                         int value_is_16bit)
{
    uint8_t data[4];
    struct deadline limit;
    int result;
    if (!valid_instance(instance))
        return PLATFORM_I2C_EINVAL;
    data[0] = (uint8_t)(reg >> 8);
    data[1] = (uint8_t)reg;
    if (value_is_16bit) {
        data[2] = (uint8_t)(value >> 8);
        data[3] = (uint8_t)value;
    } else {
        data[2] = (uint8_t)value;
    }
    deadline_start(&limit, PLATFORM_I2C_TIMEOUT_US);
    result = send_short(instance, data, value_is_16bit ? 4u : 3u, 0, &limit);
    if (result != PLATFORM_I2C_OK)
        return abort_transfer(instance, result);
    return PLATFORM_I2C_OK;
}

static int sensor_write8(void *ctx, uint16_t reg, uint8_t value)
{
    return send_register((XIicPs *)ctx, reg, value, 0);
}

static int sensor_write16(void *ctx, uint16_t reg, uint16_t value)
{
    return send_register((XIicPs *)ctx, reg, value, 1);
}

static int sensor_read8(void *ctx, uint16_t reg, uint8_t *value)
{
    XIicPs *instance = (XIicPs *)ctx;
    uint8_t address[2];
    uint8_t received;
    struct deadline limit;
    UINTPTR base;
    u32 control;
    int result;
    if (!valid_instance(instance) || value == NULL)
        return PLATFORM_I2C_EINVAL;
    base = instance->Config.BaseAddress;
    address[0] = (uint8_t)(reg >> 8);
    address[1] = (uint8_t)reg;
    deadline_start(&limit, PLATFORM_I2C_TIMEOUT_US);
    result = send_short(instance, address, 2u, 1, &limit);
    if (result != PLATFORM_I2C_OK)
        return abort_transfer(instance, result);

    /* HOLD remains set between the register-address TX and this RX START.
     * Prepare RX under HOLD, start it by writing ADDR, THEN clear HOLD so the
     * controller NACKs/STOPs after exactly one byte. Clearing it before ADDR
     * would turn the requested combined transaction into STOP then START.
     */
    control = (XIicPs_ReadReg(base, XIICPS_CR_OFFSET) & DIVIDER_MASK) |
              MASTER_CONTROL | XIICPS_CR_HOLD_MASK | XIICPS_CR_RD_WR_MASK;
    XIicPs_WriteReg(base, XIICPS_CR_OFFSET, control | XIICPS_CR_CLR_FIFO_MASK);
    XIicPs_WriteReg(base, XIICPS_ISR_OFFSET, XIICPS_IXR_ALL_INTR_MASK);
    XIicPs_WriteReg(base, XIICPS_TRANS_SIZE_OFFSET, 1u);
    XIicPs_WriteReg(base, XIICPS_ADDR_OFFSET, IMX477_I2C_ADDRESS_7BIT);
    XIicPs_WriteReg(base, XIICPS_CR_OFFSET, control & ~XIICPS_CR_HOLD_MASK);
    result = wait_complete(instance, &limit);
    if (result != PLATFORM_I2C_OK)
        return abort_transfer(instance, result);
    if ((XIicPs_ReadReg(base, XIICPS_SR_OFFSET) & XIICPS_SR_RXDV_MASK) == 0u)
        return abort_transfer(instance, PLATFORM_I2C_EFIFO);
    received = (uint8_t)XIicPs_ReadReg(base, XIICPS_DATA_OFFSET);
    XIicPs_WriteReg(base, XIICPS_ISR_OFFSET, XIICPS_IXR_COMP_MASK);
    result = wait_idle(instance, &limit);
    if (result != PLATFORM_I2C_OK)
        return abort_transfer(instance, result);
    /* Do not change caller's output on NACK, timeout, or incomplete STOP. */
    *value = received;
    return PLATFORM_I2C_OK;
}

static void sensor_delay_us(void *ctx, uint32_t usec)
{
    XTime start, now;
    XTime ticks = ((XTime)COUNTS_PER_SECOND * usec + 999999u) / 1000000u;
    (void)ctx;
    XTime_GetTime(&start);
    do {
        XTime_GetTime(&now);
    } while ((now - start) < ticks);
}

int platform_i2c_init(XIicPs *instance)
{
    XIicPs_Config *config;
    XTime first, now;
    unsigned int attempt;
    int timer_running = 0;
    int result;
    struct deadline limit;
    if (instance == NULL)
        return PLATFORM_I2C_EINVAL;
    memset(instance, 0, sizeof(*instance));
#ifdef SDT
    config = XIicPs_LookupConfig(PLATFORM_I2C_PS0_BASEADDR);
#else
#ifndef XPAR_XIICPS_0_DEVICE_ID
#error "Enable PS IIC0 via EMIO in the hardware platform and regenerate the BSP"
#endif
    config = XIicPs_LookupConfig(XPAR_XIICPS_0_DEVICE_ID);
#endif
    if (config == NULL || config->BaseAddress != PLATFORM_I2C_PS0_BASEADDR)
        return PLATFORM_I2C_ECONFIG;
    if (XIicPs_CfgInitialize(instance, config, (u32)config->BaseAddress) != XST_SUCCESS)
        return PLATFORM_I2C_ECONFIG;
#if defined(XCLOCKING)
    Xil_ClockEnable(instance->Config.RefClk);
    instance->IsClkEnabled = 1u;
#endif
    if (XIicPs_SetSClk(instance, PLATFORM_I2C_CLOCK_HZ) != XST_SUCCESS) {
        instance->IsReady = 0u;
        return PLATFORM_I2C_ECONFIG;
    }
    XIicPs_WriteReg(instance->Config.BaseAddress, XIICPS_TIME_OUT_OFFSET,
                    XIICPS_TO_RESET_VALUE);
    (void)abort_transfer(instance, PLATFORM_I2C_OK);
    /* XTime_GetTime enables the default A53 counter in EL3 BSPs. A faulty or
     * uninitialized counter must be reported before any timeout-dependent I/O.
     */
    XTime_GetTime(&first);
    for (attempt = 0u; attempt < 10000u; ++attempt) {
        XTime_GetTime(&now);
        if (now != first) {
            timer_running = 1;
            break;
        }
    }
    if (!timer_running) {
        instance->IsReady = 0u;
        return PLATFORM_I2C_ETIMER;
    }
    deadline_start(&limit, PLATFORM_I2C_TIMEOUT_US);
    result = wait_idle(instance, &limit);
    if (result != PLATFORM_I2C_OK) {
        (void)abort_transfer(instance, result);
        instance->IsReady = 0u;
    }
    return result;
}

int platform_i2c_bind(XIicPs *instance, struct imx477_io *io)
{
    if (!valid_instance(instance) || io == NULL)
        return PLATFORM_I2C_EINVAL;
    memset(io, 0, sizeof(*io));
    io->ctx = instance;
    io->write16addr8 = sensor_write8;
    io->read16addr8 = sensor_read8;
    io->delay_us = sensor_delay_us;
    io->write16addr16 = sensor_write16;
    return PLATFORM_I2C_OK;
}
