/* SPDX-License-Identifier: MIT */
#ifndef CAMERA_CAPTURE_PLATFORM_I2C_H
#define CAMERA_CAPTURE_PLATFORM_I2C_H

#include "imx477.h"
#include "xiicps.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PLATFORM_I2C_CLOCK_HZ       100000u
#define PLATFORM_I2C_TIMEOUT_US     10000u
#define PLATFORM_I2C_PS0_BASEADDR   0xff020000u

enum platform_i2c_result {
    PLATFORM_I2C_OK = 0,
    PLATFORM_I2C_EINVAL = -100,
    PLATFORM_I2C_ECONFIG = -101,
    PLATFORM_I2C_ETIMEOUT = -102,
    PLATFORM_I2C_ENACK = -103,
    PLATFORM_I2C_EARBITRATION = -104,
    PLATFORM_I2C_EFIFO = -105,
    PLATFORM_I2C_EBUSY = -106,
    PLATFORM_I2C_ETIMER = -107
};

/* Initializes Zynq UltraScale+ PS IIC0 at 100 kHz. The XSA/FSBL must enable its
 * EMIO route and reference clock. Own this controller exclusively: no ISR,
 * other core, driver call, or task may use the same XIicPs concurrently.
 * Uses the standalone A53 XTime timer, verifies that it advances, and disables
 * IIC interrupts. Both classic and SDT BSP lookup conventions are supported.
 */
int platform_i2c_init(XIicPs *instance);

/* Fills all IMX477 callbacks, including the single-transaction 16-bit write.
 * The XIicPs object must outlive the sensor driver. Slave is fixed at 0x1a.
 * Every bus transaction has a 10-ms XTime deadline; the deadline includes bus
 * idle, address write, repeated START, byte read, completion and final STOP.
 * Errors clear HOLD/FIFO/software transfer state. A physically held-low bus
 * remains an error; this function never assumes that clearing registers fixed
 * the external wiring. Missing pull-ups/NACK must not stall the application.
 */
int platform_i2c_bind(XIicPs *instance, struct imx477_io *io);

#ifdef __cplusplus
}
#endif
#endif
