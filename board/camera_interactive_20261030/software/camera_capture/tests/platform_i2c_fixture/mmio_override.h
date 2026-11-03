/* Force-include for host test: use REAL XIicPs register definitions/types,
 * replacing only physical MMIO access with the test's hardware model. */
#ifndef CAMERA_TEST_I2C_MMIO_OVERRIDE_H
#define CAMERA_TEST_I2C_MMIO_OVERRIDE_H
#include <stdint.h>
#define XIL_IO_H
#define INLINE inline
uint32_t camera_test_in32(uintptr_t address);
void camera_test_out32(uintptr_t address, uint32_t value);
#define Xil_In32 camera_test_in32
#define Xil_Out32 camera_test_out32
#endif
