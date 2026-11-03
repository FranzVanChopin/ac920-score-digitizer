#include <stdint.h>
typedef intptr_t INTPTR;
static inline void Xil_DCacheFlushRange(INTPTR address,unsigned length)
{ (void)address;(void)length; }
