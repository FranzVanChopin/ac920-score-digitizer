#ifndef TEST_XTIME_H
#define TEST_XTIME_H
#include <stdint.h>
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000u
void XTime_GetTime(XTime *t);
#endif
