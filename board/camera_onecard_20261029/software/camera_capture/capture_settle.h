/* SPDX-License-Identifier: MIT */
#ifndef CAPTURE_SETTLE_H
#define CAPTURE_SETTLE_H

#include <stdint.h>

#define CAPTURE_SETTLE_MS UINT32_C(350)

typedef struct {
    uint32_t changed_at;
    unsigned active;
    uint32_t wait_ms;
} capture_settle;

/* Both timestamps use the same monotonically advancing millisecond clock,
 * truncated to uint32_t. A zero-initialized object is ready.
 * Call changed AFTER the last successful sensor control write or stream
 * restart. Repeated changes restart the interval. AE enable/disable is not
 * a sensor change and must not clear an outstanding interval. */
void capture_settle_changed(capture_settle *state, uint32_t now);
/* Long stills require at least two actual frame periods +50ms, minimum350ms.
 * frame_period_us must be the programmed sensor period, not exposure alone. */
void capture_settle_changed_period(capture_settle *state,uint32_t now,uint32_t frame_period_us);

/* Call immediately before the capture gate request and save the returned
 * value with that frame. Use its REQUEST START time, never completion or
 * save time: a frame starting during settling remains ineligible even when
 * it finishes later. Only a complete, error-free eligible frame may be used
 * for AE, white calibration or a pending photo; previews may use other
 * complete frames. This function checks time only, not DMA/frame validity.
 *
 * Calls must follow sensor changes chronologically: do not pass an older
 * frame's timestamp after changed(), and do not write sensor controls during
 * an in-flight capture. Unsigned subtraction handles clock rollover provided
 * less than 2^32 milliseconds elapse while an interval remains active.
 * The first eligible request clears active. NULL is never ready. */
int capture_settle_frame_ready(capture_settle *state,
                               uint32_t capture_started_at);

#endif
