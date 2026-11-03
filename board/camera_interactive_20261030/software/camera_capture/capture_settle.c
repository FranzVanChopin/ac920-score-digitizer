/* SPDX-License-Identifier: MIT */
#include "capture_settle.h"

void capture_settle_changed(capture_settle *state, uint32_t now)
{
    capture_settle_changed_period(state,now,0u);
}

void capture_settle_changed_period(capture_settle *state,uint32_t now,uint32_t frame_period_us)
{
    uint32_t wait=(uint32_t)(((uint64_t)frame_period_us*2u+999u)/1000u)+50u;
    if (!state) return;
    state->changed_at = now;
    state->active = 1u;
    state->wait_ms=wait>CAPTURE_SETTLE_MS?wait:CAPTURE_SETTLE_MS;
}

int capture_settle_frame_ready(capture_settle *state,
                               uint32_t capture_started_at)
{
    if (!state) return 0;
    if (!state->active) return 1;
    if ((uint32_t)(capture_started_at - state->changed_at) < state->wait_ms)
        return 0;
    state->active = 0u;
    return 1;
}
