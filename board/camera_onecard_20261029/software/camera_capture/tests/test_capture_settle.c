/* SPDX-License-Identifier: MIT */
#include "../capture_settle.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

static void test_initial_and_boundaries(void)
{
    capture_settle state = {0};
    assert(capture_settle_frame_ready(&state, 0u));
    capture_settle_changed(&state, 100u);
    assert(state.active && state.changed_at == 100u);
    assert(!capture_settle_frame_ready(&state, 100u));
    assert(!capture_settle_frame_ready(&state, 449u));
    assert(capture_settle_frame_ready(&state, 450u));
    assert(!state.active);
    assert(capture_settle_frame_ready(&state, 451u));
}

static void test_frame_start_and_pending_photo(void)
{
    capture_settle state = {0};
    int photo_pending = 1;
    int frame_ready;
    const uint32_t first_start = 950u;
    const uint32_t first_completed = 1100u;
    capture_settle_changed(&state, 650u); /* Stable from 1000 ms. */
    frame_ready = capture_settle_frame_ready(&state, first_start);
    assert(first_completed > state.changed_at + CAPTURE_SETTLE_MS);
    /* A completed frame keeps its original eligibility. Never replace
     * first_start with first_completed to reclassify this frame. */
    assert(!frame_ready && state.active);
    if (frame_ready) photo_pending = 0;
    assert(photo_pending);
    frame_ready = capture_settle_frame_ready(&state, 1101u);
    assert(frame_ready && !state.active);
    if (frame_ready) photo_pending = 0;
    assert(!photo_pending);
}

static void test_repeated_changes_and_ae_lock(void)
{
    capture_settle state = {0};
    int auto_exposure = 1;
    capture_settle_changed(&state, 100u);
    assert(!capture_settle_frame_ready(&state, 400u));
    capture_settle_changed(&state, 420u); /* New end is 770, not 450. */
    assert(!capture_settle_frame_ready(&state, 450u));
    /* Locking AE changes caller policy, not sensor settling state. */
    auto_exposure = 0;
    assert(!auto_exposure);
    assert(!capture_settle_frame_ready(&state, 769u));
    assert(capture_settle_frame_ready(&state, 770u));
    capture_settle_changed(&state, 800u); /* Also rearms a cleared state. */
    assert(!capture_settle_frame_ready(&state, 1149u));
    assert(capture_settle_frame_ready(&state, 1150u));
}

static void test_clock_rollover(void)
{
    capture_settle state = {0};
    capture_settle_changed(&state, UINT32_MAX - UINT32_C(99));
    assert(!capture_settle_frame_ready(&state, UINT32_MAX)); /* 99 ms. */
    assert(!capture_settle_frame_ready(&state, 0u));         /* 100 ms. */
    assert(!capture_settle_frame_ready(&state, 249u));       /* 349 ms. */
    assert(capture_settle_frame_ready(&state, 250u));        /* 350 ms. */
}

int main(void)
{
    test_initial_and_boundaries();
    test_frame_start_and_pending_photo();
    test_repeated_changes_and_ae_lock();
    test_clock_rollover();
    {
        capture_settle state={0};
        capture_settle_changed_period(&state,UINT32_MAX-99u,1000629u);
        assert(state.wait_ms==2052u);
        assert(!capture_settle_frame_ready(&state,1951u));
        assert(capture_settle_frame_ready(&state,1952u));
        capture_settle_changed(&state,2000u);
        assert(state.wait_ms==350u && !capture_settle_frame_ready(&state,2349u));
        assert(capture_settle_frame_ready(&state,2350u));
    }
    assert(!capture_settle_frame_ready(NULL, 0u));
    capture_settle_changed(NULL, 0u);
    puts("PASS settle: request-start eligibility, queued photo, repeated writes, AE lock, uint32 rollover");
    return 0;
}
