#include "capture_button.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void press(capture_button *b, uint32_t start)
{
    assert(capture_button_update(b, start, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(b, start + 30u, 1) == CAPTURE_BUTTON_NONE);
}

int main(void)
{
    capture_button b;
    capture_button_init(&b, 0, 0, 30, 1500);
    /* Bounce must not create a click. */
    assert(capture_button_update(&b, 100, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 110, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 120, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 149, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 150, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 300, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 310, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 320, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 349, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 350, 0) == CAPTURE_BUTTON_PHOTO);
    assert(capture_button_update(&b, 1000, 0) == CAPTURE_BUTTON_NONE);

    press(&b, 2000);
    assert(capture_button_update(&b, 3499, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 3500, 1) == CAPTURE_BUTTON_NEXT_SONG);
    assert(capture_button_update(&b, 5500, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 5600, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 5630, 0) == CAPTURE_BUTTON_NONE);

    /* Release debounce crossing the threshold remains a short press. */
    press(&b, 6000);
    assert(capture_button_update(&b, 7499, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 7529, 0) == CAPTURE_BUTTON_PHOTO);

    /* A slow polling interval at release still reports one long press. */
    press(&b, 8000);
    assert(capture_button_update(&b, 9600, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 9630, 0) == CAPTURE_BUTTON_NEXT_SONG);
    assert(capture_button_update(&b, 9700, 0) == CAPTURE_BUTTON_NONE);

    /* Do not capture or change song from a button held during boot. */
    capture_button_init(&b, 0, 1, 30, 1500);
    assert(capture_button_update(&b, 2000, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 2100, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 2130, 0) == CAPTURE_BUTTON_NONE);
    press(&b, 2300);
    assert(capture_button_update(&b, 2400, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 2430, 0) == CAPTURE_BUTTON_PHOTO);

    /* Millisecond counter wrap. */
    capture_button_init(&b, UINT32_MAX - 100u, 0, 30, 1500);
    press(&b, UINT32_MAX - 50u);
    assert(capture_button_update(&b, 1449, 1) == CAPTURE_BUTTON_NEXT_SONG);
    assert(capture_button_update(&b, 1600, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 1630, 0) == CAPTURE_BUTTON_NONE);

    /* Simple test mode: short and long holds each produce one photo only on
     * release. Bouncing, holding and boot/busy-held keys must not repeat. */
    capture_button_init(&b, 0, 0, 30, 0);
    assert(capture_button_update(&b, 10, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 20, 0) == CAPTURE_BUTTON_NONE);
    press(&b, 100);
    assert(capture_button_update(&b, 200, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 230, 0) == CAPTURE_BUTTON_PHOTO);
    assert(capture_button_update(&b, 300, 0) == CAPTURE_BUTTON_NONE);
    press(&b, 1000);
    assert(capture_button_update(&b, 5000, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 30000, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 30100, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 30130, 0) == CAPTURE_BUTTON_PHOTO);
    assert(capture_button_update(&b, 40000, 0) == CAPTURE_BUTTON_NONE);
    capture_button_init(&b, 50000, 1, 30, 0);
    assert(capture_button_update(&b, 60000, 1) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 60100, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 60130, 0) == CAPTURE_BUTTON_NONE);
    press(&b, 61000);
    assert(capture_button_update(&b, 61100, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 61130, 0) == CAPTURE_BUTTON_PHOTO);
    capture_button_init(&b, UINT32_MAX - 100u, 0, 30, 0);
    press(&b, UINT32_MAX - 50u);
    assert(capture_button_update(&b, 100, 0) == CAPTURE_BUTTON_NONE);
    assert(capture_button_update(&b, 130, 0) == CAPTURE_BUTTON_PHOTO);
    puts("capture_button: all cases passed");
    return 0;
}
