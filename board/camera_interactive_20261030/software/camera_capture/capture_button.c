#include "capture_button.h"

void capture_button_init(capture_button *b, uint32_t now, int pressed,
                         uint32_t debounce, uint32_t long_press)
{
    b->debounce_ms = debounce;
    b->long_press_ms = long_press;
    b->raw_since_ms = now;
    b->pressed_since_ms = now;
    b->raw_pressed = pressed != 0;
    b->stable_pressed = pressed != 0;
    b->long_reported = 0;
    b->armed = !pressed;
}

capture_button_event capture_button_update(capture_button *b,
                                           uint32_t now, int pressed)
{
    unsigned level = pressed != 0;
    if (level != b->raw_pressed) {
        b->raw_pressed = level;
        b->raw_since_ms = now;
    }
    if (b->raw_pressed != b->stable_pressed &&
        (uint32_t)(now - b->raw_since_ms) >= b->debounce_ms) {
        b->stable_pressed = b->raw_pressed;
        if (b->stable_pressed) {
            b->pressed_since_ms = b->raw_since_ms;
            b->long_reported = 0;
        } else {
            if (!b->armed) {
                b->armed = 1;
                return CAPTURE_BUTTON_NONE;
            }
            if (!b->long_reported) {
                /* Classify using the first stable release edge, not the later
                 * debounce completion. A delayed poll still sees one long event. */
                if (b->long_press_ms &&
                    (uint32_t)(b->raw_since_ms - b->pressed_since_ms) >=
                    b->long_press_ms) {
                    b->long_reported = 1;
                    return CAPTURE_BUTTON_NEXT_SONG;
                }
                return CAPTURE_BUTTON_PHOTO;
            }
        }
    }
    if (b->long_press_ms && b->armed && b->stable_pressed && b->raw_pressed &&
        !b->long_reported &&
        (uint32_t)(now - b->pressed_since_ms) >= b->long_press_ms) {
        b->long_reported = 1;
        return CAPTURE_BUTTON_NEXT_SONG;
    }
    return CAPTURE_BUTTON_NONE;
}
