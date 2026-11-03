#ifndef CAPTURE_BUTTON_H
#define CAPTURE_BUTTON_H

#include <stdint.h>

typedef enum {
    CAPTURE_BUTTON_NONE = 0,
    CAPTURE_BUTTON_PHOTO,
    CAPTURE_BUTTON_NEXT_SONG,
    CAPTURE_BUTTON_LONG = CAPTURE_BUTTON_NEXT_SONG /* application-defined hold action */
} capture_button_event;

typedef struct {
    uint32_t debounce_ms;
    uint32_t long_press_ms;
    uint32_t raw_since_ms;
    uint32_t pressed_since_ms;
    unsigned raw_pressed;
    unsigned stable_pressed;
    unsigned long_reported;
    unsigned armed;
} capture_button;

/* Poll with a monotonic millisecond clock; unsigned subtraction handles wrap.
 * Start with the actual pin level. A button held at boot must first be released.
 * Call more frequently than debounce_ms. No repeat events are generated.
 * long_press_ms=0 selects photo-only mode: any debounced press/release yields
 * one PHOTO, regardless of hold duration; no NEXT_SONG event is generated. */
void capture_button_init(capture_button *button, uint32_t now_ms,
                         int initially_pressed, uint32_t debounce_ms,
                         uint32_t long_press_ms);
capture_button_event capture_button_update(capture_button *button,
                                           uint32_t now_ms, int pressed);

#endif
