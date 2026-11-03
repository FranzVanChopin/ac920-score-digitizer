#ifndef CAPTURE_PANEL_H
#define CAPTURE_PANEL_H
#include "capture_button.h"
#define PANEL_FOCUS_TOGGLE 1u
#define PANEL_UBUNTU 2u
#define PANEL_FOCUS_NEXT 4u
#define PANEL_S2_LONG_MS 1500u
typedef struct {capture_button s2,s3;} capture_panel;
/* GPIO2[8]=S2_n, GPIO2[9]=S3_n, synchronized in the PL. */
void capture_panel_init(capture_panel *,uint32_t,uint32_t);
unsigned capture_panel_update(capture_panel *,uint32_t,uint32_t);
/* Queue at a frame boundary: current is zero-based; pending is 1..5 or zero.
 * Advance from an existing pending choice so rapid taps are not collapsed. */
void capture_panel_focus_command(unsigned events,int active,unsigned current,
                                 unsigned *toggle,unsigned *pending);
#endif
