#ifndef CAPTURE_STILL_H
#define CAPTURE_STILL_H
#include "capture_exposure.h"
/* Automatic still policy only. current.lines are in FULL-resolution line units.
 * Keep analogue gain at1x; target gamma luma P90=210 with highlight protection.
 * Use gamma2.2 only to seed exposure; remeasure the new native frame on hardware.
 * Clamp shutter to max_lines, report limited rather than increasing ISO/digital
 * gain. A bounded caller may retain the last verified frame if not converged.
 * Return -1 invalid,0 unchanged,1 changed. Supports current==next. */
int capture_still_plan(const capture_exposure_stats *stats,
    const capture_exposure_setting *current,unsigned max_lines,
    capture_exposure_setting *next,unsigned *limited);
#endif
