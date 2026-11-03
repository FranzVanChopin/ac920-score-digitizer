#ifndef CAPTURE_EXPOSURE_H
#define CAPTURE_EXPOSURE_H

typedef struct {
    unsigned luma_p90;
    unsigned raw_max_p99;
    unsigned corrected_max_p99;
    unsigned raw_clipped_permille;
    unsigned corrected_clipped_permille;
    unsigned matrix_high_clip_permille;
} capture_exposure_stats;

typedef struct {
    unsigned lines;
    unsigned gain_code;
} capture_exposure_setting;

/* For stationary still capture only: trade analogue gain for exposure time,
 * preserving lines*1024/(1024-gain) to register rounding. Never exceed max_lines.
 * Caller must wait for a fresh settled frame and restore preview afterwards. */
int capture_exposure_low_noise(const capture_exposure_setting *current,
    unsigned max_lines,capture_exposure_setting *next);

enum capture_exposure_reason {
    CAPTURE_AE_HOLD, CAPTURE_AE_SATURATION, CAPTURE_AE_HIGHLIGHTS,
    CAPTURE_AE_TOO_BRIGHT, CAPTURE_AE_TOO_DARK
};

/* Pure policy; caller controls AE enable and sensor-settling intervals.
 * Return -1 for invalid input, 0 for hold, or 1 for a changed setting. */
int capture_exposure_next(const capture_exposure_stats *stats,
                          const capture_exposure_setting *current,
                          capture_exposure_setting *next,
                          enum capture_exposure_reason *reason);

/* Mode-aware policy. max_lines is the sensor's legal limit; auto_max_lines
 * is the shutter ceiling used when AE increases exposure before adding gain.
 * Require 4 <= auto_max_lines <= max_lines <= 65535. An existing manual
 * exposure above the auto ceiling remains valid up to max_lines and is not
 * shortened merely because of the policy ceiling. Gain is bounded to 4x.
 * The legacy next() above uses full-frame limits (3478 legal, 2800 auto).
 */
int capture_exposure_next_limits(const capture_exposure_stats *stats,
                                 const capture_exposure_setting *current,
                                 unsigned max_lines, unsigned auto_max_lines,
                                 capture_exposure_setting *next,
                                 enum capture_exposure_reason *reason);
#endif
