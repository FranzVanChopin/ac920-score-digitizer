#include "capture_exposure.h"

int capture_exposure_low_noise(const capture_exposure_setting *current,
    unsigned max_lines,capture_exposure_setting *next)
{
    unsigned distance,lines,new_distance;
    capture_exposure_setting old;
    if(!current || !next || max_lines<4u || max_lines>65535u || current->lines<4u ||
       current->lines>max_lines || current->gain_code>768u) return -1;
    old=*current;distance=1024u-old.gain_code;
    lines=(old.lines*1024u+distance/2u)/distance;
    if(lines>max_lines) lines=max_lines;
    new_distance=(lines*distance+old.lines/2u)/old.lines;
    if(new_distance>1024u) new_distance=1024u;
    next->lines=lines;next->gain_code=1024u-new_distance;
    return next->lines!=old.lines || next->gain_code!=old.gain_code;
}

int capture_exposure_next_limits(const capture_exposure_stats *s,
                          const capture_exposure_setting *current,
                          unsigned max_lines, unsigned auto_max_lines,
                          capture_exposure_setting *next,
                          enum capture_exposure_reason *reason)
{
    unsigned peak, numerator=0, denominator=1;
    capture_exposure_setting previous;
    enum capture_exposure_reason why=CAPTURE_AE_HOLD;
    if (!s || !current || !next || !reason || auto_max_lines<4 ||
        max_lines>65535 || auto_max_lines>max_lines || current->lines<4 ||
        current->lines>max_lines || current->gain_code>768 ||
        s->luma_p90>255 || s->raw_max_p99>255 || s->corrected_max_p99>255 ||
        s->raw_clipped_permille>1000 || s->corrected_clipped_permille>1000 ||
        s->matrix_high_clip_permille>1000) return -1;
    previous=*current;
    *next=previous;
    peak=s->raw_max_p99>s->corrected_max_p99?s->raw_max_p99:s->corrected_max_p99;
    /* A dim luma value for a saturated red/blue image must NEVER increase
     * exposure. Include pre-clamp CCM overflow, not only displayed luma. */
    if (s->raw_clipped_permille>=5 || s->corrected_clipped_permille>=5 ||
        s->matrix_high_clip_permille>=5 || peak>=250) {
        why=CAPTURE_AE_SATURATION; numerator=3; denominator=4;
    } else if (peak>245) {
        why=CAPTURE_AE_HIGHLIGHTS; numerator=7; denominator=8;
    } else if (s->luma_p90>230) {
        why=CAPTURE_AE_TOO_BRIGHT; numerator=7; denominator=8;
    } else if (s->luma_p90<205 && peak<232) {
        why=CAPTURE_AE_TOO_DARK;
        if (next->lines<auto_max_lines) {
            next->lines+=next->lines/8+1;
            if (next->lines>auto_max_lines) next->lines=auto_max_lines;
        } else {
            /* IMX477 analogue amplification = 1024/(1024-code).
             * Increase amplification by about 9/8, bounded to 4x. */
            next->gain_code=1024-(1024-next->gain_code)*8/9;
            if (next->gain_code>768) next->gain_code=768;
        }
    }
    if (numerator) {
        if (next->gain_code) {
            unsigned distance=((1024-next->gain_code)*denominator+numerator-1)/numerator;
            next->gain_code=distance>=1024?0:1024-distance;
        } else {
            next->lines=next->lines*numerator/denominator;
            if (next->lines<4) next->lines=4;
        }
    }
    *reason=why;
    return next->lines!=previous.lines || next->gain_code!=previous.gain_code;
}

int capture_exposure_next(const capture_exposure_stats *s,
                          const capture_exposure_setting *current,
                          capture_exposure_setting *next,
                          enum capture_exposure_reason *reason)
{
    return capture_exposure_next_limits(s, current, 3478u, 2800u, next, reason);
}
