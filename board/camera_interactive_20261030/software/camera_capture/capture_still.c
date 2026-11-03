#include "capture_still.h"
#include <math.h>

int capture_still_plan(const capture_exposure_stats *s,
    const capture_exposure_setting *current,unsigned max_lines,
    capture_exposure_setting *next,unsigned *limited)
{
    capture_exposure_setting old;
    unsigned peak;
    double factor=1.0,ratio,exposure;
    if(!s || !current || !next || !limited || max_lines<4u || max_lines>35000u ||
       current->lines<4u || current->lines>max_lines || current->gain_code>768u ||
       s->luma_p90>255u || s->raw_max_p99>255u || s->corrected_max_p99>255u ||
       s->raw_clipped_permille>1000u || s->corrected_clipped_permille>1000u ||
       s->matrix_high_clip_permille>1000u) return -1;
    old=*current;*limited=0u;
    peak=s->raw_max_p99>s->corrected_max_p99?s->raw_max_p99:s->corrected_max_p99;
    if(peak>=245u || s->raw_clipped_permille>=5u || s->corrected_clipped_permille>=5u ||
       s->matrix_high_clip_permille>=5u) {
        ratio=220.0/(double)(peak>250u?peak:250u);
        factor=pow(ratio,2.2);
    } else if(s->luma_p90<200u || s->luma_p90>220u) {
        ratio=210.0/(double)(s->luma_p90>16u?s->luma_p90:16u);
        /* A saturated colour may have low luma: do not chase white brightness. */
        if(peak && ratio>235.0/peak) ratio=235.0/peak;
        factor=pow(ratio,2.2);
        if(factor>16.0) factor=16.0;
        if(factor<0.25) factor=0.25;
    }
    exposure=(double)old.lines*1024.0/(1024.0-old.gain_code)*factor;
    if(exposure>max_lines) { exposure=max_lines;*limited=1u; }
    if(exposure<4.0) exposure=4.0;
    next->lines=(unsigned)(exposure+0.5);next->gain_code=0u;
    return next->lines!=old.lines || old.gain_code!=0u;
}
