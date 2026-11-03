#include "capture_still.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
int main(void)
{
    const unsigned brightness[]={164,104,77},peak[]={175,111,83};
    const unsigned low[]={6600,18000,35000},high[]={6800,18500,35000};
    capture_exposure_setting cur,next;
    capture_exposure_stats s;
    unsigned i,j,limited;
    for(i=0;i<3u;++i) {
        cur=(capture_exposure_setting){3478,115};
        s=(capture_exposure_stats){brightness[i],peak[i],peak[i],0,0,0};
        assert(capture_still_plan(&s,&cur,35000,&next,&limited)==1);
        assert(!next.gain_code && next.lines>=low[i] && next.lines<=high[i]);
        assert(limited==(i==2));
        printf("STILL aperture case%u: lines=%u gain=1x limited=%u\n",i,next.lines,limited);
    }
    /* Independent ideal response simulation over illuminances and starting gains.
     * Unclipped starts converge quickly. Clipped starts must decrease exposure;
     * their unknown overexposure cannot promise convergence in three samples. */
    for(i=0;i<5u;++i) for(j=0;j<4u;++j) {
        const double targets[]={100,1000,5000,18000,35000};
        unsigned k;
        cur=(capture_exposure_setting){980,j*256};
        for(k=0;k<3;++k) {
            double y=210.0*pow((cur.lines*1024.0/(1024.0-cur.gain_code))/targets[i],1.0/2.2);
            unsigned lum=(unsigned)(y+0.5),hi=(unsigned)(y*1.055+0.5);
            s=(capture_exposure_stats){lum>255?255:lum,hi>255?255:hi,hi>255?255:hi,hi>=255?10:0,hi>=255?10:0,0};
            assert(capture_still_plan(&s,&cur,35000,&next,&limited)>=0);
            assert(next.lines>=4 && next.lines<=35000 && !next.gain_code);
            if(hi>=255) assert(next.lines<cur.lines*1024.0/(1024.0-cur.gain_code));
            cur=next;
        }
        /* Saturated starting samples discard excess exposure information;
         * convergence remains bounded instead of pretending to infer it. */
        if(targets[i]>=5000 || (targets[i]==1000 && j==0))
            assert(cur.lines>targets[i]*0.85 && cur.lines<targets[i]*1.16);
    }
    cur=(capture_exposure_setting){35000,0};s=(capture_exposure_stats){0,0,0,0,0,0};
    assert(capture_still_plan(&s,&cur,35000,&next,&limited)==0 && limited && !next.gain_code);
    cur=(capture_exposure_setting){5000,0};s=(capture_exposure_stats){50,250,250,0,10,0};
    assert(capture_still_plan(&s,&cur,35000,&next,&limited)==1 && next.lines<cur.lines);
    s=(capture_exposure_stats){40,235,235,0,0,0};
    assert(capture_still_plan(&s,&cur,35000,&next,&limited)==0);
    s=(capture_exposure_stats){210,220,220,0,0,0};
    assert(capture_still_plan(&s,&cur,35000,&cur,&limited)==0 && cur.lines==5000);
    s.luma_p90=256;assert(capture_still_plan(&s,&cur,35000,&next,&limited)==-1);
    s.luma_p90=210;assert(capture_still_plan(&s,&cur,35001,&next,&limited)==-1);
    assert(capture_still_plan(NULL,&cur,35000,&next,&limited)==-1);
    puts("PASS low-ISO policy: real aperture cases, gamma response, fixed unity gain, exposure cap, clipping/colour guard, alias and invalid inputs");
    return 0;
}
