#include "../capture_exposure.h"
#include <assert.h>
#include <stdio.h>

static void test_preview_limits(void)
{
    capture_exposure_stats s = {0,0,0,0,0,0};
    capture_exposure_setting cur = {4,0}, next;
    enum capture_exposure_reason why;
    unsigned i;
    for (i = 0; i < 300; ++i) {
        assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&next,&why)>=0);
        assert(next.lines>=4u && next.lines<=1960u && next.gain_code<=768u);
        cur=next;
    }
    assert(cur.lines==1960u && cur.gain_code==768u);
    /* A legal manually chosen exposure above the automatic ceiling is kept. */
    cur=(capture_exposure_setting){2312u,0u};
    s=(capture_exposure_stats){220u,220u,220u,0u,0u,0u};
    assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&next,&why)==0);
    assert(next.lines==2312u && next.gain_code==0u);
    s=(capture_exposure_stats){0u,0u,0u,0u,0u,0u};
    assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&next,&why)==1);
    assert(next.lines==2312u && next.gain_code>0u);
    s=(capture_exposure_stats){10u,255u,255u,10u,10u,0u};
    for (i = 0; i < 100; ++i) {
        assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&next,&why)>=0);
        assert(next.lines>=4u && next.lines<=2312u && next.gain_code<=768u);
        cur=next;
    }
    assert(cur.lines==4u && cur.gain_code==0u);
    s=(capture_exposure_stats){0u,0u,0u,0u,0u,0u};
    for (i = 0; i < 300; ++i) {
        assert(capture_exposure_next_limits(&s,&cur,2312u,2312u,&next,&why)>=0);
        assert(next.lines>=4u && next.lines<=2312u);
        cur=next;
    }
    assert(cur.lines==2312u && cur.gain_code==768u);
    cur.lines=2313u;
    assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&next,&why)==-1);
    cur=(capture_exposure_setting){4u,0u};
    assert(capture_exposure_next_limits(&s,&cur,2312u,2313u,&next,&why)==-1);
    assert(capture_exposure_next_limits(&s,&cur,2312u,3u,&next,&why)==-1);
    assert(capture_exposure_next_limits(&s,&cur,3u,4u,&next,&why)==-1);
    assert(capture_exposure_next_limits(&s,&cur,65536u,2800u,&next,&why)==-1);
    assert(capture_exposure_next_limits(&s,&cur,4u,4u,&next,&why)==1);
    assert(next.lines==4u && next.gain_code>0u);
    cur=(capture_exposure_setting){65000u,0u};
    assert(capture_exposure_next_limits(&s,&cur,65535u,65535u,&next,&why)==1);
    assert(next.lines==65535u);
    /* Aliased current/next remains supported by the mode-aware interface. */
    cur=(capture_exposure_setting){100u,0u};
    assert(capture_exposure_next_limits(&s,&cur,2312u,1960u,&cur,&why)==1);
    assert(cur.lines==113u);
}

int main(void)
{
    capture_exposure_stats s={25,255,255,800,900,700};
    capture_exposure_setting cur={1200,0}, next={0,0};
    enum capture_exposure_reason why;
    unsigned i;
    /* Low-luma pure blue with saturated B must reduce, not raise, exposure. */
    assert(capture_exposure_next(&s,&cur,&next,&why)==1);
    assert(next.lines==900 && next.gain_code==0 && why==CAPTURE_AE_SATURATION);
    /* Sub-percent clipped highlights also take priority over dark average. */
    s=(capture_exposure_stats){80,190,220,5,0,0};
    assert(capture_exposure_next(&s,&cur,&next,&why)==1 && next.lines<cur.lines);
    s=(capture_exposure_stats){80,190,220,0,0,5};
    assert(capture_exposure_next(&s,&cur,&next,&why)==1 && next.lines<cur.lines);
    /* Preserve saturated colours at their high-channel target, even though
     * their luma can never reach a neutral-white exposure target. */
    s=(capture_exposure_stats){60,220,240,0,0,0};
    assert(capture_exposure_next(&s,&cur,&next,&why)==0 && why==CAPTURE_AE_HOLD);
    s=(capture_exposure_stats){220,210,223,0,0,0};
    assert(capture_exposure_next(&s,&cur,&next,&why)==0);
    s=(capture_exposure_stats){180,180,190,0,0,0};
    assert(capture_exposure_next(&s,&cur,&next,&why)==1 && next.lines>cur.lines);
    /* Gain is reduced before shutter time and remains in the valid range. */
    s=(capture_exposure_stats){30,255,255,1000,1000,1000}; cur.gain_code=768;
    assert(capture_exposure_next(&s,&cur,&next,&why)==1);
    assert(next.lines==cur.lines && next.gain_code<cur.gain_code);
    for(i=0;i<100;i++) { cur=next; assert(capture_exposure_next(&s,&cur,&next,&why)>=0); }
    assert(next.lines==4 && next.gain_code==0);
    s=(capture_exposure_stats){0,0,0,0,0,0};
    for(i=0;i<300;i++) { cur=next; assert(capture_exposure_next(&s,&cur,&next,&why)>=0); }
    assert(next.lines==2800 && next.gain_code==768);
    s.raw_clipped_permille=1001;
    assert(capture_exposure_next(&s,&cur,&next,&why)==-1);
    assert(capture_exposure_next(NULL,&cur,&next,&why)==-1);
    test_preview_limits();
    for(i=0;i<=768u;++i) {
        unsigned lines;
        for(lines=4;lines<=3478u;lines+=37u) {
            double before,after;
            cur=(capture_exposure_setting){lines,i};
            assert(capture_exposure_low_noise(&cur,3478u,&next)>=0);
            assert(next.lines>=cur.lines && next.lines<=3478u && next.gain_code<=cur.gain_code);
            before=(double)cur.lines/(1024u-cur.gain_code);
            after=(double)next.lines/(1024u-next.gain_code);
            /* Up to 1/2 shutter line rounding at very short exposures. */
            assert(after/before>0.89 && after/before<1.13);
            if(lines>=100u) assert(after/before>0.995 && after/before<1.005);
        }
    }
    cur=(capture_exposure_setting){1156,768};
    assert(capture_exposure_low_noise(&cur,3478,&cur)==1);
    assert(cur.lines==3478 && cur.gain_code==254);
    assert(capture_exposure_low_noise(NULL,3478,&cur)==-1);
    puts("PASS low-noise exposure: gain/shutter limits, exposure product, alias, 4x to 1.33x reference");
    puts("PASS exposure: clipped channels override luma, colour highlights hold, bounded gain/shutter convergence");
    return 0;
}
