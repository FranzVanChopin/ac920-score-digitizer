#include "capture_panel.h"
#include <assert.h>
#include <stdio.h>
int main(void)
{
    capture_panel p;uint32_t released=0x300u;
    capture_panel_init(&p,0,released);
    assert(!capture_panel_update(&p,100,0x200));
    assert(!capture_panel_update(&p,110,released)); /* bounce */
    assert(!capture_panel_update(&p,115,0x200));
    assert(!capture_panel_update(&p,145,0x200));
    assert(!capture_panel_update(&p,200,released));
    assert(capture_panel_update(&p,230,released)==PANEL_FOCUS_NEXT);
    assert(!capture_panel_update(&p,300,released));
    assert(!capture_panel_update(&p,400,0x100));
    assert(!capture_panel_update(&p,430,0x100));
    assert(!capture_panel_update(&p,10000,0x100)); /* hold never repeats */
    assert(!capture_panel_update(&p,10010,released));
    assert(capture_panel_update(&p,10040,released)==PANEL_UBUNTU);
    capture_panel_init(&p,20000,0); /* held across boot/busy interval */
    assert(!capture_panel_update(&p,20100,released));
    assert(!capture_panel_update(&p,20130,released));
    assert(!capture_panel_update(&p,20200,0));
    assert(!capture_panel_update(&p,20230,0));
    assert(!capture_panel_update(&p,20300,released));
    assert(capture_panel_update(&p,20330,released)==(PANEL_FOCUS_NEXT|PANEL_UBUNTU));
    /* Hold crosses 1500ms once; releasing it must never select another ROI. */
    capture_panel_init(&p,0,released);
    assert(!capture_panel_update(&p,100,0x200));
    assert(!capture_panel_update(&p,130,0x200));
    assert(!capture_panel_update(&p,1599,0x200));
    assert(capture_panel_update(&p,1600,0x200)==PANEL_FOCUS_TOGGLE);
    assert(!capture_panel_update(&p,1800,0x200));
    assert(!capture_panel_update(&p,2000,released));
    assert(!capture_panel_update(&p,2030,released));
    /* Mode switch re-arms while S2 is still down: its release is suppressed. */
    capture_panel_init(&p,1700,0x200);
    assert(!capture_panel_update(&p,1800,0x200));
    assert(!capture_panel_update(&p,4000,0x200));
    assert(!capture_panel_update(&p,4010,released));
    assert(!capture_panel_update(&p,4040,released));
    /* Delayed polling at release still classifies the completed hold. */
    capture_panel_init(&p,0,released);
    assert(!capture_panel_update(&p,100,0x200));
    assert(!capture_panel_update(&p,130,0x200));
    assert(!capture_panel_update(&p,1650,released));
    assert(capture_panel_update(&p,1680,released)==PANEL_FOCUS_TOGGLE);
    /* A 1499ms press stays short, even when debounce ends after 1500ms. */
    capture_panel_init(&p,0,released);
    assert(!capture_panel_update(&p,100,0x200));
    assert(!capture_panel_update(&p,130,0x200));
    assert(!capture_panel_update(&p,1599,released));
    assert(capture_panel_update(&p,1629,released)==PANEL_FOCUS_NEXT);
    {
        unsigned toggle=0,pending=0,i;
        const unsigned expected[]={2,3,4,5,1,2};
        capture_panel_focus_command(PANEL_FOCUS_NEXT,0,0,&toggle,&pending);
        assert(!toggle && !pending); /* short press outside focus is idle */
        capture_panel_focus_command(PANEL_FOCUS_TOGGLE,0,0,&toggle,&pending);
        assert(toggle==1 && !pending); /* enter */
        toggle=0;
        for(i=0;i<6;++i) {
            capture_panel_focus_command(PANEL_FOCUS_NEXT,1,0,&toggle,&pending);
            assert(pending==expected[i] && !toggle); /* queued rapid taps */
        }
        pending=0;
        capture_panel_focus_command(PANEL_FOCUS_NEXT,1,4,&toggle,&pending);
        assert(pending==1); /* actual displayed region5 wraps to1 */
        capture_panel_focus_command(PANEL_UBUNTU,1,0,&toggle,&pending);
        assert(!toggle && pending==1);
        capture_panel_focus_command(PANEL_FOCUS_TOGGLE,1,0,&toggle,&pending);
        assert(toggle==1); /* exit */
    }
    {
        uint32_t base=UINT32_MAX-1000u;
        capture_panel_init(&p,base,released);
        assert(!capture_panel_update(&p,base+10u,0x200));
        assert(!capture_panel_update(&p,base+40u,0x200));
        assert(capture_panel_update(&p,base+1510u,0x200)==PANEL_FOCUS_TOGGLE);
    }
    puts("PASS S2/S3: debounce; S2 hold1500ms toggle, short ROI1..5 wrap, standby idle, queued taps, no release double-action; S3 unchanged");
    return 0;
}
