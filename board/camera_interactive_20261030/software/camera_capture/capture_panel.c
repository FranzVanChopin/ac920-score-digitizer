#include "capture_panel.h"
void capture_panel_init(capture_panel *p,uint32_t now,uint32_t status)
{
    capture_button_init(&p->s2,now,!(status&(1u<<8)),30,PANEL_S2_LONG_MS);
    capture_button_init(&p->s3,now,!(status&(1u<<9)),30,0);
}
unsigned capture_panel_update(capture_panel *p,uint32_t now,uint32_t status)
{
    unsigned events=0;
    capture_button_event s2=capture_button_update(&p->s2,now,!(status&(1u<<8)));
    if(s2==CAPTURE_BUTTON_LONG) events|=PANEL_FOCUS_TOGGLE;
    if(s2==CAPTURE_BUTTON_PHOTO) events|=PANEL_FOCUS_NEXT;
    if(capture_button_update(&p->s3,now,!(status&(1u<<9)))==CAPTURE_BUTTON_PHOTO) events|=PANEL_UBUNTU;
    return events;
}
void capture_panel_focus_command(unsigned events,int active,unsigned current,
                                 unsigned *toggle,unsigned *pending)
{
    if(events&PANEL_FOCUS_TOGGLE) *toggle^=1u;
    if(active && (events&PANEL_FOCUS_NEXT))
        *pending=(*pending?*pending:current+1u)%5u+1u;
}
