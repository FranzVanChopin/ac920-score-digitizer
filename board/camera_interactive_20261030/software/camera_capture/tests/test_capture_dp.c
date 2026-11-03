/* Run the production backend against a timed DP/DPDMA hardware model.
 * No actual link training, PHY operation or MMIO is proven by this test. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "dp_testenv.h"
#include "../capture_dp.c"

static uint64_t clock_us,next_vsync,drop_at;
static unsigned hpd,link_enabled,no_vsync,no_ack,stuck_bus,link_bad;
static unsigned isr,eisr,hw_tag,pending_tag,train_fails,train_calls;
static unsigned triggers,retriggers,flushes,flushed_tag,services;
static XDpDma_Descriptor descriptors[2];
static unsigned desc_index;
static UINTPTR active_addr,pending_addr;
static uint8_t active_first[4],active_last[4];
static XDpPsu_Config ps_cfg={0x1000};
static XDpDma_Config dm_cfg={0x2000};
static uint8_t canvas[CAPTURE_DP_WIDTH*CAPTURE_DP_HEIGHT*3u];

static void check_active(void)
{
    if(active_addr) {
        const uint8_t *p=(const uint8_t *)active_addr;
        assert(memcmp(p,active_first,4)==0);
        assert(memcmp(p+DP_BYTES-4u,active_last,4)==0);
    }
}
void usleep(unsigned us)
{
    check_active();clock_us+=us;
    if(drop_at && clock_us>=drop_at) hpd=0;
    if(link_enabled && !no_vsync && clock_us>=next_vsync) {
        next_vsync=clock_us+16667u;isr|=XDPDMA_ISR_VSYNC_INT_MASK;
        if(pending_tag && !no_ack) {
            hw_tag=pending_tag;pending_tag=0;active_addr=pending_addr;
            memcpy(active_first,(void *)active_addr,4);
            memcpy(active_last,(uint8_t *)active_addr+DP_BYTES-4u,4);
        }
    }
}
void XTime_GetTime(XTime *t) {*t=clock_us;}
int xil_printf(const char *f,...) {(void)f;return 0;}
static void tick(void *c) {assert(c==&services);++services;}
XDpPsu_Config *XDpPsu_LookupConfig(unsigned id) {(void)id;return &ps_cfg;}
XDpDma_Config *XDpDma_LookupConfig(unsigned id) {(void)id;return &dm_cfg;}
void XDpPsu_CfgInitialize(XDpPsu *p,XDpPsu_Config *c,UINTPTR a) {p->Config=*c;assert(a==c->BaseAddr);}
void XDpDma_CfgInitialize(XDpDma *p,XDpDma_Config *c) {p->Config=*c;p->Gfx.Channel.Current=NULL;}
int XDpPsu_InitializeTx(XDpPsu *p) {(void)p;return 0;}
int XDpPsu_IsConnected(XDpPsu *p) {(void)p;return (int)hpd;}
int XDpPsu_AuxWrite(XDpPsu *p,unsigned a,unsigned n,u8 *d) {(void)p;assert(a==0x600u && n==1u && *d==1);return 0;}
int XDpPsu_GetRxCapabilities(XDpPsu *p) {p->LinkConfig.MaxLaneCount=4;p->LinkConfig.MaxLinkRate=0x14;return 0;}
int XDpPsu_SetEnhancedFrameMode(XDpPsu *p,unsigned v) {(void)p;(void)v;return 0;}
int XDpPsu_SetLaneCount(XDpPsu *p,u8 n) {assert(n==1 || n==2);p->LinkConfig.LaneCount=n;return 0;}
int XDpPsu_SetLinkRate(XDpPsu *p,u8 n) {assert(n==(p->LinkConfig.LaneCount>=2u?6u:10u));p->LinkConfig.LinkRate=n;return 0;}
int XDpPsu_SetDownspread(XDpPsu *p,unsigned n) {(void)p;(void)n;return 0;}
int XDpPsu_EstablishLink(XDpPsu *p) {(void)p;++train_calls;if(train_fails) {--train_fails;return 1;}return 0;}
void XDpPsu_EnableMainLink(XDpPsu *p,unsigned n) {(void)p;link_enabled=n;if(n) next_vsync=clock_us+16667u;}
int XDpPsu_CheckLinkStatus(XDpPsu *p,unsigned n) {(void)p;(void)n;return (int)link_bad;}
void XDpPsu_SetColorEncode(XDpPsu *p,unsigned n) {(void)p;(void)n;}
void XDpPsu_CfgMsaSetBpc(XDpPsu *p,unsigned n) {(void)p;(void)n;}
void XDpPsu_CfgMsaUseStandardVideoMode(XDpPsu *p,unsigned n) {assert(n==XVIDC_VM_1280x720_60_P);p->MsaConfig.PixelClockHz=74250000;}
void XDpPsu_CfgMsaEnSynchClkMode(XDpPsu *p,unsigned n) {(void)p;(void)n;}
void XDpPsu_SetMsaValues(XDpPsu *p) {(void)p;}
void XDpPsu_WriteReg(UINTPTR a,unsigned r,unsigned v) {(void)a;(void)r;(void)v;}
int XDpDma_SetGraphicsFormat(XDpDma *p,unsigned n) {(void)p;assert(n==RGBA8888);return 0;}
void XDpDma_SetQOS(XDpDma *p,unsigned n) {(void)p;assert(n==11);}
void XDpDma_WriteReg(UINTPTR a,unsigned r,unsigned v)
{(void)a;if(r==XDPDMA_ISR) isr&=~v;else if(r==XDPDMA_EISR) eisr&=~v;}
u32 XDpDma_ReadReg(UINTPTR a,unsigned r)
{
    (void)a;
    if(r==XDPDMA_ISR) return isr;
    if(r==XDPDMA_EISR) return eisr;
    if(r==XDPDMA_CH3_STATUS) return stuck_bus?XDPDMA_CH_STATUS_OTRAN_CNT_MASK:0;
    if(r==XDPDMA_CH3_DSCR_ID) return hw_tag;
    return 0;
}
int XDpDma_SetChannelState(XDpDma *p,unsigned c,unsigned state)
{
    (void)p;assert(c==GraphicsChan);
    if(state==XDPDMA_DISABLE) {assert(!stuck_bus);active_addr=0;pending_tag=0;}
    return 0;
}
void XDpDma_DisplayGfxFrameBuffer(XDpDma *p,XDpDma_FrameBuffer *f)
{
    assert((f->Address&255u)==0 && f->Stride==DP_STRIDE && f->Size==DP_BYTES);
    assert(f->Address!=active_addr);p->Gfx.FrameBuffer=f;
}
void XDpDma_SetupChannel(XDpDma *p,unsigned c)
{
    assert(c==GraphicsChan);desc_index^=1u;
    p->Gfx.Channel.Current=&descriptors[desc_index];flushed_tag=0;
}
void Xil_DCacheFlushRange(INTPTR a,size_t n)
{
    ++flushes;
    if(n==sizeof(XDpDma_Descriptor)) flushed_tag=((XDpDma_Descriptor *)a)->DSCR_ID;
    else assert(n==DP_BYTES && (UINTPTR)a!=active_addr);
}
static int trigger(XDpDma *p)
{
    assert(flushed_tag && flushed_tag==p->Gfx.Channel.Current->DSCR_ID);
    pending_tag=flushed_tag;pending_addr=p->Gfx.FrameBuffer->Address;
    return 0;
}
int XDpDma_Trigger(XDpDma *p,unsigned c) {(void)c;++triggers;return trigger(p);}
int XDpDma_ReTrigger(XDpDma *p,unsigned c) {(void)c;++retriggers;return trigger(p);}
void XAVBuf_CfgInitialize(XAVBuf *p,UINTPTR a,unsigned n) {(void)p;(void)a;(void)n;}
int XAVBuf_SetInputNonLiveGraphicsFormat(XAVBuf *p,unsigned n) {(void)p;(void)n;return 0;}
int XAVBuf_SetOutputVideoFormat(XAVBuf *p,unsigned n) {(void)p;(void)n;return 0;}
void XAVBuf_EnableGraphicsBuffers(XAVBuf *p,unsigned n) {(void)p;(void)n;}
void XAVBuf_InputVideoSelect(XAVBuf *p,unsigned a,unsigned b) {(void)p;(void)a;(void)b;}
void XAVBuf_ConfigureGraphicsPipeline(XAVBuf *p) {(void)p;}
void XAVBuf_ConfigureOutputVideo(XAVBuf *p) {(void)p;}
void XAVBuf_SetBlenderAlpha(XAVBuf *p,unsigned a,unsigned b) {(void)p;(void)a;(void)b;}
void XAVBuf_SetAudioVideoClkSrc(XAVBuf *p,unsigned a,unsigned b) {(void)p;(void)a;(void)b;}
void XAVBuf_SoftReset(XAVBuf *p) {(void)p;}
int XAVBuf_SetPixelClock(uint64_t n) {assert(n==74250000u);return 0;}

static void reset_model(void)
{
    initialized=online=locked=front=tag=0;presented=timeouts=retries=vsyncs=0;
    retry_at=last_check=0;clock_us=next_vsync=drop_at=0;
    hpd=link_enabled=no_vsync=no_ack=stuck_bus=link_bad=0;
    isr=eisr=hw_tag=pending_tag=train_fails=train_calls=0;
    triggers=retriggers=flushes=flushed_tag=services=0;active_addr=pending_addr=0;
    memset(&tx,0,sizeof(tx));memset(&engine,0,sizeof(engine));
    memset(canvas,50,sizeof(canvas));
    assert(capture_dp_init(tick,&services)==0);
}
static void frame(unsigned v) {memset(canvas,(int)v,sizeof(canvas));assert(capture_dp_present(canvas,CAPTURE_DP_WIDTH*3u)==0);}
int main(void)
{
    unsigned old_frames,old_flush;
    reset_model();frame(10);assert(presented==0 && clock_us==0 && !train_calls);
    hpd=1;frame(20);assert(presented==1 && triggers==1 && online && services);
    assert(active_first[0]==20 && active_first[3]==255);
    frame(30);assert(presented==2 && retriggers==1 && active_first[0]==30);
    /* CPU can immediately reuse its BGR canvas after the copy/flip. */
    memset(canvas,99,sizeof(canvas));check_active();
    hpd=0;frame(40);assert(!online && !locked);
    hpd=1;usleep(201000);frame(50);assert(presented==3 && retries==2);
    /* Link-status loss drains before retry, independently of the camera. */
    link_bad=1;usleep(2001000);frame(60);assert(!online && !locked);
    link_bad=0;usleep(201000);frame(60);assert(online);
    /* A missing VSYNC or descriptor acknowledgment must not hang. */
    no_vsync=1;old_frames=presented;frame(70);
    assert(timeouts==1 && !online && presented==old_frames && !locked);
    no_vsync=0;usleep(2001000);frame(80);assert(online);
    no_ack=1;old_frames=presented;frame(90);
    assert(timeouts==2 && !online && presented==old_frames);
    no_ack=0;usleep(2001000);frame(90);assert(online);
    /* Hardware DMA errors recover through stop/retrain. */
    eisr=1;frame(100);assert(timeouts==3 && !online);
    usleep(2001000);frame(100);assert(online && !eisr);
    /* Unplug during a flip. */
    drop_at=clock_us+1000u;frame(110);assert(!online && !locked);
    drop_at=0;hpd=1;usleep(2001000);frame(120);assert(online);
    /* If AXI reads cannot drain, neither framebuffer may be reused. */
    stuck_bus=1;hpd=0;frame(130);assert(locked && !online);
    old_flush=flushes;hpd=1;usleep(3000000);frame(140);assert(flushes==old_flush);
    /* One-lane fallback and bounded retries after total training failure. */
    reset_model();hpd=1;train_fails=1;frame(150);
    assert(presented==1 && train_calls==2 && tx.LinkConfig.LaneCount==1);
    reset_model();hpd=1;train_fails=2;frame(150);assert(!online && train_calls==2);
    frame(160);assert(train_calls==2);usleep(2001000);frame(160);assert(online);
    tag=65535u;frame(170);assert(hw_tag==1 && online);
    assert(capture_dp_present(NULL,CAPTURE_DP_WIDTH*3u)==-1);
    puts("PASS DP backend model: copy ownership, flush-before-trigger, acknowledged flips, unplug/retrain, no VSYNC/ACK, DMA errors, drain lock and lane fallback");
    return 0;
}
