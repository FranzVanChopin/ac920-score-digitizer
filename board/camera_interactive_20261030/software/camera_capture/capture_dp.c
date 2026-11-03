/* SPDX-License-Identifier: MIT
 * AC920 memory-framebuffer DP backend. Configuration follows installed AMD
 * embeddedsw 2024.2 dpdma_v1_6/examples and dppsu_v1_7 polling example.
 * VSYNC is polled; camera application's exception/GIC setup is unchanged.
 */
#include "capture_dp.h"
#include "dp_pixels.h"
#ifdef CAPTURE_DP_HOST_TEST
#include "dp_testenv.h"
#else
#include "xparameters.h"
#include "xdppsu.h"
#include "xdpdma.h"
#include "xavbuf.h"
#include "xavbuf_clk.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xtime_l.h"
#include "sleep.h"
#endif
#include <string.h>

#define DP_STRIDE (CAPTURE_DP_WIDTH*4u)
#define DP_BYTES (CAPTURE_DP_HEIGHT*DP_STRIDE)
#define RETRY_MS 2000u
#define FLIP_TIMEOUT_MS 160u

static XDpPsu tx;
static XDpDma engine;
static XAVBuf av;
static XDpDma_Config *dma_cfg;
static XDpDma_FrameBuffer planes[2];
static uint8_t pixels[2][DP_BYTES] __attribute__((aligned(256)));
static void (*service_callback)(void *);
static void *service_context;
static unsigned initialized,online,locked,front,tag;
static unsigned presented,timeouts,retries,vsyncs;
static uint32_t retry_at,last_check;

static uint32_t now_ms(void)
{
    XTime value;XTime_GetTime(&value);
    return (uint32_t)(value/(COUNTS_PER_SECOND/1000u));
}
static void service(void) { if(service_callback) service_callback(service_context); }
static int connected(void) { return XDpPsu_IsConnected(&tx)!=0; }

void capture_dp_status(void)
{
    if(!initialized) { xil_printf("[DP] not initialized\r\n");return; }
    xil_printf("[DP] HPD=%u online=%u locked=%u frames=%u vsync=%u retries=%u timeouts=%u lanes=%u rate=%02x\r\n",
        (unsigned)connected(),online,locked,presented,vsyncs,retries,timeouts,
        tx.LinkConfig.LaneCount,tx.LinkConfig.LinkRate);
    xil_printf("[DP DMA] ISR=%08x EISR=%08x CH3=%08x ID=%08x payload=%08x:%08x\r\n",
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_ISR),
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_EISR),
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_STATUS),
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_DSCR_ID),
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_PYLD_CUR_ADDRE),
        XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_PYLD_CUR_ADDR));
}

/* Never recycle descriptors/framebuffers until outstanding DMA reads drain. */
static int stop_output(void)
{
    uint32_t start=now_ms();
    XDpPsu_EnableMainLink(&tx,0);
    if(XDpDma_SetChannelState(&engine,GraphicsChan,XDPDMA_PAUSE)!=XST_SUCCESS) goto failed;
    while(XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_STATUS)&XDPDMA_CH_STATUS_OTRAN_CNT_MASK) {
        if(now_ms()-start>40u) goto failed;
        service();usleep(100);
    }
    if(XDpDma_SetChannelState(&engine,GraphicsChan,XDPDMA_DISABLE)!=XST_SUCCESS) goto failed;
    online=0;
    return 0;
failed:
    /* Leave both buffers immutable when bus completion cannot be established. */
    locked=1;online=0;
    xil_printf("[DP] DMA could not drain; display buffers locked until reboot. Capture remains available.\r\n");
    return -1;
}

static int train(void)
{
    u8 power=1;
    unsigned attempt,max_lanes;
    u32 rc;
    if(!connected()) return 0;
    ++retries;
    xil_printf("[DP] HPD=1; training 1280x720 RGB8\r\n");
    (void)XDpPsu_AuxWrite(&tx,XDPPSU_DPCD_SET_POWER_DP_PWR_VOLTAGE,1,&power);
    rc=XDpPsu_AuxWrite(&tx,XDPPSU_DPCD_SET_POWER_DP_PWR_VOLTAGE,1,&power);
    if(rc!=XST_SUCCESS) { xil_printf("[DP] AUX wake failed rc=%u\r\n",rc);return 0; }
    rc=XDpPsu_GetRxCapabilities(&tx);
    if(rc!=XST_SUCCESS) { xil_printf("[DP] RX capabilities failed rc=%u\r\n",rc);return 0; }
    max_lanes=tx.LinkConfig.MaxLaneCount>=2u?2u:1u;
    xil_printf("[DP] sink max_lanes=%u max_rate=%02x; AC920 limit=2 lanes\r\n",
        tx.LinkConfig.MaxLaneCount,tx.LinkConfig.MaxLinkRate);
    for(attempt=0;attempt<2u;++attempt) {
        unsigned lanes=attempt?1u:max_lanes;
        if(!connected()) return 0;
        XDpPsu_EnableMainLink(&tx,0);
        rc=XDpPsu_SetEnhancedFrameMode(&tx,tx.LinkConfig.SupportEnhancedFramingMode?1u:0u);
        if(rc==XST_SUCCESS) rc=XDpPsu_SetLaneCount(&tx,(u8)lanes);
        /* 720p60 RGB8 needs two RBR lanes, or one HBR lane. */
        if(rc==XST_SUCCESS) rc=XDpPsu_SetLinkRate(&tx,lanes>=2u?0x06u:0x0au);
        if(rc==XST_SUCCESS) rc=XDpPsu_SetDownspread(&tx,tx.LinkConfig.SupportDownspreadControl);
        if(rc==XST_SUCCESS) rc=XDpPsu_EstablishLink(&tx);
        xil_printf("[DP] train attempt=%u lanes=%u rate=%02x rc=%u\r\n",
            attempt+1u,tx.LinkConfig.LaneCount,tx.LinkConfig.LinkRate,rc);
        if(rc==XST_SUCCESS) break;
        service();
    }
    if(rc!=XST_SUCCESS) return 0;
    /* Reset the DMA software descriptors only after stop_output() drained it. */
    XDpDma_CfgInitialize(&engine,dma_cfg);
    if(XDpDma_SetGraphicsFormat(&engine,RGBA8888)!=XST_SUCCESS) return 0;
    XDpDma_SetQOS(&engine,11u);
    XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_IDS,0xffffffffu);
    XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_ISR,0xffffffffu);
    XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_EISR,0xffffffffu);
    XDpPsu_SetColorEncode(&tx,XDPPSU_CENC_RGB);
    XDpPsu_CfgMsaSetBpc(&tx,XVIDC_BPC_8);
    XDpPsu_CfgMsaUseStandardVideoMode(&tx,XVIDC_VM_1280x720_60_P);
    rc=(u32)XAVBuf_SetPixelClock(tx.MsaConfig.PixelClockHz);
    if(rc!=XST_SUCCESS) { xil_printf("[DP] pixel clock failed rc=%u\r\n",rc);return 0; }
    XDpPsu_WriteReg(tx.Config.BaseAddr,XDPPSU_SOFT_RESET,1);
    usleep(10);
    XDpPsu_WriteReg(tx.Config.BaseAddr,XDPPSU_SOFT_RESET,0);
    XDpPsu_SetMsaValues(&tx);
    /* AMD example's AV buffer reset after updating stream timing. */
    XAVBuf_SoftReset(&av);
    XDpPsu_EnableMainLink(&tx,1);
    online=1;last_check=now_ms();
    xil_printf("[DP] link trained; 720P pixel_clock=%u Hz; DPDMA RGBA8888 stride=%u\r\n",
        tx.MsaConfig.PixelClockHz,DP_STRIDE);
    return 1;
}

int capture_dp_shutdown(void)
{
    if(!initialized) return 0;
    if(locked) return -1;
    return stop_output();
}

int capture_dp_init(void (*callback)(void *),void *context)
{
    XDpPsu_Config *cfg=XDpPsu_LookupConfig(XPAR_XDPPSU_0_DEVICE_ID);
    u32 rc;
    unsigned i;
    service_callback=callback;service_context=context;
    dma_cfg=XDpDma_LookupConfig(XPAR_XDPDMA_0_DEVICE_ID);
    if(!cfg || !dma_cfg) return -1;
    XDpPsu_CfgInitialize(&tx,cfg,cfg->BaseAddr);
    XAVBuf_CfgInitialize(&av,cfg->BaseAddr,XPAR_XDPPSU_0_DEVICE_ID);
    XDpDma_CfgInitialize(&engine,dma_cfg);
    rc=XDpPsu_InitializeTx(&tx);
    if(rc!=XST_SUCCESS) { xil_printf("[DP] InitializeTx failed rc=%u\r\n",rc);return -1; }
    if(XDpDma_SetGraphicsFormat(&engine,RGBA8888)!=XST_SUCCESS ||
       XAVBuf_SetInputNonLiveGraphicsFormat(&av,RGBA8888)!=XST_SUCCESS ||
       XAVBuf_SetOutputVideoFormat(&av,RGB_8BPC)!=XST_SUCCESS) return -1;
    XDpDma_SetQOS(&engine,11u);
    XAVBuf_EnableGraphicsBuffers(&av,1);
    XAVBuf_InputVideoSelect(&av,XAVBUF_VIDSTREAM1_NONE,XAVBUF_VIDSTREAM2_NONLIVE_GFX);
    XAVBuf_ConfigureGraphicsPipeline(&av);
    XAVBuf_ConfigureOutputVideo(&av);
    XAVBuf_SetBlenderAlpha(&av,0,0);
    XDpPsu_CfgMsaEnSynchClkMode(&tx,0);
    XAVBuf_SetAudioVideoClkSrc(&av,XAVBUF_PS_CLK,XAVBUF_PS_CLK);
    XAVBuf_SoftReset(&av);
    XDpPsu_WriteReg(tx.Config.BaseAddr,XDPPSU_INTR_DIS,0xffffffffu);
    XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_IDS,0xffffffffu);
    for(i=0;i<2u;++i) {
        planes[i].Address=(UINTPTR)pixels[i];planes[i].Stride=DP_STRIDE;
        planes[i].LineSize=DP_STRIDE;planes[i].Size=DP_BYTES;
    }
    initialized=1;
    if(stop_output()) return -1;
    retry_at=now_ms();
    xil_printf("[DP] controller ready; HPD=%u; no monitor does not block camera capture\r\n",(unsigned)connected());
    return 0;
}

int capture_dp_present(const uint8_t *bgr,size_t stride)
{
    unsigned back,issued=0,ack=0;
    uint32_t start,irq;
    unsigned wanted;
    if(!initialized || locked) return 0;
    if(online && !connected()) {
        xil_printf("[DP] unplugged\r\n");
        if(stop_output()) return 0;
        retry_at=now_ms()+200u;
    }
    if(online && now_ms()-last_check>=2000u) {
        last_check=now_ms();
        if(XDpPsu_CheckLinkStatus(&tx,tx.LinkConfig.LaneCount)!=XST_SUCCESS) {
            xil_printf("[DP] link status lost; scheduling retrain\r\n");
            if(stop_output()) return 0;
            retry_at=now_ms()+200u;
        }
    }
    if(!online) {
        if(!connected() || (int32_t)(now_ms()-retry_at)<0) return 0;
        retry_at=now_ms()+RETRY_MS;
        if(!train()) return 0;
    }
    back=front^1u;
    if(dp_pixels_from_bgr(bgr,CAPTURE_DP_WIDTH,CAPTURE_DP_HEIGHT,stride,pixels[back],DP_STRIDE)) return -1;
    Xil_DCacheFlushRange((INTPTR)pixels[back],DP_BYTES);
    wanted=tag%65535u+1u;tag=wanted;
    XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_ISR,XDPDMA_ISR_VSYNC_INT_MASK);
    XDpDma_DisplayGfxFrameBuffer(&engine,&planes[back]);
    start=now_ms();
    while(now_ms()-start<FLIP_TIMEOUT_MS) {
        if(!connected()) break;
        if(XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_EISR)) break;
        irq=XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_ISR);
        if(irq&XDPDMA_ISR_VSYNC_INT_MASK) {
            ++vsyncs;
            XDpDma_WriteReg(engine.Config.BaseAddr,XDPDMA_ISR,XDPDMA_ISR_VSYNC_INT_MASK);
            if(ack) {
                /* One further frame boundary after hardware consumed our tag:
                 * only now is the previous front reusable by the CPU. */
                front=back;++presented;
                if(presented==1u || presented%300u==0u) capture_dp_status();
                return 0;
            }
            if(!issued) {
                unsigned first=engine.Gfx.Channel.Current==NULL;
                XDpDma_SetupChannel(&engine,GraphicsChan);
                engine.Gfx.Channel.Current->DSCR_ID=wanted;
                /* The SDK builds cached descriptors but does not flush them.
                 * Flush BEFORE trigger, not after submitting DMA work. */
                Xil_DCacheFlushRange((INTPTR)engine.Gfx.Channel.Current,sizeof(XDpDma_Descriptor));
                if(first) {
                    if(XDpDma_SetChannelState(&engine,GraphicsChan,XDPDMA_ENABLE)!=XST_SUCCESS) break;
                    if(XDpDma_Trigger(&engine,GraphicsChan)!=XST_SUCCESS) break;
                } else if(XDpDma_ReTrigger(&engine,GraphicsChan)!=XST_SUCCESS) break;
                issued=1;
            }
        }
        if(issued && (XDpDma_ReadReg(engine.Config.BaseAddr,XDPDMA_CH3_DSCR_ID)&0xffffu)==wanted) ack=1;
        service();usleep(100);
    }
    ++timeouts;
    xil_printf("[DP] flip timeout/error: issued=%u ack=%u requested_id=%u; capture continues\r\n",issued,ack,wanted);
    capture_dp_status();
    (void)stop_output();retry_at=now_ms()+RETRY_MS;
    return 0;
}
