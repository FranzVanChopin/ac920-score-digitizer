/* SPDX-License-Identifier: GPL-2.0-only */
/* AC920, A53 standalone: IMX477 -> CSI/ISP -> DDR -> PS DP and SD BMP.
 * This application belongs to camera_capture.xsa, not score_system.xsa.
 * No Linux, filesystem formatter or OMR engine is started by this image. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xil_printf.h"
#include "xil_assert.h"
#include "xtime_l.h"
#include "sleep.h"
#include "xgpio.h"
#include "xgpiops.h"
#include "xaxivdma.h"
#include "xaxivdma_hw.h"
#include "xcsiss.h"
#include "xcsi_hw.h"
#include "xv_demosaic.h"
#include "xv_gamma_lut.h"
#include "xsdps.h"
#include "xuartps_hw.h"
#include "imx477.h"
#include "platform_i2c.h"
#include "capture_store.h"
#include "capture_pair.h"
#include "capture_button.h"
#include "capture_fatfs.h"
#include "preview.h"
#include "capture_dp.h"
#include "capture_panel.h"
#include "capture_color.h"
#include "capture_exposure.h"
#include "capture_settle.h"
#include "capture_binary.h"
#include "capture_focus.h"
#include "capture_denoise.h"
#include "capture_still.h"

#define CAP_W 4056u
#define CAP_H 3040u
#define CAP_STRIDE 12224u /* align rows to cache lines; HSIZE is still 12168 */
#define DISP_W 640u
#define DISP_H 480u
#define PREVIEW_TOP 48u /* Reserve a status bar; preserve the full camera FOV. */
#define DISP_STRIDE (DISP_W*3u)
#define PS_KEY_PIN 44u
#define SD1_BASE ((UINTPTR)0xff170000u)
#define UART_BASE ((UINTPTR)0xff010000u) /* AC920 UART1 MIO32/33 */
#define GPIO_ENABLE 1u
#define GPIO_REQUEST 2u
#define GPIO_ABORT 4u
#define GPIO_PREVIEW 8u
#define GPIO_ISP_RESET 16u
#define STATUS_ISP_RESET 32u
#define STATUS_PREVIEW 64u
#define PHOTO_PAIR 1
#define ACTION_NEXT_SONG 2
#define S1_LONG_MS 1500u
#define STORE_COLOR 0u
#define STORE_GREY 1u
#define STORE_META 2u
/* PG020 v6.3 includes EOLLateErr at bit 15; the BSP's legacy mask omits it. */
#define S2MM_ERROR_MASK (XAXIVDMA_SR_ERR_ALL_MASK | 0x00008000u)
#define CSI_BAD_FRAME (XCSI_ISR_WC_MASK | XCSI_ISR_ILC_MASK | \
    XCSI_ISR_SLBF_MASK | XCSI_ISR_SOTERR_MASK | XCSI_ISR_SOTSYNCERR_MASK | \
    XCSI_ISR_ECC2BERR_MASK | XCSI_ISR_CRCERR_MASK | \
    XCSI_ISR_VC0FSYNCERR_MASK | XCSI_ISR_VC0FLVLERR_MASK)

/* Two software-owned buffers: one immutable CPU frame and one one-shot DMA
 * destination. A slow CPU drops incoming frames; DMA never wraps onto its frame. */
static uint8_t camera_buffers[2][CAP_H*CAP_STRIDE] __attribute__((aligned(64)));
static uint8_t *camera_frame=camera_buffers[0];
static uint8_t *capture_destination=camera_buffers[1];
static unsigned camera_buffer_index, pending_buffer_index;
static unsigned capture_width=2028u, capture_height=1520u, capture_stride=6144u;
static enum imx477_mode_id current_mode=IMX477_MODE_2028X1520_RAW12_30FPS;
static int capture_inflight, pending_stable;
static uint32_t pending_started_at, pending_old_done;
static unsigned pending_exposure, pending_gain, pending_wb[3];
static uint8_t binary_workspace[CAPTURE_BINARY_WORKSPACE_BYTES] __attribute__((aligned(64)));
static capture_binary_result photo_crop;
static uint8_t display_frames[2][DISP_H][DISP_STRIDE] __attribute__((aligned(64)));
static uint8_t bmp_row[CAP_W*3] __attribute__((aligned(64)));
static capture_denoise_workspace denoise_workspace;
static int denoise_enabled=1;
static XGpio control;
static XGpioPs keys;
static XIicPs iic;
static XCsiSs csi;
static XV_demosaic demosaic;
static XV_gamma_lut gamma_ip;
static XAxiVdma dma;
static struct imx477 sensor;
static capture_fatfs fatfs;
static capture_store stores[3];
static const char *const output_directories[3]={"0:/Image","0:/Grey","0:/Meta"};
static capture_button key_state;
static capture_panel panel;
static int pl_keys_ready,onecard_controls_ready,ubuntu_pending;
static capture_color color_pipeline;
static capture_color_stats color_stats;
static capture_settle sensor_settle;
static uint32_t gpio_value;
static unsigned front_index, frame_number;
static int photo_pending, controls_enabled, auto_exposure=1;
static int sensor_ready, have_frame, sd_ready, sensor_control_error;
static const char *ui_status="STARTING";
static uint32_t status_until;
static unsigned last_brightness;
static unsigned preview_color_ms, manual_lines_pending;
static unsigned capture_ms, statistics_ms, preview_ms, preview_fps_x10;
static uint32_t perf_window_start, perf_last_log;
static unsigned perf_window_frames;
static XTime photo_color_ticks;
static int frame_controls_stable;
static unsigned frame_exposure_lines, frame_gain_code, frame_wb_q8[3];
static int white_balance_pending;
static uint32_t exposure_changed_at;
static unsigned capture_attempt;
static const char *init_stage="entry";
static capture_focus_workspace focus_workspace;
static capture_focus_result focus_result, focus_five[5];
static capture_focus_history focus_history;
static uint8_t focus_mosaic[DISP_H][DISP_STRIDE];
static struct imx477 focus_preview_settings;
static int focus_active, focus_saved_ae, focus_grid;
static unsigned focus_region, focus_zoom=1u;
/* Callbacks only queue commands. Apply them to a completed frame at the main
 * loop boundary, never midway through a metric, render or DMA transfer. */
static unsigned focus_toggle_pending, focus_region_pending, focus_zoom_pending;
static unsigned focus_reset_pending, focus_grid_pending;
static uint32_t focus_edges_at, focus_log_at;
static unsigned focus_metric_ms, focus_edge_frame, focus_grid_frame;

static void focus_reset(void)
{
    capture_focus_history_reset(&focus_history);
    memset(&focus_result,0,sizeof(focus_result));
    focus_edges_at=focus_edge_frame=0;
    focus_grid=0;
}

/* A single PL MMIO access can wait forever when its AXI fabric is in reset.
 * Print BEFORE each initialization stage; a software deadline cannot recover
 * a CPU stalled inside that bus access. These checkpoints also identify a
 * driver assertion without enabling capture before a valid complete frame. */
static void startup_stage(const char *stage)
{
    init_stage=stage;
    xil_printf("[INIT] %s\r\n",stage);
}

static void startup_assert(const char8 *file,s32 line)
{
    xil_printf("FATAL ASSERT: stage=%s file=%s line=%d\r\n",init_stage,file,(int)line);
    /* Keep the BSP's fail-stop policy; never continue with an invalid driver. */
}

static uint32_t milliseconds(void)
{
    XTime t;
    XTime_GetTime(&t);
    return (uint32_t)(t / (COUNTS_PER_SECOND/1000u));
}

/* Read all status before printing. Never read the short-packet FIFO (0x30),
 * acknowledge interrupts, stop DMA or inspect the in-flight frame buffer here.
 * Snapshot registers can change between MMIO reads; they are not atomic. */
static void capture_diagnostics(const char *reason,uint32_t elapsed,uint32_t old_done)
{
    UINTPTR rx=dma.BaseAddr+XAXIVDMA_RX_OFFSET;
    UINTPTR video=dma.BaseAddr+XAXIVDMA_S2MM_ADDR_OFFSET;
    UINTPTR cb=csi.CsiPtr->Config.BaseAddr;
    uint32_t gate=XGpio_DiscreteRead(&control,2);
    uint32_t cr=Xil_In32(rx+XAXIVDMA_CR_OFFSET);
    uint32_t sr=Xil_In32(rx+XAXIVDMA_SR_OFFSET);
    uint32_t park=Xil_In32(dma.BaseAddr+XAXIVDMA_PARKPTR_OFFSET);
    uint32_t vs=Xil_In32(video+XAXIVDMA_VSIZE_OFFSET);
    uint32_t hs=Xil_In32(video+XAXIVDMA_HSIZE_OFFSET);
    uint32_t stride=Xil_In32(video+XAXIVDMA_STRD_FRMDLY_OFFSET);
    uint32_t addr=Xil_In32(video+XAXIVDMA_START_ADDR_OFFSET);
    uint32_t ccr=Xil_In32(cb+XCSI_CCR_OFFSET);
    uint32_t pcr=Xil_In32(cb+XCSI_PCR_OFFSET);
    uint32_t csr=Xil_In32(cb+XCSI_CSR_OFFSET);
    uint32_t isr=Xil_In32(cb+XCSI_ISR_OFFSET);
    uint32_t clk=Xil_In32(cb+XCSI_CLKINFR_OFFSET);
    uint32_t lane0=Xil_In32(cb+XCSI_L0INFR_OFFSET);
    uint32_t lane1=Xil_In32(cb+XCSI_L1INFR_OFFSET);
    uint32_t vc1=Xil_In32(cb+XCSI_VC0INF1R_OFFSET);
    uint32_t vc2=Xil_In32(cb+XCSI_VC0INF2R_OFFSET);
    xil_printf("[CAP %u] %s t=%u ms gate=%02x old_done=%u gpio=%02x\r\n",
               capture_attempt,reason,elapsed,gate,old_done,gpio_value);
    xil_printf("  gate: done=%u busy=%u error=%u armed=%u\r\n",
               gate&1u,(gate>>1)&1u,(gate>>2)&1u,(gate>>3)&1u);
    xil_printf("  S2MM CR=%08x SR=%08x PARK=%08x\r\n",cr,sr,park);
    xil_printf("  run=%u frame_count_en=%u threshold=%u count_status=%u halted=%u frame_irq=%u errors=%08x\r\n",
               cr&1u,(cr>>4)&1u,(cr>>16)&255u,(sr>>16)&255u,sr&1u,
               (sr>>12)&1u,sr&S2MM_ERROR_MASK);
    xil_printf("  S2MM VSIZE=%u HSIZE=%u STRIDE=%u START0=%08x expected=%08x\r\n",
               vs,hs,stride&65535u,addr,(unsigned)(UINTPTR)capture_destination);
    xil_printf("  CSI CCR=%08x PCR=%08x CSR=%08x ISR=%08x frame_received=%u bad=%08x\r\n",
               ccr,pcr,csr,isr,(isr>>31)&1u,isr&CSI_BAD_FRAME);
    xil_printf("  CSI CLK=%08x L0=%08x L1=%08x VC0_INFO1=%08x INFO2=%08x\r\n",
               clk,lane0,lane1,vc1,vc2);
    xil_printf("  CSI snapshot: packets=%u lines=%u last_word_bytes=%u datatype=%02x\r\n",
               csr>>16,vc1>>16,vc1&65535u,vc2&63u);
}

static void sensor_diagnostics(const struct imx477_io *io)
{
    static const uint16_t regs[]={0x0100,0x0112,0x0113,0x0114,
        0x034c,0x034d,0x034e,0x034f,0x0340,0x0341,0x0342,0x0343};
    uint8_t values[sizeof(regs)/sizeof(regs[0])];
    unsigned i;
    for(i=0;i<sizeof(regs)/sizeof(regs[0]);i++) {
        int rc=io->read16addr8(io->ctx,regs[i],&values[i]);
        if(rc) {
            xil_printf("[SENSOR] diagnostic read failed reg=%04x bus=%d\r\n",regs[i],rc);
            return;
        }
    }
    xil_printf("[SENSOR] MODE=%02x RAW=%02x/%02x LANES=%02x output=%ux%u frame_lines=%u line_clocks=%u\r\n",
               values[0],values[1],values[2],values[3],
               ((unsigned)values[4]<<8)|values[5],((unsigned)values[6]<<8)|values[7],
               ((unsigned)values[8]<<8)|values[9],((unsigned)values[10]<<8)|values[11]);
}

static void print_next_name(void)
{
    char name[CAPTURE_NAME_CAPACITY];
    if(capture_store_current_name(&stores[0],name)==CAPTURE_OK)
        xil_printf("Next: %s (shared Image/Grey/Meta song-page)\r\n",name);
    else xil_printf("Image numbering full; existing photos will not be overwritten.\r\n");
}

static void print_color_status(void)
{
    xil_printf("[COLOR] frame=%u sensor_WB_Q8_RGB=%u/%u/%u exposure=%u gain_code=%u stable=%d\r\n",
        frame_number,frame_wb_q8[0],frame_wb_q8[1],frame_wb_q8[2],
        frame_exposure_lines,frame_gain_code,frame_controls_stable);
    xil_printf("[COLOR] input_p99_RGB=%u/%u/%u input_nearclip_permille=%u/%u/%u output_luma90=%u max99=%u nearclip=%u matrix_overflow=%u samples=%u\r\n",
        color_stats.raw_p99_rgb[0],color_stats.raw_p99_rgb[1],color_stats.raw_p99_rgb[2],
        color_stats.raw_saturated_permille_rgb[0],color_stats.raw_saturated_permille_rgb[1],
        color_stats.raw_saturated_permille_rgb[2],color_stats.corrected_luma_p90,
        color_stats.corrected_max_p99,color_stats.corrected_saturated_permille,
        color_stats.corrected_high_clip_permille,color_stats.samples);
}

static void print_performance(void)
{
    xil_printf("[PERF] live_fps=%u.%u capture_ms=%u stats_ms=%u preview_ms=%u preview_CCM_ms=%u photo_pending=%d stable=%d framing=FIXED_A4\r\n",
        preview_fps_x10/10u,preview_fps_x10%10u,capture_ms,statistics_ms,
        preview_ms,preview_color_ms,photo_pending,frame_controls_stable);
}

static void count_preview_frame(void)
{
    uint32_t now=milliseconds(), elapsed=now-perf_window_start;
    ++perf_window_frames;
    if(elapsed>=1000u) {
        preview_fps_x10=(unsigned)((uint64_t)perf_window_frames*10000u/elapsed);
        perf_window_start=now; perf_window_frames=0;
    }
    if(now-perf_last_log>=5000u) {
        print_performance(); perf_last_log=now;
    }
}

static void service_controls(void *unused)
{
    capture_button_event event;
    unsigned panel_events=0;
    (void)unused;
    if(pl_keys_ready) panel_events=capture_panel_update(&panel,milliseconds(),XGpio_DiscreteRead(&control,2));
    /* S3 can queue while a photo is being saved; the main loop switches only
     * after the entire Image/Grey/Meta transaction and preview restoration. */
    if(onecard_controls_ready && (panel_events&PANEL_UBUNTU)) ubuntu_pending=1;
    if(controls_enabled && !ubuntu_pending)
        capture_panel_focus_command(panel_events,focus_active,focus_region,
                                   &focus_toggle_pending,&focus_region_pending);
    event=capture_button_update(&key_state,milliseconds(),
                                !XGpioPs_ReadPin(&keys,PS_KEY_PIN));
    if(!controls_enabled) {
        if(XUartPs_IsReceiveData(UART_BASE))
            (void)XUartPs_ReadReg(UART_BASE,XUARTPS_FIFO_OFFSET);
        return;
    }
    /* First request wins until completion; saving/mode changes ignore keys. */
    if(!ubuntu_pending && !photo_pending && event==CAPTURE_BUTTON_PHOTO) photo_pending=PHOTO_PAIR;
    if(!ubuntu_pending && !photo_pending && event==CAPTURE_BUTTON_LONG) photo_pending=ACTION_NEXT_SONG;
    if(XUartPs_IsReceiveData(UART_BASE)) {
        unsigned char ch=(unsigned char)XUartPs_ReadReg(UART_BASE,XUARTPS_FIFO_OFFSET);
        if(!ubuntu_pending && !photo_pending && (ch==' '||ch=='p'||ch=='b')) photo_pending=PHOTO_PAIR;
        if(!ubuntu_pending && !photo_pending && ch=='t') photo_pending=ACTION_NEXT_SONG;
        if(ch=='u' && onecard_controls_ready) ubuntu_pending=1;
        if(ch=='n') {
            denoise_enabled=!denoise_enabled;
            xil_printf("[DENOISE] saved color chroma filter %s; focus and raw frame unchanged.\r\n",denoise_enabled?"ON":"OFF");
        }
        if(ch=='f') focus_toggle_pending^=1u;
        if(focus_active) {
            if(ch>='1' && ch<='5') focus_region_pending=(unsigned)(ch-'0');
            if(ch=='z') focus_zoom_pending^=1u;
            if(ch=='r') focus_reset_pending=1u;
            if(ch=='g') focus_grid_pending=1u;
        }
        if(ch=='a') {
            if(focus_active) xil_printf("[FOCUS] AE locked; use [ ] or f to exit.\r\n");
            else { auto_exposure=!auto_exposure; xil_printf("AE %s\r\n",auto_exposure?"ON":"LOCKED"); }
        }
        if(ch=='s') {
            xil_printf("Frames=%u brightness90=%u current_exposure=%u current_gain_code=%u SD=%d AE=%d\r\n",
                       frame_number,last_brightness,sensor.exposure_lines,
                       sensor.analogue_gain,sd_ready,auto_exposure);
            print_color_status();
            print_performance();
            capture_dp_status();
        }
        if(ch=='w' && focus_active) xil_printf("[FOCUS] WB locked; calibrate outside focus mode.\r\n");
        if(ch=='w' && !focus_active) {
            white_balance_pending=1;
            xil_printf("WB requested: center 20%% must contain only neutral white/gray, below clipping.\r\n");
        }
        if(sensor_ready && (ch=='['||ch==']')) {
            unsigned lines=manual_lines_pending?manual_lines_pending:sensor.exposure_lines;
            lines=ch==']'?lines+lines/4+1:lines-lines/5;
            if(lines<4) lines=4;
            if(lines>sensor.mode->frame_length_lines-22u) lines=sensor.mode->frame_length_lines-22u;
            auto_exposure=0;
            xil_printf("[CONTROL] AE LOCKED by manual exposure key; 'a' re-enables automatic low-ISO stills outside focus\r\n");
            /* This callback also runs during DMA and SD writes. Queue controls
             * here; all sensor changes run at the main-loop frame boundary. */
            manual_lines_pending=lines;
        }
    }
}

static void serviced_delay(unsigned ms)
{
    uint32_t start=milliseconds();
    do { service_controls(NULL); usleep(1000); } while(milliseconds()-start<ms);
}

static int init_gpio(void)
{
    XGpioPs_Config *cfg=XGpioPs_LookupConfig(XPAR_XGPIOPS_0_DEVICE_ID);
    startup_stage("PS GPIO / S1");
    if(!cfg || XGpioPs_CfgInitialize(&keys,cfg,cfg->BaseAddr)!=XST_SUCCESS) return -1;
    XGpioPs_SetDirectionPin(&keys,PS_KEY_PIN,0);
    XGpioPs_SetOutputEnablePin(&keys,PS_KEY_PIN,0);
    capture_button_init(&key_state,milliseconds(),!XGpioPs_ReadPin(&keys,PS_KEY_PIN),30,S1_LONG_MS);
    xil_printf("[INIT] PS GPIO OK; S1 level=%u (released=1)\r\n",
               (unsigned)XGpioPs_ReadPin(&keys,PS_KEY_PIN));
    startup_stage("PL AXI GPIO @0x80050000 (first PL access)");
    if(XGpio_Initialize(&control,XPAR_GPIO_0_DEVICE_ID)!=XST_SUCCESS) return -1;
    XGpio_SetDataDirection(&control,1,0);
    XGpio_SetDataDirection(&control,2,0x3ff);
    gpio_value=GPIO_ABORT|GPIO_PREVIEW;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    capture_panel_init(&panel,milliseconds(),XGpio_DiscreteRead(&control,2));pl_keys_ready=1;
    xil_printf("[INIT] PL GPIO OK; S2_n=%u S3_n=%u; synchronized Bank44 3.3V inputs; S1 long=%u ms\r\n",
        (XGpio_DiscreteRead(&control,2)>>8)&1u,(XGpio_DiscreteRead(&control,2)>>9)&1u,S1_LONG_MS);
    return 0;
}

static int mount_sd(void)
{
    XSdPs_Config *cfg=XSdPs_LookupConfig(0);
    int rc;
    sd_ready=0;
    /* Fail closed if the BSP maps drive 0 to the soldered eMMC controller. */
    if(!cfg || cfg->BaseAddress!=SD1_BASE) {
        xil_printf("ERROR: drive 0 is not PS SD1; refusing filesystem writes.\r\n");
        return -1;
    }
    rc=capture_fatfs_mount(&fatfs,"0:");
    if(rc==CAPTURE_OK) rc=capture_store_init_album(stores,&capture_fatfs_operations,&fatfs,output_directories);
    if(rc!=CAPTURE_OK) {
        xil_printf("SD mount/directory error %d; insert a FAT32 SD card. No format performed.\r\n",rc);
        ui_status="SD ERROR";
        return rc;
    }
    sd_ready=1;
    {
        unsigned hc=Xil_In8(SD1_BASE+0x28u), clk=Xil_In16(SD1_BASE+0x2cu);
        unsigned divider=((clk>>8)&255u)|((clk&0xc0u)<<2);
        unsigned divisor=divider?2u*divider:1u;
        xil_printf("[SD BUS] width=%u high_speed=%u HC2=%04x CLK=%04x ref_hz=%u derived_hz=%u cluster_bytes=%u buffer_bytes=%u\r\n",
            (hc&0x20u)?8u:((hc&2u)?4u:1u),!!(hc&4u),Xil_In16(SD1_BASE+0x3eu),clk,
            (unsigned)cfg->InputClockHz,(clk&0x20u)?0u:(unsigned)cfg->InputClockHz/divisor,
            (unsigned)fatfs.volume.csize*512u,CAPTURE_SD_BUFFER_BYTES);
    }
    print_next_name();
    return 0;
}

static int reset_dma_channel(int direction)
{
    uint32_t start=milliseconds();
    XAxiVdma_Reset(&dma,direction);
    while(XAxiVdma_ResetNotDone(&dma,direction)) {
        if(milliseconds()-start>200) return -1;
        service_controls(NULL);
    }
    return 0;
}

static int setup_write_channel(void)
{
    XAxiVdma_DmaSetup setup;
    UINTPTR addresses[XAXIVDMA_MAX_FRAMESTORE];
    unsigned i;
    if(reset_dma_channel(XAXIVDMA_WRITE)) return -1;
    memset(&setup,0,sizeof(setup));
    setup.VertSizeInput=capture_height;
    setup.HoriSizeInput=capture_width*3;
    setup.Stride=capture_stride;
    setup.EnableCircularBuf=0;
    setup.EnableFrameCounter=1;
    setup.FixedFrameStoreAddr=0;
    if(XAxiVdma_DmaConfig(&dma,XAXIVDMA_WRITE,&setup)!=XST_SUCCESS) return -1;
    for(i=0;i<XAXIVDMA_MAX_FRAMESTORE;i++) addresses[i]=(UINTPTR)capture_destination;
    return XAxiVdma_DmaSetBufferAddr(&dma,XAXIVDMA_WRITE,addresses)==XST_SUCCESS?0:-1;
}

static int init_display(void)
{
    XAxiVdma_Config *cfg=XAxiVdma_LookupConfig(XPAR_AXIVDMA_0_DEVICE_ID);
    unsigned x,y;
    startup_stage("capture VDMA driver/reset; DP display uses PS DPDMA");
    if(!cfg || XAxiVdma_CfgInitialize(&dma,cfg,cfg->BaseAddress)!=XST_SUCCESS) return -1;
    if(setup_write_channel()) return -1;
    memset(display_frames,0,sizeof(display_frames));
    preview_text(&display_frames[0][0][0],DISP_W,DISP_H,DISP_STRIDE,40,210,
                 "AC920 IMX477 DP STARTING",2,0xffffff);
    for(y=290;y<330;++y) for(x=80;x<560;++x) {
        uint8_t *p=&display_frames[0][y][x*3u];
        p[0]=x>=400u?255u:0u;p[1]=x>=240u && x<400u?255u:0u;p[2]=x<240u?255u:0u;
    }
    memcpy(display_frames[1],display_frames[0],sizeof(display_frames[0]));
    startup_stage("PS DisplayPort / DPDMA initialization");
    if(capture_dp_init(service_controls,NULL)) return -1;
    return capture_dp_present(&display_frames[0][0][0],DISP_STRIDE);
}

static int init_isp(void)
{
    XV_demosaic_Config *dcfg=XV_demosaic_LookupConfig(XPAR_XV_DEMOSAIC_0_DEVICE_ID);
    XV_gamma_lut_Config *gcfg=XV_gamma_lut_LookupConfig(XPAR_XV_GAMMA_LUT_0_DEVICE_ID);
    XCsiSs_Config *ccfg=XCsiSs_LookupConfig(XPAR_CSISS_0_DEVICE_ID);
    int lut[512];
    unsigned i;
    startup_stage("ISP driver lookup");
    if(!dcfg||!gcfg||!ccfg) return -1;
    startup_stage("ISP demosaic driver");
    if(XV_demosaic_CfgInitialize(&demosaic,dcfg,dcfg->BaseAddress)!=XST_SUCCESS) return -1;
    startup_stage("ISP gamma driver");
    if(XV_gamma_lut_CfgInitialize(&gamma_ip,gcfg,gcfg->BaseAddress)!=XST_SUCCESS) return -1;
    startup_stage("CSI receiver driver/reset");
    if(XCsiSs_CfgInitialize(&csi,ccfg,ccfg->BaseAddr)!=XST_SUCCESS) return -1;
    startup_stage("ISP dimensions and gamma LUT");
    XV_demosaic_Set_HwReg_width(&demosaic,capture_width);
    XV_demosaic_Set_HwReg_height(&demosaic,capture_height);
    XV_demosaic_Set_HwReg_bayer_phase(&demosaic,0); /* RGGB, no sensor flips */
    XV_gamma_lut_Set_HwReg_width(&gamma_ip,capture_width);
    XV_gamma_lut_Set_HwReg_height(&gamma_ip,capture_height);
    XV_gamma_lut_Set_HwReg_video_format(&gamma_ip,0);
    /* IMX477 tuning black level is 4096/65536 = 64/1024.
     * Keep the tested PL transfer. Sensor supplies WB; PS approximately
     * inverts this quantizer to apply CCM, then re-encodes gamma 2.2. */
    for(i=0;i<512;i++) {
        unsigned j, packed=0;
        for(j=0;j<2;j++) {
            unsigned sample=2*i+j;
            double linear=sample>64?(double)(sample-64)/959.0:0.0;
            unsigned value=(unsigned)(1023.0*pow(linear,1.0/2.2)+0.5);
            packed|=value<<(16*j);
        }
        lut[i]=(int)packed;
    }
    if(XV_gamma_lut_Write_HwReg_gamma_lut_0_Words(&gamma_ip,0,lut,512)!=512 ||
       XV_gamma_lut_Write_HwReg_gamma_lut_1_Words(&gamma_ip,0,lut,512)!=512 ||
       XV_gamma_lut_Write_HwReg_gamma_lut_2_Words(&gamma_ip,0,lut,512)!=512) return -1;
    XV_gamma_lut_EnableAutoRestart(&gamma_ip); XV_gamma_lut_Start(&gamma_ip);
    XV_demosaic_EnableAutoRestart(&demosaic); XV_demosaic_Start(&demosaic);
    startup_stage("CSI deactivate");
    if(XCsiSs_Activate(&csi,0)!=XST_SUCCESS) return -1;
    startup_stage("CSI configure: 2 lanes, interrupt mask=0; VC0 fixed in hardware");
    if(XCsiSs_Configure(&csi,2,0)!=XST_SUCCESS) return -1;
    startup_stage("CSI activate");
    if(XCsiSs_Activate(&csi,1)!=XST_SUCCESS) return -1;
    xil_printf("[INIT] CSI/ISP OK\r\n");
    return 0;
}

static int abort_capture(void)
{
    gpio_value|=GPIO_ABORT;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    usleep(100);
    XAxiVdma_DmaStop(&dma,XAXIVDMA_WRITE);
    capture_inflight=0; have_frame=0;
    if(setup_write_channel()) return -2;
    gpio_value&=~GPIO_ABORT;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    usleep(100);
    return 0;
}

static int begin_capture(void)
{
    XAxiVdma_FrameCounter counter={0};
    UINTPTR csi_base=csi.CsiPtr->Config.BaseAddr;
    if(capture_inflight) return -2;
    pending_buffer_index=camera_buffer_index^1u;
    capture_destination=camera_buffers[pending_buffer_index];
    ++capture_attempt;
    if(setup_write_channel()) return -2;
    counter.ReadFrameCount=1; counter.WriteFrameCount=1;
    if(XAxiVdma_SetFrameCounter(&dma,&counter)!=XST_SUCCESS) return -2;
    /* Startup cleans both buffers once. CPU never writes either source, so
     * no dirty CPU lines can overwrite DMA; invalidate only AFTER completion.
     * Destination and immutable CPU frame are different aligned allocations. */
    Xil_Out32(dma.BaseAddr+XAXIVDMA_RX_OFFSET+XAXIVDMA_SR_OFFSET,
              XAXIVDMA_IXR_ALL_MASK|S2MM_ERROR_MASK);
    Xil_Out32(csi_base+XCSI_ISR_OFFSET,XCSI_ISR_ALLINTR_MASK);
    pending_old_done=XGpio_DiscreteRead(&control,2)&1u;
    if(XAxiVdma_DmaStart(&dma,XAXIVDMA_WRITE)!=XST_SUCCESS) return -2;
    if(capture_attempt<=3u) capture_diagnostics("armed before gate request",0,pending_old_done);
    pending_stable=capture_settle_frame_ready(&sensor_settle,milliseconds());
    pending_exposure=sensor.exposure_lines; pending_gain=sensor.analogue_gain;
    pending_wb[0]=sensor.red_gain_q8; pending_wb[1]=sensor.green_gain_q8;
    pending_wb[2]=sensor.blue_gain_q8;
    gpio_value^=GPIO_REQUEST;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    pending_started_at=milliseconds(); capture_inflight=1;
    return 0;
}

static int wait_capture(void)
{
    uint32_t status,ds,csi_errors;
    int done_logged=0;
    UINTPTR csi_base=csi.CsiPtr->Config.BaseAddr;
    if(!capture_inflight) return -2;
    for(;;) {
        service_controls(NULL);
        status=XGpio_DiscreteRead(&control,2);
        ds=Xil_In32(dma.BaseAddr+XAXIVDMA_RX_OFFSET+XAXIVDMA_SR_OFFSET);
        if(ds&S2MM_ERROR_MASK) {
            capture_diagnostics("VDMA error",milliseconds()-pending_started_at,pending_old_done);
            break;
        }
        if((status&1u)!=pending_old_done && !done_logged) {
            done_logged=1;
            if(capture_attempt<=3u)
                capture_diagnostics("gate frame transferred",milliseconds()-pending_started_at,pending_old_done);
        }
        if((status&1u)!=pending_old_done && (ds&XAXIVDMA_SR_HALTED_MASK)) {
            usleep(1);
            status=XGpio_DiscreteRead(&control,2);
            ds=Xil_In32(dma.BaseAddr+XAXIVDMA_RX_OFFSET+XAXIVDMA_SR_OFFSET);
            csi_errors=Xil_In32(csi_base+XCSI_ISR_OFFSET)&CSI_BAD_FRAME;
            if(!(status&4u) && !csi_errors && !(ds&S2MM_ERROR_MASK) &&
               (ds&XAXIVDMA_SR_HALTED_MASK) &&
               !!(status&STATUS_PREVIEW)==(current_mode==IMX477_MODE_2028X1520_RAW12_30FPS)) {
                Xil_DCacheInvalidateRange((INTPTR)capture_destination,capture_height*capture_stride);
                camera_buffer_index=pending_buffer_index;
                camera_frame=capture_destination;
                capture_inflight=0; have_frame=1; frame_number++;
                frame_controls_stable=pending_stable;
                frame_exposure_lines=pending_exposure; frame_gain_code=pending_gain;
                memcpy(frame_wb_q8,pending_wb,sizeof(frame_wb_q8));
                capture_ms=milliseconds()-pending_started_at;
                if(capture_attempt<=3u) {
                    const uint8_t *tail=camera_frame+(size_t)(capture_height-1u)*capture_stride+capture_width*3u-3u;
                    capture_diagnostics("complete and DMA halted",capture_ms,pending_old_done);
                    xil_printf("[FRAME %u] size=%ux%u buffer=%u head_BGR=%02x%02x%02x tail_BGR=%02x%02x%02x\r\n",
                        frame_number,capture_width,capture_height,camera_buffer_index,
                        camera_frame[0],camera_frame[1],camera_frame[2],tail[0],tail[1],tail[2]);
                }
                return 0;
            }
            xil_printf("Rejected frame: gate=%02x VDMA=%08x CSI=%08x\r\n",status,ds,csi_errors);
            capture_diagnostics("frame rejected",milliseconds()-pending_started_at,pending_old_done);
            break;
        }
        if(milliseconds()-pending_started_at>2500u) {
            capture_diagnostics("timeout before recovery",milliseconds()-pending_started_at,pending_old_done);
            break;
        }
        usleep(100);
    }
    return abort_capture()?-2:-1;
}

static int acquire_frame(void)
{
    int rc=capture_inflight?0:begin_capture();
    return rc?rc:wait_capture();
}

/* The same content-independent sensor-long-edge crop anchors preview, stills and all
 * five focus positions. Keep the native DMA stride; never copy/modify source. */
static const uint8_t *a4_frame_source(unsigned crop[4])
{
    if(capture_a4_crop(capture_width,capture_height,crop)!=CAPTURE_BINARY_OK) return NULL;
    return camera_frame+(size_t)crop[1]*capture_stride+3u*crop[0];
}

static int measure_frame(void)
{
    unsigned crop[4];
    const uint8_t *source=a4_frame_source(crop);
    uint32_t start=milliseconds();
    int rc;
    if(!source) return -1;
    rc=capture_color_measure(&color_pipeline,source,crop[2],crop[3],
                                capture_stride,16u,&color_stats);
    statistics_ms=milliseconds()-start;
    last_brightness=color_stats.corrected_luma_p90;
    return rc==CAPTURE_COLOR_OK?0:-1;
}

static int focus_frame_eligible(void)
{
    return have_frame && frame_controls_stable &&
        current_mode==IMX477_MODE_4056X3040_RAW12_10FPS &&
        frame_exposure_lines==sensor.exposure_lines && frame_gain_code==sensor.analogue_gain &&
        frame_wb_q8[0]==sensor.red_gain_q8 && frame_wb_q8[1]==sensor.green_gain_q8 &&
        frame_wb_q8[2]==sensor.blue_gain_q8;
}

static const char *focus_quality(const capture_focus_result *r)
{
    if(r->flags&CAPTURE_FOCUS_CLIPPED) return "CLIPPED";
    if(r->flags&CAPTURE_FOCUS_LOW_TEXTURE) return "LOW TEXTURE";
    return "RELATIVE SCORE";
}

static void focus_edge_text(char *text,size_t size,unsigned value,unsigned count)
{
    if(count<CAPTURE_FOCUS_MIN_EDGES) (void)snprintf(text,size,"-- N%u",count);
    else (void)snprintf(text,size,"%u.%02u N%u",value/1000u,(value%1000u)/10u,count);
}

static void focus_log_result(unsigned region,const capture_focus_result *r,const char *kind)
{
    char h[24],v[24];
    unsigned crop[4];
    if(capture_a4_crop(capture_width,capture_height,crop)) return;
    focus_edge_text(h,sizeof(h),r->edge_h_milli,r->edge_h_count);
    focus_edge_text(v,sizeof(v),r->edge_v_milli,r->edge_v_count);
    xil_printf("[FOCUS %s] frame=%u A4_roi=%s sensor_xy=%u,%u size=500x500 G_pre_CCM T=%u.%04u H=%s V=%s flags=%u clip_permille=%u\r\n",
        kind,frame_number,capture_focus_region_name(region),crop[0]+r->roi.x,crop[1]+r->roi.y,
        r->tenengrad_x10000/10000u,r->tenengrad_x10000%10000u,h,v,r->flags,r->clip_permille);
}

/* All five metrics and native thumbnails come from this SAME immutable frame.
 * Cache pixels, not DMA pointers, so the explicitly frozen mosaic stays valid. */
static int focus_make_grid(void)
{
    static const unsigned xs[5]={220u,8u,432u,8u,432u};
    static const unsigned ys[5]={202u,52u,52u,352u,352u};
    unsigned i,crop[4];
    const uint8_t *source=a4_frame_source(crop);
    char text[64],h[24],v[24];
    if(!source) return -1;
    memset(focus_mosaic,0,sizeof(focus_mosaic));
    for(i=0;i<5u;++i) {
        capture_focus_rect r;
        uint8_t *tile=&focus_mosaic[ys[i]][xs[i]*3u];
        if(capture_focus_measure(source,crop[2],crop[3],capture_stride,i,1,
            &focus_workspace,&focus_five[i],service_controls,NULL) ||
           capture_focus_region(crop[2],crop[3],i,200u,96u,&r)) return -1;
        if(preview_letterbox(source+(size_t)r.y*capture_stride+r.x*3u,200u,96u,capture_stride,
            tile,200u,96u,DISP_STRIDE,service_controls,NULL) ||
           capture_color_process_bgr(&color_pipeline,tile,200u,96u,DISP_STRIDE,
               service_controls,NULL)!=CAPTURE_COLOR_OK) return -1;
        (void)snprintf(text,sizeof(text),"%s T%u.%04u %s",capture_focus_region_name(i),
            focus_five[i].tenengrad_x10000/10000u,focus_five[i].tenengrad_x10000%10000u,
            focus_five[i].flags?"INVALID":"");
        preview_text(&focus_mosaic[0][0],DISP_W,DISP_H,DISP_STRIDE,xs[i],ys[i]+97u,text,1,0xffffff);
        focus_edge_text(h,sizeof(h),focus_five[i].edge_h_milli,focus_five[i].edge_h_count);
        focus_edge_text(v,sizeof(v),focus_five[i].edge_v_milli,focus_five[i].edge_v_count);
        (void)snprintf(text,sizeof(text),"H %s V %s",h,v);
        preview_text(&focus_mosaic[0][0],DISP_W,DISP_H,DISP_STRIDE,xs[i],ys[i]+108u,text,1,0xffff80);
        focus_log_result(i,&focus_five[i],"FIVE");
    }
    i=capture_focus_uneven(focus_five);
    xil_printf("[FOCUS FIVE] uneven_top_bottom=%u uneven_left_right=%u; >30 percent, comparable content required; not an optical tilt diagnosis.\r\n",
        i&1u,(i>>1u)&1u);
    focus_grid_frame=frame_number;focus_grid=1;return 0;
}

static int focus_measure_frame(void)
{
    capture_focus_result result;
    unsigned crop[4];
    const uint8_t *source;
    uint32_t start=milliseconds();
    int edges;
    if(!focus_active || !focus_frame_eligible()) return 0;
    source=a4_frame_source(crop);if(!source) return -1;
    if(focus_grid_pending) {
        focus_grid_pending=0;
        if(focus_make_grid()) return -1;
        xil_printf("[FOCUS FIVE] frozen frame=%u elapsed_ms=%u; g refresh, 1-5/z resume live. Different content is not proof of tilt.\r\n",
            focus_grid_frame,milliseconds()-start);
    }
    if(focus_grid) return 0;
    edges=!focus_edge_frame || (uint32_t)(start-focus_edges_at)>=1000u;
    if(capture_focus_measure(source,crop[2],crop[3],capture_stride,focus_region,edges,
        &focus_workspace,&result,service_controls,NULL)) return -1;
    if(!edges) {
        result.edge_h_milli=focus_result.edge_h_milli;result.edge_h_count=focus_result.edge_h_count;
        result.edge_v_milli=focus_result.edge_v_milli;result.edge_v_count=focus_result.edge_v_count;
    } else { focus_edges_at=start;focus_edge_frame=frame_number; }
    focus_result=result;capture_focus_history_push(&focus_history,&focus_result);
    focus_metric_ms=milliseconds()-start;
    if((uint32_t)(start-focus_log_at)>=500u) {
        focus_log_result(focus_region,&focus_result,"LIVE");
        xil_printf("[FOCUS] avg3=%u.%04u peak=%u.%04u zoom=%u metric_ms=%u edges_updated=%u edge_frame=%u exposure=%u gain=%u live_fps=%u.%u\r\n",
            focus_history.current/10000u,focus_history.current%10000u,
            focus_history.peak/10000u,focus_history.peak%10000u,focus_zoom,focus_metric_ms,
            (unsigned)edges,focus_edge_frame,sensor.exposure_lines,sensor.analogue_gain,
            preview_fps_x10/10u,preview_fps_x10%10u);
        focus_log_at=start;
    }
    return 0;
}

static int focus_render(uint8_t *dest)
{
    capture_focus_rect r;
    unsigned crop[4];
    const uint8_t *source;
    if(focus_grid) { memcpy(dest,focus_mosaic,sizeof(focus_mosaic));return 0; }
    source=a4_frame_source(crop);if(!source) return -1;
    if(capture_focus_region(crop[2],crop[3],focus_region,DISP_W/focus_zoom,
        (DISP_H-PREVIEW_TOP)/focus_zoom,&r)) return -1;
    if(preview_letterbox(source+(size_t)r.y*capture_stride+3u*r.x,r.width,r.height,
        capture_stride,dest+PREVIEW_TOP*DISP_STRIDE,DISP_W,DISP_H-PREVIEW_TOP,DISP_STRIDE,
        service_controls,NULL)) return -1;
    return capture_color_process_bgr(&color_pipeline,dest+PREVIEW_TOP*DISP_STRIDE,
        DISP_W,DISP_H-PREVIEW_TOP,DISP_STRIDE,service_controls,NULL)==CAPTURE_COLOR_OK?0:-1;
}

static void focus_header(uint8_t *dest)
{
    char text[112],h[24],v[24];
    unsigned x,y,bar=focus_history.peak?(unsigned)((uint64_t)focus_history.current*620u/focus_history.peak):0u;
    if(focus_grid) {
        (void)snprintf(text,sizeof(text),"FOCUS A4 FIVE SNAPSHOT FRAME %u  %s",focus_grid_frame,ui_status);
        preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,4,text,1,0x00ff80);
        preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,18,"G REFRESH S2 NEXT LIVE/HOLD EXIT 1-5/Z LIVE AE/WB LOCK",1,0xffffff);
        x=capture_focus_uneven(focus_five);
        (void)snprintf(text,sizeof(text),"500X500 METRIC / NATIVE TILES  %s %s  COMPARE SIMILAR CONTENT",
            x&1u?"UNEVEN TB":"",x&2u?"UNEVEN LR":"");
        preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,32,text,1,0xffff80);
        return;
    }
    (void)snprintf(text,sizeof(text),"FOCUS A4 ROI%u %s %uX  %s  AE/WB LOCK  %u.%uFPS  %s",
        focus_region+1u,capture_focus_region_name(focus_region),focus_zoom,ui_status,preview_fps_x10/10u,preview_fps_x10%10u,
        focus_frame_eligible()?focus_quality(&focus_result):"SETTLING");
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,2,text,1,0x00ff80);
    if(focus_frame_eligible() && focus_history.valid)
        (void)snprintf(text,sizeof(text),"T AVG3 %u.%04u PEAK %u.%04u R RESET S2 NEXT/HOLD EXIT Z ZOOM G FIVE",
            focus_history.current/10000u,focus_history.current%10000u,
            focus_history.peak/10000u,focus_history.peak%10000u);
    else (void)snprintf(text,sizeof(text),"T -- R RESET S2 NEXT/HOLD EXIT 1-5 ROI Z ZOOM G FIVE");
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,15,text,1,0xffffff);
    focus_edge_text(h,sizeof(h),focus_result.edge_h_milli,focus_result.edge_h_count);
    focus_edge_text(v,sizeof(v),focus_result.edge_v_milli,focus_result.edge_v_count);
    (void)snprintf(text,sizeof(text),"EDGE PX H %s  V %s  ROI500  METRIC %uMS  S1 TAP PHOTO HOLD NEXT",h,v,focus_metric_ms);
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,28,text,1,0xffff80);
    for(y=42u;y<46u;++y) for(x=8u;x<628u;++x) {
        uint8_t *p=dest+y*DISP_STRIDE+x*3u;
        p[0]=40u;p[1]=(focus_frame_eligible() && focus_history.valid && x-8u<bar)?220u:40u;p[2]=40u;
    }
}

static int show_preview(void)
{
    unsigned back=front_index^1u;
    uint8_t *dest=&display_frames[back][0][0];
    char name[CAPTURE_NAME_CAPACITY];
    char fps_text[16];
    uint32_t render_start=milliseconds(), color_start;
    if(have_frame && focus_active) {
        color_start=milliseconds();
        if(focus_render(dest)) return -1;
        preview_color_ms=milliseconds()-color_start;
    } else if(have_frame) {
        unsigned crop[4];
        const uint8_t *source=a4_frame_source(crop);
        if(!source) return -1;
        if(preview_letterbox(source,crop[2],crop[3],capture_stride,
                             dest+PREVIEW_TOP*DISP_STRIDE,DISP_W,DISP_H-PREVIEW_TOP,
                             DISP_STRIDE,service_controls,NULL)) return -1;
        /* Nearest-neighbour sampling commutes exactly with our per-pixel
         * CCM. Correct only the small preview, BEFORE drawing UI text.
         * The zero-valued letterbox bars remain zero under this transform. */
        color_start=milliseconds();
        if(capture_color_process_bgr(&color_pipeline,dest+PREVIEW_TOP*DISP_STRIDE,
            DISP_W,DISP_H-PREVIEW_TOP,DISP_STRIDE,service_controls,NULL)!=CAPTURE_COLOR_OK)
            return -1;
        preview_color_ms=milliseconds()-color_start;
    } else {
        /* Preserve the last displayed image after a failed/partial capture.
         * Never read camera_frame until both the gate and DMA completed. */
        memcpy(dest,display_frames[front_index],sizeof(display_frames[back]));
    }
    memset(dest,0,PREVIEW_TOP*DISP_STRIDE);
    if(focus_active) focus_header(dest);
    else {
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,4,"IMX477",2,0xffffff);
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,104,4,
        strcmp(ui_status,"LIVE")==0?"A4 FIXED":ui_status,2,0x00ff80);
    if(sd_ready && capture_store_current_name(&stores[0],name)==CAPTURE_OK) {
        name[5]=0;
        preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,280,4,name,2,0xffffff);
    }
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,8,26,"S1 PHOTO/HOLD SONG S2 HOLD FOCUS S3 OMR",2,0xffffff);
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,408,4,auto_exposure?"AE ON":"AE LOCK",2,0xffff80);
    (void)snprintf(fps_text,sizeof(fps_text),"%u.%uFPS",preview_fps_x10/10u,preview_fps_x10%10u);
    preview_text(dest,DISP_W,DISP_H,DISP_STRIDE,516,4,fps_text,2,0xffff80);
    }
    if(capture_dp_present(dest,DISP_STRIDE)) return -1;
    /* DPDMA reads its own RGBA copies, never these BGR render canvases. */
    front_index=back;
    preview_ms=milliseconds()-render_start;
    return 0;
}

/* Controls stay disabled through save and preview restoration. */
static int save_photo(void)
{
    char name[CAPTURE_NAME_CAPACITY], json_name[CAPTURE_NAME_CAPACITY], metadata[4096];
    uint32_t start=milliseconds(), noise_start;
    unsigned width,height,y,exposure_us;
    size_t stride;
    int rc,metadata_length;
    XTime begin,end;
    capture_pair_frame image;
    photo_color_ticks=0;
    if(capture_inflight || !have_frame || !frame_controls_stable ||
       current_mode!=IMX477_MODE_4056X3040_RAW12_10FPS) return -1;
    if(!sd_ready && mount_sd()) return 0;
    rc=capture_store_current_name(stores,name);
    if(rc!=CAPTURE_OK) {
        xil_printf("[ALBUM] No available page rc=%d; hold S1 for next song.\r\n",rc);
        ui_status=rc==CAPTURE_PAGE_FULL?"PAGE FULL":"SD ERROR";
        status_until=milliseconds()+2000u;return show_preview();
    }
    memcpy(json_name,name,5u);memcpy(json_name+5u,".json",6u);
    memset(&photo_crop,0,sizeof(photo_crop));
    photo_crop.source_width=capture_width;photo_crop.source_height=capture_height;
    rc=capture_a4_crop(capture_width,capture_height,photo_crop.crop_xywh);
    if(rc!=CAPTURE_BINARY_OK) return -1;
    width=photo_crop.crop_xywh[2];height=photo_crop.crop_xywh[3];
    stride=(size_t)width*3u;
    if(stride*height>sizeof(binary_workspace)) return -1;
    ui_status="PROCESS";
    if(show_preview()) return -1;
    XTime_GetTime(&begin);
    for(y=0;y<height;++y) memcpy(binary_workspace+(size_t)y*stride,
        camera_frame+(size_t)(y+photo_crop.crop_xywh[1])*capture_stride+3u*photo_crop.crop_xywh[0],stride);
    if(capture_color_process_bgr(&color_pipeline,binary_workspace,width,height,stride,
        service_controls,NULL)!=CAPTURE_COLOR_OK) return -1;
    XTime_GetTime(&end);photo_color_ticks=end-begin;
    noise_start=milliseconds();
    if(denoise_enabled && capture_denoise_bgr(binary_workspace,width,height,stride,
        &denoise_workspace,service_controls,NULL)) return -1;
    xil_printf("[DENOISE] chroma=%s radius=4 variance=25 native=%ux%u filter_ms=%u; luma not spatially filtered\r\n",
        denoise_enabled?"ON":"OFF",width,height,milliseconds()-noise_start);
    image.bgr=binary_workspace;image.width=width;image.height=height;image.stride=stride;
    image.service=service_controls;image.service_context=NULL;
    exposure_us=(unsigned)((uint64_t)frame_exposure_lines*sensor.mode->line_length_clocks*1000000u/
                           sensor.mode->timing_pixel_rate_hz);
    metadata_length=snprintf(metadata,sizeof(metadata),
        "{\n\"version\":\"a4_onecard_dp2\",\"complete\":true,\"song\":%u,\"page\":%u,\n"
        "\"image_file\":\"Image/%.5s.bmp\",\"gray_file\":\"Grey/%.5s.bmp\",\n"
        "\"frame\":%u,\"uptime_ms\":%u,\"same_frame\":true,\"controls_stable\":%s,\n"
        "\"sensor\":\"IMX477\",\"sensor_size\":[%u,%u],\"sensor_raw_bits\":12,\"binning\":false,\n"
        "\"crop_mode\":\"fixed_a4\",\"crop_xywh\":[%u,%u,%u,%u],\"output_size\":[%u,%u],\n"
        "\"image_format\":\"BMP_RGB24\",\"gray_format\":\"BMP_GRAY8\",\n"
        "\"gray_from_saved_color\":true,\"gray_coefficients_rgb\":[4899,9617,1868],"
        "\"gray_divisor\":16384,\"gray_rounding\":8192,\n"
        "\"resize\":false,\"clahe\":false,\"binarize\":false,\n"
        "\"exposure_lines\":%u,\"exposure_us\":%u,\"frame_length_lines\":%u,"
        "\"line_length_clocks\":%u,\"timing_pixel_rate_hz\":%u,\n"
        "\"analogue_gain_code\":%u,\"analogue_gain_x1000\":%u,\"digital_gain_q8\":%u,\n"
        "\"sensor_wb_q8_rgb\":[%u,%u,%u],\"auto_exposure\":%s,\"focus_mode\":%s,\n"
        "\"color_chroma_denoise\":%s,\"ccm_rgb\":[%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f],\n"
        "\"orientation\":\"sensor_native\",\"aperture_f\":null,\"focal_length_mm\":null\n}\n",
        stores[0].song,stores[0].next_page,name,name,frame_number,milliseconds(),frame_controls_stable?"true":"false",
        capture_width,capture_height,photo_crop.crop_xywh[0],photo_crop.crop_xywh[1],width,height,width,height,
        frame_exposure_lines,exposure_us,sensor.frame_length_lines,sensor.mode->line_length_clocks,
        sensor.mode->timing_pixel_rate_hz,frame_gain_code,1024000u/(1024u-frame_gain_code),sensor.digital_gain,
        frame_wb_q8[0],frame_wb_q8[1],frame_wb_q8[2],auto_exposure?"true":"false",focus_active?"true":"false",
        denoise_enabled?"true":"false",capture_color_ccm_rgb[0],capture_color_ccm_rgb[1],capture_color_ccm_rgb[2],
        capture_color_ccm_rgb[3],capture_color_ccm_rgb[4],capture_color_ccm_rgb[5],capture_color_ccm_rgb[6],
        capture_color_ccm_rgb[7],capture_color_ccm_rgb[8]);
    if(metadata_length<=0 || (size_t)metadata_length>=sizeof(metadata)) return -1;
    xil_printf("[A4] FIXED crop=%u,%u,%u,%u; native Image+Grey; no CLAHE/binary/resize; processing_ms=%u\r\n",
        photo_crop.crop_xywh[0],photo_crop.crop_xywh[1],width,height,milliseconds()-start);
    ui_status="WRITING";
    if(show_preview()) return -1;
    xil_printf("WRITING Image+Grey/%s and Meta/%s dimensions=%ux%u\r\n",name,json_name,width,height);
    print_color_status();
    if(capture_fatfs_reset_stats(&fatfs)!=CAPTURE_OK) return -1;
    rc=capture_store_save_album(stores,width,height,capture_pair_rgb_row,&image,
        capture_pair_gray_row,&image,(const uint8_t *)metadata,(size_t)metadata_length,
        bmp_row,sizeof(bmp_row),name);
    if(rc==CAPTURE_OK) {
        xil_printf("SAVED Image/%s + Grey/%s + Meta/%s %ux%u (%u ms)\r\n",
            name,name,json_name,width,height,milliseconds()-start);
        print_next_name();ui_status="SAVED";
    } else {
        xil_printf("SAVE FAILED %d; rescan Image/Grey/Meta before retry; existing files retained.\r\n",rc);
        if(rc==CAPTURE_PAIR_PARTIAL)
            xil_printf("[SD] Rollback incomplete; surviving filenames will be skipped on rescan.\r\n");
        ui_status=rc==CAPTURE_PAGE_FULL?"PAGE FULL":"SAVE ERROR";
        if(rc!=CAPTURE_PAGE_FULL) sd_ready=0;
    }
    xil_printf("[PERF PHOTO] color_crop_CCM_ms=%u\r\n",(unsigned)(photo_color_ticks/(COUNTS_PER_SECOND/1000u)));
    xil_printf("[SD PERF] bytes=%u row_calls=%u f_write_calls=%u write_ms=%u reserve_ms=%u sync_ms=%u close_ms=%u max_write_ms=%u\r\n",
        (unsigned)fatfs.stats.bytes,fatfs.stats.logical_writes,fatfs.stats.writes,
        (unsigned)(fatfs.stats.write_ticks/(COUNTS_PER_SECOND/1000u)),
        (unsigned)(fatfs.stats.reserve_ticks/(COUNTS_PER_SECOND/1000u)),
        (unsigned)(fatfs.stats.sync_ticks/(COUNTS_PER_SECOND/1000u)),
        (unsigned)(fatfs.stats.close_ticks/(COUNTS_PER_SECOND/1000u)),fatfs.stats.max_write_ms);
    status_until=milliseconds()+2000u;
    return show_preview();
}

static void update_exposure(void)
{
    capture_exposure_stats stats;
    capture_exposure_setting current={sensor.exposure_lines,sensor.analogue_gain}, next;
    enum capture_exposure_reason reason=CAPTURE_AE_HOLD;
    unsigned i;
    int rc=IMX477_OK;
    if(capture_inflight || photo_pending || !frame_controls_stable) return;
    next=current;
    if(manual_lines_pending) {
        next.lines=manual_lines_pending;
        manual_lines_pending=0;
    } else {
        if(!auto_exposure || milliseconds()-exposure_changed_at<350u) return;
        memset(&stats,0,sizeof(stats));
        stats.luma_p90=color_stats.corrected_luma_p90;
        stats.corrected_max_p99=color_stats.corrected_max_p99;
        stats.corrected_clipped_permille=color_stats.corrected_saturated_permille;
        stats.matrix_high_clip_permille=color_stats.corrected_high_clip_permille;
        for(i=0;i<3;i++) {
            if(color_stats.raw_p99_rgb[i]>stats.raw_max_p99)
                stats.raw_max_p99=color_stats.raw_p99_rgb[i];
            if(color_stats.raw_saturated_permille_rgb[i]>stats.raw_clipped_permille)
                stats.raw_clipped_permille=color_stats.raw_saturated_permille_rgb[i];
        }
        if(capture_exposure_next_limits(&stats,&current,sensor.mode->frame_length_lines-22u,
            current_mode==IMX477_MODE_2028X1520_RAW12_30FPS?1960u:2800u,&next,&reason)<=0) return;
    }
    if(next.lines!=current.lines) rc=imx477_set_exposure_lines(&sensor,(uint16_t)next.lines);
    if(rc==IMX477_OK && next.gain_code!=current.gain_code)
        rc=imx477_set_analogue_gain(&sensor,(uint16_t)next.gain_code);
    if(rc!=IMX477_OK) {
        auto_exposure=0; sensor_control_error=1;
        xil_printf("Exposure I2C error %d reg=%04x; capture stopped.\r\n",rc,sensor.last_failed_register);
    } else if(next.lines!=current.lines || next.gain_code!=current.gain_code) {
        if(focus_active) focus_reset();
        exposure_changed_at=milliseconds();
        capture_settle_changed_period(&sensor_settle,exposure_changed_at,imx477_frame_period_us(&sensor));
        xil_printf("[AE] reason=%u lines=%u->%u gain=%u->%u luma90=%u max99=%u\r\n",
            (unsigned)reason,current.lines,next.lines,current.gain_code,next.gain_code,
            color_stats.corrected_luma_p90,color_stats.corrected_max_p99);
        /* Keep displaying during settling. Gate subsequent AE/WB/photos by
         * the capture-start readiness snapshot instead of blocking here. */
    }
}

static int update_white_balance(void)
{
    capture_color_white_result result;
    const unsigned gains[3]={sensor.red_gain_q8,sensor.green_gain_q8,sensor.blue_gain_q8};
    int rc;
    if(focus_active || capture_inflight || !white_balance_pending || photo_pending || manual_lines_pending ||
       !frame_controls_stable) return 0;
    white_balance_pending=0;
    rc=capture_color_calibrate_white(&color_pipeline,camera_frame,capture_width,capture_height,
        capture_stride,capture_width*2u/5u,capture_height*2u/5u,capture_width/5u,capture_height/5u,gains,&result);
    if(rc!=CAPTURE_COLOR_OK) {
        xil_printf("[WB] rejected: %s; gains unchanged. Use a neutral target; '[' reduces exposure.\r\n",
                   capture_color_result_string(rc));
        return 0;
    }
    xil_printf("[WB] neutral ROI samples=%u RGB_Q8 %u/%u/%u -> %u/%u/%u\r\n",
        result.samples,gains[0],gains[1],gains[2],result.proposed_sensor_gain_q8[0],
        result.proposed_sensor_gain_q8[1],result.proposed_sensor_gain_q8[2]);
    /* DMA is halted and the complete frame has not been modified by CCM.
     * Stop streaming before touching mode or per-color gains. */
    have_frame=0;
    rc=imx477_stop(&sensor);
    if(rc==IMX477_OK) rc=imx477_set_colour_gains(&sensor,
        (uint16_t)result.proposed_sensor_gain_q8[0],
        (uint16_t)result.proposed_sensor_gain_q8[1],
        (uint16_t)result.proposed_sensor_gain_q8[2]);
    if(rc==IMX477_OK) rc=imx477_start(&sensor);
    if(rc!=IMX477_OK) {
        sensor_control_error=1;
        xil_printf("[WB] failed rc=%d reg=%04x bus=%d expected=%02x read=%02x; capture stopped.\r\n",
            rc,sensor.last_failed_register,sensor.last_bus_result,
            sensor.last_expected_value,sensor.last_read_value);
        return -1;
    }
    exposure_changed_at=milliseconds();
    capture_settle_changed_period(&sensor_settle,exposure_changed_at,imx477_frame_period_us(&sensor));
    xil_printf("[WB] sensor writes/readback OK; discard old frame, wait for fresh data.\r\n");
    return 1;
}

static int isp_reset_request(int asserted)
{
    uint32_t start=milliseconds();
    if(asserted) gpio_value|=GPIO_ISP_RESET;
    else gpio_value&=~GPIO_ISP_RESET;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    while(!!(XGpio_DiscreteRead(&control,2)&STATUS_ISP_RESET)!=!!asserted) {
        if(milliseconds()-start>100u) {
            xil_printf("[MODE] ISP reset acknowledgement timeout asserted=%d\r\n",asserted);
            return -1;
        }
        usleep(10);
    }
    /* HLS AXI-Lite must never be touched while its reset is asserted. */
    usleep(2); return 0;
}

static int switch_sensor_mode(enum imx477_mode_id mode,const struct imx477 *restore,
                              const capture_exposure_setting *still)
{
    const struct imx477_mode_info *info=imx477_get_mode_info(mode);
    int rc;
    uint32_t start=milliseconds();
    if(!info || capture_inflight || (restore && still)) return -1;
    gpio_value|=GPIO_ABORT;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    have_frame=0;
    memset(&photo_crop,0,sizeof(photo_crop));
    rc=imx477_stop(&sensor); /* Driver waits for current frame drain. */
    if(rc!=IMX477_OK) goto sensor_failed;
    if(XCsiSs_Activate(&csi,0)!=XST_SUCCESS || isp_reset_request(1)) return -1;
    if(mode==IMX477_MODE_2028X1520_RAW12_30FPS) gpio_value|=GPIO_PREVIEW;
    else gpio_value&=~GPIO_PREVIEW;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    capture_width=info->width; capture_height=info->height;
    capture_stride=(capture_width*3u+63u)&~63u;
    current_mode=mode;
    rc=imx477_reconfigure_preserving_controls(&sensor,mode);
    if(rc!=IMX477_OK) goto sensor_failed;
    if(still) {
        if(still->gain_code || still->lines>IMX477_STILL_EXPOSURE_MAX_LINES) return -1;
        rc=imx477_set_still_low_gain(&sensor,(uint16_t)still->lines);
        if(rc!=IMX477_OK) goto sensor_failed;
    }
    if(restore) {
        /* Restore the exact preview settings; time-conversion rounding or
         * full->preview exposure clamping must not accumulate on each photo. */
        rc=imx477_set_exposure_lines(&sensor,restore->exposure_lines);
        if(rc==IMX477_OK) rc=imx477_set_analogue_gain(&sensor,restore->analogue_gain);
        if(rc!=IMX477_OK) goto sensor_failed;
        if(sensor.red_gain_q8!=restore->red_gain_q8 ||
           sensor.green_gain_q8!=restore->green_gain_q8 ||
           sensor.blue_gain_q8!=restore->blue_gain_q8) {
            rc=imx477_set_colour_gains(&sensor,restore->red_gain_q8,
                                     restore->green_gain_q8,restore->blue_gain_q8);
            if(rc!=IMX477_OK) goto sensor_failed;
        }
    }
    if(isp_reset_request(0) || init_isp() || setup_write_channel()) return -1;
    gpio_value&=~GPIO_ABORT;
    XGpio_DiscreteWrite(&control,1,gpio_value); usleep(2);
    rc=imx477_start(&sensor);
    if(rc!=IMX477_OK) goto sensor_failed;
    exposure_changed_at=milliseconds();
    capture_settle_changed_period(&sensor_settle,exposure_changed_at,imx477_frame_period_us(&sensor));
    xil_printf("[MODE] %ux%u stride=%u exposure=%u gain=%u WB=%u/%u/%u switch_ms=%u\r\n",
        capture_width,capture_height,capture_stride,sensor.exposure_lines,sensor.analogue_gain,
        sensor.red_gain_q8,sensor.green_gain_q8,sensor.blue_gain_q8,milliseconds()-start);
    xil_printf("[MODE TIMING] frame_lines=%u frame_us=%u settle_ms=%u\r\n",
        sensor.frame_length_lines,imx477_frame_period_us(&sensor),sensor_settle.wait_ms);
    return 0;
sensor_failed:
    xil_printf("[MODE] sensor failed rc=%d reg=%04x bus=%d expected=%02x read=%02x\r\n",
        rc,sensor.last_failed_register,sensor.last_bus_result,
        sensor.last_expected_value,sensor.last_read_value);
    sensor_control_error=1; return -1;
}

static void rearm_keys(void)
{
    unsigned discarded=0;
    capture_panel_init(&panel,milliseconds(),XGpio_DiscreteRead(&control,2));
    capture_button_init(&key_state,milliseconds(),!XGpioPs_ReadPin(&keys,PS_KEY_PIN),30,S1_LONG_MS);
    /* Mode programming can run without a service callback. Discard commands
     * typed during the busy interval; UART RX has a 64-byte hardware FIFO.
     * Bound the drain even if an external sender transmits continuously. */
    while(discarded<64u && XUartPs_IsReceiveData(UART_BASE)) {
        (void)XUartPs_ReadReg(UART_BASE,XUARTPS_FIFO_OFFSET); ++discarded;
    }
}

/* Return1 after a mode change: old completed frame was invalidated. */
static int focus_apply_commands(void)
{
    if(capture_inflight) return -1;
    if(focus_toggle_pending && frame_controls_stable) {
        focus_toggle_pending=0;controls_enabled=0;
        white_balance_pending=0;manual_lines_pending=0;
        focus_region_pending=focus_zoom_pending=focus_reset_pending=focus_grid_pending=0;
        if(!focus_active) {
            focus_preview_settings=sensor;focus_saved_ae=auto_exposure;auto_exposure=0;
            if(switch_sensor_mode(IMX477_MODE_4056X3040_RAW12_10FPS,NULL,NULL)) return -1;
            focus_active=1;focus_region=0;focus_zoom=1;
            xil_printf("[FOCUS] ENTER 4056x3040 RAW12 sensor10fps; positions inside fixed4056x2868 A4 crop; G before PS CCM; fixed500x500 metric; native display640x432 or320x216 doubled.\r\n");
            xil_printf("[FOCUS] AE/WB locked; S2 tap=next ROI1..5, hold1.5s/f=exit; 1 C,2 TL,3 TR,4 BL,5 BR,z zoom,r peak reset,g five-region snapshot,[ ] exposure. S1 photos unchanged.\r\n");
        } else {
            if(switch_sensor_mode(IMX477_MODE_2028X1520_RAW12_30FPS,&focus_preview_settings,NULL)) return -1;
            focus_active=0;auto_exposure=focus_saved_ae;
            xil_printf("[FOCUS] EXIT; fixed A4 preview and exact previous exposure/gain/WB restored; AE=%u.\r\n",(unsigned)auto_exposure);
        }
        focus_reset();ui_status="SETTLING";status_until=0;
        rearm_keys();controls_enabled=1;
        perf_window_start=milliseconds();perf_window_frames=preview_fps_x10=0;
        return 1;
    }
    if(focus_active) {
        if(focus_region_pending) {
            focus_region=focus_region_pending-1u;focus_region_pending=0;focus_reset();
            xil_printf("[FOCUS] ROI=%u %s\r\n",focus_region+1u,capture_focus_region_name(focus_region));
        }
        if(focus_zoom_pending) { focus_zoom=3u-focus_zoom;focus_zoom_pending=0;focus_reset(); }
        if(focus_reset_pending) { focus_reset_pending=0;focus_reset(); }
    }
    return 0;
}

static capture_exposure_stats still_statistics(void)
{
    capture_exposure_stats result={0};
    unsigned i;
    result.luma_p90=color_stats.corrected_luma_p90;
    result.corrected_max_p99=color_stats.corrected_max_p99;
    result.corrected_clipped_permille=color_stats.corrected_saturated_permille;
    result.matrix_high_clip_permille=color_stats.corrected_high_clip_permille;
    for(i=0;i<3u;++i) {
        if(color_stats.raw_p99_rgb[i]>result.raw_max_p99) result.raw_max_p99=color_stats.raw_p99_rgb[i];
        if(color_stats.raw_saturated_permille_rgb[i]>result.raw_clipped_permille)
            result.raw_clipped_permille=color_stats.raw_saturated_permille_rgb[i];
    }
    return result;
}

static int shoot_and_return(void)
{
    struct imx477 preview_settings=sensor;
    unsigned errors=0,pass,limited=0;
    capture_exposure_stats stats;
    capture_exposure_setting current,next;
    const capture_exposure_setting *still=NULL;
    uint32_t start;
    int rc;
    if(capture_inflight) return -1;
    if(focus_active) {
        if(!focus_frame_eligible()) return -1;
        controls_enabled=0;photo_pending=0;
        /* Zoom/metric ROI never enters the capture pipeline. save_photo still
         * uses the fixed A4 rectangle in the native full-resolution frame. */
        if(save_photo()) return -1;
        rearm_keys();controls_enabled=1;
        perf_window_start=milliseconds();perf_window_frames=0;
        xil_printf("[FOCUS] photo complete; full-resolution focus and locked controls retained.\r\n");
        return 0;
    }
    if(current_mode!=IMX477_MODE_2028X1520_RAW12_30FPS) return -1;
    controls_enabled=0; photo_pending=0;
    ui_status="CAPTURE";
    if(auto_exposure) {
        /* Preview line period is half the full-mode period. Seed with its
         * latest completed-frame stats, then meter the actual native still. */
        current.lines=(sensor.exposure_lines+1u)/2u;
        if(current.lines<4u) current.lines=4u;
        current.gain_code=sensor.analogue_gain;stats=still_statistics();
        if(capture_still_plan(&stats,&current,IMX477_STILL_EXPOSURE_MAX_LINES,&next,&limited)<0) return -1;
        still=&next;
        xil_printf("[PHOTO ISO] AUTO base_gain=1.00x seed_lines=%u max_exposure_us=1000000 limited=%u; WB preserved\r\n",next.lines,limited);
    } else {
        xil_printf("[PHOTO ISO] MANUAL LOCK: retaining user exposure and gain\r\n");
    }
    if(show_preview() || switch_sensor_mode(IMX477_MODE_4056X3040_RAW12_10FPS,NULL,still)) return -1;
    for(pass=0;pass<3u;++pass) {
        start=milliseconds();
        do {
            rc=acquire_frame();
            if(rc) {
                if(rc==-2 || ++errors>=3u) return -1;
            } else if(frame_controls_stable) break;
            if(milliseconds()-start>7000u) return -1;
        } while(1);
        if(measure_frame()) return -1;
        if(!auto_exposure) break;
        stats=still_statistics();current.lines=sensor.exposure_lines;current.gain_code=sensor.analogue_gain;
        rc=capture_still_plan(&stats,&current,IMX477_STILL_EXPOSURE_MAX_LINES,&next,&limited);
        if(rc<0) return -1;
        xil_printf("[PHOTO AE] pass=%u exposure_us=%u gain_code=%u luma90=%u max99=%u next_lines=%u limited=%u\r\n",
            pass+1u,(unsigned)((uint64_t)sensor.exposure_lines*sensor.mode->line_length_clocks*1000000u/sensor.mode->timing_pixel_rate_hz),
            sensor.analogue_gain,stats.luma_p90,stats.corrected_max_p99,next.lines,limited);
        if(limited) xil_printf("[PHOTO AE] shutter limit: add light or open aperture; no extra analogue/digital gain\r\n");
        if(!rc) break;
        if(pass==2u) { xil_printf("[PHOTO AE] metering limit: saving last verified frame; target may not be reached\r\n");break; }
        /* Every new setting is applied/read back in standby, ISP reset/drained,
         * and starts a fresh period-aware settling interval. No live DMA writes. */
        if(switch_sensor_mode(IMX477_MODE_4056X3040_RAW12_10FPS,NULL,&next)) return -1;
    }
    if(save_photo()) return -1;
    if(switch_sensor_mode(IMX477_MODE_2028X1520_RAW12_30FPS,&preview_settings,NULL)) return -1;
    rearm_keys(); controls_enabled=1;
    perf_window_start=milliseconds(); perf_window_frames=0;
    xil_printf("[MODE] fixed A4 preview restored; S1 short=Image+Grey+Meta, hold1.5s=next song.\r\n");
    return 0;
}

#include "onecard_platform.h"

int main(void)
{
    struct imx477_io sensor_io;
    unsigned failed_frames=0;
    int init_rc;
    Xil_AssertSetCallback(startup_assert);
    xil_printf("\r\nAC920 IMX477 a4_onecard_dp2 2026-10-29\r\n");
    xil_printf("FIXED A4 SENSOR LONG EDGE: preview2028x1434; Image/Grey both4056x2868; no detection or photo resize\r\n");
    xil_printf("S1 short=Image+Grey+Meta; hold1.5s=next song; native crop x0 y86 w4056 h2868\r\n");
    xil_printf("Shared 00-00.bmp onward; blank, dark and no-paper scenes can all be saved.\r\n");
    xil_printf("UART p/b/space=Image+Grey+Meta, t=next song, s=status; n=color denoise; a/[ ]=exposure; w=neutral-target sensor WB\r\n");
    xil_printf("DISPLAY: PS DisplayPort 640x480 RGB8; pure DP, no PL HDMI; S2 hold1.5s=focus, tap in focus=next ROI; S3/u Ubuntu\r\n");
    xil_printf("FOCUS f=enter/exit, 1-5=region, z=1X/2X, r=peak reset, g=five-region snapshot\r\n");
    xil_printf("[BUILD] VDMA use_fsync=%u SG=%u DRE=%u stream_bits=%u memory_bits=%u parameter_reads=%u\r\n",
               (unsigned)XPAR_AXIVDMA_0_USE_FSYNC,(unsigned)XPAR_AXIVDMA_0_INCLUDE_SG,
               (unsigned)XPAR_AXIVDMA_0_INCLUDE_S2MM_DRE,
               (unsigned)XPAR_AXIVDMA_0_S_AXIS_S2MM_TDATA_WIDTH,
               (unsigned)XPAR_AXIVDMA_0_M_AXI_S2MM_DATA_WIDTH,
               (unsigned)XPAR_AXIVDMA_0_ENABLE_VIDPRMTR_READS);
    xil_printf("[BUILD] capture=%ux%u HSIZE=%u stride=%u; detailed capture log for first 3 attempts and every failure\r\n",
               capture_width,capture_height,capture_width*3u,capture_stride);
    xil_printf("[BUILD] 2028x1520 sensor ~30fps, two-buffer preview; full-resolution mode for both photo types\r\n");
    startup_stage("clean double capture buffers; CPU source access stays read-only");
    Xil_DCacheFlushRange((INTPTR)camera_buffers,sizeof(camera_buffers));
    if(init_gpio()) { xil_printf("GPIO initialization failed: %s\r\n",init_stage); return 1; }
    if(init_display()) { xil_printf("DP/capture VDMA initialization failed: %s\r\n",init_stage); return 1; }
    xil_printf("[INIT] DP/capture VDMA OK\r\n");
    if(init_isp()) {
        xil_printf("CSI/ISP initialization failed: %s\r\n",init_stage);
        ui_status="CSI ERROR"; (void)show_preview(); return 1;
    }
    startup_stage("PS CCM lookup tables (sensor performs WB)");
    if(capture_color_init(&color_pipeline)!=CAPTURE_COLOR_OK) {
        xil_printf("Color pipeline initialization failed.\r\n"); return 1;
    }
    startup_stage("PS I2C0 at 100 kHz");
    init_rc=platform_i2c_init(&iic);
    if(!init_rc) init_rc=platform_i2c_bind(&iic,&sensor_io);
    if(init_rc) {
        xil_printf("I2C initialization failed: %d\r\n",init_rc);
        ui_status="I2C ERROR"; (void)show_preview(); return 1;
    }
    startup_stage("camera ENABLE high");
    gpio_value=GPIO_ENABLE|GPIO_ABORT|GPIO_PREVIEW;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    startup_stage("IMX477 power-up delay and chip ID");
    init_rc=imx477_init(&sensor,&sensor_io);
    if(init_rc==IMX477_OK) {
        xil_printf("[INIT] IMX477 chip ID 0x0477 verified\r\n");
        startup_stage("IMX477 configure 2028x1520 RAW12 ~30fps");
        init_rc=imx477_configure(&sensor,current_mode);
    }
    if(init_rc==IMX477_OK) {
        startup_stage("IMX477 independent color gains and register readback");
        /* Provisional 6500 K RPi tuning; explicit 'w' measures the real light.
         * See COLOR_SOURCE.txt and IMX477_COLOR_REGISTERS.txt for limitations. */
        init_rc=imx477_set_colour_gains(&sensor,830u,256u,357u);
        if(init_rc==IMX477_OK)
            xil_printf("[WB] SENSOR independent mode readback OK; RGB_Q8=%u/%u/%u; PS CCM only\r\n",
                sensor.red_gain_q8,sensor.green_gain_q8,sensor.blue_gain_q8);
    }
    if(init_rc!=IMX477_OK) {
        xil_printf("IMX477 init failed: rc=%d reg=%04x bus=%d expected=%02x read=%02x; verify adapter power, ENABLE and I2C.\r\n",
                   init_rc,sensor.last_failed_register,sensor.last_bus_result,
                   sensor.last_expected_value,sensor.last_read_value);
        ui_status="CAM ERROR"; (void)show_preview(); return 1;
    }
    xil_printf("[INIT] IMX477 configuration OK\r\n");
    startup_stage("SD1 FAT32 mount and Image/Grey/Meta song-page numbering");
    (void)mount_sd();
    startup_stage("release capture gate and start IMX477");
    gpio_value&=~GPIO_ABORT;
    XGpio_DiscreteWrite(&control,1,gpio_value);
    usleep(100);
    init_rc=imx477_start(&sensor);
    if(init_rc!=IMX477_OK) {
        xil_printf("IMX477 stream start failed: rc=%d reg=%04x bus=%d\r\n",
                   init_rc,sensor.last_failed_register,sensor.last_bus_result);
        ui_status="CAM ERROR"; (void)show_preview(); return 1;
    }
    sensor_ready=1; controls_enabled=0;
    sensor_diagnostics(&sensor_io);
    ui_status=sd_ready?"LIVE":"SD ERROR";
    xil_printf("Camera streaming; waiting for the first complete frame.\r\n");
    perf_window_start=perf_last_log=milliseconds();
    for(;;) {
        int frame_result;
        service_controls(NULL);
        frame_result=acquire_frame();
        if(frame_result) {
            ui_status="FRAME ERROR";
            if(show_preview() || frame_result==-2 || ++failed_frames>=3) goto stopped;
            serviced_delay(100); continue;
        }
        failed_frames=0;
        if(measure_frame()) goto stopped;
        if(!controls_enabled) {
            photo_pending=0; rearm_keys(); controls_enabled=1;onecard_controls_ready=1;
            xil_printf("READY: fixed A4 preview; S1 short=Image+Grey+Meta,long=next song; S2 hold=focus,tap=ROI1..5; S3/u=Ubuntu.\r\n");
        }
        if(ubuntu_pending && !photo_pending) {
            if(switch_to_ubuntu()) goto stopped;
            continue;
        }
        /* All sensor writes/mode transitions happen with DMA halted, before
         * priming the other buffer. Requests arriving during rendering wait
         * for its already-started capture to complete on the next iteration. */
        frame_result=focus_apply_commands();
        if(frame_result<0) goto stopped;
        if(frame_result>0) continue;
        if(photo_pending==ACTION_NEXT_SONG) {
            int song_rc;
            controls_enabled=0;photo_pending=0;
            song_rc=(!sd_ready && mount_sd())?CAPTURE_NOT_READY:capture_store_album_next_song(stores);
            if(song_rc==CAPTURE_OK) {
                xil_printf("[SONG] selected %02u; page=00\r\n",stores[0].song);
                print_next_name();ui_status="NEXT SONG";
            } else {
                xil_printf("[SONG] change failed rc=%d; photos retained\r\n",song_rc);
                ui_status=song_rc==CAPTURE_SONG_FULL?"SONG FULL":"SD ERROR";
                if(song_rc!=CAPTURE_SONG_FULL) sd_ready=0;
            }
            status_until=milliseconds()+2000u;
            rearm_keys();controls_enabled=1;
            if(show_preview()) goto stopped;
            continue;
        }
        if(photo_pending && frame_controls_stable) {
            if(shoot_and_return()) goto stopped;
            continue;
        }
        frame_result=update_white_balance();
        if(frame_result<0) goto stopped;
        if(frame_result>0) continue;
        update_exposure();
        if(sensor_control_error) goto stopped;
        if(!status_until || (int32_t)(milliseconds()-status_until)>=0) {
            ui_status=photo_pending&&!frame_controls_stable?"SETTLING":(sd_ready?"LIVE":"SD ERROR");
            status_until=0;
        }
        /* Arm during vertical blank, then render the immutable completed
         * buffer while the next one-shot DMA writes the other allocation.
         * If rendering is slow, that DMA halts and waits; no CPU frame reuse. */
        if(begin_capture()) goto stopped;
        if(focus_measure_frame()) goto stopped;
        if(show_preview()) goto stopped;
        count_preview_frame();
    }
stopped:
    xil_printf("Capture/mode/display failure; stopping without saving an unverified frame.\r\n");
    controls_enabled=0;
    gpio_value|=GPIO_ABORT; XGpio_DiscreteWrite(&control,1,gpio_value);
    XAxiVdma_DmaStop(&dma,XAXIVDMA_WRITE);
    (void)imx477_stop(&sensor);
    return 1;
}
