/* Host hardware model: only the SDK boundary is replaced. */
#ifndef DP_TESTENV_H
#define DP_TESTENV_H
#include <stdint.h>
#include <stddef.h>
typedef uint8_t u8;
typedef uint32_t u32;
typedef uintptr_t UINTPTR;
typedef intptr_t INTPTR;
typedef uint64_t XTime;
#define COUNTS_PER_SECOND 1000000u
#define XST_SUCCESS 0
#define XPAR_XDPPSU_0_DEVICE_ID 0
#define XPAR_XDPDMA_0_DEVICE_ID 0
#define XDPDMA_ISR 0x004u
#define XDPDMA_IDS 0x00cu
#define XDPDMA_EISR 0x010u
#define XDPDMA_CH3_STATUS 0x51cu
#define XDPDMA_CH3_DSCR_ID 0x528u
#define XDPDMA_CH3_PYLD_CUR_ADDRE 0x514u
#define XDPDMA_CH3_PYLD_CUR_ADDR 0x510u
#define XDPDMA_CH_STATUS_OTRAN_CNT_MASK 0x01e00000u
#define XDPDMA_ISR_VSYNC_INT_MASK 0x08000000u
#define XDPPSU_DPCD_SET_POWER_DP_PWR_VOLTAGE 0x600u
#define XDPPSU_SOFT_RESET 0x01cu
#define XDPPSU_INTR_DIS 0x3acu
enum {GraphicsChan, XDPDMA_PAUSE, XDPDMA_DISABLE, XDPDMA_ENABLE,
      RGBA8888, RGB_8BPC, XDPPSU_CENC_RGB, XVIDC_BPC_8,
      XVIDC_VM_1280x720_60_P, XAVBUF_VIDSTREAM1_NONE,
      XAVBUF_VIDSTREAM2_NONLIVE_GFX, XAVBUF_PS_CLK};
typedef struct {UINTPTR BaseAddr;} XDpPsu_Config;
typedef struct {UINTPTR BaseAddr;} XDpDma_Config;
typedef struct {
    XDpPsu_Config Config;
    struct {unsigned LaneCount,LinkRate,MaxLaneCount,MaxLinkRate;
            unsigned SupportEnhancedFramingMode,SupportDownspreadControl;} LinkConfig;
    struct {unsigned PixelClockHz;} MsaConfig;
} XDpPsu;
typedef struct {unsigned DSCR_ID;} XDpDma_Descriptor;
typedef struct {UINTPTR Address;unsigned Stride,LineSize,Size;} XDpDma_FrameBuffer;
typedef struct {
    XDpDma_Config Config;
    struct {struct {XDpDma_Descriptor *Current;} Channel;
            XDpDma_FrameBuffer *FrameBuffer;} Gfx;
} XDpDma;
typedef struct {unsigned dummy;} XAVBuf;

XDpPsu_Config *XDpPsu_LookupConfig(unsigned id);
XDpDma_Config *XDpDma_LookupConfig(unsigned id);
void XDpPsu_CfgInitialize(XDpPsu *,XDpPsu_Config *,UINTPTR);
void XDpDma_CfgInitialize(XDpDma *,XDpDma_Config *);
int XDpPsu_InitializeTx(XDpPsu *);
int XDpPsu_IsConnected(XDpPsu *);
int XDpPsu_AuxWrite(XDpPsu *,unsigned,unsigned,u8 *);
int XDpPsu_GetRxCapabilities(XDpPsu *);
int XDpPsu_SetEnhancedFrameMode(XDpPsu *,unsigned);
int XDpPsu_SetLaneCount(XDpPsu *,u8);
int XDpPsu_SetLinkRate(XDpPsu *,u8);
int XDpPsu_SetDownspread(XDpPsu *,unsigned);
int XDpPsu_EstablishLink(XDpPsu *);
void XDpPsu_EnableMainLink(XDpPsu *,unsigned);
int XDpPsu_CheckLinkStatus(XDpPsu *,unsigned);
void XDpPsu_SetColorEncode(XDpPsu *,unsigned);
void XDpPsu_CfgMsaSetBpc(XDpPsu *,unsigned);
void XDpPsu_CfgMsaUseStandardVideoMode(XDpPsu *,unsigned);
void XDpPsu_CfgMsaEnSynchClkMode(XDpPsu *,unsigned);
void XDpPsu_SetMsaValues(XDpPsu *);
void XDpPsu_WriteReg(UINTPTR,unsigned,unsigned);
int XDpDma_SetGraphicsFormat(XDpDma *,unsigned);
void XDpDma_SetQOS(XDpDma *,unsigned);
void XDpDma_WriteReg(UINTPTR,unsigned,unsigned);
u32 XDpDma_ReadReg(UINTPTR,unsigned);
int XDpDma_SetChannelState(XDpDma *,unsigned,unsigned);
void XDpDma_DisplayGfxFrameBuffer(XDpDma *,XDpDma_FrameBuffer *);
void XDpDma_SetupChannel(XDpDma *,unsigned);
int XDpDma_Trigger(XDpDma *,unsigned);
int XDpDma_ReTrigger(XDpDma *,unsigned);
void XAVBuf_CfgInitialize(XAVBuf *,UINTPTR,unsigned);
int XAVBuf_SetInputNonLiveGraphicsFormat(XAVBuf *,unsigned);
int XAVBuf_SetOutputVideoFormat(XAVBuf *,unsigned);
void XAVBuf_EnableGraphicsBuffers(XAVBuf *,unsigned);
void XAVBuf_InputVideoSelect(XAVBuf *,unsigned,unsigned);
void XAVBuf_ConfigureGraphicsPipeline(XAVBuf *);
void XAVBuf_ConfigureOutputVideo(XAVBuf *);
void XAVBuf_SetBlenderAlpha(XAVBuf *,unsigned,unsigned);
void XAVBuf_SetAudioVideoClkSrc(XAVBuf *,unsigned,unsigned);
void XAVBuf_SoftReset(XAVBuf *);
int XAVBuf_SetPixelClock(uint64_t);
void Xil_DCacheFlushRange(INTPTR,size_t);
int xil_printf(const char *,...);
void XTime_GetTime(XTime *);
void usleep(unsigned);
#endif
