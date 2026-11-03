/* Included after main.c's camera helpers: this layer alone touches board MMIO.
 * Sequence follows AMD FSBL XFsbl_UpdateMultiBoot (Vitis 2024.2).
 * Linux must use PMU config_reg, not /dev/mem, on the installed Ubuntu system. */
#include "capture_onecard.h"
#include "diskio.h"
#include "xplatform_info.h"
#define ONECARD_MULTI_BOOT ((UINTPTR)0xffca0010u)
#define ONECARD_RESET_CTRL ((UINTPTR)0xff5e0218u)

static int onecard_preflight(void *unused)
{
    FIL file;UINT got;FRESULT rc,closed;
    char text[1024];uint8_t block[4096];FSIZE_t length;
    capture_onecard_ready config;uint32_t crc=0xffffffffu;
    (void)unused;
    if(capture_inflight || fatfs.output_open || !sd_ready) return -1;
    if((Xil_In32((UINTPTR)0xff5e0200u)&15u)!=5u || (Xil_In32(ONECARD_MULTI_BOOT)&0xfffu)!=0u) {
        xil_printf("[ONECARD] Expected SD1 boot and capture slot0; refusing switch\r\n");return -1;
    }
    rc=f_open(&file,"0:/OneCard/READY.TXT",FA_READ);
    if(rc!=FR_OK) {xil_printf("[ONECARD] READY.TXT missing; card must be provisioned first\r\n");return -1;}
    length=f_size(&file);got=0;
    rc=length<sizeof(text)?f_read(&file,text,(UINT)length,&got):FR_INVALID_PARAMETER;
    closed=f_close(&file);
    if(rc!=FR_OK || closed!=FR_OK || got!=length || capture_onecard_parse_ready(text,got,&config)) return -1;
    rc=f_open(&file,"0:/BOOT0001.BIN",FA_READ);
    if(rc!=FR_OK) return -1;
    if(f_size(&file)!=config.linux_bytes) {f_close(&file);return -1;}
    do {
        rc=f_read(&file,block,sizeof(block),&got);
        if(rc!=FR_OK) break;
        if(crc==0xffffffffu && f_tell(&file)==got && capture_onecard_boot_header(block,got)) {rc=FR_INVALID_OBJECT;break;}
        crc=capture_onecard_crc32(crc,block,got);
    } while(got);
    closed=f_close(&file);
    if(rc!=FR_OK || closed!=FR_OK || (crc^0xffffffffu)!=config.linux_crc32) {
        xil_printf("[ONECARD] Linux BOOT size/header/CRC check failed; capture retained\r\n");return -1;
    }
    xil_printf("[ONECARD] Linux BOOT header/CRC verified; bytes=%u\r\n",config.linux_bytes);
    return 0;
}
static int onecard_quiesce(void *unused)
{
    uint32_t start=milliseconds();(void)unused;
    gpio_value|=GPIO_ABORT;XGpio_DiscreteWrite(&control,1,gpio_value);
    if(imx477_stop(&sensor)!=IMX477_OK) return -1;
    XAxiVdma_DmaStop(&dma,XAXIVDMA_WRITE);
    while(XAxiVdma_IsBusy(&dma,XAXIVDMA_WRITE)) {
        if(milliseconds()-start>500u) return -1;
        usleep(100);
    }
    capture_inflight=0;
    if(XCsiSs_Activate(&csi,0)!=XST_SUCCESS || capture_dp_shutdown()) return -1;
    xil_printf("[ONECARD] sensor, CSI, VDMA and DP DMA stopped/drained\r\n");
    return 0;
}
static int onecard_close_storage(void *unused)
{
    FIL file;UINT written;char text[256];int count;FRESULT rc,closed;FILINFO info;
    (void)unused;
    if(fatfs.output_open || fatfs.buffered) return -1;
    count=snprintf(text,sizeof(text),"{\"schema\":1,\"action\":\"recognize\",\"version\":\"a4_onecard_dp2\",\"song\":%u,\"next_page\":%u,\"uptime_ms\":%u}\n",
                   stores[0].song,stores[0].next_page,milliseconds());
    if(count<=0 || (size_t)count>=sizeof(text)) return -1;
    rc=f_open(&file,"0:/OneCard/REQUEST.TMP",FA_WRITE|FA_CREATE_ALWAYS);
    if(rc!=FR_OK) return -1;
    rc=f_write(&file,text,(UINT)count,&written);
    if(rc==FR_OK && written!=(UINT)count) rc=FR_DISK_ERR;
    if(rc==FR_OK) rc=f_sync(&file);
    closed=f_close(&file);
    if(rc!=FR_OK || closed!=FR_OK) return -1;
    rc=f_stat("0:/OneCard/REQUEST.JSON",&info);
    if(rc==FR_OK) rc=f_unlink("0:/OneCard/REQUEST.JSON");
    else if(rc==FR_NO_FILE) rc=FR_OK;
    if(rc!=FR_OK || f_rename("0:/OneCard/REQUEST.TMP","0:/OneCard/REQUEST.JSON")!=FR_OK) return -1;
    if(disk_ioctl(0,CTRL_SYNC,NULL)!=RES_OK || f_mount(NULL,"0:",0)!=FR_OK) return -1;
    sd_ready=0;
    xil_printf("[ONECARD] request committed; files closed, SD synchronized and FAT unmounted\r\n");
    return 0;
}
static int onecard_select_linux(void *unused)
{
    (void)unused;
    Xil_Out32(ONECARD_MULTI_BOOT,1u);dsb();
    if(Xil_In32(ONECARD_MULTI_BOOT)!=1u) {
        Xil_Out32(ONECARD_MULTI_BOOT,0u);dsb();return -1;
    }
    xil_printf("[ONECARD] CSU_MULTI_BOOT=1 readback OK -> BOOT0001.BIN\r\n");
    return 0;
}
static int onecard_reset(void *unused)
{
    (void)unused;
    xil_printf("[ONECARD] system soft reset to Ubuntu; power cycle always returns to capture\r\n");
    usleep(5000);Xil_DCacheFlush();
    /* Silicon v1.0 workaround from the installed FSBL. */
    if(XGetPSVersion_Info()==XPS_VERSION_1)
        Xil_Out32((UINTPTR)0xff5e0030u,Xil_In32((UINTPTR)0xff5e0030u)|8u);
    dsb();isb();Xil_Out32(ONECARD_RESET_CTRL,Xil_In32(ONECARD_RESET_CTRL)|0x10u);
    dsb();usleep(1000000);
    /* A reset that did not occur is a failure, never silently leave slot1 set. */
    Xil_Out32(ONECARD_MULTI_BOOT,0u);dsb();return -1;
}
static int switch_to_ubuntu(void)
{
    const capture_onecard_ops ops={{onecard_preflight,onecard_quiesce,onecard_close_storage,
                                    onecard_select_linux,onecard_reset},NULL};
    enum capture_onecard_stage stage=ONECARD_PREFLIGHT;
    struct imx477 settings=sensor;
    controls_enabled=0;onecard_controls_ready=0;ubuntu_pending=0;
    ui_status="UBUNTU";(void)show_preview();
    xil_printf("[ONECARD] S3/u requested; capture transaction complete\r\n");
    (void)capture_onecard_switch(&ops,&stage); /* successful reset cannot return */
    xil_printf("[ONECARD] ABORT stage=%u; no handoff completed\r\n",(unsigned)stage);
    if(stage>=ONECARD_QUIESCE) {
        if(!sd_ready && mount_sd()) return -1;
        if(switch_sensor_mode(current_mode,&settings,NULL)) return -1;
    }
    ui_status="SWITCH ERROR";status_until=milliseconds()+5000u;
    rearm_keys();onecard_controls_ready=1;controls_enabled=1;
    return 0;
}
