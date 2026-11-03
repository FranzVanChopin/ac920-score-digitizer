# AC920 + Raspberry Pi HQ IMX477 full-frame capture and HDMI preview.
# Vivado 2024.2. Run from any directory; output stays in build/vivado_camera.
# Set CAMERA_GENERATE=1 in the environment to generate HDL/IP output products.
# Synthesis/implementation are separate explicit build steps.
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_camera]
# A failed rebuild must not leave an old bitstream eligible for Bootgen.
file delete -force [file join $project_dir hardware.complete]
file delete -force [file join $project_dir hardware_validation.json]
set gate_file [file join $repo_root src rtl camera axis_frame_gate.v]
if {![file exists $gate_file]} { error "Missing frame gate RTL: $gate_file" }
create_project camera_capture $project_dir -part xczu4ev-sfvc784-1L-i -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
add_files -norecurse [list $gate_file \
    [file join $repo_root src rtl camera camera_isp_control.v] \
    [file join $repo_root src rtl camera camera_dvi_wrapper.v] \
    [file join $repo_root src rtl vendor dvi_tx dvi_transmitter_top.v] \
    [file join $repo_root src rtl vendor dvi_tx dvi_encoder.v] \
    [file join $repo_root src rtl vendor dvi_tx encode.v] \
    [file join $repo_root src rtl vendor dvi_tx serdes_4b_10to1.v]]
update_compile_order -fileset sources_1
create_bd_design camera_capture
current_bd_design camera_capture

proc camera_ip {name type version props} {
    set cell [create_bd_cell -type ip -vlnv xilinx.com:ip:${type}:${version} $name]
    if {[llength $props]} {set_property -dict $props $cell}
    return $cell
}
proc camera_net {args} {
    set pins {}
    foreach pin $args {lappend pins [get_bd_pins $pin]}
    connect_bd_net {*}$pins
}
proc camera_axis {a b} {
    connect_bd_intf_net [get_bd_intf_pins $a] [get_bd_intf_pins $b]
}

set ps [camera_ip ps zynq_ultra_ps_e 3.5 {}]
source -notrace [file join $repo_root build vivado ac920_ps_preset.tcl]
apply_ac920_ps_preset $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 0 CONFIG.PSU__USE__M_AXI_GP1 0 \
    CONFIG.PSU__USE__M_AXI_GP2 1 CONFIG.PSU__USE__S_AXI_GP2 1 \
    CONFIG.PSU__SAXIGP2__DATA_WIDTH 128 \
    CONFIG.PSU__FPGA_PL0_ENABLE 1 CONFIG.PSU__FPGA_PL1_ENABLE 1 \
    CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ 100 \
    CONFIG.PSU__CRL_APB__PL1_REF_CTRL__FREQMHZ 200 \
    CONFIG.PSU__I2C0__PERIPHERAL__ENABLE 1 CONFIG.PSU__I2C0__PERIPHERAL__IO EMIO \
    CONFIG.PSU__SD0__PERIPHERAL__ENABLE 0 \
    CONFIG.PSU__GPIO_EMIO__PERIPHERAL__ENABLE 0 \
    CONFIG.PSU__GPIO0_MIO__PERIPHERAL__ENABLE 1 \
    CONFIG.PSU__GPIO1_MIO__PERIPHERAL__ENABLE 1 \
    CONFIG.PSU__GPIO2_MIO__PERIPHERAL__ENABLE 1] $ps
# PS_KEY is active-low MIO44. UART1 MIO32/33 and SD1 MIO46..51
# retain the board preset; camera I2C is PS I2C0 through EMIO.
make_bd_intf_pins_external -name camera_i2c [get_bd_intf_pins ps/IIC_0]

camera_ip csi_rx mipi_csi2_rx_subsystem 6.0 [list \
    CONFIG.CMN_NUM_LANES 2 CONFIG.CMN_NUM_PIXELS 1 \
    CONFIG.CMN_PXL_FORMAT RAW12 CONFIG.C_EN_CSI_V2_0 false \
    CONFIG.CMN_VC 0 CONFIG.C_CSI_EN_CRC true \
    CONFIG.CSI_EMB_NON_IMG false CONFIG.C_CSI_FILTER_USERDATATYPE true \
    CONFIG.DPY_LINE_RATE 900 CONFIG.HP_IO_BANK_SELECTION 65 \
    CONFIG.CLK_LANE_IO_LOC W8 CONFIG.DATA_LANE0_IO_LOC U9 \
    CONFIG.DATA_LANE1_IO_LOC U8 CONFIG.SupportLevel 1]
# MIPI IP creates its external PHY interface during customization.
set phy_net [get_bd_intf_nets -quiet -of_objects [get_bd_intf_pins csi_rx/mipi_phy_if]]
if {[llength $phy_net] == 0} {
    make_bd_intf_pins_external -name camera_mipi [get_bd_intf_pins csi_rx/mipi_phy_if]
} else {
    set phy_port [get_bd_intf_ports -quiet -of_objects $phy_net]
    if {[llength $phy_port]} {set_property name camera_mipi $phy_port}
}
camera_ip demosaic v_demosaic 1.1 [list CONFIG.MAX_COLS 4056 \
    CONFIG.MAX_ROWS 3040 CONFIG.MAX_DATA_WIDTH 12 CONFIG.SAMPLES_PER_CLOCK 1 \
    CONFIG.USE_URAM 1]
camera_ip gamma v_gamma_lut 1.1 [list CONFIG.MAX_COLS 4056 \
    CONFIG.MAX_ROWS 3040 CONFIG.MAX_DATA_WIDTH 10 CONFIG.SAMPLES_PER_CLOCK 1]

# PG286 packs 12-bit components as {padding[3:0], R[11:0], B[11:0], G[11:0]}.
# Gamma LUT v1.1 accepts 8/10 bpc, so preserve demosaic precision then reduce
# each component to its most significant 10 bits before gamma correction.
camera_ip rgb12_to_rgb10 axis_subset_converter 1.1 [list \
    CONFIG.S_TDATA_NUM_BYTES 5 CONFIG.M_TDATA_NUM_BYTES 4 \
    CONFIG.S_HAS_TREADY 1 CONFIG.M_HAS_TREADY 1 \
    CONFIG.S_HAS_TLAST 1 CONFIG.M_HAS_TLAST 1 \
    CONFIG.S_TUSER_WIDTH 1 CONFIG.M_TUSER_WIDTH 1 \
    CONFIG.TDATA_REMAP {2'b00,tdata[35:26],tdata[23:14],tdata[11:2]}]
# Output 0xRRGGBB; the little-endian DDR byte order is B,G,R (BMP compatible).
camera_ip rgb10_to_rgb8 axis_subset_converter 1.1 [list \
    CONFIG.S_TDATA_NUM_BYTES 4 CONFIG.M_TDATA_NUM_BYTES 3 \
    CONFIG.S_HAS_TREADY 1 CONFIG.M_HAS_TREADY 1 \
    CONFIG.S_HAS_TLAST 1 CONFIG.M_HAS_TLAST 1 \
    CONFIG.S_TUSER_WIDTH 1 CONFIG.M_TUSER_WIDTH 1 \
    CONFIG.TDATA_REMAP {tdata[29:22],tdata[9:2],tdata[19:12]}]
create_bd_cell -type module -reference axis_frame_gate frame_gate
set_property -dict [list CONFIG.FRAME_WIDTH 4056 CONFIG.FRAME_HEIGHT 3040 \
    CONFIG.PREVIEW_WIDTH 2028 CONFIG.PREVIEW_HEIGHT 1520] [get_bd_cells frame_gate]
create_bd_cell -type module -reference camera_isp_control isp_control

camera_ip vdma axi_vdma 6.3 [list \
    CONFIG.c_num_fstores 3 CONFIG.c_include_mm2s 1 CONFIG.c_include_s2mm 1 \
    CONFIG.c_m_axi_mm2s_data_width 128 CONFIG.c_m_axi_s2mm_data_width 128 \
    CONFIG.c_m_axis_mm2s_tdata_width 24 \
    CONFIG.c_include_mm2s_dre 1 CONFIG.c_include_s2mm_dre 1 \
    CONFIG.c_mm2s_linebuffer_depth 4096 CONFIG.c_s2mm_linebuffer_depth 8192 \
    CONFIG.c_mm2s_max_burst_length 64 CONFIG.c_s2mm_max_burst_length 64 \
    CONFIG.c_use_fsync 0 CONFIG.c_use_mm2s_fsync 0 CONFIG.c_use_s2mm_fsync 0 \
    CONFIG.c_flush_on_fsync 0 \
    CONFIG.c_mm2s_sof_enable 1 CONFIG.c_s2mm_sof_enable 1 \
    CONFIG.c_include_internal_genlock 0 \
    CONFIG.c_mm2s_genlock_mode 0 CONFIG.c_s2mm_genlock_mode 0 \
    CONFIG.c_addr_width 32 CONFIG.c_include_sg 0]
camera_ip camera_gpio axi_gpio 2.0 [list \
    CONFIG.C_GPIO_WIDTH 5 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_DOUT_DEFAULT 0x00000000 \
    CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH 8 CONFIG.C_ALL_INPUTS_2 1]
foreach {name bit} {enable_bit 0 request_bit 1 abort_bit 2 mode_bit 3 isp_reset_bit 4} {
    camera_ip $name xlslice 1.0 [list CONFIG.DIN_WIDTH 5 CONFIG.DIN_FROM $bit \
        CONFIG.DIN_TO $bit CONFIG.DOUT_WIDTH 1]
    camera_net camera_gpio/gpio_io_o $name/Din
}
make_bd_pins_external -name camera_enable [get_bd_pins enable_bit/Dout]
camera_net request_bit/Dout frame_gate/frame_req_toggle
camera_net abort_bit/Dout frame_gate/frame_abort
camera_net mode_bit/Dout frame_gate/preview_mode
camera_net isp_reset_bit/Dout isp_control/reset_request
camera_net frame_gate/capture_status isp_control/gate_status
camera_net isp_control/control_status camera_gpio/gpio2_io_i
# Only PS S1 is used. Do not bring S3/AH14 into the HDMI I/O bank.

camera_ip video_out v_axi4s_vid_out 4.0 [list CONFIG.C_HAS_ASYNC_CLK 1]
# The board HDMI pins are HDIO: maximum DDR output rate is 250 Mb/s.
# 25 MHz pixels / 125 MHz serializer respect the 8 ns ODDR minimum period.
# VGA totals 800 x 525 give 59.5238 Hz; monitor acceptance needs board testing.
camera_ip vtc v_tc 6.2 [list CONFIG.VIDEO_MODE 480p CONFIG.enable_detection false]
camera_ip video_clk clk_wiz 6.0 [list CONFIG.PRIM_IN_FREQ 100 \
    CONFIG.PRIM_SOURCE Differential_clock_capable_pin CONFIG.NUM_OUT_CLKS 2 \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ 25 CONFIG.CLKOUT2_USED true \
    CONFIG.CLKOUT2_REQUESTED_OUT_FREQ 125 \
    CONFIG.RESET_TYPE ACTIVE_LOW CONFIG.RESET_PORT resetn]
make_bd_intf_pins_external -name board_clk [get_bd_intf_pins video_clk/CLK_IN1_D]
create_bd_cell -type module -reference camera_dvi_wrapper hdmi
foreach {internal external} {tmds_clk_p hdmi_clk_p tmds_clk_n hdmi_clk_n \
    tmds_data_p hdmi_data_p tmds_data_n hdmi_data_n} {
    make_bd_pins_external -name $external [get_bd_pins hdmi/$internal]
}

camera_ip control_axi smartconnect 1.0 [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 6 CONFIG.NUM_CLKS 2]
camera_ip memory_axi smartconnect 1.0 [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1]
camera_axis ps/M_AXI_HPM0_LPD control_axi/S00_AXI
foreach {index slave} {0 vdma/S_AXI_LITE 1 csi_rx/csirxss_s_axi 2 demosaic/s_axi_CTRL \
    3 gamma/s_axi_CTRL 4 vtc/ctrl 5 camera_gpio/S_AXI} {
    camera_axis [format control_axi/M%02d_AXI $index] $slave
}
camera_axis vdma/M_AXI_MM2S memory_axi/S00_AXI
camera_axis vdma/M_AXI_S2MM memory_axi/S01_AXI
camera_axis memory_axi/M00_AXI ps/S_AXI_HP0_FPD
camera_axis csi_rx/video_out demosaic/s_axis_video
camera_axis demosaic/m_axis_video rgb12_to_rgb10/S_AXIS
camera_axis rgb12_to_rgb10/M_AXIS gamma/s_axis_video
camera_axis gamma/m_axis_video rgb10_to_rgb8/S_AXIS
camera_axis rgb10_to_rgb8/M_AXIS frame_gate/s_axis
camera_axis frame_gate/m_axis vdma/S_AXIS_S2MM
camera_axis vdma/M_AXIS_MM2S video_out/video_in
camera_axis vtc/vtiming_out video_out/vtiming_in
camera_net video_out/vtg_ce vtc/gen_clken
camera_net video_out/vid_data hdmi/video_in_data
camera_net video_out/vid_active_video hdmi/video_in_de
camera_net video_out/vid_hsync hdmi/video_in_hs
camera_net video_out/vid_vsync hdmi/video_in_vs

camera_ip one xlconstant 1.1 [list CONFIG.CONST_VAL 1 CONFIG.CONST_WIDTH 1]
camera_ip zero xlconstant 1.1 [list CONFIG.CONST_VAL 0 CONFIG.CONST_WIDTH 1]
foreach name {ctrl_reset video_reset pixel_reset} {
    # ext_reset_in follows the active-low PS reset metadata. The unused aux
    # input is tied low, so explicitly make it active-high: the IP default is
    # active-low and would otherwise hold every AXI/video peripheral in reset.
    camera_ip $name proc_sys_reset 5.0 [list CONFIG.C_AUX_RESET_HIGH 1]
    camera_net ps/pl_resetn0 $name/ext_reset_in
    camera_net zero/dout $name/aux_reset_in $name/mb_debug_sys_rst
}
camera_net one/dout ctrl_reset/dcm_locked video_reset/dcm_locked
camera_net video_clk/locked pixel_reset/dcm_locked
camera_net ps/pl_clk0 ps/maxihpm0_lpd_aclk control_axi/aclk ctrl_reset/slowest_sync_clk \
    csi_rx/lite_aclk vdma/s_axi_lite_aclk vtc/s_axi_aclk camera_gpio/s_axi_aclk frame_gate/ctrl_clk isp_control/ctrl_clk
camera_net ps/pl_clk1 ps/saxihp0_fpd_aclk control_axi/aclk1 memory_axi/aclk \
    video_reset/slowest_sync_clk csi_rx/video_aclk csi_rx/dphy_clk_200M \
    demosaic/ap_clk gamma/ap_clk rgb12_to_rgb10/aclk rgb10_to_rgb8/aclk frame_gate/clk isp_control/clk \
    vdma/m_axi_mm2s_aclk vdma/m_axis_mm2s_aclk vdma/m_axi_s2mm_aclk vdma/s_axis_s2mm_aclk \
    video_out/aclk
camera_net video_clk/clk_out1 pixel_reset/slowest_sync_clk vtc/clk video_out/vid_io_out_clk hdmi/pixel_clk
camera_net video_clk/clk_out2 hdmi/pixelclk5x
camera_net ctrl_reset/peripheral_aresetn control_axi/aresetn csi_rx/lite_aresetn \
    vdma/axi_resetn vtc/s_axi_aresetn camera_gpio/s_axi_aresetn video_clk/resetn frame_gate/ctrl_resetn isp_control/ctrl_resetn
camera_net video_reset/peripheral_aresetn memory_axi/aresetn csi_rx/video_aresetn \
    frame_gate/resetn video_out/aresetn isp_control/resetn
camera_net isp_control/isp_resetn demosaic/ap_rst_n gamma/ap_rst_n \
    rgb12_to_rgb10/aresetn rgb10_to_rgb8/aresetn
camera_net pixel_reset/peripheral_aresetn vtc/resetn hdmi/rst_n
camera_net pixel_reset/peripheral_reset video_out/vid_io_out_reset
camera_net one/dout video_out/aclken video_out/vid_io_out_ce vtc/clken vtc/s_axi_aclken
camera_net zero/dout video_out/fid

foreach {address segment range} {
    0x80000000 vdma/S_AXI_LITE/Reg 64K
    0x80010000 csi_rx/csirxss_s_axi/Reg 4K
    0x80020000 demosaic/s_axi_CTRL/Reg 64K
    0x80030000 gamma/s_axi_CTRL/Reg 64K
    0x80040000 vtc/ctrl/Reg 64K
    0x80050000 camera_gpio/S_AXI/Reg 64K
} {
    assign_bd_address -offset $address -range $range -target_address_space \
        [get_bd_addr_spaces ps/Data] [get_bd_addr_segs $segment]
}
foreach space {vdma/Data_MM2S vdma/Data_S2MM} {
    assign_bd_address -offset 0x00000000 -range 2G -target_address_space \
        [get_bd_addr_spaces $space] [get_bd_addr_segs ps/SAXIGP2/HP0_DDR_LOW]
}
validate_bd_design
# ISP reset is separate from every DMA, CSI receiver and display reset.
# Software must assert gate abort, stop/drain the sensor, assert bit4 and poll
# status bit5 before reset release; then reinitialize HLS dimensions/LUTs.
foreach sink {demosaic/ap_rst_n gamma/ap_rst_n rgb12_to_rgb10/aresetn rgb10_to_rgb8/aresetn} {
    if {[get_bd_nets -of_objects [get_bd_pins $sink]] ne
        [get_bd_nets -of_objects [get_bd_pins isp_control/isp_resetn]]} {
        error "ISP reset wiring mismatch: $sink"
    }
}
foreach sink {vdma/axi_resetn csi_rx/video_aresetn video_out/aresetn frame_gate/resetn} {
    if {[get_bd_nets -of_objects [get_bd_pins $sink]] eq
        [get_bd_nets -of_objects [get_bd_pins isp_control/isp_resetn]]} {
        error "ISP reset must not reset $sink"
    }
}
if {[get_property CONFIG.C_GPIO_WIDTH [get_bd_cells camera_gpio]] != 5 ||
    [get_property CONFIG.PREVIEW_WIDTH [get_bd_cells frame_gate]] != 2028 ||
    [get_property CONFIG.PREVIEW_HEIGHT [get_bd_cells frame_gate]] != 1520} {
    error "Dual-mode GPIO/frame-gate configuration mismatch"
}
# C_EXT_RESET_HIGH is propagated-only in AMD's IP Tcl; C_AUX_RESET_HIGH is
# user-overridable. Check both the resulting polarity and physical BD wiring.
# Fail generation if reset metadata or data-clock connections ever change.
# VDMA permits a slower AXI-Lite clock while all four data clocks share pl_clk1;
# c_prmry_is_aclk_async describes the data clocks, not the AXI-Lite crossing.
foreach name {ctrl_reset video_reset pixel_reset} {
    if {[get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells $name]] != 0} {
        error "$name must have an active-low external reset from ps/pl_resetn0"
    }
    if {[get_property CONFIG.C_AUX_RESET_HIGH [get_bd_cells $name]] != 1 ||
        [get_property CONFIG.POLARITY [get_bd_pins $name/aux_reset_in]] ne "ACTIVE_HIGH"} {
        error "$name aux_reset_in must be active-high because it is tied to zero"
    }
    foreach {driver sink} [list ps/pl_resetn0 $name/ext_reset_in \
        zero/dout $name/aux_reset_in zero/dout $name/mb_debug_sys_rst] {
        set driver_net [get_bd_nets -of_objects [get_bd_pins $driver]]
        set sink_net [get_bd_nets -of_objects [get_bd_pins $sink]]
        if {[llength $driver_net] != 1 || $driver_net ne $sink_net} {
            error "Reset connection changed: $sink must be driven by $driver"
        }
    }
    if {[get_property CONFIG.POLARITY [get_bd_pins $name/mb_debug_sys_rst]] ne "ACTIVE_HIGH"} {
        error "$name mb_debug_sys_rst must be active-high because it is tied to zero"
    }
}
if {[get_property CONFIG.CONST_VAL [get_bd_cells zero]] != 0} {
    error "Unused auxiliary/debug resets require the zero constant"
}
if {[get_property CONFIG.c_prmry_is_aclk_async [get_bd_cells vdma]] != 0} {
    error "VDMA data clocks must all share ps/pl_clk1"
}
if {[get_property CONFIG.c_s_axis_s2mm_tdata_width [get_bd_cells vdma]] != 24} {
    error "VDMA capture stream must remain 24-bit packed BGR"
}
# The gate supplies exactly one complete frame. TUSER-based S2MM frame counting
# waits for the following SOF, which the closed gate cannot provide. Free-run
# completes from programmed VSIZE after commands/status and FIFOs drain.
foreach parameter {c_use_fsync c_use_mm2s_fsync c_use_s2mm_fsync c_flush_on_fsync} {
    if {[get_property CONFIG.$parameter [get_bd_cells vdma]] != 0} {
        error "VDMA $parameter must be zero for gated single-frame capture"
    }
}
save_bd_design
add_files -fileset constrs_1 [file join $repo_root build vivado ac920_camera_capture_pins.xdc]
set cdc_file [file join $repo_root build vivado ac920_camera_capture_cdc.xdc]
add_files -fileset constrs_1 $cdc_file
# Module-reference OOC registers are linked at implementation, not top synthesis.
set_property USED_IN_SYNTHESIS false [get_files $cdc_file]
set bd_file [get_files [file join $project_dir camera_capture.srcs sources_1 bd camera_capture camera_capture.bd]]
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top camera_capture_wrapper [current_fileset]
update_compile_order -fileset sources_1
report_ip_status -file [file join $project_dir ip_status.rpt]
if {[info exists ::env(CAMERA_GENERATE)] && $::env(CAMERA_GENERATE) eq "1"} {
    generate_target all $bd_file
    export_ip_user_files -of_objects $bd_file -no_script -sync -force -quiet
    write_hw_platform -fixed -force -file [file join $project_dir camera_capture.xsa]
}
puts "CAMERA_CAPTURE_BD_VALIDATED: $project_dir"
close_project
