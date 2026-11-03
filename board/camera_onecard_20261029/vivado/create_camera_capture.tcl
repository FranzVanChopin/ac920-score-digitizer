# AC920 pure PS DP + IMX477 capture + synchronized S2/S3. No HDMI PL logic.
# Vivado 2024.2. Run from any directory; output stays in build/vivado_camera.
# Set CAMERA_GENERATE=1 in the environment to generate HDL/IP output products.
# Synthesis/implementation are separate explicit build steps.
set repo_root [file normalize [file join [file dirname [info script]] ..]]
set project_dir [file join $repo_root build vivado_camera]
# A failed rebuild must not leave an old bitstream eligible for Bootgen.
file delete -force [file join $project_dir hardware.complete]
file delete -force [file join $project_dir hardware_validation.json]
set gate_file [file join $repo_root rtl camera axis_frame_gate.v]
if {![file exists $gate_file]} { error "Missing frame gate RTL: $gate_file" }
create_project camera_capture $project_dir -part xczu4ev-sfvc784-1L-i -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
add_files -norecurse [list $gate_file \
    [file join $repo_root rtl camera camera_isp_control.v] \
    [file join $repo_root rtl camera camera_key_sync.v]]
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
source -notrace [file join $repo_root vivado ac920_ps_preset.tcl]
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
    CONFIG.c_num_fstores 3 CONFIG.c_include_mm2s 0 CONFIG.c_include_s2mm 1 \
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
    CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH 10 CONFIG.C_ALL_INPUTS_2 1]
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
camera_ip capture_status xlconcat 2.1 [list CONFIG.NUM_PORTS 2 CONFIG.IN0_WIDTH 8 CONFIG.IN1_WIDTH 2]
create_bd_cell -type module -reference camera_key_sync key_sync
foreach key {s2 s3} {
    create_bd_port -dir I key_${key}_n
    connect_bd_net [get_bd_ports key_${key}_n] [get_bd_pins key_sync/key_${key}_n]
}
camera_net isp_control/control_status capture_status/In0
camera_net key_sync/keys_n capture_status/In1
camera_net capture_status/dout camera_gpio/gpio2_io_i
camera_ip control_axi smartconnect 1.0 [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 5 CONFIG.NUM_CLKS 2]
camera_ip memory_axi smartconnect 1.0 [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 1]
camera_axis ps/M_AXI_HPM0_LPD control_axi/S00_AXI
foreach {index slave} {0 vdma/S_AXI_LITE 1 csi_rx/csirxss_s_axi 2 demosaic/s_axi_CTRL \
    3 gamma/s_axi_CTRL 4 camera_gpio/S_AXI} {
    camera_axis [format control_axi/M%02d_AXI $index] $slave
}
camera_axis vdma/M_AXI_S2MM memory_axi/S00_AXI
camera_axis memory_axi/M00_AXI ps/S_AXI_HP0_FPD
camera_axis csi_rx/video_out demosaic/s_axis_video
camera_axis demosaic/m_axis_video rgb12_to_rgb10/S_AXIS
camera_axis rgb12_to_rgb10/M_AXIS gamma/s_axis_video
camera_axis gamma/m_axis_video rgb10_to_rgb8/S_AXIS
camera_axis rgb10_to_rgb8/M_AXIS frame_gate/s_axis
camera_axis frame_gate/m_axis vdma/S_AXIS_S2MM
camera_ip one xlconstant 1.1 [list CONFIG.CONST_VAL 1 CONFIG.CONST_WIDTH 1]
camera_ip zero xlconstant 1.1 [list CONFIG.CONST_VAL 0 CONFIG.CONST_WIDTH 1]
foreach name {ctrl_reset video_reset} {
    # ext_reset_in follows the active-low PS reset metadata. The unused aux
    # input is tied low, so explicitly make it active-high: the IP default is
    # active-low and would otherwise hold every AXI/video peripheral in reset.
    camera_ip $name proc_sys_reset 5.0 [list CONFIG.C_AUX_RESET_HIGH 1]
    camera_net ps/pl_resetn0 $name/ext_reset_in
    camera_net zero/dout $name/aux_reset_in $name/mb_debug_sys_rst
}
camera_net one/dout ctrl_reset/dcm_locked video_reset/dcm_locked
camera_net ps/pl_clk0 ps/maxihpm0_lpd_aclk control_axi/aclk ctrl_reset/slowest_sync_clk \
    csi_rx/lite_aclk vdma/s_axi_lite_aclk camera_gpio/s_axi_aclk frame_gate/ctrl_clk isp_control/ctrl_clk key_sync/clk
camera_net ps/pl_clk1 ps/saxihp0_fpd_aclk control_axi/aclk1 memory_axi/aclk \
    video_reset/slowest_sync_clk csi_rx/video_aclk csi_rx/dphy_clk_200M \
    demosaic/ap_clk gamma/ap_clk rgb12_to_rgb10/aclk rgb10_to_rgb8/aclk frame_gate/clk isp_control/clk \
    vdma/m_axi_s2mm_aclk vdma/s_axis_s2mm_aclk
camera_net ctrl_reset/peripheral_aresetn control_axi/aresetn csi_rx/lite_aresetn \
    vdma/axi_resetn camera_gpio/s_axi_aresetn frame_gate/ctrl_resetn isp_control/ctrl_resetn key_sync/resetn
camera_net video_reset/peripheral_aresetn memory_axi/aresetn csi_rx/video_aresetn \
    frame_gate/resetn isp_control/resetn
camera_net isp_control/isp_resetn demosaic/ap_rst_n gamma/ap_rst_n \
    rgb12_to_rgb10/aresetn rgb10_to_rgb8/aresetn
foreach {address segment range} {
    0x80000000 vdma/S_AXI_LITE/Reg 64K
    0x80010000 csi_rx/csirxss_s_axi/Reg 4K
    0x80020000 demosaic/s_axi_CTRL/Reg 64K
    0x80030000 gamma/s_axi_CTRL/Reg 64K
    0x80050000 camera_gpio/S_AXI/Reg 64K
} {
    assign_bd_address -offset $address -range $range -target_address_space \
        [get_bd_addr_spaces ps/Data] [get_bd_addr_segs $segment]
}
foreach space {vdma/Data_S2MM} {
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
foreach sink {vdma/axi_resetn csi_rx/video_aresetn frame_gate/resetn} {
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
# VDMA permits a slower AXI-Lite clock while both capture data clocks share pl_clk1;
# c_prmry_is_aclk_async describes the data clocks, not the AXI-Lite crossing.
foreach name {ctrl_reset video_reset} {
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
add_files -fileset constrs_1 [file join $repo_root vivado ac920_camera_capture_pins.xdc]
set cdc_file [file join $repo_root vivado ac920_camera_capture_cdc.xdc]
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
