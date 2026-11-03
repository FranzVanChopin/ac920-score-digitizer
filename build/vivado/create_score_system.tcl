# Build the AC920 Processing System + dual AXI score peripheral block design.
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_score_system]
create_project score_system $project_dir -part xczu4ev-sfvc784-1L-i -force
set_property target_language Verilog [current_project]

add_files -norecurse [list \
    [file join $repo_root src rtl score_video_core.sv] \
    [file join $repo_root src rtl score_video_axi_control.sv] \
    [file join $repo_root src rtl score_event_player.sv] \
    [file join $repo_root src rtl axi_score_event_player.sv] \
    [file join $repo_root src rtl score_hdmi_top.sv] \
    [file join $repo_root src rtl score_hdmi_bd_wrapper.v] \
    [file join $repo_root src rtl vendor dvi_tx dvi_transmitter_top.v] \
    [file join $repo_root src rtl vendor dvi_tx dvi_encoder.v] \
    [file join $repo_root src rtl vendor dvi_tx encode.v] \
    [file join $repo_root src rtl vendor dvi_tx serdes_4b_10to1.v]]
update_compile_order -fileset sources_1

create_bd_design score_system
current_bd_design score_system
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.5 ps]
source -notrace [file join $repo_root build vivado ac920_ps_preset.tcl]
apply_ac920_ps_preset $ps

set display [create_bd_cell -type module -reference score_hdmi_bd_wrapper display]
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 axi_smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc

# PS HPM0_LPD is a 32-bit, low-bandwidth master suited to these AXI4-Lite blocks.
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_LPD] [get_bd_intf_pins axi_smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M00_AXI] [get_bd_intf_pins display/S_AXI_DISPLAY]
connect_bd_intf_net [get_bd_intf_pins axi_smc/M01_AXI] [get_bd_intf_pins display/S_AXI_AUDIO]

# Synchronize PS reset release to the 99.999 MHz PL0 clock before releasing PL logic.
set reset_inv [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 resetn_invert]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $reset_inv
set reset_sync [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 pl_reset_sync]
set reset_ready [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 reset_ready]
set reset_debug [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 reset_debug]
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {1}] $reset_ready
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] $reset_debug

connect_bd_net [get_bd_pins ps/pl_clk0] \
    [get_bd_pins ps/maxihpm0_lpd_aclk] [get_bd_pins axi_smc/aclk] \
    [get_bd_pins display/pl_clk_100] [get_bd_pins pl_reset_sync/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins resetn_invert/Op1]
connect_bd_net [get_bd_pins resetn_invert/Res] [get_bd_pins pl_reset_sync/ext_reset_in]
connect_bd_net [get_bd_pins reset_ready/dout] [get_bd_pins pl_reset_sync/dcm_locked]
connect_bd_net [get_bd_pins reset_debug/dout] \
    [get_bd_pins pl_reset_sync/aux_reset_in] [get_bd_pins pl_reset_sync/mb_debug_sys_rst]
connect_bd_net [get_bd_pins pl_reset_sync/peripheral_aresetn] \
    [get_bd_pins axi_smc/aresetn] [get_bd_pins display/resetn]

# Use distinct page-aligned 4 KiB windows for Linux UIO mmap map0.
assign_bd_address -offset 0x80000000 -range 4K \
    -target_address_space [get_bd_addr_spaces ps/Data] \
    [get_bd_addr_segs display/S_AXI_DISPLAY/reg0]
assign_bd_address -offset 0x80001000 -range 4K \
    -target_address_space [get_bd_addr_spaces ps/Data] \
    [get_bd_addr_segs display/S_AXI_AUDIO/reg0]

make_bd_pins_external -name beep_out [get_bd_pins display/beep_out]
make_bd_pins_external -name hdmi_clk_p [get_bd_pins display/hdmi_clk_p]
make_bd_pins_external -name hdmi_clk_n [get_bd_pins display/hdmi_clk_n]
make_bd_pins_external -name hdmi_data_p [get_bd_pins display/hdmi_data_p]
make_bd_pins_external -name hdmi_data_n [get_bd_pins display/hdmi_data_n]

validate_bd_design
save_bd_design
add_files -fileset constrs_1 [file join $repo_root build vivado ac920_score_system_pins.xdc]
set bd_file [get_files [file join $project_dir score_system.srcs sources_1 bd score_system score_system.bd]]
generate_target all $bd_file
set wrapper_file [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper_file
set_property top score_system_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -scripts_only
close_project
