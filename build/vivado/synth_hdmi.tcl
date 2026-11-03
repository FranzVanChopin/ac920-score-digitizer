set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_hdmi]
create_project score_hdmi $project_dir -part xczu4ev-sfvc784-1L-i -force
add_files -norecurse [file join $repo_root src rtl score_video_core.sv]
add_files -norecurse [file join $repo_root src rtl score_hdmi_top.sv]
add_files -norecurse [file join $repo_root src rtl score_video_axi_control.sv]
add_files -norecurse [file join $repo_root src rtl score_event_player.sv]
add_files -norecurse [file join $repo_root src rtl axi_score_event_player.sv]
add_files -norecurse [file join $repo_root src rtl vendor dvi_tx dvi_transmitter_top.v]
add_files -norecurse [file join $repo_root src rtl vendor dvi_tx dvi_encoder.v]
add_files -norecurse [file join $repo_root src rtl vendor dvi_tx encode.v]
add_files -norecurse [file join $repo_root src rtl vendor dvi_tx serdes_4b_10to1.v]
add_files -fileset constrs_1 [file join $repo_root build vivado ac920_hdmi_pins.xdc]
set_property top score_hdmi_top [current_fileset]
update_compile_order -fileset sources_1
synth_design -top score_hdmi_top -part xczu4ev-sfvc784-1L-i -flatten_hierarchy none
report_utilization -file [file join $project_dir score_hdmi_utilization.rpt]
report_timing_summary -file [file join $project_dir score_hdmi_timing.rpt]
close_project
