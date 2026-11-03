set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_video]
create_project score_video $project_dir -part xczu4ev-sfvc784-1L-i -force
add_files -norecurse [file join $repo_root src rtl score_video_core.sv]
set_property top score_video_core [current_fileset]
update_compile_order -fileset sources_1
synth_design -top score_video_core -part xczu4ev-sfvc784-1L-i -flatten_hierarchy none
create_clock -name pixel_clk -period 41.667 [get_ports pixel_clk]
report_utilization -file [file join $project_dir score_video_utilization.rpt]
report_timing_summary -file [file join $project_dir score_video_timing.rpt]
close_project
