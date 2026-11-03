set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_player]
create_project score_player $project_dir -part xczu4ev-sfvc784-1L-i -force
add_files -norecurse [file join $repo_root src rtl score_event_player.sv]
add_files -norecurse [file join $repo_root src rtl axi_score_event_player.sv]
set_property top axi_score_event_player [current_fileset]
update_compile_order -fileset sources_1
synth_design -top axi_score_event_player -part xczu4ev-sfvc784-1L-i -flatten_hierarchy none
report_utilization -file [file join $project_dir score_player_utilization.rpt]
report_timing_summary -file [file join $project_dir score_player_timing.rpt]
close_project
