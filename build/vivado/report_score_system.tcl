set project_dir [file normalize [file join [file dirname [info script]] .. .. build vivado_score_system]]
set checkpoint [file join $project_dir score_system.runs impl_1 score_system_wrapper_routed.dcp]
if {![file exists $checkpoint]} {
    error "Routed checkpoint not found: $checkpoint; run build_score_system.bat first"
}
open_checkpoint $checkpoint
report_utilization -file [file join $project_dir score_system_utilization.rpt]
report_timing_summary -file [file join $project_dir score_system_timing.rpt]
close_design
