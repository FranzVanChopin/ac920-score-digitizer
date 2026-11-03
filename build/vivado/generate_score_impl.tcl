set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_file [file join $repo_root build vivado_score_system score_system.xpr]
if {![file exists $project_file]} {
    error "Vivado project not found: $project_file"
}
open_project $project_file
puts "Available runs: [get_runs]"
launch_runs impl_1 -scripts_only
close_project
