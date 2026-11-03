# Write a PL bitstream from the previously fully routed AC920 score-system DCP.
# This does not flash the board or modify an SD image.
set project_dir [file normalize [file join [file dirname [info script]] .. .. build vivado_score_system]]
set checkpoint [file join $project_dir score_system.runs impl_1 score_system_wrapper_routed.dcp]
set output [file join $project_dir score_system_wrapper.bit]
if {![file exists $checkpoint]} {
    error "Routed checkpoint not found: $checkpoint; run build/vivado/build_score_system.bat first"
}
open_checkpoint $checkpoint
report_drc -file [file join $project_dir score_system_bitstream_drc.rpt]
write_bitstream -force -file $output
puts "Wrote bitstream: $output"
close_design
