# Run through build_camera_capture.ps1 on Windows (short-path HLS workaround).
set repo_root [file normalize [file join [file dirname [info script]] ..]]
set project_dir [file join $repo_root build vivado_camera]
file delete -force [file join $project_dir hardware.complete]
file delete -force [file join $project_dir hardware_validation.json]
open_project [file join $project_dir camera_capture.xpr]
set bd_file [get_files */camera_capture.bd]
create_ip_run $bd_file
# Rebuild the top level when reusing a generated project after a BD edit.
reset_run synth_1
launch_runs synth_1 -jobs 8
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
    error "Camera synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
open_run synth_1
set request_sync_d [get_pins {camera_capture_i/frame_gate/inst/req_meta_reg/D camera_capture_i/frame_gate/inst/abort_meta_reg/D camera_capture_i/frame_gate/inst/mode_meta_reg/D camera_capture_i/isp_control/inst/reset_meta_reg/D}]
set status_sync_d [get_pins {camera_capture_i/frame_gate/inst/status_meta_reg[0]/D camera_capture_i/frame_gate/inst/status_meta_reg[1]/D camera_capture_i/frame_gate/inst/status_meta_reg[2]/D camera_capture_i/frame_gate/inst/status_meta_reg[3]/D camera_capture_i/frame_gate/inst/status_meta_reg[4]/D camera_capture_i/isp_control/inst/active_meta_reg/D}]
if {[llength $request_sync_d] != 4 || [llength $status_sync_d] != 6} {
    error "Camera CDC endpoint count changed; inspect gate synchronizers before building"
}
report_utilization -file [file join $project_dir synthesis_utilization.rpt]
report_drc -file [file join $project_dir synthesis_drc.rpt]
close_design
launch_runs impl_1 -to_step route_design -jobs 8
wait_on_run impl_1
open_run impl_1
if {[llength [get_ports -quiet hdmi*]] || [llength [get_ports -quiet board_clk*]]} {
    error "Pure-DP design must not contain PL HDMI or its oscillator input"
}
foreach {port pin} {key_s2_n AG14 key_s3_n AH14} {
    if {[get_property PACKAGE_PIN [get_ports $port]] ne $pin ||
        [get_property IOSTANDARD [get_ports $port]] ne "LVCMOS33" ||
        [get_property DIRECTION [get_ports $port]] ne "IN"} {
        error "Button location, voltage or direction mismatch: $port"
    }
}
report_timing_summary -delay_type min_max -check_timing_verbose -file [file join $project_dir implementation_timing.rpt]
check_timing -verbose -file [file join $project_dir check_timing.rpt]
report_drc -file [file join $project_dir implementation_drc.rpt]
report_methodology -file [file join $project_dir implementation_methodology.rpt]
report_cdc -details -file [file join $project_dir implementation_cdc.rpt]
report_bus_skew -warn_on_violation -file [file join $project_dir implementation_bus_skew.rpt]
report_utilization -file [file join $project_dir implementation_utilization.rpt]
report_io -file [file join $project_dir implementation_io.rpt]
report_clocks -file [file join $project_dir implementation_clocks.rpt]
# The summary also covers pulse width / minimum period. Setup and hold alone
# cannot catch the HDIO ODDR's 125 MHz limit.
set timing_file [open [file join $project_dir implementation_timing.rpt] r]
set timing_text [read $timing_file]
close $timing_file
if {![string match {*All user specified timing constraints are met.*} $timing_text]} {
    error "Camera design fails full timing acceptance (including pulse width); inspect implementation_timing.rpt"
}
set skew_file [open [file join $project_dir implementation_bus_skew.rpt] r]
set skew_text [read $skew_file]
close $skew_file
if {[string match {*VIOLATED*} $skew_text]} {
    error "Camera design fails bus-skew constraints; inspect implementation_bus_skew.rpt"
}
set cdc_report [open [file join $project_dir implementation_cdc.rpt] r]
set cdc_text [read $cdc_report]
close $cdc_report
if {[regexp -line {^CDC-[0-9]+[[:space:]]+(Critical|Error)} $cdc_text]} {
    error "Camera design has critical CDC findings; inspect implementation_cdc.rpt"
}
set check_report [open [file join $project_dir check_timing.rpt] r]
set check_text [read $check_report]
close $check_report
foreach check {no_clock unconstrained_internal_endpoints multiple_clock generated_clocks loops latch_loops} {
    if {![string match "*checking $check (0)*" $check_text]} {
        error "Camera design fails check_timing $check; inspect check_timing.rpt"
    }
}
foreach delay_type {max min} {
    set worst [get_timing_paths -delay_type $delay_type -max_paths 1]
    if {[llength $worst] && [get_property SLACK $worst] < 0} {
        error "Camera design fails $delay_type timing; inspect implementation_timing.rpt"
    }
}
set bad_drc [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == "Critical Warning"}]
if {[llength $bad_drc]} {error "Unresolved critical DRC violations: $bad_drc"}
set bad_methodology [get_methodology_violations -quiet -filter {SEVERITY == Error || SEVERITY == "Critical Warning"}]
if {[llength $bad_methodology]} {error "Unresolved critical methodology violations: $bad_methodology"}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Camera bitstream failed: [get_property STATUS [get_runs impl_1]]"
}
write_hw_platform -fixed -include_bit -force -file [file join $project_dir camera_capture.xsa]
# certutil is bundled with Windows; exec passes paths as separate arguments.
proc camera_sha256 {path} {
    set output [exec certutil.exe -hashfile $path SHA256]
    foreach line [split $output \n] {
        set line [string trim $line]
        if {[regexp {^[[:xdigit:]]{64}$} $line]} {return [string toupper $line]}
    }
    error "Could not compute SHA256 for $path"
}
set bit_hash [camera_sha256 [file join $project_dir camera_capture.runs impl_1 camera_capture_wrapper.bit]]
set xsa_hash [camera_sha256 [file join $project_dir camera_capture.xsa]]
set complete_file [open [file join $project_dir hardware.complete] w]
puts $complete_file "Camera hardware accepted by Vivado [version -short]"
puts $complete_file "Completed UTC: [clock format [clock seconds] -gmt 1 -format {%Y-%m-%dT%H:%M:%SZ}]"
puts $complete_file "Part: [get_property PART [current_project]]"
puts $complete_file "Checks: setup, hold, pulse width, bus skew, DRC, methodology, CDC, internal timing coverage"
puts $complete_file "Bitstream: camera_capture.runs/impl_1/camera_capture_wrapper.bit"
puts $complete_file "XSA: camera_capture.xsa (includes bitstream)"
puts $complete_file "Bitstream SHA256: $bit_hash"
puts $complete_file "XSA SHA256: $xsa_hash"
close $complete_file
puts "CAMERA_BUILD_COMPLETE: $project_dir"
close_project
