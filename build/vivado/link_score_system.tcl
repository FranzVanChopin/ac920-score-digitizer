# Link the top-level synthesized BD netlist with each OOC IP checkpoint.
# Vivado's generated synth_1 checkpoint intentionally contains black-box
# stubs for hierarchical IP, so it must not be reported as a complete design.
set repo_root [file normalize [file join [file dirname [info script]] .. ..]]
set project_dir [file join $repo_root build vivado_score_system]
set runs_dir [file join $project_dir score_system.runs]
set checkpoint [file join $runs_dir synth_1 score_system_wrapper.dcp]
if {![file exists $checkpoint]} {
    error "Top synthesis checkpoint not found: $checkpoint"
}

open_checkpoint $checkpoint
set ooc_cells [list \
    [list score_system_i/ps score_system_ps_0_synth_1 score_system_ps_0] \
    [list score_system_i/display score_system_display_0_synth_1 score_system_display_0] \
    [list score_system_i/axi_smc score_system_axi_smc_0_synth_1 score_system_axi_smc_0] \
    [list score_system_i/pl_reset_sync score_system_pl_reset_sync_0_synth_1 score_system_pl_reset_sync_0] \
    [list score_system_i/resetn_invert score_system_resetn_invert_0_synth_1 score_system_resetn_invert_0]]

foreach entry $ooc_cells {
    lassign $entry cell run_name dcp_name
    set dcp [file join $runs_dir $run_name ${dcp_name}.dcp]
    if {![file exists $dcp]} {
        error "OOC checkpoint not found for $cell: $dcp"
    }
    puts "Linking $cell <- $dcp"
    read_checkpoint -cell $cell $dcp
}

# The generated synth run excludes board constraints from its checkpoint.
# Reload pin/CDC constraints and constrain the PS-generated PL clock for a
# useful pre-route timing report. The clock period is 99.999001 MHz from the
# AC920 factory PS preset (rounded to 10 ns here).
read_xdc [file join $repo_root build vivado ac920_score_system_pins.xdc]
create_clock -name pl_clk0 -period 10.000 [get_pins score_system_i/ps/pl_clk0]

set blackboxes [get_cells -hier -filter IS_BLACKBOX]
if {[llength $blackboxes] != 0} {
    puts "ERROR: unresolved black boxes remain: $blackboxes"
    error "Score system checkpoint is not fully linked"
}

set linked_checkpoint [file join $project_dir score_system_linked.dcp]
write_checkpoint -force $linked_checkpoint
report_utilization -file [file join $project_dir score_system_utilization.rpt]
report_timing_summary -file [file join $project_dir score_system_timing.rpt]
close_design
