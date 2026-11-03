# PS pl_clk0/pl_clk1 have no guaranteed phase relationship in the PL timing model.
# AMD IPs already constrain their internal CDC handshakes. Do not apply a broad
# clock-group exception: it would override the IPs' max-delay datapath limits.
# Only the first stages of our two-flop single-bit synchronizers need exceptions.
# Their remaining stages retain ordinary synchronous setup/hold checks.
set request_sync_d [get_pins {camera_capture_i/frame_gate/inst/req_meta_reg/D camera_capture_i/frame_gate/inst/abort_meta_reg/D camera_capture_i/frame_gate/inst/mode_meta_reg/D camera_capture_i/isp_control/inst/reset_meta_reg/D}]
set status_sync_d [get_pins {camera_capture_i/frame_gate/inst/status_meta_reg[0]/D camera_capture_i/frame_gate/inst/status_meta_reg[1]/D camera_capture_i/frame_gate/inst/status_meta_reg[2]/D camera_capture_i/frame_gate/inst/status_meta_reg[3]/D camera_capture_i/frame_gate/inst/status_meta_reg[4]/D camera_capture_i/isp_control/inst/active_meta_reg/D}]
set_max_delay -datapath_only 5.0 -from [get_clocks clk_pl_0] -to $request_sync_d
set_max_delay -datapath_only 10.0 -from [get_clocks clk_pl_1] -to $status_sync_d
# Asynchronous physical keys terminate at their first synchronizer stages.
set key_meta_d [get_pins {camera_capture_i/key_sync/inst/meta_reg[0]/D camera_capture_i/key_sync/inst/meta_reg[1]/D}]
set_false_path -from [get_ports {key_s2_n key_s3_n}] -to $key_meta_d
