# AC920 camera control bank46 is 3.3 V; CSI bank65 is 1.2 V.
set_property PACKAGE_PIN F13 [get_ports {camera_enable[0]}]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_enable[0]}]
set_property PACKAGE_PIN G14 [get_ports camera_i2c_scl_io]
set_property PACKAGE_PIN G15 [get_ports camera_i2c_sda_io]
set_property IOSTANDARD LVCMOS33 [get_ports {camera_i2c_scl_io camera_i2c_sda_io}]
# Board Bank44 is supplied from VCC3P3 via FB32. All HDMI ports removed.
set_property PACKAGE_PIN AG14 [get_ports key_s2_n]
set_property PACKAGE_PIN AH14 [get_ports key_s3_n]
set_property IOSTANDARD LVCMOS33 [get_ports {key_s2_n key_s3_n}]
# Inputs have board pull-ups; never configure these as outputs.
# DP lanes/AUX/HPD use PS GTR/MIO, configured in ac920_ps_preset.tcl.
