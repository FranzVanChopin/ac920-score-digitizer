# AC920 carrier-board I/O from the supplied factory project Pin.xdc.
# Add only when the top-level port names below are used.

# HDMI TMDS differential outputs
set_property PACKAGE_PIN AC14 [get_ports hdmi_clk_p]
set_property PACKAGE_PIN AC13 [get_ports hdmi_clk_n]
set_property PACKAGE_PIN Y14  [get_ports {hdmi_data_p[0]}]
set_property PACKAGE_PIN Y13  [get_ports {hdmi_data_n[0]}]
set_property PACKAGE_PIN AA13 [get_ports {hdmi_data_p[1]}]
set_property PACKAGE_PIN AB13 [get_ports {hdmi_data_n[1]}]
set_property PACKAGE_PIN AD15 [get_ports {hdmi_data_p[2]}]
set_property PACKAGE_PIN AD14 [get_ports {hdmi_data_n[2]}]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports {hdmi_data_p[*]}]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports {hdmi_data_n[*]}]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports hdmi_clk_p]
set_property IOSTANDARD DIFF_HSTL_I_18 [get_ports hdmi_clk_n]
set_property SLEW FAST [get_ports {hdmi_data_p[*]}]
set_property SLEW FAST [get_ports {hdmi_data_n[*]}]
set_property SLEW FAST [get_ports hdmi_clk_p]
set_property SLEW FAST [get_ports hdmi_clk_n]

# HDMI DDC I2C (optional inout ports for EDID handling)
set_property PACKAGE_PIN AE14 [get_ports hdmi_scl]
set_property PACKAGE_PIN AB15 [get_ports hdmi_sda]
set_property IOSTANDARD LVCMOS18 [get_ports hdmi_scl]
set_property IOSTANDARD LVCMOS18 [get_ports hdmi_sda]

# On-board buzzer output for the initial monophonic audition path
set_property PACKAGE_PIN AA12 [get_ports beep_out]
set_property IOSTANDARD LVCMOS18 [get_ports beep_out]
