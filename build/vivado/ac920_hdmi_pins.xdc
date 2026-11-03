# AC920 carrier HDMI/DVI TMDS pins from the supplied factory project.
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

# AC920 on-board buzzer output.
set_property PACKAGE_PIN AA12 [get_ports beep_out]
set_property IOSTANDARD LVCMOS18 [get_ports beep_out]

# This clock is intended to be driven by the Zynq MPSoC PS FCLK in a block design.
create_clock -name pl_clk_100 -period 10.000 [get_ports pl_clk_100]

# The request/ack toggles are two-flop synchronizers. The wider transaction
# payload is held stable until the pixel-domain acknowledge returns.
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/request_toggle_reg}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/request_sync_1_reg}]
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/ack_toggle_reg}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/ack_sync_1_reg}]
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/transaction_*_reg*}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/transaction_*_pix_reg*}]
