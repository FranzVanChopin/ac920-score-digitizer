# Top-level AC920 HDMI TMDS and buzzer pins for the generated score_system wrapper.
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

set_property PACKAGE_PIN AA12 [get_ports beep_out]
set_property IOSTANDARD LVCMOS18 [get_ports beep_out]

# The request/ack toggles synchronize across clock domains. Transaction data
# remains stable until the acknowledgement returns to the AXI clock domain.
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/request_toggle_reg}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/request_sync_1_reg}]
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/ack_toggle_reg}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/ack_sync_1_reg}]
set_false_path -from [get_cells -hier -filter {NAME =~ *display_control/transaction_*_reg*}] \
               -to   [get_cells -hier -filter {NAME =~ *display_control/transaction_*_pix_reg*}]
