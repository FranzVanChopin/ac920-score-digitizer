"""Derive the pure-DP design without changing the sealed HDMI/DP1 builds."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
s=(ROOT/'vivado/create_camera_capture.tcl').read_text()
def cut(a,b,replacement=''):
    global s
    start=s.index(a);end=s.index(b,start)
    s=s[:start]+replacement+s[end:]
cut('add_files -norecurse','\nupdate_compile_order', '''add_files -norecurse [list $gate_file \\
    [file join $repo_root rtl camera camera_isp_control.v] \\
    [file join $repo_root rtl camera camera_key_sync.v]]''')
s=s.replace('CONFIG.c_include_mm2s 1','CONFIG.c_include_mm2s 0')
s=s.replace('CONFIG.C_GPIO2_WIDTH 8','CONFIG.C_GPIO2_WIDTH 10')
s=s.replace('camera_net isp_control/control_status camera_gpio/gpio2_io_i','''camera_ip capture_status xlconcat 2.1 [list CONFIG.NUM_PORTS 2 CONFIG.IN0_WIDTH 8 CONFIG.IN1_WIDTH 2]
create_bd_cell -type module -reference camera_key_sync key_sync
foreach key {s2 s3} {
    create_bd_port -dir I key_${key}_n
    connect_bd_net [get_bd_ports key_${key}_n] [get_bd_pins key_sync/key_${key}_n]
}
camera_net isp_control/control_status capture_status/In0
camera_net key_sync/keys_n capture_status/In1
camera_net capture_status/dout camera_gpio/gpio2_io_i''')
cut('# Only PS S1 is used.','camera_ip control_axi')
s=s.replace('CONFIG.NUM_MI 6','CONFIG.NUM_MI 5').replace('CONFIG.NUM_SI 2 CONFIG.NUM_MI 1','CONFIG.NUM_SI 1 CONFIG.NUM_MI 1')
s=s.replace('3 gamma/s_axi_CTRL 4 vtc/ctrl 5 camera_gpio/S_AXI','3 gamma/s_axi_CTRL 4 camera_gpio/S_AXI')
s=s.replace('camera_axis vdma/M_AXI_MM2S memory_axi/S00_AXI\n','')
s=s.replace('camera_axis vdma/M_AXI_S2MM memory_axi/S01_AXI','camera_axis vdma/M_AXI_S2MM memory_axi/S00_AXI')
cut('camera_axis vdma/M_AXIS_MM2S','camera_ip one ')
s=s.replace('foreach name {ctrl_reset video_reset pixel_reset}','foreach name {ctrl_reset video_reset}')
s=s.replace('camera_net video_clk/locked pixel_reset/dcm_locked\n','')
s=s.replace('vdma/s_axi_lite_aclk vtc/s_axi_aclk camera_gpio/s_axi_aclk frame_gate/ctrl_clk isp_control/ctrl_clk','vdma/s_axi_lite_aclk camera_gpio/s_axi_aclk frame_gate/ctrl_clk isp_control/ctrl_clk key_sync/clk')
s=s.replace('vdma/m_axi_mm2s_aclk vdma/m_axis_mm2s_aclk ','')
s=s.replace(' \\\n    video_out/aclk','')
cut('camera_net video_clk/clk_out1','camera_net ctrl_reset/peripheral_aresetn')
s=s.replace('vdma/axi_resetn vtc/s_axi_aresetn camera_gpio/s_axi_aresetn video_clk/resetn frame_gate/ctrl_resetn isp_control/ctrl_resetn','vdma/axi_resetn camera_gpio/s_axi_aresetn frame_gate/ctrl_resetn isp_control/ctrl_resetn key_sync/resetn')
s=s.replace(' video_out/aresetn','')
cut('camera_net pixel_reset/','foreach {address segment range}')
s=s.replace('    0x80040000 vtc/ctrl/Reg 64K\n','')
s=s.replace('foreach space {vdma/Data_MM2S vdma/Data_S2MM}','foreach space {vdma/Data_S2MM}')
s=s.replace('all four data clocks','both capture data clocks')
s=s.replace('# AC920 + Raspberry Pi HQ IMX477 full-frame capture and HDMI preview.','# AC920 pure PS DP + IMX477 capture + synchronized S2/S3. No HDMI PL logic.')
for token in ('video_clk','camera_ip video_out','pixel_reset','vtc/','camera_dvi_wrapper','hdmi/'):
    assert token not in s,token
(HERE/'vivado/create_camera_capture.tcl').write_text(s,encoding='utf-8')
