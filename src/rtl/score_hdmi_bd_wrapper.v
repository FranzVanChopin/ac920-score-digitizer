// Verilog-2001 shell for Vivado IP Integrator module-reference inference.
// The SystemVerilog implementation remains score_hdmi_top.sv.
module score_hdmi_bd_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 pl_clk_100 CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME pl_clk_100, ASSOCIATED_BUSIF S_AXI_DISPLAY:S_AXI_AUDIO, ASSOCIATED_RESET resetn, FREQ_HZ 99999001" *)
    input pl_clk_100,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME resetn, POLARITY ACTIVE_LOW" *)
    input resetn,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY AWADDR",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_DISPLAY, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, FREQ_HZ 99999001" *) input [11:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY AWVALID" *) input s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY AWREADY" *) output s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY WDATA" *) input [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY WSTRB" *) input [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY WVALID" *) input s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY WREADY" *) output s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY BRESP" *) output [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY BVALID" *) output s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY BREADY" *) input s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY ARADDR" *) input [11:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY ARVALID" *) input s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY ARREADY" *) output s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY RDATA" *) output [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY RRESP" *) output [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY RVALID" *) output s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_DISPLAY RREADY" *) input s_axi_rready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO AWADDR",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI_AUDIO, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, FREQ_HZ 99999001" *) input [11:0] audio_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO AWVALID" *) input audio_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO AWREADY" *) output audio_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO WDATA" *) input [31:0] audio_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO WSTRB" *) input [3:0] audio_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO WVALID" *) input audio_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO WREADY" *) output audio_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO BRESP" *) output [1:0] audio_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO BVALID" *) output audio_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO BREADY" *) input audio_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO ARADDR" *) input [11:0] audio_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO ARVALID" *) input audio_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO ARREADY" *) output audio_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO RDATA" *) output [31:0] audio_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO RRESP" *) output [1:0] audio_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO RVALID" *) output audio_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI_AUDIO RREADY" *) input audio_axi_rready,

    output beep_out,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 hdmi_clk_p CLK",
       X_INTERFACE_PARAMETER = "XIL_INTERFACENAME hdmi_clk_p, FREQ_HZ 24000000" *) output hdmi_clk_p,
    (* X_INTERFACE_INFO = "xilinx.com:signal:data:1.0 hdmi_clk_n DATA" *) output hdmi_clk_n,
    output [2:0] hdmi_data_p,
    output [2:0] hdmi_data_n
);
    score_hdmi_top implementation (
        .pl_clk_100(pl_clk_100), .resetn(resetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb), .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid), .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp), .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .audio_axi_awaddr(audio_axi_awaddr), .audio_axi_awvalid(audio_axi_awvalid), .audio_axi_awready(audio_axi_awready),
        .audio_axi_wdata(audio_axi_wdata), .audio_axi_wstrb(audio_axi_wstrb), .audio_axi_wvalid(audio_axi_wvalid), .audio_axi_wready(audio_axi_wready),
        .audio_axi_bresp(audio_axi_bresp), .audio_axi_bvalid(audio_axi_bvalid), .audio_axi_bready(audio_axi_bready),
        .audio_axi_araddr(audio_axi_araddr), .audio_axi_arvalid(audio_axi_arvalid), .audio_axi_arready(audio_axi_arready),
        .audio_axi_rdata(audio_axi_rdata), .audio_axi_rresp(audio_axi_rresp), .audio_axi_rvalid(audio_axi_rvalid), .audio_axi_rready(audio_axi_rready),
        .beep_out(beep_out), .hdmi_clk_p(hdmi_clk_p), .hdmi_clk_n(hdmi_clk_n),
        .hdmi_data_p(hdmi_data_p), .hdmi_data_n(hdmi_data_n)
    );
endmodule
