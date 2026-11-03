`timescale 1ns/1ps

// AC920 display wrapper. Connect pl_clk_100 and the AXI-Lite slave to a
// 100 MHz PS FCLK/interconnect in the block design; register updates cross
// into the generated pixel-clock domain through the display control handshake.
module score_hdmi_top (
    input  logic        pl_clk_100,
    input  logic        resetn,
    input  logic [11:0] s_axi_awaddr,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [11:0] s_axi_araddr,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,
    input  logic [11:0] audio_axi_awaddr,
    input  logic        audio_axi_awvalid,
    output logic        audio_axi_awready,
    input  logic [31:0] audio_axi_wdata,
    input  logic [3:0]  audio_axi_wstrb,
    input  logic        audio_axi_wvalid,
    output logic        audio_axi_wready,
    output logic [1:0]  audio_axi_bresp,
    output logic        audio_axi_bvalid,
    input  logic        audio_axi_bready,
    input  logic [11:0] audio_axi_araddr,
    input  logic        audio_axi_arvalid,
    output logic        audio_axi_arready,
    output logic [31:0] audio_axi_rdata,
    output logic [1:0]  audio_axi_rresp,
    output logic        audio_axi_rvalid,
    input  logic        audio_axi_rready,
    output logic        beep_out,
    output logic        hdmi_clk_p,
    output logic        hdmi_clk_n,
    output logic [2:0]  hdmi_data_p,
    output logic [2:0]  hdmi_data_n
);
    logic mmcm_feedback;
    logic pixel_clk_unbuffered;
    logic serial_clk_unbuffered;
    logic pixel_clk;
    logic serial_clk;
    logic mmcm_locked;
    logic [7:0] red;
    logic [7:0] green;
    logic [7:0] blue;
    logic hsync;
    logic vsync;
    logic video_active;
    logic [10:0] pixel_x;
    logic [9:0] pixel_y;
    logic video_resetn;
    logic note_we_pix;
    logic [4:0] note_index_pix;
    logic [10:0] note_x_pix;
    logic [9:0] note_y_pix;
    logic [4:0] current_note_index_pix;
    logic current_note_valid_pix;

    // 100 MHz input -> 1.2 GHz VCO -> 24 MHz pixel / 120 MHz 5x clocks.
    MMCME4_BASE #(
        .BANDWIDTH("OPTIMIZED"),
        // Use an UltraScale+ legal 1/8-step multiplier. 100 MHz * 12 =
        // 1.2 GHz VCO; /50 gives 24 MHz pixels and /10 gives 120 MHz 5x.
        .CLKFBOUT_MULT_F(12.0),
        .CLKIN1_PERIOD(10.000),
        .CLKOUT0_DIVIDE_F(50.0),
        .CLKOUT0_DUTY_CYCLE(0.5),
        .CLKOUT0_PHASE(0.0),
        .CLKOUT1_DIVIDE(10),
        .CLKOUT1_DUTY_CYCLE(0.5),
        .CLKOUT1_PHASE(0.0),
        .DIVCLK_DIVIDE(1),
        .STARTUP_WAIT("FALSE")
    ) pixel_clock_mmcm (
        .CLKIN1(pl_clk_100),
        .CLKFBIN(mmcm_feedback),
        .CLKFBOUT(mmcm_feedback),
        .CLKOUT0(pixel_clk_unbuffered),
        .CLKOUT1(serial_clk_unbuffered),
        .CLKFBOUTB(), .CLKOUT0B(), .CLKOUT1B(), .CLKOUT2B(),
        .CLKOUT3B(),
        .CLKOUT2(), .CLKOUT3(), .CLKOUT4(), .CLKOUT5(), .CLKOUT6(),
        .LOCKED(mmcm_locked),
        .PWRDWN(1'b0),
        .RST(~resetn)
    );

    BUFG pixel_clock_buffer (.I(pixel_clk_unbuffered), .O(pixel_clk));
    BUFG serial_clock_buffer (.I(serial_clk_unbuffered), .O(serial_clk));
    assign video_resetn = resetn && mmcm_locked;

    // Assert video reset asynchronously, but release it synchronously in the
    // pixel domain. The PS reset and MMCM LOCKED edge are not pixel-clock
    // aligned, so feeding them directly to pixel-clocked async-reset pins
    // creates recovery/removal violations across the whole video datapath.
    (* ASYNC_REG = "TRUE" *) logic [1:0] pixel_reset_sync;
    logic pixel_resetn;
    always_ff @(posedge pixel_clk or negedge video_resetn) begin
        if (!video_resetn)
            pixel_reset_sync <= 2'b00;
        else
            pixel_reset_sync <= {pixel_reset_sync[0], 1'b1};
    end
    assign pixel_resetn = pixel_reset_sync[1];

    score_video_axi_control display_control (
        .s_axi_aclk(pl_clk_100),
        .s_axi_aresetn(resetn),
        .s_axi_awaddr, .s_axi_awvalid, .s_axi_awready,
        .s_axi_wdata, .s_axi_wstrb, .s_axi_wvalid, .s_axi_wready,
        .s_axi_bresp, .s_axi_bvalid, .s_axi_bready,
        .s_axi_araddr, .s_axi_arvalid, .s_axi_arready,
        .s_axi_rdata, .s_axi_rresp, .s_axi_rvalid, .s_axi_rready,
        .pixel_clk,
        .pixel_resetn(pixel_resetn),
        .note_we_pix, .note_index_pix, .note_x_pix, .note_y_pix,
        .current_note_index_pix, .current_note_valid_pix
    );

    score_video_core score_pixels (
        .pixel_clk(pixel_clk),
        .resetn(pixel_resetn),
        .note_we(note_we_pix),
        .note_index(note_index_pix),
        .note_x_write(note_x_pix),
        .note_y_write(note_y_pix),
        .current_note_index(current_note_index_pix),
        .current_note_valid(current_note_valid_pix),
        .red(red), .green(green), .blue(blue),
        .hsync(hsync), .vsync(vsync), .video_active(video_active),
        .pixel_x(pixel_x), .pixel_y(pixel_y)
    );

    axi_score_event_player #(.CLK_HZ(100_000_000)) audio_player (
        .s_axi_aclk(pl_clk_100),
        .s_axi_aresetn(resetn),
        .s_axi_awaddr(audio_axi_awaddr), .s_axi_awvalid(audio_axi_awvalid), .s_axi_awready(audio_axi_awready),
        .s_axi_wdata(audio_axi_wdata), .s_axi_wstrb(audio_axi_wstrb),
        .s_axi_wvalid(audio_axi_wvalid), .s_axi_wready(audio_axi_wready),
        .s_axi_bresp(audio_axi_bresp), .s_axi_bvalid(audio_axi_bvalid), .s_axi_bready(audio_axi_bready),
        .s_axi_araddr(audio_axi_araddr), .s_axi_arvalid(audio_axi_arvalid), .s_axi_arready(audio_axi_arready),
        .s_axi_rdata(audio_axi_rdata), .s_axi_rresp(audio_axi_rresp), .s_axi_rvalid(audio_axi_rvalid), .s_axi_rready(audio_axi_rready),
        .beep_out(beep_out)
    );

    dvi_transmitter_top dvi_tx (
        .pixelclk5x(serial_clk),
        .pixel_clk(pixel_clk),
        .rst_n(pixel_resetn),
        .video_in_data({red, green, blue}),
        .video_in_hs(hsync),
        .video_in_vs(vsync),
        .video_in_de(video_active),
        .tmds_clk_p(hdmi_clk_p),
        .tmds_clk_n(hdmi_clk_n),
        .tmds_data_p(hdmi_data_p),
        .tmds_data_n(hdmi_data_n)
    );
endmodule
