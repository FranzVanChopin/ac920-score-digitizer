`timescale 1ns / 1ps
// SPDX-License-Identifier: MIT
// Explicit metadata prevents IP Integrator from mistaking the serialized TMDS
// clock outputs for clocks of the pixel-domain reset input.
module camera_dvi_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 pixel_clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET rst_n" *)
    input wire pixel_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 pixelclk5x CLK" *)
    input wire pixelclk5x,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 rst_n RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire rst_n,
    input wire [23:0] video_in_data,
    input wire video_in_hs, video_in_vs, video_in_de,
    (* X_INTERFACE_IGNORE = "true" *) output wire tmds_clk_p,
    (* X_INTERFACE_IGNORE = "true" *) output wire tmds_clk_n,
    output wire [2:0] tmds_data_p, tmds_data_n
);
    dvi_transmitter_top transmitter (
        .pixelclk5x(pixelclk5x), .pixel_clk(pixel_clk), .rst_n(rst_n),
        .video_in_data(video_in_data), .video_in_hs(video_in_hs),
        .video_in_vs(video_in_vs), .video_in_de(video_in_de),
        .tmds_clk_p(tmds_clk_p), .tmds_clk_n(tmds_clk_n),
        .tmds_data_p(tmds_data_p), .tmds_data_n(tmds_data_n)
    );
endmodule
