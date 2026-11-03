`timescale 1ns / 1ps
// SPDX-License-Identifier: MIT
// ISP-only reset. Neither VDMA nor HDMI is reset. S3 is not connected.
// Reset request first crosses to video_clk; assertion and three-cycle release
// are synchronous there. GPIO status bit5 acknowledges the ACTUAL reset state.
module camera_isp_control (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET resetn:isp_resetn" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire resetn,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ctrl_clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ctrl_resetn" *)
    input wire ctrl_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ctrl_resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire ctrl_resetn,
    input wire reset_request,
    input wire [7:0] gate_status,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 isp_resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    output wire isp_resetn,
    output wire [7:0] control_status
);
    (* ASYNC_REG = "TRUE" *) reg reset_meta = 1, reset_sync = 1;
    (* ASYNC_REG = "TRUE" *) reg active_meta = 1, active_sync = 1;
    reg [2:0] release_pipe = 0;
    assign isp_resetn = release_pipe[2];
    // [6]=accepted gate mode, [5]=ISP reset asserted, [4]=reserved/released.
    assign control_status = {1'b0, gate_status[4], active_sync, 1'b1, gate_status[3:0]};
    always @(posedge clk) begin
        if (!resetn) begin
            reset_meta <= 1;
            reset_sync <= 1;
            release_pipe <= 0;
        end else begin
            reset_meta <= reset_request;
            reset_sync <= reset_meta;
            if (reset_sync) release_pipe <= 0;
            else release_pipe <= {release_pipe[1:0], 1'b1};
        end
    end
    always @(posedge ctrl_clk) begin
        if (!ctrl_resetn) begin
            active_meta <= 1;
            active_sync <= 1;
        end else begin
            active_meta <= !isp_resetn;
            active_sync <= active_meta;
        end
    end
endmodule
