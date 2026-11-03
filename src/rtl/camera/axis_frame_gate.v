`timescale 1ns / 1ps
// SPDX-License-Identifier: MIT
// Request one complete RGB888 frame; consume/discard all other sensor frames.
// The downstream S2MM engine MUST be armed before changing frame_req_toggle.
// done means the last AXIS beat was accepted, NOT that DDR writes have drained.
// Software must also wait for VDMA completion before touching the frame buffer.
module axis_frame_gate #(
    parameter integer FRAME_WIDTH = 4056,
    parameter integer FRAME_HEIGHT = 3040,
    parameter integer PREVIEW_WIDTH = 2028,
    parameter integer PREVIEW_HEIGHT = 1520
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET resetn" *)
    input wire clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire resetn,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ctrl_clk CLK", X_INTERFACE_PARAMETER = "ASSOCIATED_RESET ctrl_resetn" *)
    input wire ctrl_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 ctrl_resetn RST", X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire ctrl_resetn,
    input wire frame_req_toggle,
    input wire frame_abort,
    input wire preview_mode,
    output wire [7:0] capture_status,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TDATA" *) input wire [23:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TVALID" *) input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TREADY" *) output wire s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TUSER" *) input wire s_axis_tuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 s_axis TLAST" *) input wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TDATA" *) output wire [23:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TVALID" *) output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TREADY" *) input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TUSER" *) output wire m_axis_tuser,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 m_axis TLAST" *) output wire m_axis_tlast
);
    (* ASYNC_REG = "TRUE" *) reg req_meta = 0, req_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg abort_meta = 0, abort_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg mode_meta = 0, mode_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg [7:0] status_meta = 0, status_sync = 0;
    reg req_seen = 0, done_toggle = 0;
    reg armed = 0, busy = 0, frame_error = 0;
    reg active_preview = 0;
    reg [15:0] pixel_count = 0, line_count = 0;
    wire [15:0] width_last = active_preview ? PREVIEW_WIDTH - 1 : FRAME_WIDTH - 1;
    wire [15:0] height_last = active_preview ? PREVIEW_HEIGHT - 1 : FRAME_HEIGHT - 1;
    wire forward_frame = !abort_sync && (busy || (armed && s_axis_tuser));
    wire transfer = s_axis_tvalid && s_axis_tready;
    wire [7:0] status_video = {3'b000, active_preview, armed, frame_error, busy, done_toggle};

    assign m_axis_tdata = s_axis_tdata;
    assign m_axis_tuser = s_axis_tuser;
    assign m_axis_tlast = s_axis_tlast;
    assign m_axis_tvalid = s_axis_tvalid && forward_frame;
    assign s_axis_tready = forward_frame ? m_axis_tready : 1'b1;
    assign capture_status = status_sync;

    always @(posedge ctrl_clk) begin
        if (!ctrl_resetn) begin
            status_meta <= 0;
            status_sync <= 0;
        end else begin
            status_meta <= status_video;
            status_sync <= status_meta;
        end
    end

    always @(posedge clk) begin
        if (!resetn) begin
            req_meta <= 0;
            req_sync <= 0;
            abort_meta <= 0;
            abort_sync <= 0;
            mode_meta <= 0;
            mode_sync <= 0;
            active_preview <= 0;
            req_seen <= 0;
            done_toggle <= 0;
            armed <= 0;
            busy <= 0;
            frame_error <= 0;
            pixel_count <= 0;
            line_count <= 0;
        end else begin
            req_meta <= frame_req_toggle;
            req_sync <= req_meta;
            abort_meta <= frame_abort;
            abort_sync <= abort_meta;
            mode_meta <= preview_mode;
            mode_sync <= mode_meta;
            if (abort_sync) begin
                req_seen <= req_sync;
                armed <= 0;
                busy <= 0;
                frame_error <= 1;
                pixel_count <= 0;
                line_count <= 0;
            end else begin
                if (req_sync != req_seen) begin
                    req_seen <= req_sync;
                    if (!busy && !armed) begin
                        // Software holds mode stable before toggling request.
                        // Later mode changes never resize an in-flight frame.
                        active_preview <= mode_sync;
                        armed <= 1;
                        frame_error <= 0;
                        pixel_count <= 0;
                        line_count <= 0;
                    end else begin
                        // Overlapping requests are rejected, never queued.
                        frame_error <= 1;
                    end
                end
                if (transfer && forward_frame) begin
                    armed <= 0;
                    busy <= 1;
                    if (busy && s_axis_tuser) begin
                        // An early new SOF means the captured frame is incomplete.
                        frame_error <= 1;
                        busy <= 0;
                        done_toggle <= !done_toggle;
                    end else if (s_axis_tlast) begin
                        if (pixel_count != width_last)
                            frame_error <= 1;
                        pixel_count <= 0;
                        if (line_count == height_last) begin
                            line_count <= 0;
                            busy <= 0;
                            done_toggle <= !done_toggle;
                        end else begin
                            line_count <= line_count + 1'b1;
                        end
                    end else begin
                        if (pixel_count >= width_last) begin
                            // Saturate instead of allowing a malformed row to wrap.
                            frame_error <= 1;
                        end else begin
                            pixel_count <= pixel_count + 1'b1;
                        end
                    end
                end
            end
        end
    end
endmodule
