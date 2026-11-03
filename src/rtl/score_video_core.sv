`timescale 1ns/1ps

// 640x480 score preview raster and a small programmable note scene.
// Pixel clock is 25 MHz; the AC920 DVI TX uses a 125 MHz 5x clock.
module score_video_core #(
    parameter integer NOTE_SLOTS = 32
) (
    input  logic        pixel_clk,
    input  logic        resetn,
    input  logic        note_we,
    input  logic [4:0]  note_index,
    input  logic [10:0] note_x_write,
    input  logic [9:0]  note_y_write,
    input  logic [4:0]  current_note_index,
    input  logic        current_note_valid,
    output logic [7:0]  red,
    output logic [7:0]  green,
    output logic [7:0]  blue,
    output logic        hsync,
    output logic        vsync,
    output logic        video_active,
    output logic [10:0] pixel_x,
    output logic [9:0]  pixel_y
);
    localparam integer H_ACTIVE = 640;
    localparam integer H_FRONT  = 16;
    localparam integer H_SYNC   = 96;
    localparam integer H_BACK   = 48;
    localparam integer H_TOTAL  = H_ACTIVE + H_FRONT + H_SYNC + H_BACK;
    localparam integer V_ACTIVE = 480;
    localparam integer V_FRONT  = 10;
    localparam integer V_SYNC   = 2;
    localparam integer V_BACK   = 33;
    localparam integer V_TOTAL  = V_ACTIVE + V_FRONT + V_SYNC + V_BACK;

    logic [10:0] note_x [0:NOTE_SLOTS-1];
    logic [9:0]  note_y [0:NOTE_SLOTS-1];
    logic [10:0] h_count;
    logic [9:0]  v_count;
    logic note_pixel;
    logic active_note_pixel;
    logic staff_pixel;
    logic bar_pixel;
    integer i;
    integer dx;
    integer dy;

    assign pixel_x = h_count;
    assign pixel_y = v_count;
    assign video_active = (h_count < H_ACTIVE) && (v_count < V_ACTIVE);
    // 640x480 VGA timing uses negative sync polarity.
    assign hsync = !((h_count >= H_ACTIVE + H_FRONT) &&
                     (h_count <  H_ACTIVE + H_FRONT + H_SYNC));
    assign vsync = !((v_count >= V_ACTIVE + V_FRONT) &&
                     (v_count <  V_ACTIVE + V_FRONT + V_SYNC));

    always_ff @(posedge pixel_clk or negedge resetn) begin
        if (!resetn) begin
            h_count <= 0;
            v_count <= 0;
        end else if (h_count == H_TOTAL - 1) begin
            h_count <= 0;
            if (v_count == V_TOTAL - 1)
                v_count <= 0;
            else
                v_count <= v_count + 1'b1;
        end else begin
            h_count <= h_count + 1'b1;
        end
    end

    always_ff @(posedge pixel_clk or negedge resetn) begin
        if (!resetn) begin
            for (integer reset_index = 0; reset_index < NOTE_SLOTS; reset_index = reset_index + 1) begin
                note_x[reset_index] <= 11'd0;
                note_y[reset_index] <= 10'd0;
            end
        end else if (note_we && note_index < NOTE_SLOTS) begin
            note_x[note_index] <= note_x_write;
            note_y[note_index] <= note_y_write;
        end
    end

    always_comb begin
        note_pixel = 1'b0;
        active_note_pixel = 1'b0;
        for (i = 0; i < NOTE_SLOTS; i = i + 1) begin
            dx = $signed({1'b0, h_count}) - $signed({1'b0, note_x[i]});
            dy = $signed({1'b0, v_count}) - $signed({1'b0, note_y[i]});
            // A compact filled notehead approximation, without multipliers.
            if ((note_x[i] != 0) &&
                (((dx >= -3) && (dx <= 3) && (dy >= -4) && (dy <= 4)) ||
                 ((dx >= -5) && (dx <= 5) && (dy >= -3) && (dy <= 3)) ||
                 ((dx >= -7) && (dx <= 7) && (dy >= -1) && (dy <= 1)))) begin
                note_pixel = 1'b1;
                if (current_note_valid && (i == current_note_index))
                    active_note_pixel = 1'b1;
            end
        end

        staff_pixel = 1'b0;
        if ((h_count >= 48) && (h_count <= 592) && (v_count < V_ACTIVE)) begin
            if ((v_count == 120) || (v_count == 130) || (v_count == 140) ||
                (v_count == 150) || (v_count == 160) || (v_count == 270) ||
                (v_count == 280) || (v_count == 290) || (v_count == 300) ||
                (v_count == 310))
                staff_pixel = 1'b1;
        end

        bar_pixel = 1'b0;
        if (((h_count == 48) || (h_count == 592) || (h_count == 184) ||
             (h_count == 320) || (h_count == 456)) &&
            (((v_count >= 120) && (v_count <= 160)) ||
             ((v_count >= 270) && (v_count <= 310))))
            bar_pixel = 1'b1;

        // Warm paper, dark notation, muted barlines, and a red current note.
        if (!video_active) begin
            red = 8'h00;
            green = 8'h00;
            blue = 8'h00;
        end else if ((h_count < 640) && (v_count < 44)) begin
            red = 8'h20;
            green = 8'h3B;
            blue = 8'h5B;
        end else if ((h_count < 640) && (v_count >= 440)) begin
            red = 8'h20;
            green = 8'h3B;
            blue = 8'h5B;
        end else if (active_note_pixel) begin
            red = 8'hD8;
            green = 8'h32;
            blue = 8'h36;
        end else if (note_pixel || staff_pixel) begin
            red = 8'h24;
            green = 8'h27;
            blue = 8'h2A;
        end else if (bar_pixel) begin
            red = 8'h70;
            green = 8'h76;
            blue = 8'h7A;
        end else begin
            red = 8'hF7;
            green = 8'hF4;
            blue = 8'hEC;
        end
    end
endmodule
