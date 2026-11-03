`timescale 1ns/1ps

module tb_score_video_core;
    logic clk = 1'b0;
    logic resetn = 1'b0;
    logic note_we = 1'b0;
    logic [4:0] note_index = 0;
    logic [10:0] note_x_write = 0;
    logic [9:0] note_y_write = 0;
    logic [4:0] current_note_index = 0;
    logic current_note_valid = 1'b0;
    logic [7:0] red, green, blue;
    logic hsync, vsync, video_active;
    logic [10:0] pixel_x;
    logic [9:0] pixel_y;

    always #12.5 clk = ~clk;

    score_video_core #(.NOTE_SLOTS(1)) dut (
        .pixel_clk(clk), .resetn,
        .note_we, .note_index, .note_x_write, .note_y_write,
        .current_note_index, .current_note_valid,
        .red, .green, .blue, .hsync, .vsync, .video_active,
        .pixel_x, .pixel_y
    );

    initial begin
        repeat (3) @(negedge clk);
        resetn = 1'b1;
        @(negedge clk);
        note_index = 0;
        note_x_write = 11'd100;
        note_y_write = 10'd140;
        note_we = 1'b1;
        @(negedge clk);
        note_we = 1'b0;
        current_note_valid = 1'b1;

        wait ((pixel_x == 100) && (pixel_y == 140));
        #1;
        if ({red, green, blue} != 24'hD83236)
            $fatal(1, "current note was not highlighted at its configured position");

        wait ((pixel_x == 60) && (pixel_y == 150));
        #1;
        if ({red, green, blue} != 24'h24272A)
            $fatal(1, "staff line was not rendered");

        wait ((pixel_x == 656) && (pixel_y == 150));
        #1;
        if (hsync || video_active)
            $fatal(1, "horizontal sync timing is incorrect");

        $display("PASS: 640x480 pixel clocking, staff rendering, and active-note highlight");
        $finish;
    end

    initial begin
        #8_000_000;
        $fatal(1, "timeout waiting for video raster, current coordinate %0d,%0d", pixel_x, pixel_y);
    end
endmodule
