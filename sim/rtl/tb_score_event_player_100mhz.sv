`timescale 1ns/1ps

module tb_score_event_player_100mhz;
    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic event_valid = 1'b0;
    logic event_ready;
    logic [6:0] midi_note = 7'd0;
    logic [31:0] duration_ms = 32'd0;
    logic event_done;
    logic tone_out;
    time first_edge;
    time half_period_ns;

    always #5 clk = ~clk; // 100 MHz production clock

    score_event_player #(.CLK_HZ(100_000_000)) dut (
        .clk(clk), .rst_n(rst_n),
        .event_valid(event_valid), .event_ready(event_ready),
        .midi_note(midi_note), .duration_ms(duration_ms),
        .event_done(event_done), .tone_out(tone_out)
    );

    initial begin
        repeat (3) @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        midi_note = 7'd69; // A4
        duration_ms = 32'd3;
        event_valid = 1'b1;
        @(negedge clk);
        event_valid = 1'b0;

        wait (tone_out === 1'b1);
        first_edge = $time;
        wait (tone_out === 1'b0);
        half_period_ns = $time - first_edge;
        if (half_period_ns != 1_136_350)
            $fatal(1, "100 MHz A4 half-period mismatch: %0t ns", half_period_ns);

        wait (event_done === 1'b1);
        $display("PASS: production 100 MHz A4 lookup/scaling path");
        $finish;
    end

    initial begin
        #4_000_000;
        $fatal(1, "timeout waiting for 100 MHz event playback");
    end
endmodule
