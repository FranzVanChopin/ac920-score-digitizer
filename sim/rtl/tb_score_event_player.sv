`timescale 1ns/1ps

module tb_score_event_player;
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

    always #5000 clk = ~clk; // 100 kHz simulation clock

    score_event_player #(.CLK_HZ(100_000)) dut (
        .clk(clk), .rst_n(rst_n),
        .event_valid(event_valid), .event_ready(event_ready),
        .midi_note(midi_note), .duration_ms(duration_ms),
        .event_done(event_done), .tone_out(tone_out)
    );

    initial begin
        repeat (3) @(posedge clk);
        rst_n <= 1'b1;
        @(posedge clk);
        midi_note <= 7'd69; // A4
        duration_ms <= 32'd3;
        event_valid <= 1'b1;
        @(posedge clk);
        event_valid <= 1'b0;

        wait (tone_out === 1'b1);
        first_edge = $time;
        wait (tone_out === 1'b0);
        half_period_ns = $time - first_edge;
        if (half_period_ns != 1_130_000)
            $fatal(1, "A4 half-period mismatch: %0t ns", half_period_ns);

        wait (event_done === 1'b1);
        if (event_ready !== 1'b1)
            $fatal(1, "event_ready did not return after event completion");
        $display("PASS: A4 tone divider and event duration");
        $finish;
    end

    initial begin
        #5_000_000;
        $fatal(1, "timeout waiting for event playback");
    end
endmodule
