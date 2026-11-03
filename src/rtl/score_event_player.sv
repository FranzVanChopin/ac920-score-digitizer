`timescale 1ns/1ps

// Single-voice event timer/tone source. Requires a known clk frequency and
// an external audio-safe output stage; this is not a direct headphone output.
module score_event_player #(
    parameter integer CLK_HZ = 100_000_000
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        event_valid,
    output logic        event_ready,
    input  logic [6:0]  midi_note,       // MIDI 0 means rest/silence
    input  logic [31:0] duration_ms,
    output logic        event_done,
    output logic        tone_out
);
    localparam integer MS_DIV = CLK_HZ / 1000;

    logic [31:0] ms_remaining;
    logic [31:0] ms_div_count;
    logic [31:0] half_period;
    logic [31:0] phase_count;
    logic        active;
    logic        sounding;
    logic        tone_out_reg;

    function automatic [31:0] midi_base_half_period(input logic [6:0] note);
        begin
            // Five times smaller than the 100 MHz half-period lookup values.
            case (note)
                7'd48: midi_base_half_period = 76336; // C3
                7'd49: midi_base_half_period = 72067;
                7'd50: midi_base_half_period = 68027;
                7'd51: midi_base_half_period = 64225;
                7'd52: midi_base_half_period = 60606; // E3
                7'd53: midi_base_half_period = 57196;
                7'd54: midi_base_half_period = 54054;
                7'd55: midi_base_half_period = 51020; // G3
                7'd56: midi_base_half_period = 48193;
                7'd57: midi_base_half_period = 45455;
                7'd58: midi_base_half_period = 42918;
                7'd59: midi_base_half_period = 40568;
                7'd60: midi_base_half_period = 38223; // C4
                7'd61: midi_base_half_period = 36075;
                7'd62: midi_base_half_period = 34053;
                7'd63: midi_base_half_period = 32154;
                7'd64: midi_base_half_period = 30395; // E4
                7'd65: midi_base_half_period = 28653;
                7'd66: midi_base_half_period = 27027;
                7'd67: midi_base_half_period = 25510; // G4
                7'd68: midi_base_half_period = 24096;
                7'd69: midi_base_half_period = 22727; // A4 = 440 Hz
                7'd70: midi_base_half_period = 21459;
                7'd71: midi_base_half_period = 20283;
                7'd72: midi_base_half_period = 19111; // C5
                7'd73: midi_base_half_period = 18038;
                7'd74: midi_base_half_period = 17026;
                7'd75: midi_base_half_period = 16077;
                7'd76: midi_base_half_period = 15198; // E5
                7'd77: midi_base_half_period = 14327;
                7'd78: midi_base_half_period = 13514;
                7'd79: midi_base_half_period = 12755; // G5
                7'd80: midi_base_half_period = 12048;
                7'd81: midi_base_half_period = 11364;
                7'd82: midi_base_half_period = 10729;
                7'd83: midi_base_half_period = 10138;
                7'd84: midi_base_half_period = 9556;  // C6
                default: midi_base_half_period = 0;
            endcase
        end
    endfunction

    function automatic [31:0] midi_half_period(input logic [6:0] note);
        logic [63:0] scaled;
        begin
            scaled = midi_base_half_period(note);
            if (CLK_HZ == 100_000_000) begin
                // The production clock uses the unscaled lookup directly.
                // Avoid a wide multiply/divide in the AXI-write-to-tone path.
                scaled = scaled * 5;
            end else begin
                // Preserve exact parameter-scaled behavior for simulation and
                // alternate clock configurations.
                scaled = scaled * CLK_HZ * 5;
                scaled = scaled / 100_000_000;
            end
            midi_half_period = scaled[31:0];
        end
    endfunction

    assign event_ready = !active;
    assign tone_out = sounding ? tone_out_reg : 1'b0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ms_remaining <= 32'd0;
            ms_div_count  <= 32'd0;
            half_period   <= 32'd0;
            phase_count   <= 32'd0;
            active        <= 1'b0;
            sounding      <= 1'b0;
            tone_out_reg  <= 1'b0;
            event_done    <= 1'b0;
        end else begin
            event_done <= 1'b0;
            if (event_valid && event_ready) begin
                if (duration_ms == 0) begin
                    event_done <= 1'b1;
                end else begin
                    ms_remaining <= duration_ms;
                    ms_div_count <= 32'd0;
                    half_period  <= midi_half_period(midi_note);
                    phase_count  <= 32'd0;
                    active       <= 1'b1;
                    sounding     <= (midi_base_half_period(midi_note) != 0);
                    tone_out_reg <= 1'b0;
                end
            end else if (active) begin
                if (ms_div_count >= MS_DIV - 1) begin
                    ms_div_count <= 32'd0;
                    if (ms_remaining <= 1) begin
                        ms_remaining <= 32'd0;
                        active       <= 1'b0;
                        sounding     <= 1'b0;
                        tone_out_reg <= 1'b0;
                        event_done   <= 1'b1;
                    end else begin
                        ms_remaining <= ms_remaining - 1'b1;
                    end
                end else begin
                    ms_div_count <= ms_div_count + 1'b1;
                end
                if (sounding) begin
                    if (phase_count >= half_period - 1'b1) begin
                        phase_count  <= 32'd0;
                        tone_out_reg <= ~tone_out_reg;
                    end else begin
                        phase_count <= phase_count + 1'b1;
                    end
                end
            end
        end
    end
endmodule
