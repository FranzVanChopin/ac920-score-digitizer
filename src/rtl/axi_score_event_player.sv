`timescale 1ns/1ps

// AXI4-Lite control plane for the single-voice score_event_player.
// Register map (word aligned): 0x00 STATUS / CONTROL, 0x04 MIDI, 0x08 DURATION_MS.
module axi_score_event_player #(
    parameter integer CLK_HZ = 100_000_000,
    parameter integer ADDR_WIDTH = 12
) (
    input  logic                   s_axi_aclk,
    input  logic                   s_axi_aresetn,
    input  logic [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  logic                   s_axi_awvalid,
    output logic                   s_axi_awready,
    input  logic [31:0]            s_axi_wdata,
    input  logic [3:0]             s_axi_wstrb,
    input  logic                   s_axi_wvalid,
    output logic                   s_axi_wready,
    output logic [1:0]             s_axi_bresp,
    output logic                   s_axi_bvalid,
    input  logic                   s_axi_bready,
    input  logic [ADDR_WIDTH-1:0]  s_axi_araddr,
    input  logic                   s_axi_arvalid,
    output logic                   s_axi_arready,
    output logic [31:0]            s_axi_rdata,
    output logic [1:0]             s_axi_rresp,
    output logic                   s_axi_rvalid,
    input  logic                   s_axi_rready,
    output logic                   beep_out
);
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS   = 'h00;
    localparam logic [ADDR_WIDTH-1:0] REG_MIDI     = 'h04;
    localparam logic [ADDR_WIDTH-1:0] REG_DURATION = 'h08;

    logic aw_pending;
    logic w_pending;
    logic [ADDR_WIDTH-1:0] awaddr_hold;
    logic [31:0] wdata_hold;
    logic [3:0] wstrb_hold;
    logic [6:0] midi_reg;
    logic [31:0] duration_reg;
    logic start_pending;
    logic done_sticky;
    logic player_ready;
    logic player_done;
    logic player_tone;
    logic [31:0] read_mux;

    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready  = !w_pending && !s_axi_bvalid;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = !s_axi_rvalid;
    assign s_axi_rresp   = 2'b00;
    assign beep_out      = player_tone;

    always_comb begin
        if ((s_axi_araddr[ADDR_WIDTH-1:6] == '0) && (s_axi_araddr[1:0] == 2'b00)) begin
            case (s_axi_araddr[5:2])
                4'h0: begin
                    read_mux = 32'd0;
                    read_mux[0] = player_ready;
                    read_mux[1] = !player_ready;
                    read_mux[2] = player_tone;
                    read_mux[3] = done_sticky;
                end
                4'h1: read_mux = {25'd0, midi_reg};
                4'h2: read_mux = duration_reg;
                default: read_mux = 32'd0;
            endcase
        end else begin
            read_mux = 32'd0;
        end
    end

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            aw_pending    <= 1'b0;
            w_pending     <= 1'b0;
            awaddr_hold   <= '0;
            wdata_hold    <= '0;
            wstrb_hold    <= '0;
            midi_reg      <= 7'd0;
            duration_reg  <= 32'd0;
            start_pending <= 1'b0;
            done_sticky   <= 1'b0;
            s_axi_bvalid  <= 1'b0;
            s_axi_rvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
        end else begin
            if (s_axi_awvalid && s_axi_awready) begin
                awaddr_hold <= s_axi_awaddr;
                aw_pending  <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_hold <= s_axi_wdata;
                wstrb_hold <= s_axi_wstrb;
                w_pending  <= 1'b1;
            end

            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending   <= 1'b0;
                w_pending    <= 1'b0;
                s_axi_bvalid <= 1'b1;
                if ((awaddr_hold[ADDR_WIDTH-1:6] == '0) && (awaddr_hold[1:0] == 2'b00) && awaddr_hold[5:2] == 4'h0) begin
                    if (wstrb_hold[0] && wdata_hold[0])
                        start_pending <= 1'b1;
                    if (wstrb_hold[0] && wdata_hold[1])
                        done_sticky <= 1'b0;
                end else if ((awaddr_hold[ADDR_WIDTH-1:6] == '0) && (awaddr_hold[1:0] == 2'b00) && awaddr_hold[5:2] == 4'h1) begin
                    for (int byte_index = 0; byte_index < 4; byte_index++) begin
                        if (wstrb_hold[byte_index]) begin
                            for (int bit_index = 0; bit_index < 8; bit_index++) begin
                                if ((byte_index * 8 + bit_index) < 7)
                                    midi_reg[byte_index * 8 + bit_index] <= wdata_hold[byte_index * 8 + bit_index];
                            end
                        end
                    end
                end else if ((awaddr_hold[ADDR_WIDTH-1:6] == '0) && (awaddr_hold[1:0] == 2'b00) && awaddr_hold[5:2] == 4'h2) begin
                    for (int byte_index = 0; byte_index < 4; byte_index++) begin
                        if (wstrb_hold[byte_index])
                            duration_reg[byte_index * 8 +: 8] <= wdata_hold[byte_index * 8 +: 8];
                    end
                end
            end
            if (s_axi_bvalid && s_axi_bready)
                s_axi_bvalid <= 1'b0;

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rdata  <= read_mux;
                s_axi_rvalid <= 1'b1;
            end
            if (s_axi_rvalid && s_axi_rready)
                s_axi_rvalid <= 1'b0;

            if (start_pending && player_ready)
                start_pending <= 1'b0;
            if (player_done)
                done_sticky <= 1'b1;
        end
    end

    score_event_player #(.CLK_HZ(CLK_HZ)) player (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .event_valid(start_pending),
        .event_ready(player_ready),
        .midi_note(midi_reg),
        .duration_ms(duration_reg),
        .event_done(player_done),
        .tone_out(player_tone)
    );
endmodule
