`timescale 1ns/1ps

// AXI4-Lite score-scene registers in the PS FCLK domain, with a bundled-data
// toggle handshake into the pixel-clock domain. Software waits for BUSY=0
// before issuing another COMMIT so the payload remains stable across the CDC.
module score_video_axi_control #(
    parameter integer ADDR_WIDTH = 12
) (
    input  logic                  s_axi_aclk,
    input  logic                  s_axi_aresetn,
    input  logic [ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic                  s_axi_awvalid,
    output logic                  s_axi_awready,
    input  logic [31:0]           s_axi_wdata,
    input  logic [3:0]            s_axi_wstrb,
    input  logic                  s_axi_wvalid,
    output logic                  s_axi_wready,
    output logic [1:0]            s_axi_bresp,
    output logic                  s_axi_bvalid,
    input  logic                  s_axi_bready,
    input  logic [ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic                  s_axi_arvalid,
    output logic                  s_axi_arready,
    output logic [31:0]           s_axi_rdata,
    output logic [1:0]            s_axi_rresp,
    output logic                  s_axi_rvalid,
    input  logic                  s_axi_rready,
    input  logic                  pixel_clk,
    input  logic                  pixel_resetn,
    output logic                  note_we_pix,
    output logic [4:0]            note_index_pix,
    output logic [10:0]           note_x_pix,
    output logic [9:0]            note_y_pix,
    output logic [4:0]            current_note_index_pix,
    output logic                  current_note_valid_pix
);
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS       = 'h00;
    localparam logic [ADDR_WIDTH-1:0] REG_SLOT         = 'h04;
    localparam logic [ADDR_WIDTH-1:0] REG_X            = 'h08;
    localparam logic [ADDR_WIDTH-1:0] REG_Y            = 'h0C;
    localparam logic [ADDR_WIDTH-1:0] REG_CURSOR       = 'h10;
    localparam logic [ADDR_WIDTH-1:0] REG_NOTE_COMMIT  = 'h14;
    localparam logic [ADDR_WIDTH-1:0] REG_VIEW_COMMIT  = 'h18;

    logic aw_pending;
    logic w_pending;
    logic [ADDR_WIDTH-1:0] awaddr_hold;
    logic [31:0] wdata_hold;
    logic [3:0] wstrb_hold;
    logic [4:0] slot_reg;
    logic [10:0] x_reg;
    logic [9:0] y_reg;
    logic [4:0] cursor_reg;
    logic cursor_valid_reg;
    logic request_toggle;
    (* ASYNC_REG = "TRUE" *) logic ack_sync_1;
    (* ASYNC_REG = "TRUE" *) logic ack_sync_2;
    (* ASYNC_REG = "TRUE" *) logic request_sync_1;
    (* ASYNC_REG = "TRUE" *) logic request_sync_2;
    logic request_seen;
    logic ack_toggle;
    logic pending_pix;
    logic transaction_is_note;
    logic [4:0] transaction_slot;
    logic [10:0] transaction_x;
    logic [9:0] transaction_y;
    logic [4:0] transaction_cursor;
    logic transaction_cursor_valid;
    logic transaction_is_note_pix;
    logic [4:0] transaction_slot_pix;
    logic [10:0] transaction_x_pix;
    logic [9:0] transaction_y_pix;
    logic [4:0] transaction_cursor_pix;
    logic transaction_cursor_valid_pix;
    logic [31:0] read_mux;
    logic write_known;

    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready  = !w_pending && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    assign note_we_pix = pending_pix && transaction_is_note_pix;
    assign note_index_pix = transaction_slot_pix;
    assign note_x_pix = transaction_x_pix;
    assign note_y_pix = transaction_y_pix;
    assign current_note_index_pix = transaction_cursor_pix;
    assign current_note_valid_pix = transaction_cursor_valid_pix;
    assign write_known = (awaddr_hold == REG_SLOT) || (awaddr_hold == REG_X) ||
                         (awaddr_hold == REG_Y) || (awaddr_hold == REG_CURSOR) ||
                         (awaddr_hold == REG_NOTE_COMMIT) || (awaddr_hold == REG_VIEW_COMMIT);

    always_comb begin
        case (s_axi_araddr)
            REG_STATUS: begin
                read_mux = 32'd0;
                read_mux[0] = (request_toggle != ack_sync_2);
            end
            REG_SLOT:   read_mux = {27'd0, slot_reg};
            REG_X:      read_mux = {21'd0, x_reg};
            REG_Y:      read_mux = {22'd0, y_reg};
            REG_CURSOR: read_mux = {23'd0, cursor_valid_reg, 3'd0, cursor_reg};
            default:    read_mux = 32'd0;
        endcase
    end

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            aw_pending   <= 1'b0;
            w_pending    <= 1'b0;
            awaddr_hold  <= '0;
            wdata_hold   <= '0;
            wstrb_hold   <= '0;
            slot_reg     <= '0;
            x_reg        <= '0;
            y_reg        <= '0;
            cursor_reg   <= '0;
            cursor_valid_reg <= 1'b0;
            request_toggle <= 1'b0;
            ack_sync_1   <= 1'b0;
            ack_sync_2   <= 1'b0;
            transaction_is_note <= 1'b0;
            transaction_slot <= '0;
            transaction_x <= '0;
            transaction_y <= '0;
            transaction_cursor <= '0;
            transaction_cursor_valid <= 1'b0;
            s_axi_bresp   <= 2'b00;
            s_axi_bvalid  <= 1'b0;
            s_axi_rdata   <= 32'd0;
            s_axi_rresp   <= 2'b00;
            s_axi_rvalid  <= 1'b0;
        end else begin
            ack_sync_1 <= ack_toggle;
            ack_sync_2 <= ack_sync_1;

            if (s_axi_awvalid && s_axi_awready) begin
                awaddr_hold <= s_axi_awaddr;
                aw_pending <= 1'b1;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                wdata_hold <= s_axi_wdata;
                wstrb_hold <= s_axi_wstrb;
                w_pending <= 1'b1;
            end
            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending <= 1'b0;
                w_pending <= 1'b0;
                s_axi_bvalid <= 1'b1;
                s_axi_bresp <= write_known ? 2'b00 : 2'b10;
                case (awaddr_hold)
                    REG_SLOT: begin
                        if (wstrb_hold[0]) slot_reg[4:0] <= wdata_hold[4:0];
                    end
                    REG_X: begin
                        if (wstrb_hold[0]) x_reg[7:0] <= wdata_hold[7:0];
                        if (wstrb_hold[1]) x_reg[10:8] <= wdata_hold[10:8];
                    end
                    REG_Y: begin
                        if (wstrb_hold[0]) y_reg[7:0] <= wdata_hold[7:0];
                        if (wstrb_hold[1]) y_reg[9:8] <= wdata_hold[9:8];
                    end
                    REG_CURSOR: begin
                        if (wstrb_hold[0]) cursor_reg <= wdata_hold[4:0];
                        if (wstrb_hold[1]) cursor_valid_reg <= wdata_hold[8];
                    end
                    REG_NOTE_COMMIT, REG_VIEW_COMMIT: begin
                        if (wstrb_hold[0] && wdata_hold[0] && write_known &&
                            (request_toggle == ack_sync_2)) begin
                            transaction_is_note <= (awaddr_hold == REG_NOTE_COMMIT);
                            transaction_slot <= slot_reg;
                            transaction_x <= x_reg;
                            transaction_y <= y_reg;
                            transaction_cursor <= cursor_reg;
                            transaction_cursor_valid <= cursor_valid_reg;
                            request_toggle <= ~request_toggle;
                            s_axi_bresp <= 2'b00;
                        end else if (wstrb_hold[0] && wdata_hold[0]) begin
                            s_axi_bresp <= 2'b10;
                        end
                    end
                    default: begin end
                endcase
                if (s_axi_bvalid && s_axi_bready)
                    s_axi_bvalid <= 1'b0;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid <= 1'b1;
                s_axi_rdata <= read_mux;
                s_axi_rresp <= (s_axi_araddr == REG_STATUS || s_axi_araddr == REG_SLOT ||
                                s_axi_araddr == REG_X || s_axi_araddr == REG_Y ||
                                s_axi_araddr == REG_CURSOR) ? 2'b00 : 2'b10;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end

    always_ff @(posedge pixel_clk or negedge pixel_resetn) begin
        if (!pixel_resetn) begin
            request_sync_1 <= 1'b0;
            request_sync_2 <= 1'b0;
            request_seen <= 1'b0;
            ack_toggle <= 1'b0;
            pending_pix <= 1'b0;
            transaction_is_note_pix <= 1'b0;
            transaction_slot_pix <= '0;
            transaction_x_pix <= '0;
            transaction_y_pix <= '0;
            transaction_cursor_pix <= '0;
            transaction_cursor_valid_pix <= 1'b0;
        end else begin
            request_sync_1 <= request_toggle;
            request_sync_2 <= request_sync_1;
            if (pending_pix) begin
                pending_pix <= 1'b0;
                ack_toggle <= request_seen;
            end else if (request_sync_2 != request_seen) begin
                request_seen <= request_sync_2;
                transaction_is_note_pix <= transaction_is_note;
                transaction_slot_pix <= transaction_slot;
                transaction_x_pix <= transaction_x;
                transaction_y_pix <= transaction_y;
                transaction_cursor_pix <= transaction_cursor;
                transaction_cursor_valid_pix <= transaction_cursor_valid;
                pending_pix <= 1'b1;
            end
        end
    end
endmodule
