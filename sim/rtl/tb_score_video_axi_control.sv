`timescale 1ns/1ps

module tb_score_video_axi_control;
    logic axi_clk = 1'b0;
    logic pixel_clk = 1'b0;
    logic resetn = 1'b0;
    logic [11:0] awaddr = 0;
    logic awvalid = 0;
    logic awready;
    logic [31:0] wdata = 0;
    logic [3:0] wstrb = 4'hF;
    logic wvalid = 0;
    logic wready;
    logic [1:0] bresp;
    logic bvalid;
    logic bready = 0;
    logic [11:0] araddr = 0;
    logic arvalid = 0;
    logic arready;
    logic [31:0] rdata;
    logic [1:0] rresp;
    logic rvalid;
    logic rready = 0;
    logic note_we_pix;
    logic [4:0] note_index_pix;
    logic [10:0] note_x_pix;
    logic [9:0] note_y_pix;
    logic [4:0] current_note_index_pix;
    logic current_note_valid_pix;
    logic [31:0] status;

    always #5 axi_clk = ~axi_clk;
    always #20.833 pixel_clk = ~pixel_clk;

    score_video_axi_control dut (
        .s_axi_aclk(axi_clk), .s_axi_aresetn(resetn),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
        .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
        .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
        .pixel_clk(pixel_clk), .pixel_resetn(resetn),
        .note_we_pix, .note_index_pix, .note_x_pix, .note_y_pix,
        .current_note_index_pix, .current_note_valid_pix
    );

    task automatic write_reg(input logic [11:0] address, input logic [31:0] data);
        begin
            @(negedge axi_clk);
            awaddr = address;
            awvalid = 1'b1;
            while (!awready) @(negedge axi_clk);
            @(negedge axi_clk);
            awvalid = 1'b0;
            wdata = data;
            wvalid = 1'b1;
            while (!wready) @(negedge axi_clk);
            @(negedge axi_clk);
            wvalid = 1'b0;
            while (!bvalid) @(negedge axi_clk);
            if (bresp != 2'b00) $fatal(1, "AXI write returned an error at %h", address);
            bready = 1'b1;
            @(negedge axi_clk);
            bready = 1'b0;
        end
    endtask

    task automatic read_reg(input logic [11:0] address, output logic [31:0] data);
        begin
            @(negedge axi_clk);
            araddr = address;
            arvalid = 1'b1;
            while (!arready) @(negedge axi_clk);
            @(negedge axi_clk);
            arvalid = 1'b0;
            while (!rvalid) @(negedge axi_clk);
            if (rresp != 2'b00) $fatal(1, "AXI read returned an error at %h", address);
            data = rdata;
            rready = 1'b1;
            @(negedge axi_clk);
            rready = 1'b0;
        end
    endtask

    initial begin
        repeat (4) @(negedge axi_clk);
        resetn = 1'b1;
        write_reg(6'h04, 32'd3);
        write_reg(6'h08, 32'd123);
        write_reg(6'h0C, 32'd150);
        write_reg(6'h10, 32'h0000_0103); // active slot 3
        write_reg(6'h14, 32'd1);         // commit note + cursor snapshot

        wait (note_we_pix === 1'b1);
        #1;
        if (note_index_pix != 3 || note_x_pix != 123 || note_y_pix != 150)
            $fatal(1, "pixel-domain note payload mismatch");
        if (current_note_index_pix != 3 || !current_note_valid_pix)
            $fatal(1, "pixel-domain cursor payload mismatch");
        wait (note_we_pix === 1'b0);
        repeat (3) @(posedge axi_clk);
        read_reg(6'h00, status);
        if (status[0]) $fatal(1, "AXI busy status did not clear after note commit");

        write_reg(6'h10, 32'h0000_0107); // active slot 7
        write_reg(6'h18, 32'd1);         // commit cursor only
        wait (current_note_index_pix == 7);
        if (!current_note_valid_pix)
            $fatal(1, "cursor-valid flag was lost across the clock-domain crossing");
        $display("PASS: AXI score registers and 100-to-24 MHz update handshake");
        $finish;
    end

    initial begin
        #100_000;
        $fatal(1, "timeout waiting for score display AXI transactions");
    end
endmodule
